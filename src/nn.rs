use ::rand;
use ::na;
use na::{DMatrix, DVector, Norm, IterableMut};

#[derive(Clone)]
pub struct Network {
  pub layer_sizes: Vec<usize>,
  pub activation_coeffs: Vec<f32>,
  pub weights: Vec<DMatrix<f32>>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TrainConfig {
  pub learning_rate: f32,
  pub momentum_rate: f32,
  pub validation_ratio: f32,
  pub sequential_validation_failures_required: usize,
  pub max_epochs: Option<usize>,
}

pub type TrainData = Vec<(Vec<f32>, Vec<f32>)>;

impl Network {
  pub fn from_definition(layer_sizes: Vec<usize>, activation_coeffs: Vec<f32>) -> Network {
    let mut net = Network {
      layer_sizes: layer_sizes.clone(),
      activation_coeffs: activation_coeffs,
      weights: layer_sizes.windows(2).map(|w| DMatrix::new_zeros(w[0], w[1])).collect::<Vec<_>>(),
    };
    net.activation_coeffs.insert(0, 0.0);
    net.weights.insert(0, DMatrix::new_zeros(0, 0));
    net
  }

  pub fn assign_random_weights<R: rand::Rng>(&mut self, rng: &mut R) {
    use rand::distributions::{Normal, IndependentSample};

    let weight_dist = Normal::new(0., 1.);
    for matrix in &mut self.weights {
      for weight in matrix.as_mut_vector() {
        *weight = weight_dist.ind_sample(rng) as f32;
      }
    }
  }

  fn uninitialized_layers(&self) -> Vec<DVector<f32>> {
    self.layer_sizes.iter().map(|&sz| DVector::new_zeros(sz)).collect::<Vec<_>>()
  }

  pub fn train(&mut self, train_data: TrainData, validation_data: TrainData, conf: &TrainConfig) {
    let mut epochs_since_validation_improvement = 0usize;
    let mut epoch = 0usize;
    let mut layers = self.uninitialized_layers();
    let mut delta = self.uninitialized_layers();

    while epochs_since_validation_improvement < conf.sequential_validation_failures_required && conf.max_epochs.map(|max| epoch < max).unwrap_or(true) {
      epoch += 1;
      let mut train_error = 0.0;
      let mut validation_error = ::std::f32::INFINITY;

      for (input, output) in train_data.iter().map(|&(ref ex, ref ta)| (DVector::from_slice(ex.len(), &ex[..]),
                                                                        DVector::from_slice(ta.len(), &ta[..]))) {
        layers[0] = input.clone();
        self.feed_forward(&mut layers);
        let mut out_layer_diff = output - layers.last().unwrap().clone();
        train_error += out_layer_diff.norm_squared() / out_layer_diff.len() as f32;
        self.backpropagate(layers.clone(), out_layer_diff, &mut delta, conf);
        self.update_weights(&layers, &delta, conf);
      }

      train_error /= train_data.len() as f32;
      let new_validation_error = validation_data.iter()
        .map(|&(ref ex, ref ta)| self.validation_error_of(&mut layers, DVector::from_slice(ex.len(), &ex[..]), DVector::from_slice(ta.len(), &ta[..]))).sum::<f32>() / validation_data.len() as f32;
      if new_validation_error >= validation_error {
        epochs_since_validation_improvement += 1;
      } else {
        epochs_since_validation_improvement = 0;
      }
      validation_error = new_validation_error;
      println!("epoch {} - train err: {}, validation err: {}", epoch, train_error, validation_error);
    }
  }

  fn validation_error_of(&self, layers: &mut Vec<DVector<f32>>, example: DVector<f32>, target: DVector<f32>) -> f32 {
    assert_eq!(layers[0].len(), example.len());
    assert_eq!(layers.last().unwrap().len(), target.len());

    self.eval_impl(layers, example);
    
    (target - layers.last().unwrap().clone()).norm_squared() / layers.last().unwrap().len() as f32
  }

  fn eval_impl(&self, layers: &mut Vec<DVector<f32>>, example: DVector<f32>) {
    layers[0] = example;
    self.feed_forward(layers);
  }

  pub fn eval(&self, example: DVector<f32>) -> Vec<f32> {
    use na::Iterable;
    let mut layers = self.uninitialized_layers();
    assert_eq!(layers[0].len(), example.len());
    self.eval_impl(&mut layers, example);
    layers.last().unwrap().iter().cloned().collect()
  }

  fn feed_forward(&self, layers: &mut Vec<DVector<f32>>) {
    use ::mmul;
    use na::Iterable;

    for it in 0..(layers.len() - 1) {
      let last_comp = layers[it].len() - 1;
      layers[it][last_comp] = 1.0;
      let input = layers[it].clone() * self.weights[it + 1].clone();
      assert_eq!(layers[it + 1].len(), input.len());
      layers[it + 1] = input.iter().map(|&net| Network::sigmoid(net, self.activation_coeffs[it + 1])).collect();
    }
  }

  fn backpropagate(&mut self, mut layers: Vec<DVector<f32>>, out_layer_diff: DVector<f32>, delta: &mut Vec<DVector<f32>>, conf: &TrainConfig) {
    use ::mmul;
    use na::{Iterable, Transpose};
    // NOTE(msniegocki): the initial net activation can be reused due
    // the useful derivative propperty of the sigmoid function
    for (layer, coeff) in layers.iter_mut().zip(self.activation_coeffs.iter().skip(1)) {
      for out in layer.iter_mut() {
        *out = Network::sigmoid_prime_from_sigmoid(*out, *coeff);
      }
    }

    *delta.last_mut().unwrap() = out_layer_diff
      .iter()
      .zip(layers.last().unwrap().iter())
      .map(|(e, fz)| e * fz)
      .collect();
    for it in (0..(layers.len() - 1)).rev() {
      let next_delta = &self.weights[it + 1] * &delta[it + 1];
      assert_eq!(next_delta.len(), delta[it].len());
      delta[it] = next_delta.iter().zip(layers[it].iter()).map(|(&d, &x)| d * x).collect();
    }
  }

  fn update_weights(&mut self, layers: &Vec<DVector<f32>>, delta: &Vec<DVector<f32>>, conf: &TrainConfig) {
    use ::mmul;
    use na::Outer;

    for it in 1..layers.len() {
      self.weights[it] -= conf.learning_rate * delta[it-1].outer(&layers[it]);
    }
  }

  fn sigmoid(t: f32, beta: f32) -> f32 {
    1.0 / (1.0 + (-t * beta).exp())
  }

  fn sigmoid_prime(t: f32, beta: f32) -> f32 {
    beta * Network::sigmoid(t, beta) * (1.0 - Network::sigmoid(t, beta))
  }

  fn sigmoid_prime_from_sigmoid(sig: f32, beta: f32) -> f32 {
    beta * sig * (1.0 - sig)
  }

  // pub fn write(&self, filename: &str) {
  //   use ::{std, bc};
  //   use std::error::Error;
  //   use std::fs::File;
  //   use std::io::Write;

  //   let bytes = bc::serde::serialize(self, bc::SizeLimit::Infinite)
  //     .unwrap_or_else(
  //       |err| {
  //         let _ = writeln!(
  //           std::io::stderr(),
  //           "bincode serialization error: {}\n{}",
  //           err.description(),
  //           err.cause().map(Error::description).unwrap_or(""));
  //       Vec::new()
  //     });
    
  //   let mut file: File = match File::create(filename) {
  //     Err(err) =>
  //       panic!(
  //         "failed to create file {}: {}\n{}",
  //         filename,
  //         err.description(),
  //         err.cause().map(Error::description).unwrap_or("")),
  //     Ok(f) => f,
  //   };

  //   file.write_all(&bytes).unwrap_or_else(
  //     |err| panic!(
  //       "failed to write to file {}: {}\n{}",
  //       filename,
  //       err.description(),
  //       err.cause().map(Error::description).unwrap_or("")));
  // }
}