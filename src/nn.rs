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

  fn add_delta(delta1: &mut Vec<DVector<f32>>, delta2: Vec<DVector<f32>>) {
    for (dw1, dw2) in delta1.iter_mut().zip(delta2.iter().cloned()) {
      *dw1 += dw2;
    }
  }

  pub fn train(&mut self, train_data: TrainData, validation_data: TrainData, conf: &TrainConfig) {
    let mut epochs_since_validation_improvement = 0usize;
    let mut epoch = 0usize;
    let mut layers = self.uninitialized_layers();
    let mut delta = self.uninitialized_layers();
    let mut best_known_net = self.clone();

    let mut validation_error = ::std::f32::INFINITY;
    while epochs_since_validation_improvement < conf.sequential_validation_failures_required && conf.max_epochs.map(|max| epoch < max).unwrap_or(true) {
      epoch += 1;
      let mut train_error = 0.0;

      println!("### EPOCH {} ###", epoch);

      for (input, output) in train_data.iter()
          .map(|&(ref ex, ref ta)| (DVector::from_slice(ex.len(), &ex[..]),
                                    DVector::from_slice(ta.len(), &ta[..]))) {
        *layers.get_mut(0).unwrap() = input.clone();
        self.feed_forward(&mut layers);
        // println!("output: {:?}", output);
        let out_layer_diff = layers.last().unwrap().clone() - output;
        train_error += out_layer_diff.norm_squared() / out_layer_diff.len() as f32;
        Network::add_delta(&mut delta, self.backpropagate(layers.clone(), out_layer_diff, conf));
      }

      self.update_weights(&layers, &delta, train_data.len(), conf);

      use std::io::Write;

      // writeln!(::std::io::stderr(), "epoch {}", epoch);
      // for (it, ref w) in self.weights.iter().enumerate() {
      //   writeln!(::std::io::stderr(), "w_{} = {:?}", it, w);
      // }

      // let mut map = String::new();
      // for it in (0..51isize).map(|x| x as f32 / 50.0) {
      //   for jt in (0..51isize).map(|x| x as f32 / 50.0) {
      //     write!(map, "{}", if self.eval(DVector::from_slice(3, &[it, jt, 1.0]))[0] > 0.5 {'#'} else {'.'});
      //   }
      //   write!(map, "\n");
      // }

      // ::std::io::Write::write_all(&mut ::std::io::stderr(), map.as_bytes());

      train_error /= train_data.len() as f32;
      let new_validation_error = validation_data.iter()
        .map(|&(ref ex, ref ta)| self.validation_error_of(&mut layers, DVector::from_slice(ex.len(), &ex[..]), DVector::from_slice(ta.len(), &ta[..]))).sum::<f32>() / validation_data.len() as f32;
      if new_validation_error < validation_error {
        epochs_since_validation_improvement = 0;
        best_known_net = self.clone();
      } else {
        epochs_since_validation_improvement += 1;
      }
      validation_error = new_validation_error;
      println!("epoch {} - train err: {}, validation err: {}, stability: {}", epoch, train_error, validation_error, epochs_since_validation_improvement);
    }
    *self = best_known_net;
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

    // println!("==============================");
    // println!("feed_forward begin.");
    // println!("layers:");
    for (it, ref layer) in layers.iter().enumerate() {
      // println!("    #{} = {:?}", it, layer);
    }

    for it in 0..(layers.len() - 1) {
      let last_comp = layers[it].len() - 1;
      layers[it][last_comp] = 1.0;
      let input = layers[it].clone() * self.weights[it + 1].clone();
      assert_eq!(layers[it + 1].len(), input.len());
      layers[it + 1] = input.iter().map(|&net| Network::sigmoid(net, self.activation_coeffs[it + 1])).collect();
    }

    // println!("feed_forward end.");
    // println!("layers:");
    for (it, ref layer) in layers.iter().enumerate() {
      // println!("    #{} = {:?}", it, layer);
    }
    // println!("==============================");
  }

  fn backpropagate(&mut self, mut layers: Vec<DVector<f32>>, out_layer_diff: DVector<f32>, conf: &TrainConfig) -> Vec<DVector<f32>> {
    use ::mmul;
    use na::{Iterable, Transpose};
    // NOTE(msniegocki): the initial net activation can be reused due
    // the useful derivative propperty of the sigmoid function
  
    // println!("==============================");
    // println!("backpropagate begin.");
    // println!("diff: {:?}", out_layer_diff);
    // println!("layers:");
    for (it, ref layer) in layers.iter().enumerate() {
      // println!("    #{} = {:?}", it, layer);
    }

    for (layer, coeff) in layers.iter_mut().zip(self.activation_coeffs.iter().skip(1)) {
      for out in layer.iter_mut() {
        *out = Network::sigmoid_prime(*out, *coeff);
      }
    }

    // println!("layers post-diff:");
    for (it, ref layer) in layers.iter().enumerate() {
      // println!("    #{} = {:?}", it, layer);
    }

    let mut delta = self.uninitialized_layers();

    *delta.last_mut().unwrap() = out_layer_diff
      .iter()
      .zip(layers.last().unwrap().iter())
      .map(|(e, fz)| e * fz)
      .collect();
    for it in (0..(layers.len() - 1)).rev() {
      let next_delta: DVector<f32> = &self.weights[it + 1] * &delta[it + 1];
      assert_eq!(next_delta.len(), delta[it].len());
      // println!("layers[{}] = {:?}", it, layers[it]);
      // println!("next_delta = {:?}", next_delta);

      delta[it] = next_delta.iter().zip(layers[it].iter()).map(|(&d, &x)| d * x).collect();
      // delta[it] = next_delta * layers[it].clone();

      // println!("delta[{}] = {:?}", it, delta[it]);
    }

    // println!("backpropagate end.");
    // println!("delta:");
    for (it, ref d) in delta.iter().enumerate() {
      // println!("    #{} = {:?}", it, d);
    }
    // println!("==============================");

    delta
  }

  fn update_weights(&mut self, layers: &Vec<DVector<f32>>, delta: &Vec<DVector<f32>>, examples: usize, conf: &TrainConfig) {
    use ::mmul;
    use na::Outer;

    // println!("==============================");
    // println!("weight update begin.");
    // println!("weights:");
    for (it, ref w) in self.weights.iter().enumerate() {
      // println!("    #{} = {:?}", it, w);
    }

    for it in 1..layers.len() {
      // println!("weight update: {:?} -= {:?}", self.weights[it], conf.learning_rate * delta[it-1].outer(&layers[it]));
      // println!("(from delta[{}] = {:?}, layers[{}] = {:?})", it-1, delta[it-1], it, layers[it]);
      self.weights[it] += conf.learning_rate / examples as f32 * delta[it-1].outer(&layers[it]);
    }

    // println!("weight update end.");
    // println!("weights:");
    for (it, ref w) in self.weights.iter().enumerate() {
      // println!("    #{} = {:?}", it, w);
    }
    // println!("==============================");
  }

  fn sigmoid(t: f32, beta: f32) -> f32 {
    1.0 / (1.0 + (-t * beta).exp())
  }

  fn sigmoid_prime(t: f32, beta: f32) -> f32 {
    beta * Network::sigmoid(t, beta) * (1.0 - Network::sigmoid(t, beta))
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