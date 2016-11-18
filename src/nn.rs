#![allow(unused_variable)]
use ::rand;
use na::{DMatrix, DVector, Norm, IterableMut};

#[derive(Clone, Debug)]
pub struct Network {
  pub layer_sizes: Vec<usize>,
  pub activation_coeffs: Vec<f32>,
  pub weights: Vec<DMatrix<f32>>,
  pub biases: Vec<DVector<f32>>,
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
      biases: layer_sizes.iter().map(|&s| DVector::new_zeros(s)).collect::<Vec<_>>(),
    };
    net.activation_coeffs.insert(0, 0.0);
    net.weights.insert(0, DMatrix::new_zeros(0, 0));
    net
  }

  pub fn assign_random_weights<R: rand::Rng>(&mut self, rng: &mut R) {
    use rand::distributions::{Normal, IndependentSample};

    let dist = Normal::new(0.0, 0.1);
    for matrix in &mut self.weights {
      for weight in matrix.as_mut_vector() {
        *weight = dist.ind_sample(rng) as f32;
      }
    }
    for bias_v in &mut self.biases {
      for b in bias_v.iter_mut() {
        *b = dist.ind_sample(rng) as f32;
      }
    }
  }

  fn zero_layers(&self) -> Vec<DVector<f32>> {
    self.layer_sizes.iter().map(|&sz| DVector::new_zeros(sz)).collect::<Vec<_>>()
  }

  fn zero_weights(&self) -> Vec<DMatrix<f32>> {
    let mut weights = self.layer_sizes.windows(2).map(|w| DMatrix::new_zeros(w[0], w[1])).collect::<Vec<_>>();
    weights.insert(0, DMatrix::new_zeros(0, 0));
    weights
  }

  fn add_weights(delta1: &mut Vec<DMatrix<f32>>, delta2: Vec<DMatrix<f32>>) {
    for (dw1, dw2) in delta1.iter_mut().zip(delta2.iter().cloned()) {
      *dw1 += dw2;
    }
  }

  fn add_biases(bias1: &mut Vec<DVector<f32>>, bias2: Vec<DVector<f32>>) {
    for (dw1, dw2) in bias1.iter_mut().zip(bias2.iter().cloned()) {
      *dw1 += dw2;
    }
  }

  pub fn train(&mut self, train_data: TrainData, validation_data: TrainData, conf: &TrainConfig) {
    let mut epochs_since_validation_improvement = 0usize;
    let mut epoch = 0usize;
    let mut layers = self.zero_layers();
    let mut weight_update_sum = self.zero_weights();
    let mut bias_update_sum = self.zero_layers();
    let mut best_known_net = self.clone();

    let mut validation_error = ::std::f32::INFINITY;
    while epochs_since_validation_improvement < conf.sequential_validation_failures_required && conf.max_epochs.map(|max| epoch < max).unwrap_or(true) {
      epoch += 1;
      let mut train_error = 0.0;
      for wu in &mut weight_update_sum {
        for dw in wu.as_mut_vector() {
          *dw = 0.0;
        }
      }

      for bu in &mut bias_update_sum {
        for db in bu.iter_mut() {
          *db = 0.0;
        }
      }

      println!("\n\n### EPOCH {} ###", epoch);

      for (input, output) in train_data.iter()
          .map(|&(ref ex, ref ta)| (DVector::from_slice(ex.len(), &ex[..]),
                                    DVector::from_slice(ta.len(), &ta[..]))) {
        *layers.get_mut(0).unwrap() = input.clone();
        self.feed_forward(&mut layers);
        let out_layer_diff = layers.last().unwrap().clone() - output;
        train_error += out_layer_diff.norm_squared() / out_layer_diff.len() as f32;
        let residual_errors = self.backpropagate(layers.clone(), out_layer_diff, conf);
        let (weight_update, bias_update) = self.compute_weight_update(&layers, residual_errors, conf);
        Network::add_weights(&mut weight_update_sum, weight_update);
        Network::add_biases(&mut bias_update_sum, bias_update);
      }
      self.update_weights(&weight_update_sum, &bias_update_sum, train_data.len(), conf);

      train_error /= train_data.len() as f32;
      let new_validation_error = validation_data.iter()
        .map(|&(ref ex, ref ta)| self.validation_error_of(&mut layers, DVector::from_slice(ex.len(), &ex[..]), DVector::from_slice(ta.len(), &ta[..]))).sum::<f32>() / validation_data.len() as f32;
      if new_validation_error < validation_error {
        epochs_since_validation_improvement = 0;
        best_known_net = self.clone();
        validation_error = new_validation_error;
      } else {
        epochs_since_validation_improvement += 1;
      }
      println!("epoch {} - train err: {}, validation err: {} ({} this epoch), stability: {}", epoch, train_error, validation_error, new_validation_error, epochs_since_validation_improvement);
    }
    *self = best_known_net;
  }

  fn compute_weight_update(&self, layers: &[DVector<f32>], delta: Vec<DVector<f32>>, conf: &TrainConfig) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
    use na::Outer;

    let mut weight_update = self.zero_weights();
    let mut bias_update = self.zero_layers();

    for it in 1..layers.len() {
      let correction = layers[it-1].outer(&delta[it]);
      weight_update[it] = conf.learning_rate * correction;
      bias_update[it] = conf.learning_rate * delta[it].clone();
    }

    (weight_update, bias_update)
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
    let mut layers = self.zero_layers();
    assert_eq!(layers[0].len(), example.len());
    self.eval_impl(&mut layers, example);
    layers.last().unwrap().iter().cloned().collect()
  }

  fn feed_forward(&self, layers: &mut Vec<DVector<f32>>) {
    use na::Iterable;

    for it in 0..(layers.len() - 1) {
      let input = layers[it].clone() * self.weights[it + 1].clone();
      assert_eq!(layers[it + 1].len(), input.len());
      assert_eq!(layers[it + 1].len(), self.biases[it + 1].len());
      layers[it + 1] = input.iter().zip(self.biases[it + 1].iter()).map(|(&net, &b)| Network::sigmoid(net + b, self.activation_coeffs[it + 1])).collect();
    }
  }

  fn backpropagate(&mut self, mut layers: Vec<DVector<f32>>, out_layer_diff: DVector<f32>, conf: &TrainConfig) -> Vec<DVector<f32>> {
    use na::Iterable;
    for ((layer, coeff), bias_v) in layers.iter_mut().zip(&self.activation_coeffs).zip(&self.biases) {
      for (out, b) in layer.iter_mut().zip(bias_v.iter()) {
        *out = Network::sigmoid_prime_from_sigmoid(*out + b, *coeff);
      }
    }

    let mut delta = self.zero_layers();

    *delta.last_mut().unwrap() = out_layer_diff
      .iter()
      .zip(layers.last().unwrap().iter())
      .map(|(e, fz)| e * fz)
      .collect();
    for it in (0..(layers.len() - 1)).rev() {
      let next_delta: DVector<f32> = &self.weights[it + 1] * &delta[it + 1];
      assert_eq!(next_delta.len(), delta[it].len());
      delta[it] = next_delta.iter().zip(layers[it].iter()).map(|(&d, &x)| d * x).collect();
    }

    delta
  }

  fn update_weights(&mut self, weight_update_sum: &[DMatrix<f32>], bias_update_sum: &[DVector<f32>], examples: usize, conf: &TrainConfig) {
    use na::Iterable;

    for it in 0..self.weights.len() {
      for (w, dw) in self.weights[it].as_mut_vector().iter_mut().zip(weight_update_sum[it].as_vector()) {
        *w -= dw / examples as f32;
      }
    }

    for it in 0..self.biases.len() {
      for (b, db) in self.biases[it].iter_mut().zip(bias_update_sum[it].iter()) {
        *b -= db / examples as f32;
      }
    }
  }

  pub fn sigmoid(t: f32, beta: f32) -> f32 {
    1.0 / (1.0 + (-t * beta).exp())
  }

  pub fn sigmoid_prime(t: f32, beta: f32) -> f32 {
    beta * Network::sigmoid(t, beta) * (1.0 - Network::sigmoid(t, beta))
  }

  fn sigmoid_prime_from_sigmoid(sig: f32, beta: f32) -> f32 {
    beta * sig * (1.0 - sig)
  }
}