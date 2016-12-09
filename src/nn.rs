#![allow(unused_variables)]

use ::*;
use na::{DMatrix, DVector, Norm, IterableMut};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Network {
  pub layer_sizes: Vec<usize>,
  pub activation_coeffs: Vec<f32>,
  pub weights: Vec<DMatrix<f32>>,
  pub biases: Vec<DVector<f32>>,
  pub activation_fn: ActivationFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NetworkDefn {
  layers: Vec<usize>,
  activation_coeffs: Vec<f32>,
  activation_fn: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum ActivationFunction {
  Sigmoid,
  Tanh,
}

impl ActivationFunction {
  pub fn function(&self, x: f32, coeff: f32) -> f32 {
    match self {
      &ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x * coeff).exp()),
      &ActivationFunction::Tanh => (x * coeff).tanh(),
    }
  }

  pub fn derivative(&self, x: f32, coeff: f32) -> f32 {
    match self {
      &ActivationFunction::Sigmoid => coeff * self.function(x, coeff) * (1.0 - self.function(x, coeff)),
      &ActivationFunction::Tanh => coeff / (x * coeff).cosh(),
    }
  }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct TrainConfig {
  pub learning_rate: f32,
  pub momentum_rate: Option<f32>,
  pub validation_ratio: f32,
  pub sequential_validation_failures_required: usize,
  pub max_epochs: Option<usize>,
  pub epoch_log_period: Option<usize>,
  pub batch_size: Option<f64>,
  pub sparsity_param: Option<f32>,
  pub sparsity_weight: Option<f32>,
  pub regularization_param: f32,
  pub rayon_weight: f64,
}

pub type TrainData = Vec<Vec<f32>>;

impl Network {
  pub fn from_definition(defn: &NetworkDefn) -> Network {
    let mut net = Network {
      layer_sizes: defn.layers.clone(),
      activation_coeffs: defn.activation_coeffs.clone(),
      weights: defn.layers.windows(2).map(|w| DMatrix::new_zeros(w[0], w[1])).collect::<Vec<_>>(),
      biases: defn.layers.iter().map(|&s| DVector::new_zeros(s)).collect::<Vec<_>>(),
      activation_fn: match defn.activation_fn.as_str() {
        "sigmoid" => ActivationFunction::Sigmoid,
        "tanh" => ActivationFunction::Tanh,
        _ => panic!("unrecognized activation function: {}", defn.activation_fn),
      },
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

  fn weight_sum(mut delta1: Vec<DMatrix<f32>>, delta2: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
    for (dw1, dw2) in delta1.iter_mut().zip(delta2.iter().cloned()) {
      *dw1 += dw2;
    }
    delta1
  }

  fn bias_sum(mut bias1: Vec<DVector<f32>>, bias2: Vec<DVector<f32>>) -> Vec<DVector<f32>> {
    for (dw1, dw2) in bias1.iter_mut().zip(bias2.iter().cloned()) {
      *dw1 += dw2;
    }
    bias1
  }

  pub fn split_data_sequences<R: ::rand::Rng>(rng: &mut R, all_data: TrainData, conf: &TrainConfig) -> (TrainData, TrainData) {
    use ::rand::sample;

    let amt = (conf.validation_ratio * all_data.len() as f32) as usize;
    let validation_idx = sample(rng, 0..all_data.len(), amt);

    let mut train_data = Vec::with_capacity(all_data.len() - amt);
    let mut val_data = Vec::with_capacity(amt);

    for (it, ex) in all_data.into_iter().enumerate() {
      if validation_idx.contains(&it) {
        &mut val_data
      } else {
        &mut train_data
      }.push(ex);
    }

    (train_data, val_data)
  }

  fn cost(&mut self, output_error: f32, avg_activations: &[f32], examples: usize, conf: &TrainConfig) -> f32 {
    let rho = conf.sparsity_param.unwrap_or(1.0);
    let beta = conf.sparsity_weight.unwrap_or(0.0);
    
    output_error +
    if conf.sparsity_weight.is_some() { beta * avg_activations.iter().map(|rho_j| rho * (rho / rho_j).ln() + (1.0 - rho) * ((1.0 - rho) / (1.0 - rho_j)).ln()).sum::<f32>() } else { 0.0 } +
    conf.regularization_param * self.weights.iter().map(|mat| mat.as_vector().iter().map(|w| w*w).sum::<f32>() / examples as f32).sum::<f32>() / self.weights.len() as f32
  }

  pub fn train<R: ::rand::Rng>(&mut self, all_data: TrainData, conf: &TrainConfig, rng: &mut R, learning: Arc<AtomicBool>) {
    use rayon::prelude::*;

    let mut epochs_since_validation_improvement = 0usize;
    let mut epoch = 0usize;
    let mut last_weight_update_sum = self.zero_weights();
    let mut last_bias_update_sum = self.zero_layers();
    let mut best_known_net = self.clone();

    let mut validation_cost = ::std::f32::INFINITY;

    let (train_data, validation_data) = Network::split_data_sequences(rng, all_data, conf);
    let batch_size = (conf.batch_size.unwrap_or(1.0) * train_data.len() as f64) as usize;

    while learning.load(Ordering::SeqCst) &&
        epochs_since_validation_improvement < conf.sequential_validation_failures_required &&
        conf.max_epochs.map(|max| epoch < max).unwrap_or(true) {
      epoch += 1;
      let batch_indices = ::rand::sample(rng, 0..train_data.len(), batch_size);

      let average_activations = if conf.sparsity_weight.is_some() {
        train_data.par_iter()
          .map(|ref ex| DVector::from_slice(ex.len(), &ex[..]))
          .map(|mut input| {
            input *= &self.weights[1];
            for x in input.iter_mut() {
              *x = self.activation_fn.function(*x, self.activation_coeffs[1]);
            }
            input
        })
        .reduce(|| DVector::new_zeros(self.layer_sizes[1]), |a, b| a + b)
        / train_data.len() as f32
      } else {
        DVector::new_zeros(self.layer_sizes[1])
      };

      let (weight_update_sum, bias_update_sum, mut train_error) = batch_indices.par_iter()
        .weight(conf.rayon_weight)
        .map(|&idx| &train_data[idx])
        .map(|ref ex| DVector::from_slice(ex.len(), &ex[..]))
        .map(|input| {
          let mut layers = self.zero_layers();
          *layers.get_mut(0).unwrap() = input.clone();
          let mut layer_inputs = self.zero_layers();
          self.feed_forward(&mut layers, &mut layer_inputs);
          let out_layer_diff = layers.last().unwrap().clone() - input;
          let train_error = out_layer_diff.norm_squared() / out_layer_diff.len() as f32;
          let residual_errors = self.backpropagate(layer_inputs.clone(), out_layer_diff, &average_activations, conf);
          let updates = self.compute_weight_update(&layers, residual_errors, conf);
          (updates.0, updates.1, train_error)
        })
        .reduce(|| (self.zero_weights(), self.zero_layers(), 0.0),
          |(a_w, a_b, a_err), (b_w, b_b, b_err)| (Network::weight_sum(a_w, b_w), Network::bias_sum(a_b, b_b), a_err + b_err));

      train_error /= batch_indices.len() as f32;

      self.update_weights(&weight_update_sum, &bias_update_sum, &last_weight_update_sum, &last_bias_update_sum, batch_size, conf);

      let train_cost = self.cost(train_error, &average_activations[..], batch_indices.len(), conf);

      let average_activations_val = if conf.sparsity_weight.is_some() {
        validation_data.par_iter()
          .map(|ref ex| DVector::from_slice(ex.len(), &ex[..]))
          .map(|mut input| {
            input *= &self.weights[1];
            for x in input.iter_mut() {
              *x = self.activation_fn.function(*x, self.activation_coeffs[1]);
            }
            input
        })
        .reduce(|| DVector::new_zeros(self.layer_sizes[1]), |a, b| a + b)
        / validation_data.len() as f32
      } else {
        DVector::new_zeros(self.layer_sizes[1])
      };

      let validation_error = validation_data.par_iter()
        .weight(conf.rayon_weight)
        .map(|ref ex| {
          let mut layers = self.zero_layers();
          self.validation_error_of(&mut layers, DVector::from_slice(ex.len(), &ex[..]))
        })
        .sum()
        / validation_data.len() as f32;
      let new_validation_cost = self.cost(validation_error, &average_activations_val[..], validation_data.len(), conf);

      if new_validation_cost < validation_cost {
        epochs_since_validation_improvement = 0;
        best_known_net = self.clone();
        validation_cost = new_validation_cost;
      } else {
        epochs_since_validation_improvement += 1;
      }

      if epoch % conf.epoch_log_period.unwrap_or(10) == 0 {
        println!("#{} - train err: {}, val err: {} (last best: {}, stability: {})", epoch, train_cost, new_validation_cost, validation_cost, epochs_since_validation_improvement);
      }

      if conf.momentum_rate.is_some() {
        last_weight_update_sum = weight_update_sum;
        last_bias_update_sum = bias_update_sum;
      }
    }
    *self = best_known_net;
  }

  fn compute_weight_update(&self, layers: &[DVector<f32>], delta: Vec<DVector<f32>>, conf: &TrainConfig) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
    use na::Outer;

    let mut weight_update = self.zero_weights();
    let mut bias_update = self.zero_layers();

    for it in 1..layers.len() {
      let correction = layers[it-1].outer(&delta[it]);
      weight_update[it] = correction;
      bias_update[it] = delta[it].clone();
    }

    (weight_update, bias_update)
  }

  fn validation_error_of(&self, layers: &mut Vec<DVector<f32>>, example: DVector<f32>) -> f32 {
    debug_assert_eq!(layers[0].len(), example.len());

    self.eval_impl(layers, example.clone());
    
    (example - layers.last().unwrap().clone()).norm_squared() / layers.last().unwrap().len() as f32
  }

  fn eval_impl(&self, layers: &mut Vec<DVector<f32>>, example: DVector<f32>) {
    layers[0] = example;
    let mut _li = self.zero_layers();
    self.feed_forward(layers, &mut _li);
  }

  pub fn eval(&self, example: DVector<f32>) -> Vec<f32> {
    use na::Iterable;
    let mut layers = self.zero_layers();
    assert_eq!(layers[0].len(), example.len());
    self.eval_impl(&mut layers, example);
    layers.last().unwrap().iter().cloned().collect()
  }

  fn feed_forward(&self, layers: &mut Vec<DVector<f32>>, layer_inputs: &mut Vec<DVector<f32>>) {
    use na::Iterable;

    for it in 0..(layers.len() - 1) {
      let input = {
        let mut clone = layers[it].clone();
        clone *= &self.weights[it + 1];
        clone
      };
      debug_assert_eq!(layers[it + 1].len(), input.len());
      debug_assert_eq!(layers[it + 1].len(), self.biases[it + 1].len());
      // println!("layer_inputs is {}, biases is {}", layer_inputs.len(), self.biases.len());
      layer_inputs[it + 1] = input.iter().zip(self.biases[it + 1].iter()).map(|(&net, &b)| net + b).collect(); 
      layers[it + 1] = layer_inputs[it + 1].iter().map(|&inp| self.activation_fn.function(inp, self.activation_coeffs[it + 1])).collect();
    }
  }

  fn backpropagate(&self, mut layers: Vec<DVector<f32>>, out_layer_diff: DVector<f32>, avg_activations: &DVector<f32>, conf: &TrainConfig) -> Vec<DVector<f32>> {
    use na::Iterable;

    for (layer, coeff) in layers.iter_mut().zip(&self.activation_coeffs) {
      for out in layer.iter_mut() {
        *out = self.activation_fn.derivative(*out, *coeff);
      }
    }

    let beta = conf.sparsity_weight.unwrap_or(0.0);
    let rho = conf.sparsity_param.unwrap_or(1.0);

    let mut delta = self.zero_layers();

    *delta.last_mut().unwrap() = out_layer_diff
      .iter()
      .zip(layers.last().unwrap().iter())
      .map(|(e, fz)| e * fz)
      .collect();
    for it in (0..(layers.len() - 1)).rev() {
      let next_delta: DVector<f32> = &self.weights[it + 1] * &delta[it + 1];
      debug_assert_eq!(next_delta.len(), delta[it].len());

      if it == 1 && beta != 0.0 {
        delta[it] = next_delta.iter().zip(avg_activations.iter()).map(|(&x, rho_i)| x + beta * (- rho / *rho_i + (1.0 - rho) / (1.0 - rho_i))).zip(layers[it].iter()).map(|(d, &x)| d * x).collect();
      } else {
        delta[it] = next_delta.iter().zip(layers[it].iter()).map(|(&d, x)| d * *x).collect();
      }
    }

    delta
  }

  fn update_weights(&mut self, weight_update_sum: &[DMatrix<f32>], bias_update_sum: &[DVector<f32>], last_weight_update_sum: &[DMatrix<f32>], last_bias_update_sum: &[DVector<f32>], examples: usize, conf: &TrainConfig) {
    use na::Iterable;

    for it in 0..self.weights.len() {
      for (w, dw) in self.weights[it].as_mut_vector().iter_mut().zip(weight_update_sum[it].as_vector()) {
        *w *= 1.0 - conf.regularization_param * conf.learning_rate / examples as f32;
        *w -= dw / examples as f32 * conf.learning_rate;
      }
    }

    for it in 0..self.biases.len() {
      for (b, db) in self.biases[it].iter_mut().zip(bias_update_sum[it].iter()) {
        *b -= db / examples as f32 * conf.learning_rate;
      }
    }

    if let Some(momentum) = conf.momentum_rate {
      for it in 0..self.weights.len() {
        for (w, dw) in self.weights[it].as_mut_vector().iter_mut().zip(last_weight_update_sum[it].as_vector()) {
          *w -= dw / examples as f32 * momentum;
        }
      }

      for it in 0..self.biases.len() {
        for (b, db) in self.biases[it].iter_mut().zip(last_bias_update_sum[it].iter()) {
          *b -= db / examples as f32 * momentum;
        }
      }
    }
  }
}