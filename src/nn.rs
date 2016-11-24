#![allow(unused_variables)]
use ::rand;
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
}

pub type TrainData = Vec<(Vec<f32>, usize)>;

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

  pub fn split_data_sequences<R: ::rand::Rng>(rng: &mut R, all_data: TrainData, conf: &TrainConfig) -> (TrainData, TrainData) {
    use std::collections::HashMap;

    let mut buckets: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();

    for (input, label) in all_data {
      if buckets.contains_key(&label) {
        buckets.get_mut(&label).unwrap().push(input);
      } else {
        buckets.insert(label, vec![input]);
      }
    }

    for (input, mut outputs) in &mut buckets {
      rng.shuffle(&mut outputs);
    }

    let mut validation_data: TrainData = Vec::new();
    let mut train_data: TrainData = Vec::new();

    for (label, inputs) in buckets {
      let split_index = (conf.validation_ratio * inputs.len() as f32) as usize;
      for (it, entry) in inputs.into_iter().enumerate() {
        if it < split_index {
          &mut validation_data
        } else {
          &mut train_data
        // }.push((entry, (0..10).map(|x| if x == label { 1.0 } else { 0.0 }).collect::<Vec<f32>>()));
        }.push((entry, label));
      }
    }

    rng.shuffle(&mut validation_data);
    rng.shuffle(&mut train_data);

    (train_data, validation_data)
  }

  pub fn train<R: ::rand::Rng>(&mut self, all_data: TrainData, conf: &TrainConfig, rng: &mut R) {
    use std::iter::FromIterator;

    let mut epochs_since_validation_improvement = 0usize;
    let mut epoch = 0usize;
    let mut layers = self.zero_layers();
    let mut weight_update_sum = self.zero_weights();
    let mut last_weight_update_sum = self.zero_weights();
    let mut bias_update_sum = self.zero_layers();
    let mut last_bias_update_sum = self.zero_layers();
    let mut best_known_net = self.clone();

    let mut validation_error = ::std::f32::INFINITY;

    let (train_data, validation_data) = Network::split_data_sequences(rng, all_data, conf);

    while epochs_since_validation_improvement < conf.sequential_validation_failures_required && conf.max_epochs.map(|max| epoch < max).unwrap_or(true) {
      epoch += 1;
      let mut train_error = 0.0;
      if conf.momentum_rate.is_some() {
        last_weight_update_sum = weight_update_sum;
        last_bias_update_sum = bias_update_sum;
      }
      weight_update_sum = self.zero_weights();
      bias_update_sum = self.zero_layers();

      for (input, output) in train_data.iter()
          .map(|&(ref ex, ta)| (DVector::from_slice(ex.len(), &ex[..]),
                                DVector::from_iter((0..10).map(|x| if x == ta { 1.0 } else { 0.0 })))) {
        *layers.get_mut(0).unwrap() = input.clone();
        let mut layer_inputs = self.zero_layers();
        self.feed_forward(&mut layers, &mut layer_inputs);
        let out_layer_diff = layers.last().unwrap().clone() - output;
        train_error += out_layer_diff.norm_squared() / out_layer_diff.len() as f32;
        let residual_errors = self.backpropagate(layer_inputs.clone(), out_layer_diff, conf);
        let (weight_update, bias_update) = self.compute_weight_update(&layers, residual_errors, conf);
        Network::add_weights(&mut weight_update_sum, weight_update);
        Network::add_biases(&mut bias_update_sum, bias_update);
      }
      self.update_weights(&weight_update_sum, &bias_update_sum, &last_weight_update_sum, &last_bias_update_sum, train_data.len(), conf);

      train_error /= train_data.len() as f32;
      let new_validation_error = validation_data.iter()
        .map(|&(ref ex, ta)| self.validation_error_of(&mut layers, DVector::from_slice(ex.len(), &ex[..]), DVector::from_iter((0..10).map(|x| if x == ta { 1.0 } else { 0.0 })))).sum::<f32>() / validation_data.len() as f32;
      if new_validation_error < validation_error {
        epochs_since_validation_improvement = 0;
        best_known_net = self.clone();
        validation_error = new_validation_error;
      } else {
        epochs_since_validation_improvement += 1;
      }
      // println!("epoch {} - train err: {}, validation err: {} ({} this epoch), stability: {}", epoch, train_error, validation_error, new_validation_error, epochs_since_validation_improvement);

      // {
      //   use std::fmt::Write;

      //   let mut map = String::new();
      //   for it in (0..11isize).map(|x| x as f32 / 10.0) {
      //     for jt in (0..11isize).map(|x| x as f32 / 10.0) {
      //       write!(map, "{}", if self.eval(DVector::from_slice(2, &[it, jt]))[0] > 0.5 {'#'} else {'.'});
      //     }
      //     write!(map, "\n");
      //   }

      //   ::std::io::Write::write_all(&mut ::std::io::stderr(), map.as_bytes());
      //   ::std::thread::sleep_ms(10);
      // }
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

  fn validation_error_of(&self, layers: &mut Vec<DVector<f32>>, example: DVector<f32>, target: DVector<f32>) -> f32 {
    assert_eq!(layers[0].len(), example.len());
    assert_eq!(layers.last().unwrap().len(), target.len());

    self.eval_impl(layers, example);
    
    (target - layers.last().unwrap().clone()).norm_squared() / layers.last().unwrap().len() as f32
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
      let input = layers[it].clone() * self.weights[it + 1].clone();
      assert_eq!(layers[it + 1].len(), input.len());
      assert_eq!(layers[it + 1].len(), self.biases[it + 1].len());
      // println!("layer_inputs is {}, biases is {}", layer_inputs.len(), self.biases.len());
      layer_inputs[it + 1] = input.iter().zip(self.biases[it + 1].iter()).map(|(&net, &b)| net + b).collect(); 
      layers[it + 1] = layer_inputs[it + 1].iter().map(|&inp| self.activation_fn.function(inp, self.activation_coeffs[it + 1])).collect();
    }
  }

  fn backpropagate(&mut self, mut layers: Vec<DVector<f32>>, out_layer_diff: DVector<f32>, conf: &TrainConfig) -> Vec<DVector<f32>> {
    use na::Iterable;

    for (layer, coeff) in layers.iter_mut().zip(&self.activation_coeffs) {
      for out in layer.iter_mut() {
        *out = self.activation_fn.derivative(*out, *coeff);
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

  fn update_weights(&mut self, weight_update_sum: &[DMatrix<f32>], bias_update_sum: &[DVector<f32>], last_weight_update_sum: &[DMatrix<f32>], last_bias_update_sum: &[DVector<f32>], examples: usize, conf: &TrainConfig) {
    use na::Iterable;

    for it in 0..self.weights.len() {
      for (w, dw) in self.weights[it].as_mut_vector().iter_mut().zip(weight_update_sum[it].as_vector()) {
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