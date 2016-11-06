#![feature(proc_macro)]

extern crate bincode as bc;
extern crate clap;
extern crate matrixmultiply as mmul;
extern crate nalgebra as na;
extern crate rand;
extern crate serde;
extern crate serde_json as sj;
#[macro_use] extern crate serde_derive;

mod nn;
mod program_args;

fn main() {
  // let args = program_args::get();
  use nn::*;
  use rand::{XorShiftRng, SeedableRng};

  let mut net = Network::from_definition(vec![3, 1], vec![1.0, 1.0]);

  let mut rng = XorShiftRng::from_seed(rand::random());
  let train_data = (0..100).map(|_| {
    let a = rand::random::<bool>();
    let b = rand::random::<bool>();
    (
      vec![
        if a { 1.0 } else { 0.0 },
        if b { 1.0 } else { 0.0 },
        1.0,
      ],
      vec![
        if a != b { 1.0 } else { 0.0 },
      ]
    )
  }).collect::<Vec<_>>();
  let validation_data = (0..20).map(|_| {
    let a = rand::random::<bool>();
    let b = rand::random::<bool>();
    (
      vec![
        if a { 1.0 } else { 0.0 },
        if b { 1.0 } else { 0.0 },
        1.0,
      ],
      vec![
        if a && b { 1.0 } else { 0.0 },
      ]
    )
  }).collect::<Vec<_>>();

  net.assign_random_weights(&mut rng);
  net.train(train_data, validation_data, &TrainConfig {
    learning_rate: 0.05,
    momentum_rate: 0.0,
    validation_ratio: 0.0,
    sequential_validation_failures_required: 5,
    max_epochs: Some(1000),
  });

  // net.weights[1] = na::DMatrix::from_column_vector(3, 1, &[0.25531432, 0.01867515, -0.26619542]);

  for it in (0..51isize).map(|x| x as f32 / 50.0) {
    for jt in (0..51isize).map(|x| x as f32 / 50.0) {
      print!("{}", if net.eval(na::DVector::from_slice(3, &[it, jt, 1.0]))[0] > 0.5 {'#'} else {'.'});
    }
    print!("\n");
  }
}