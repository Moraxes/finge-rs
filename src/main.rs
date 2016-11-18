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
  use nn::*;
  use rand::{Rng, XorShiftRng, SeedableRng};

  let mut net = Network::from_definition(vec![2, 1], vec![1.0]);

  let mut rng: XorShiftRng = XorShiftRng::from_seed(rand::random());
  let train_data = (0..100).map(|_| {
    let a = rng.gen();
    let b = rng.gen();
    (
      vec![
        if a { 1.0 } else { 0.0 },
        if b { 1.0 } else { 0.0 },
      ],
      vec![
        if a && b { 1.0 } else { 0.0 },
      ]
    )
  }).collect::<Vec<_>>();
  let validation_data = (0..20).map(|_| {
    let a = rng.gen();
    let b = rng.gen();
    (
      vec![
        if a { 1.0 } else { 0.0 },
        if b { 1.0 } else { 0.0 },
      ],
      vec![
        if a && b { 1.0 } else { 0.0 },
      ]
    )
  }).collect::<Vec<_>>();

  net.assign_random_weights(&mut rng);
  net.train(train_data, validation_data, &TrainConfig {
    learning_rate: 1.0,
    momentum_rate: 0.0,
    validation_ratio: 0.0,
    sequential_validation_failures_required: 100,
    max_epochs: Some(100),
  });

  let demo_data = vec![
    (vec![0.0, 0.0], vec![0.0]),
    (vec![0.0, 1.0], vec![0.0]),
    (vec![1.0, 0.0], vec![0.0]),
    (vec![1.0, 1.0], vec![1.0]),
  ];
  for (ex, ta) in demo_data {
    println!("{:?} -> {:?} (target: {:?})", ex, net.eval(na::DVector::from_slice(2, &ex)), ta);
  }
}
