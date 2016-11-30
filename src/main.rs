#![feature(proc_macro)]
#![feature(iter_max_by)]

extern crate bincode as bc;
extern crate clap;
extern crate matrixmultiply as mmul;
extern crate nalgebra as na;
extern crate rand;
extern crate serde;
extern crate serde_json as sj;
#[macro_use] extern crate serde_derive;
extern crate image as img;
extern crate byteorder as bo;

mod nn;
mod program_args;
mod mnist;

use clap::ArgMatches;

fn main() {
  let args: clap::ArgMatches = program_args::get();
  match args.subcommand_name() {
    Some("train") => train(args.subcommand_matches("train").unwrap()),
    Some("test") => test(args.subcommand_matches("test").unwrap()),
    Some("dump-features") => dump_features(args.subcommand_matches("dump-features").unwrap()),
    _ => {},
  }
}

fn train<'a>(args: &ArgMatches<'a>) {
  use nn::*;
  use rand::SeedableRng;

  let conf = {
    use std::fs::File;
    match File::open(args.value_of("config").unwrap()) {
      Ok(file) => sj::from_reader(file).unwrap(),
      Err(_) => TrainConfig {
        learning_rate: 0.1,
        momentum_rate: None,
        validation_ratio: 0.2,
        sequential_validation_failures_required: 5,
        max_epochs: Some(1000),
        epoch_log_period: Some(10),
      },
    }
  };

  let images = mnist::load_idx_images("mnist/train-images.idx3-ubyte").unwrap();
  let labels = mnist::load_idx_labels("mnist/train-labels.idx1-ubyte").unwrap();

  let train_data: Vec<(Vec<f32>, usize)> = images.into_iter().zip(labels).collect();

  let mut net = if let Some(model_path) = args.value_of("model") {
    use bc::serde as bcs;
    use std::fs::File;
    use std::io::BufReader;

    let mut file = BufReader::new(File::open(model_path).unwrap());
    bcs::deserialize_from(&mut file, bc::SizeLimit::Infinite).unwrap()
  } else {
    let defn = {
      use std::fs::File;
      match File::open(args.value_of("net_defn").unwrap()) {
        Ok(file) => sj::from_reader(file).unwrap(),
        Err(_) => panic!("no network definition found"),
      }
    };
    Network::from_definition(&defn)
  };
  let mut rng: rand::XorShiftRng = rand::XorShiftRng::from_seed(rand::random());
  net.assign_random_weights(&mut rng);
  for err in net.train(train_data, &conf, &mut rng) {
    println!("{}", err);
  }

  {
    use bc::serde as bcs;
    use std::fs::File;
    use std::io::{Write, BufWriter};

    let bytes = bcs::serialize(&net, bc::SizeLimit::Infinite).unwrap();
    let mut file = BufWriter::new(File::create(args.value_of("output").unwrap()).unwrap());
    file.write(&bytes).unwrap();
    println!("Model written to {}", args.value_of("output").unwrap());
  }
}

fn test<'a>(args: &ArgMatches<'a>) {
  use nn::*;

  let images = mnist::load_idx_images("mnist/train-images.idx3-ubyte").unwrap();
  let labels = mnist::load_idx_labels("mnist/train-labels.idx1-ubyte").unwrap();

  let test_data: Vec<(Vec<f32>, usize)> = images.into_iter().zip(labels).collect();

  let net: Network = {
    use bc::serde as bcs;
    use std::fs::File;
    use std::io::BufReader;

    let mut file = BufReader::new(File::open(args.value_of("model").unwrap()).unwrap());
    bcs::deserialize_from(&mut file, bc::SizeLimit::Infinite).unwrap()
  };

  let successful_predictions = test_data.iter().filter(|&&(ref example, label): &&(Vec<f32>, usize)| {
    use std::cmp::Ordering;
    let output = net.eval(na::DVector::from_slice(example.len(), &example[..]));
    let output_lbl = output.iter().enumerate()
      .max_by(|&(_, &x), &(_, &y)| if x < y { Ordering::Less } else if x > y { Ordering::Greater } else { Ordering::Equal }).unwrap_or((255, &0.0)).0;
    output_lbl == label
  }).count();

  for (it, case) in test_data.iter().enumerate() {
    use std::cmp::Ordering;
    let output = net.eval(na::DVector::from_slice(case.0.len(), &case.0[..]));
    let output_lbl = output.iter().enumerate()
      .max_by(|&(_, &x), &(_, &y)| if x < y { Ordering::Less } else if x > y { Ordering::Greater } else { Ordering::Equal }).unwrap_or((255, &0.0)).0;
    if output_lbl != case.1 {
      println!("misprediction: item {} as {:?}", it, output_lbl);
    }
  }
  let percentage = successful_predictions as f32 / test_data.len() as f32 * 100.0;
  println!("{} / {} ({:.*})", successful_predictions, test_data.len(), 2, percentage);
}

fn dump_features<'a>(args: &ArgMatches<'a>) {
  use std::path::PathBuf;
  use nn::*;

  let net: Network = {
    use bc::serde as bcs;
    use std::fs::File;
    use std::io::BufReader;

    let mut file = BufReader::new(File::open(args.value_of("model").unwrap()).unwrap());
    bcs::deserialize_from(&mut file, bc::SizeLimit::Infinite).unwrap()
  };

  let mut base_pb = PathBuf::new();
  base_pb.push(args.value_of("dir").unwrap());

  for col_it in 0..net.weights[1].ncols() {
    use na::{Iterable, Column};

    let col: na::DVector<f32> = net.weights[1].column(col_it);

    let min = col.iter().fold(std::f32::INFINITY, |acc, &x| if x < acc { x } else { acc });
    let max = col.iter().fold(std::f32::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc });
    
    let bytes = col.iter().map(|x| ((x - min) / (max - min) * 255.0) as u8).collect::<Vec<_>>();
    base_pb.push(format!("feature-0-{:06}.png", col_it));
    img::save_buffer(base_pb.to_str().unwrap(), &bytes[..], 28, 28, img::ColorType::Gray(8)).unwrap();
    base_pb.pop();
  }

  for col_it in 0..net.weights[2].ncols() {
    use na::{Iterable, Column};
    use na::Shape;
    use na::Transpose;

    let col: na::DVector<f32> = net.weights[2].column(col_it);
    println!("len: {}", col.len());
    println!("shape: {:?}", net.weights[1].shape());

    let preimage: na::DVector<f32> = col * net.weights[1].transpose();

    let min = preimage.iter().fold(std::f32::INFINITY, |acc, &x| if x < acc { x } else { acc });
    let max = preimage.iter().fold(std::f32::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc });
    
    let bytes = preimage.iter().map(|x| ((x - min) / (max - min) * 255.0) as u8).collect::<Vec<_>>();
    base_pb.push(format!("feature-1-{:06}.png", col_it));
    img::save_buffer(base_pb.to_str().unwrap(), &bytes[..], 28, 28, img::ColorType::Gray(8)).unwrap();
    base_pb.pop();
  }
}