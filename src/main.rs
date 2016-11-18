#![feature(proc_macro)]

extern crate bincode as bc;
extern crate clap;
extern crate matrixmultiply as mmul;
extern crate nalgebra as na;
extern crate rand;
extern crate serde;
extern crate serde_json as sj;
#[macro_use] extern crate serde_derive;
extern crate image as img;

mod nn;
mod program_args;

use clap::ArgMatches;

fn main() {
  let args: clap::ArgMatches = program_args::get();
  match args.subcommand_name() {
    Some("train") => train(args.subcommand_matches("train").unwrap()),
    Some("test") => test(args.subcommand_matches("test").unwrap()),
    _ => {},
  }
}

fn train<'a>(args: &ArgMatches<'a>) {
  use nn::*;
  use rand::{Rng, SeedableRng};

  let conf = {
    use std::fs::File;
    match File::open(args.value_of("config").unwrap()) {
      Ok(file) => sj::from_reader(file).unwrap(),
      Err(_) => TrainConfig {
        learning_rate: 1.0,
        momentum_rate: 0.0,
        validation_ratio: 0.2,
        sequential_validation_failures_required: 5,
        max_epochs: Some(1000),
      },
    }
  };

  let mut train_data = Vec::new();

  for maybe_entry in std::fs::read_dir("./data").unwrap() {
    let entry = maybe_entry.unwrap();
    let label = match entry.file_name().into_string().unwrap().chars().next().unwrap() {
      '0' => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '1' => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '2' => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '3' => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '4' => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '5' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      '6' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      '7' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      '8' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      '9' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      _ => panic!("excuse me"),
    };
    let data = {
      use std::fs::File;
      use std::io::BufReader;

      let file = BufReader::new(File::open(entry.path()).unwrap());
      let pngdata: img::DynamicImage = img::load(file, img::ImageFormat::PNG).unwrap();
      pngdata.to_luma().pixels().map(|&p| 1.0 - p.data[0] as f32 / 255.0).collect()
    };
    train_data.push((data, label));
  }


  let mut net = Network::from_definition(vec![10*7, 5*7, 10], vec![2.0, 4.0], ActivationFunction::Sigmoid);
  let mut rng: rand::XorShiftRng = rand::XorShiftRng::from_seed(rand::random());
  rng.shuffle(&mut train_data);
  net.assign_random_weights(&mut rng);
  net.train(train_data, &conf);

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

  let mut train_data = Vec::new();
  let mut train_names = Vec::new();

  for maybe_entry in std::fs::read_dir("./data").unwrap() {
    let entry: std::fs::DirEntry = maybe_entry.unwrap();
    train_names.push(entry.file_name().into_string().unwrap());
    let label = match entry.file_name().into_string().unwrap().chars().next().unwrap() {
      '0' => vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '1' => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '2' => vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '3' => vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '4' => vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      '5' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
      '6' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
      '7' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
      '8' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      '9' => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      _ => panic!("excuse me"),
    };
    let data = {
      use std::fs::File;
      use std::io::BufReader;

      let file = BufReader::new(File::open(entry.path()).unwrap());
      let pngdata: img::DynamicImage = img::load(file, img::ImageFormat::PNG).unwrap();
      pngdata.to_luma().pixels().map(|&p| 1.0 - p.data[0] as f32 / 255.0).collect()
    };
    train_data.push((data, label));
  }

  let net: Network = {
    use bc::serde as bcs;
    use std::fs::File;
    use std::io::BufReader;

    let mut file = BufReader::new(File::open(args.value_of("model").unwrap()).unwrap());
    bcs::deserialize_from(&mut file, bc::SizeLimit::Infinite).unwrap()
  };

  let successful_predictions = train_data.iter().filter(|&&(ref example, ref label): &&(Vec<f32>, Vec<f32>)| {
    let output = net.eval(na::DVector::from_slice(example.len(), &example[..]));
    let output_th: Vec<f32> = output.iter().map(|&x| if x < 0.5 { 0.0 } else { 1.0 }).collect();
    output_th.iter().zip(label).all(|(&out, &lbl)| out == lbl)
  }).count();

  for (it, case) in train_data.iter().enumerate() {
    let output = net.eval(na::DVector::from_slice(case.0.len(), &case.0[..]));
    let output_th: Vec<f32> = output.iter().map(|&x| if x < 0.5 { 0.0 } else { 1.0 }).collect();
    if !output_th.iter().zip(&case.1).all(|(&out, &lbl)| out == lbl) {
      println!("misprediction: {} as {:?}", train_names[it], output_th);
    }
  }
  println!("{} / {}", successful_predictions, train_data.len());
}