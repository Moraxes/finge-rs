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
        learning_rate: 0.1,
        momentum_rate: None,
        validation_ratio: 0.2,
        sequential_validation_failures_required: 5,
        max_epochs: Some(1000),
      },
    }
  };

  let mut train_data = Vec::new();

  for maybe_entry in std::fs::read_dir(args.value_of("data_dir").unwrap()).unwrap() {
    let entry = maybe_entry.unwrap();
    let label = match entry.file_name().into_string().unwrap().chars().next().unwrap() {
      '0' => 0,
      '1' => 1,
      '2' => 2,
      '3' => 3,
      '4' => 4,
      '5' => 5,
      '6' => 6,
      '7' => 7,
      '8' => 8,
      '9' => 9,
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
  rng.shuffle(&mut train_data);
  net.assign_random_weights(&mut rng);
  net.train(train_data, &conf, &mut rng);

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

  for maybe_entry in std::fs::read_dir(args.value_of("data_dir").unwrap()).unwrap() {
    let entry: std::fs::DirEntry = maybe_entry.unwrap();
    train_names.push(entry.file_name().into_string().unwrap());
    let label = match entry.file_name().into_string().unwrap().chars().next().unwrap() {
      '0' => 0,
      '1' => 1,
      '2' => 2,
      '3' => 3,
      '4' => 4,
      '5' => 5,
      '6' => 6,
      '7' => 7,
      '8' => 8,
      '9' => 9,
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

  let successful_predictions = train_data.iter().filter(|&&(ref example, label): &&(Vec<f32>, usize)| {
    let output = net.eval(na::DVector::from_slice(example.len(), &example[..]));
    let output_th: Vec<f32> = output.iter().map(|&x| if x < 0.5 { 0.0 } else { 1.0 }).collect();
    output_th.iter().zip((0..10).map(|x| if x == label { 1.0 } else { 0.0 })).all(|(&out, lbl)| out == lbl)
  }).count();

  for (it, case) in train_data.iter().enumerate() {
    let output = net.eval(na::DVector::from_slice(case.0.len(), &case.0[..]));
    let output_th: Vec<f32> = output.iter().map(|&x| if x < 0.5 { 0.0 } else { 1.0 }).collect();
    if !output_th.iter().zip((0..10).map(|x| if x == case.1 { 1.0 } else { 0.0 })).all(|(&out, lbl)| out == lbl) {
      println!("misprediction: {} as {:?}", train_names[it], output_th);
    }
  }
  println!("{} / {}", successful_predictions, train_data.len());
}