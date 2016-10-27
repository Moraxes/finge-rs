#![feature(proc_macro)]

extern crate bincode as bc;
extern crate clap;
extern crate matrixmultiply as mmul;
extern crate rand;
extern crate serde;
extern crate serde_json as sj;
#[macro_use] extern crate serde_derive;

mod nn;
mod program_args;

fn main() {
  let args = program_args::get();
}
