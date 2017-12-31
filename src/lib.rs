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
extern crate rayon;
extern crate ctrlc;

pub mod nn;
pub mod mnist;
pub mod program_args;

pub use nn::*;