use clap::{Arg, ArgMatches, SubCommand, App};

pub fn get<'a>() -> ArgMatches<'a> {
  App::new("finge-rs")
    .version("0.1")
    .author("Maciej Śniegocki <m.w.sniegocki@gmail.com")
    .subcommand(SubCommand::with_name("train")
      .arg(Arg::with_name("config")
        .long("config")
        .short("c")
        .takes_value(true)
        .default_value("Fingers.json")
        .help("training configuration file"))
      .arg(Arg::with_name("output")
        .long("output")
        .short("o")
        .takes_value(true)
        .default_value("Model.bc")
        .help("output file for the model"))
      .help("train a model"))
    .subcommand(SubCommand::with_name("test")
      .arg(Arg::with_name("model")
        .long("model")
        .short("m")
        .takes_value(true)
        .default_value("Model.bc")
        .help("model to be evaluated")))
    .get_matches()
}
