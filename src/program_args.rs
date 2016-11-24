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
      .arg(Arg::with_name("model")
        .long("model")
        .short("m")
        .takes_value(true)
        .help("if defined, retrains given model"))
      .arg(Arg::with_name("net_defn")
        .long("net-defn")
        .short("n")
        .takes_value(true)
        .default_value("Network.json")
        .help("network definition file"))
      .arg(Arg::with_name("output")
        .long("output")
        .short("o")
        .takes_value(true)
        .default_value("Model.bc")
        .help("output file for the model"))
      .arg(Arg::with_name("data_dir")
        .long("data-dir")
        .short("d")
        .takes_value(true)
        .default_value("./data/")
        .help("path to directory with training data"))
      .help("train a model"))
    .subcommand(SubCommand::with_name("test")
      .arg(Arg::with_name("model")
        .long("model")
        .short("m")
        .takes_value(true)
        .default_value("Model.bc")
        .help("model to be evaluated"))
      .arg(Arg::with_name("data_dir")
        .long("data-dir")
        .short("d")
        .takes_value(true)
        .default_value("./test_data/")
        .help("path to directory with test data")))
    .get_matches()
}
