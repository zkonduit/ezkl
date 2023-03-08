use chrono::Local;
use colored::*;
use colored_json::prelude::*;
use env_logger::Builder;
use ezkl::commands::Cli;
use ezkl::execute::run;
use log::{error, info, Level, LevelFilter, Record};
use rand::seq::SliceRandom;
use std::env;
use std::error::Error;
use std::fmt::Formatter;
use std::io::Write;

#[allow(dead_code)]
pub fn level_color(level: &log::Level, msg: &str) -> String {
    match level {
        Level::Error => msg.red(),
        Level::Warn => msg.yellow(),
        Level::Info => msg.green(),
        Level::Debug => msg.green(),
        Level::Trace => msg.magenta(),
    }
    .bold()
    .to_string()
}

fn level_token(level: &Level) -> &str {
    match *level {
        Level::Error => "E",
        Level::Warn => "W",
        Level::Info => "*",
        Level::Debug => "D",
        Level::Trace => "T",
    }
}

fn prefix_token(level: &Level) -> String {
    format!(
        "{}{}{}",
        "[".blue().bold(),
        level_color(level, level_token(level)),
        "]".blue().bold()
    )
}

pub fn format(buf: &mut Formatter, record: &Record<'_>) -> Result<(), std::fmt::Error> {
    let sep = format!("\n{} ", " | ".white().bold());
    writeln!(
        buf,
        "{} {}",
        prefix_token(&record.level()),
        format!("{}", record.args()).replace("\n", &sep),
    )
}

pub fn init_logger() {
    let mut builder = Builder::new();
    builder.format(|buf, record| {
        writeln!(
            buf,
            "{} [{}, {}] - {}",
            prefix_token(&record.level()),
            Local::now().format("%Y-%m-%dT%H:%M:%S"),
            record.metadata().target(),
            format!("{}", record.args()).replace("\n", &format!("\n{} ", " | ".white().bold())),
        )
    });
    builder.target(env_logger::Target::Stdout);
    builder.filter(None, LevelFilter::Info);
    if env::var("RUST_LOG").is_ok() {
        builder.parse_filters(&env::var("RUST_LOG").unwrap());
    }
    builder.init();
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::create().unwrap();
    init_logger();
    banner();
    info!("command: \n {}", &args.as_json()?.to_colored_json_auto()?);
    let res = run(args).await;
    match &res {
        Ok(_) => info!("succeeded"),
        Err(e) => error!("failed: {}", e),
    };
    res
}

fn banner() {
    let ell: Vec<&str> = vec![
        "for Neural Networks",
        "Linear Algebra",
        "for Layers",
        "for the Laconic",
        "Learning",
        "for Liberty",
        "for the Lyrical",
    ];
    info!(
        "{}",
        format!(
            "

        ███████╗███████╗██╗  ██╗██╗
        ██╔════╝╚══███╔╝██║ ██╔╝██║
        █████╗    ███╔╝ █████╔╝ ██║
        ██╔══╝   ███╔╝  ██╔═██╗ ██║
        ███████╗███████╗██║  ██╗███████╗
        ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝

        -----------------------------------------------------------
        Easy Zero Knowledge {}.
        -----------------------------------------------------------
        
        ",
            ell.choose(&mut rand::thread_rng()).unwrap()
        )
    );
}
