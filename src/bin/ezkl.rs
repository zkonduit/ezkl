use clap::Parser;
use colored::*;
use colored_json::prelude::*;
use env_logger::Builder;
use ezkl_lib::commands::Cli;
use ezkl_lib::execute::run;
use log::{error, info, Level, LevelFilter, Record};
use rand::seq::SliceRandom;
use std::env;
use std::error::Error;
use std::fmt::Formatter;
use std::io::Write;
use std::time::Instant;

#[allow(dead_code)]
pub fn level_color(level: &log::Level, msg: &str) -> String {
    match level {
        Level::Error => msg.red(),
        Level::Warn => msg.yellow(),
        Level::Info => msg.blue(),
        Level::Debug => msg.green(),
        Level::Trace => msg.magenta(),
    }
    .bold()
    .to_string()
}

pub fn level_text_color(level: &log::Level, msg: &str) -> String {
    match level {
        Level::Error => msg.red(),
        Level::Warn => msg.yellow(),
        Level::Info => msg.white(),
        Level::Debug => msg.white(),
        Level::Trace => msg.white(),
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
    let level = record.level();
    writeln!(
        buf,
        "{} {}",
        prefix_token(&level),
        level_color(&level, record.args().as_str().unwrap()).replace('\n', &sep),
    )
}

pub fn init_logger() {
    let start = Instant::now();
    let mut builder = Builder::new();

    builder.format(move |buf, record| {
        writeln!(
            buf,
            "{} [{}s, {}] - {}",
            prefix_token(&record.level()),
            start.elapsed().as_secs(),
            record.metadata().target(),
            level_text_color(&record.level(), &format!("{}", record.args()))
                .replace('\n', &format!("\n{} ", " | ".white().bold()))
        )
    });
    builder.target(env_logger::Target::Stdout);
    builder.filter(None, LevelFilter::Info);
    if env::var("RUST_LOG").is_ok() {
        builder.parse_filters(&env::var("RUST_LOG").unwrap());
    }
    builder.init();
}

#[tokio::main(flavor = "current_thread")]
pub async fn main() -> Result<(), Box<dyn Error>> {
    let args = Cli::parse();
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
