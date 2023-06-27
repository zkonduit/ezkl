use clap::Parser;
use colored_json::ToColoredJson;
use ezkl_lib::commands::Cli;
use ezkl_lib::execute::run;
use ezkl_lib::logger::init_logger;
use log::{error, info};
use rand::prelude::SliceRandom;
use std::error::Error;

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
