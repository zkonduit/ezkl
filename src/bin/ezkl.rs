// ignore file if compiling for wasm

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use mimalloc::MiMalloc;

#[global_allocator]
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use clap::{CommandFactory, Parser};
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use colored_json::ToColoredJson;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use ezkl::commands::Cli;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use ezkl::execute::run;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use ezkl::logger::init_logger;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use log::{error, info};
#[cfg(not(any(target_arch = "wasm32", feature = "no-banner")))]
use rand::prelude::SliceRandom;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
#[cfg(feature = "icicle")]
use std::env;

#[tokio::main(flavor = "current_thread")]
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
pub async fn main() {
    let args = Cli::parse();

    if let Some(generator) = args.generator {
        ezkl::commands::print_completions(generator, &mut Cli::command());
    } else if let Some(command) = args.command {
        init_logger();
        #[cfg(not(any(target_arch = "wasm32", feature = "no-banner")))]
        banner();
        #[cfg(feature = "icicle")]
        if env::var("ENABLE_ICICLE_GPU").is_ok() {
            info!("Running with ICICLE GPU");
        } else {
            info!("Running with CPU");
        }
        info!(
            "command: \n {}",
            &command.as_json().to_colored_json_auto().unwrap()
        );
        let res = run(command).await;
        match &res {
            Ok(_) => {
                info!("succeeded");
            }
            Err(e) => {
                error!("{}", e);
                std::process::exit(1)
            }
        }
    } else {
        init_logger();
        error!("No command provided");
        std::process::exit(1)
    }
}

#[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
pub fn main() {}

#[cfg(not(any(target_arch = "wasm32", feature = "no-banner")))]
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
