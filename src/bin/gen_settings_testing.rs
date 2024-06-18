use std::path::PathBuf;
// use ezkl::graph::GraphCircuit;
use ezkl::graph::{GraphCircuit, GraphSettings, GraphWitness, Model};
// use ezkl::commands::*, EZKLError;
use {ezkl::commands::*, ezkl::EZKLError};
use std::env;
use ezkl::logger;
use ezkl::Commitments;
use ezkl::RunArgs;
use log::{debug, info};
use env_logger;



// Generate a circuit settings file
pub(crate) fn gen_circuit_settings(
    model_path: PathBuf,
    params_output: PathBuf,
    run_args:RunArgs ,
) -> Result<String, EZKLError> {
    let circuit = GraphCircuit::from_run_args(&run_args, &model_path)?;
    let params = circuit.settings();
    params.save(&params_output)?;
    Ok(String::from("Settings generated successfully")) // Return actual output if needed
}

fn main() {
    // Prepare the arguments for the function
    let model_path = PathBuf::from("examples/onnx/simple_cnn/simple_cnn.onnx");
    let params_output = PathBuf::from("examples/onnx/simple_cnn/settings_1.json");
    let default_run_args = RunArgs::default();
    env::set_var("RUST_LOG", "debug");
    // env_logger::init();
    // Initialize the logger
    logger::init_logger();

    // Example debug and info messages
    info!("Starting the application...");
    // Call gen_circuit_settings and handle errors
    match gen_circuit_settings(model_path, params_output, default_run_args) {
        Ok(output) => println!("Function executed successfully: {}", output),
        Err(e) => println!("Error occurred: {:?}", e),
    }
}
