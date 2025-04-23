//! Age Verification Circuit Optimizer Example
//! 
//! This example demonstrates how to optimize the age verification circuit
//! for the Subnet 2 competition using Metal acceleration on Apple silicon.

use std::path::PathBuf;
use std::time::Instant;

use ezkl::circuit::metal_optimize::{
    AgeVerificationOptimizer,
    optimize_age_verification_circuit,
    optimize_age_verification_circuit_for_memory,
    optimize_age_verification_circuit_for_performance,
};
use ezkl::execute::{run, Command, run_args::InputOutputSettings};
use ezkl::graph::Settings;
use ezkl::pfsys::metal_msm_accelerator::{
    MetalMSMAccelerator,
    MetalMSMConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Print a welcome message
    println!("Age Verification Circuit Optimizer for Subnet 2 Competition");
    println!("----------------------------------------------------------");
    
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).cloned().unwrap_or_else(|| {
        println!("No model path provided, using default path");
        "./neurons/_validator/competitions/1/age.onnx".to_string()
    });
    let input_path = args.get(2).cloned().unwrap_or_else(|| {
        println!("No input path provided, using default path");
        "./neurons/_validator/competitions/1/input.json".to_string()
    });
    
    // Determine optimization mode
    let optimization_mode = args.get(3).cloned().unwrap_or_else(|| {
        println!("No optimization mode provided, using 'balanced'");
        "balanced".to_string()
    });
    
    println!("Using model: {}", model_path);
    println!("Using input: {}", input_path);
    println!("Optimization mode: {}", optimization_mode);
    
    // Load the model and input
    let model_path = PathBuf::from(model_path);
    let input_path = PathBuf::from(input_path);
    
    // Generate settings
    println!("Generating settings...");
    let mut settings = generate_settings(&model_path, &input_path)?;
    
    // Apply optimizations based on the mode
    match optimization_mode.as_str() {
        "memory" => {
            println!("Applying memory optimizations...");
            optimize_age_verification_circuit_for_memory(&mut settings);
        },
        "performance" => {
            println!("Applying performance optimizations...");
            optimize_age_verification_circuit_for_performance(&mut settings);
        },
        _ => {
            println!("Applying balanced optimizations...");
            optimize_age_verification_circuit(&mut settings);
        }
    }
    
    // Save optimized settings
    println!("Saving optimized settings...");
    let settings_path = PathBuf::from("optimized_settings.json");
    save_settings(&settings, &settings_path)?;
    
    // Run calibration
    println!("Calibrating circuit...");
    calibrate_circuit(&model_path, &input_path, &settings_path)?;
    
    // Compile circuit
    println!("Compiling circuit...");
    compile_circuit(&model_path, &settings_path)?;
    
    // Setup the circuit
    println!("Setting up the circuit...");
    setup_circuit(&settings_path)?;
    
    // Benchmark proving time and memory usage
    println!("Benchmarking circuit...");
    benchmark_circuit(&input_path, &settings_path)?;
    
    println!("Optimization complete! The circuit is ready for the competition.");
    println!("Optimized settings saved to: {}", settings_path.display());
    
    Ok(())
}

/// Generate settings for the age verification circuit
fn generate_settings(
    model_path: &PathBuf,
    input_path: &PathBuf,
) -> Result<Settings, Box<dyn std::error::Error>> {
    let command = Command::GenSettings { 
        model_path: model_path.clone(),
        vk_path: None,
        output_path: PathBuf::from("settings.json"),
    };
    
    let start = Instant::now();
    run(command).await?;
    let duration = start.elapsed();
    
    println!("Settings generation took: {:?}", duration);
    
    // Load the generated settings
    let settings_path = PathBuf::from("settings.json");
    let settings = std::fs::read_to_string(settings_path)?;
    let settings: Settings = serde_json::from_str(&settings)?;
    
    Ok(settings)
}

/// Save settings to a file
fn save_settings(
    settings: &Settings,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let settings_json = serde_json::to_string_pretty(settings)?;
    std::fs::write(output_path, settings_json)?;
    
    Ok(())
}

/// Calibrate the circuit
fn calibrate_circuit(
    model_path: &PathBuf,
    input_path: &PathBuf,
    settings_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let command = Command::CalibrateSettings { 
        model_path: Some(model_path.clone()),
        input_path: input_path.clone(),
        settings_path: settings_path.clone(),
        output_path: settings_path.clone(),
        calibrate_scale: false,
        scale_factor: None,
        param_scale: None,
        input_scale: None,
    };
    
    let start = Instant::now();
    run(command).await?;
    let duration = start.elapsed();
    
    println!("Circuit calibration took: {:?}", duration);
    
    Ok(())
}

/// Compile the circuit
fn compile_circuit(
    model_path: &PathBuf,
    settings_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let command = Command::Compile { 
        model_path: model_path.clone(),
        output_path: PathBuf::from("model.compiled"),
        settings_path: settings_path.clone(),
    };
    
    let start = Instant::now();
    run(command).await?;
    let duration = start.elapsed();
    
    println!("Circuit compilation took: {:?}", duration);
    
    Ok(())
}

/// Setup the circuit
fn setup_circuit(
    settings_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let command = Command::Setup { 
        compiled_path: PathBuf::from("model.compiled"),
        srs_path: PathBuf::from("kzg.srs"),
        vk_path: PathBuf::from("vk.key"),
        pk_path: PathBuf::from("pk.key"),
        settings_path: settings_path.clone(),
        skip_param_check: false,
    };
    
    let start = Instant::now();
    run(command).await?;
    let duration = start.elapsed();
    
    println!("Circuit setup took: {:?}", duration);
    
    Ok(())
}

/// Benchmark the circuit
fn benchmark_circuit(
    input_path: &PathBuf,
    settings_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    // First generate a witness
    let command = Command::GenWitness { 
        compiled_path: PathBuf::from("model.compiled"),
        input_path: input_path.clone(),
        output_path: PathBuf::from("witness.json"),
        settings_path: settings_path.clone(),
        scales_path: None,
    };
    
    let start = Instant::now();
    run(command).await?;
    let witness_duration = start.elapsed();
    
    println!("Witness generation took: {:?}", witness_duration);
    
    // Then generate a proof
    let command = Command::Prove { 
        compiled_path: PathBuf::from("model.compiled"),
        witness_path: PathBuf::from("witness.json"),
        pk_path: PathBuf::from("pk.key"),
        proof_path: PathBuf::from("proof.json"),
        srs_path: Some(PathBuf::from("kzg.srs")),
        settings_path: settings_path.clone(),
        transcript: None,
    };
    
    let start = Instant::now();
    run(command).await?;
    let prove_duration = start.elapsed();
    
    println!("Proof generation took: {:?}", prove_duration);
    
    // Finally verify the proof
    let command = Command::Verify { 
        compiled_path: PathBuf::from("model.compiled"),
        proof_path: PathBuf::from("proof.json"),
        vk_path: PathBuf::from("vk.key"),
        settings_path: settings_path.clone(),
        srs_path: Some(PathBuf::from("kzg.srs")),
    };
    
    let start = Instant::now();
    run(command).await?;
    let verify_duration = start.elapsed();
    
    println!("Proof verification took: {:?}", verify_duration);
    
    // Print a performance summary
    println!("\nPerformance Summary:");
    println!("-----------------");
    println!("Witness generation: {:?}", witness_duration);
    println!("Proof generation:   {:?}", prove_duration);
    println!("Verification:       {:?}", verify_duration);
    println!("Total:              {:?}", witness_duration + prove_duration + verify_duration);
    
    // Measure memory usage
    let mut cmd = std::process::Command::new("ps");
    cmd.arg("-o").arg("rss=").arg(format!("{}", std::process::id()));
    let output = cmd.output()?;
    let memory_usage = String::from_utf8_lossy(&output.stdout).trim().parse::<u64>().unwrap_or(0);
    
    println!("Memory usage: {} KB", memory_usage);
    
    Ok(())
} 