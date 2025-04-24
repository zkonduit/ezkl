//! Age Verification Circuit Optimizer Example
//! 
//! This example demonstrates how to optimize the age verification circuit
//! for the Subnet 2 competition using Metal acceleration on Apple silicon.

use std::path::PathBuf;
use ezkl::RunArgs;

#[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
use ezkl::circuit::metal_optimize::{
    optimize_age_verification_circuit,
    optimize_age_verification_circuit_for_memory,
    optimize_age_verification_circuit_for_performance,
};

fn main() {
    // Print a welcome message
    println!("Age Verification Circuit Optimizer for Subnet 2 Competition");
    println!("----------------------------------------------------------");
    
    // Create default RunArgs
    let mut run_args = RunArgs::default();
    
    // Print initial settings
    println!("Initial settings:");
    print_settings(&run_args);
    
    // Apply balanced optimizations
    #[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
    {
        println!("\nApplying balanced optimizations...");
        optimize_age_verification_circuit(&mut run_args);
        println!("Optimized settings (balanced):");
        print_settings(&run_args);
        
        // Reset for memory optimization demo
        run_args = RunArgs::default();
        
        println!("\nApplying memory optimizations...");
        optimize_age_verification_circuit_for_memory(&mut run_args);
        println!("Optimized settings (memory):");
        print_settings(&run_args);
        
        // Reset for performance optimization demo
        run_args = RunArgs::default();
        
        println!("\nApplying performance optimizations...");
        optimize_age_verification_circuit_for_performance(&mut run_args);
        println!("Optimized settings (performance):");
        print_settings(&run_args);
    }
    
    #[cfg(not(any(feature = "macos-metal", feature = "ios-metal")))]
    println!("Warning: Metal acceleration not enabled. Compile with --features macos-metal for optimizations.");
    
    println!("\nNote: To use these optimizations in your EZKL project:");
    println!("1. Add the metal_optimize module to your dependencies");
    println!("2. Call the appropriate optimization function on your RunArgs");
    println!("3. Use the optimized RunArgs in your circuit generation");
    
    println!("\nMetal acceleration status:");
    #[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
    println!("Metal acceleration: ENABLED ✅");
    
    #[cfg(not(any(feature = "macos-metal", feature = "ios-metal")))]
    println!("Metal acceleration: DISABLED ❌");
}

// Helper function to print settings
fn print_settings(run_args: &RunArgs) {
    println!("  - input_scale: {}", run_args.input_scale);
    println!("  - param_scale: {}", run_args.param_scale);
    println!("  - logrows: {}", run_args.logrows);
    println!("  - decomp_legs: {}", run_args.decomp_legs);
    println!("  - lookup_range: ({}, {})", run_args.lookup_range.0, run_args.lookup_range.1);
} 