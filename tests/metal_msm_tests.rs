#![cfg(any(feature = "macos-metal", feature = "ios-metal"))]

//! Tests for Metal MSM acceleration
//! 
//! These tests verify that the Metal MSM acceleration works correctly
//! for the age verification circuit.

use ezkl::pfsys::metal_msm_accelerator::{
    MetalMSMAccelerator,
    MetalMSMConfig,
    accelerate_age_verification_msm,
    memory_optimized_age_verification_msm,
    performance_optimized_age_verification_msm,
    msm_optimized,
};
use ezkl::circuit::metal_optimize::{
    AgeVerificationOptimizer,
    optimize_age_verification_circuit,
    optimize_age_verification_circuit_for_memory,
    optimize_age_verification_circuit_for_performance,
};
use halo2curves::bn256::{Fr, G1Affine, G1};
use rand::rngs::OsRng;
use std::time::Instant;
use ezkl::RunArgs;
use rand::thread_rng;
use halo2_proofs::arithmetic::Field;

/// Generate random data for MSM testing
fn generate_test_data(size: usize) -> (Vec<G1Affine>, Vec<Fr>) {
    let mut rng = OsRng;
    
    let scalars: Vec<Fr> = (0..size)
        .map(|_| Fr::random(&mut rng))
        .collect();
    
    let bases: Vec<G1Affine> = (0..size)
        .map(|_| G1::random(&mut rng).to_affine())
        .collect();
    
    (bases, scalars)
}

/// Test that the Metal MSM acceleration produces the correct result
#[test]
fn test_metal_msm_correctness() {
    let sizes = [64, 128, 256];
    
    for size in sizes {
        let (bases, scalars) = generate_test_data(size);
        
        // Compute the expected result using the standard implementation
        let expected = G1::msm_unchecked(&bases, &scalars);
        
        // Compute the result using our accelerated implementation
        let actual = accelerate_age_verification_msm(&bases, &scalars);
        
        // Check that the results match
        assert_eq!(expected, actual, "MSM results don't match for size {}", size);
    }
}

/// Test that the memory-optimized Metal MSM acceleration produces the correct result
#[test]
fn test_memory_optimized_msm_correctness() {
    let sizes = [64, 128, 256];
    
    for size in sizes {
        let (bases, scalars) = generate_test_data(size);
        
        // Compute the expected result using the standard implementation
        let expected = G1::msm_unchecked(&bases, &scalars);
        
        // Compute the result using our memory-optimized implementation
        let actual = memory_optimized_age_verification_msm(&bases, &scalars);
        
        // Check that the results match
        assert_eq!(expected, actual, "Memory-optimized MSM results don't match for size {}", size);
    }
}

/// Test that the performance-optimized Metal MSM acceleration produces the correct result
#[test]
fn test_performance_optimized_msm_correctness() {
    let sizes = [64, 128, 256];
    
    for size in sizes {
        let (bases, scalars) = generate_test_data(size);
        
        // Compute the expected result using the standard implementation
        let expected = G1::msm_unchecked(&bases, &scalars);
        
        // Compute the result using our performance-optimized implementation
        let actual = performance_optimized_age_verification_msm(&bases, &scalars);
        
        // Check that the results match
        assert_eq!(expected, actual, "Performance-optimized MSM results don't match for size {}", size);
    }
}

/// Benchmark the different MSM implementations for the age verification circuit
#[test]
fn benchmark_age_verification_msm() {
    println!("Benchmarking different MSM implementations for age verification circuit...");
    
    // Sizes representative of age verification circuit operations
    let sizes = [4096, 8192, 16384];
    
    for size in sizes {
        println!("\nBenchmarking MSM with size {}", size);
        
        let (bases, scalars) = generate_test_data(size);
        
        // Benchmark standard CPU implementation
        let start = Instant::now();
        let _ = G1::msm_unchecked(&bases, &scalars);
        let cpu_time = start.elapsed();
        println!("  Standard CPU implementation: {:?}", cpu_time);
        
        // Benchmark standard Metal implementation
        let start = Instant::now();
        let _ = accelerate_age_verification_msm(&bases, &scalars);
        let metal_time = start.elapsed();
        println!("  Standard Metal implementation: {:?}", metal_time);
        
        // Benchmark memory-optimized implementation
        let start = Instant::now();
        let _ = memory_optimized_age_verification_msm(&bases, &scalars);
        let memory_opt_time = start.elapsed();
        println!("  Memory-optimized implementation: {:?}", memory_opt_time);
        
        // Benchmark performance-optimized implementation
        let start = Instant::now();
        let _ = performance_optimized_age_verification_msm(&bases, &scalars);
        let perf_opt_time = start.elapsed();
        println!("  Performance-optimized implementation: {:?}", perf_opt_time);
        
        // Calculate improvements
        if cpu_time.as_micros() > 0 {
            let metal_speedup = cpu_time.as_micros() as f64 / metal_time.as_micros() as f64;
            let memory_opt_speedup = cpu_time.as_micros() as f64 / memory_opt_time.as_micros() as f64;
            let perf_opt_speedup = cpu_time.as_micros() as f64 / perf_opt_time.as_micros() as f64;
            
            println!("  Speedup with standard Metal: {:.2}x", metal_speedup);
            println!("  Speedup with memory-optimized: {:.2}x", memory_opt_speedup);
            println!("  Speedup with performance-optimized: {:.2}x", perf_opt_speedup);
        }
    }
}

/// Benchmark the different MSM implementations with more realistic data sizes
#[test]
fn benchmark_realistic_age_verification_msm() {
    println!("Benchmarking with realistic age verification circuit sizes...");
    
    // The age verification model has a 64x64x3 input (12,288 elements)
    // and multiple outputs of smaller sizes
    // These sizes are representative of the actual circuit operations
    let sizes = [12288, 24576, 49152];
    
    for size in sizes {
        println!("\nBenchmarking MSM with size {}", size);
        
        let (bases, scalars) = generate_test_data(size);
        
        // Benchmark standard CPU implementation
        let start = Instant::now();
        let _ = G1::msm_unchecked(&bases, &scalars);
        let cpu_time = start.elapsed();
        println!("  Standard CPU implementation: {:?}", cpu_time);
        
        // Benchmark standard Metal implementation
        let start = Instant::now();
        let _ = accelerate_age_verification_msm(&bases, &scalars);
        let metal_time = start.elapsed();
        println!("  Standard Metal implementation: {:?}", metal_time);
        
        // Benchmark memory-optimized implementation
        let start = Instant::now();
        let _ = memory_optimized_age_verification_msm(&bases, &scalars);
        let memory_opt_time = start.elapsed();
        println!("  Memory-optimized implementation: {:?}", memory_opt_time);
        
        // Benchmark performance-optimized implementation
        let start = Instant::now();
        let _ = performance_optimized_age_verification_msm(&bases, &scalars);
        let perf_opt_time = start.elapsed();
        println!("  Performance-optimized implementation: {:?}", perf_opt_time);
        
        // Calculate improvements
        if cpu_time.as_micros() > 0 {
            let metal_speedup = cpu_time.as_micros() as f64 / metal_time.as_micros() as f64;
            let memory_opt_speedup = cpu_time.as_micros() as f64 / memory_opt_time.as_micros() as f64;
            let perf_opt_speedup = cpu_time.as_micros() as f64 / perf_opt_time.as_micros() as f64;
            
            println!("  Speedup with standard Metal: {:.2}x", metal_speedup);
            println!("  Speedup with memory-optimized: {:.2}x", memory_opt_speedup);
            println!("  Speedup with performance-optimized: {:.2}x", perf_opt_speedup);
        }
    }
}

/// Measure the memory usage of different MSM implementations
#[test]
#[ignore] // This test is resource-intensive and should be run manually
fn measure_memory_usage() {
    println!("Measuring memory usage of different MSM implementations...");
    
    // Size representative of age verification circuit
    let size = 12288;
    
    let (bases, scalars) = generate_test_data(size);
    
    // Create accelerators with different configurations
    let standard_accelerator = MetalMSMAccelerator::with_config(MetalMSMConfig {
        use_metal: true,
        age_verification_optimized: false,
        ..MetalMSMConfig::default()
    });
    
    let age_optimized_accelerator = MetalMSMAccelerator::with_config(MetalMSMConfig {
        use_metal: true,
        age_verification_optimized: true,
        ..MetalMSMConfig::default()
    });
    
    let memory_optimized_accelerator = MetalMSMAccelerator::with_config(MetalMSMConfig {
        use_metal: true,
        age_verification_optimized: true,
        optimize_for_memory: true,
        batch_size: 512,
        window_size: 20,
        ..MetalMSMConfig::default()
    });
    
    // Use the accelerators (actual memory measurement would be done externally)
    println!("Running standard accelerator...");
    standard_accelerator.msm(&bases, &scalars);
    
    println!("Running age-optimized accelerator...");
    age_optimized_accelerator.msm(&bases, &scalars);
    
    println!("Running memory-optimized accelerator...");
    memory_optimized_accelerator.memory_optimized_msm(&bases, &scalars);
}

#[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
mod metal_msm_tests {
    use ezkl::pfsys::metal_msm_accelerator::{MetalMSMAccelerator, MetalMSMConfig, msm_optimized};
    use ezkl::RunArgs;
    use ezkl::circuit::metal_optimize::{
        optimize_age_verification_circuit,
        optimize_age_verification_circuit_for_memory,
        optimize_age_verification_circuit_for_performance,
    };
    use halo2curves::bn256::{G1Affine, Fr};
    use rand::thread_rng;
    use halo2_proofs::arithmetic::Field;
    use std::time::Instant;

    #[test]
    fn test_metal_msm_accelerator_setup() {
        let config = MetalMSMConfig {
            window_size: 22,
            batch_size: 64,
            use_metal: true,
            num_threads: 6,
            age_verification_optimized: true,
            max_threads: 6,
            optimization_target: "balanced".to_string(),
        };
        
        MetalMSMAccelerator::setup_for_age_verification(config);
        
        // Just testing that the setup doesn't panic
        assert!(true);
    }
    
    #[test]
    fn test_optimize_age_verification_circuit() {
        let mut run_args = RunArgs::default();
        
        // Apply optimization
        optimize_age_verification_circuit(&mut run_args);
        
        // Check that settings were updated
        assert_eq!(run_args.bits, 16);
        assert_eq!(run_args.lookup_bits, 10);
    }
    
    #[test]
    fn test_memory_optimization() {
        let mut run_args = RunArgs::default();
        
        // Apply memory optimization
        optimize_age_verification_circuit_for_memory(&mut run_args);
        
        // Check that settings were updated for memory optimization
        assert_eq!(run_args.bits, 13);
        assert_eq!(run_args.lookup_bits, 9);
    }
    
    #[test]
    fn test_performance_optimization() {
        let mut run_args = RunArgs::default();
        
        // Apply performance optimization
        optimize_age_verification_circuit_for_performance(&mut run_args);
        
        // Check that settings were updated for performance optimization
        assert_eq!(run_args.bits, 18);
        assert_eq!(run_args.lookup_bits, 12);
    }
    
    #[test]
    fn benchmark_metal_msm() {
        // Create test data
        let n = 10000;
        let mut rng = thread_rng();
        
        let scalars: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let bases: Vec<G1Affine> = (0..n)
            .map(|_| G1Affine::random(&mut rng))
            .collect();
        
        // Benchmark Metal MSM
        let start = Instant::now();
        let _result = msm_optimized(&bases, &scalars);
        let duration = start.elapsed();
        
        println!("Metal MSM with {} points took: {:?}", n, duration);
    }
} 