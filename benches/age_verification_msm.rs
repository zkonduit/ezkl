//! Benchmarks for the age verification circuit MSM optimizations.
//! 
//! This benchmark compares the performance of different MSM implementations
//! for the age verification circuit.

use std::time::Duration;
use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ezkl::pfsys::metal_msm_accelerator::{
    accelerate_age_verification_msm,
    memory_optimized_age_verification_msm,
    performance_optimized_age_verification_msm,
    MetalMSMAccelerator,
    MetalMSMConfig
};
use halo2curves::bn256::{Fr, G1Affine, G1};
use rand::rngs::OsRng;

/// Generate random bases and scalars for benchmarking
fn generate_test_msm_data(size: usize) -> (Vec<G1Affine>, Vec<Fr>) {
    let mut rng = OsRng;
    
    let scalars: Vec<Fr> = (0..size)
        .map(|_| Fr::random(&mut rng))
        .collect();
    
    let bases: Vec<G1Affine> = (0..size)
        .map(|_| G1::random(&mut rng).to_affine())
        .collect();
    
    (bases, scalars)
}

/// Benchmark the standard MSM implementation
fn bench_standard_msm(bases: &[G1Affine], scalars: &[Fr]) -> G1 {
    G1::msm_unchecked(bases, scalars)
}

/// Benchmark the Metal-accelerated MSM implementation
fn bench_metal_msm(bases: &[G1Affine], scalars: &[Fr]) -> G1 {
    accelerate_age_verification_msm(bases, scalars)
}

/// Benchmark the memory-optimized Metal MSM implementation
fn bench_memory_optimized_msm(bases: &[G1Affine], scalars: &[Fr]) -> G1 {
    memory_optimized_age_verification_msm(bases, scalars)
}

/// Benchmark the performance-optimized Metal MSM implementation
fn bench_performance_optimized_msm(bases: &[G1Affine], scalars: &[Fr]) -> G1 {
    performance_optimized_age_verification_msm(bases, scalars)
}

/// Run different configurations of the MSM benchmark
fn criterion_benchmark(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536];
    
    let mut group = c.benchmark_group("age_verification_msm");
    group.measurement_time(Duration::from_secs(10));
    
    for size in sizes.iter() {
        let (bases, scalars) = generate_test_msm_data(*size);
        
        group.bench_with_input(BenchmarkId::new("standard", size), &size, |b, _| {
            b.iter(|| bench_standard_msm(black_box(&bases), black_box(&scalars)))
        });
        
        #[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
        {
            group.bench_with_input(BenchmarkId::new("metal", size), &size, |b, _| {
                b.iter(|| bench_metal_msm(black_box(&bases), black_box(&scalars)))
            });
            
            group.bench_with_input(BenchmarkId::new("memory_optimized", size), &size, |b, _| {
                b.iter(|| bench_memory_optimized_msm(black_box(&bases), black_box(&scalars)))
            });
            
            group.bench_with_input(BenchmarkId::new("performance_optimized", size), &size, |b, _| {
                b.iter(|| bench_performance_optimized_msm(black_box(&bases), black_box(&scalars)))
            });
        }
    }
    
    group.finish();
}

/// Measure memory usage for different MSM implementations
fn measure_memory_usage() {
    println!("Measuring memory usage for different MSM implementations");
    
    let sizes = [1024, 4096, 16384, 65536];
    
    for size in sizes.iter() {
        println!("Testing with size: {}", size);
        
        let (bases, scalars) = generate_test_msm_data(*size);
        
        // Standard MSM
        let start = Instant::now();
        let _ = bench_standard_msm(&bases, &scalars);
        let duration = start.elapsed();
        println!("Standard MSM: {:?}", duration);
        
        #[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
        {
            // Metal-accelerated MSM
            let start = Instant::now();
            let _ = bench_metal_msm(&bases, &scalars);
            let duration = start.elapsed();
            println!("Metal MSM: {:?}", duration);
            
            // Memory-optimized MSM
            let start = Instant::now();
            let _ = bench_memory_optimized_msm(&bases, &scalars);
            let duration = start.elapsed();
            println!("Memory-optimized MSM: {:?}", duration);
            
            // Performance-optimized MSM
            let start = Instant::now();
            let _ = bench_performance_optimized_msm(&bases, &scalars);
            let duration = start.elapsed();
            println!("Performance-optimized MSM: {:?}", duration);
        }
        
        println!("--------------------------------------------------------");
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

/// Main function for running the memory measurement directly
#[allow(dead_code)]
fn main() {
    measure_memory_usage();
} 