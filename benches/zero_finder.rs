use std::thread;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use halo2curves::{bn256::Fr as F, ff::Field};
use maybe_rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use rand::Rng;

// Assuming these are your types
#[derive(Clone)]
#[allow(dead_code)]
enum ValType {
    Constant(F),
    AssignedConstant(usize, F),
    Other,
}

// Helper to generate test data
fn generate_test_data(size: usize, zero_probability: f64) -> Vec<ValType> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_i| {
            if rng.r#gen::<f64>() < zero_probability {
                ValType::Constant(F::ZERO)
            } else {
                ValType::Constant(F::ONE) // Or some other non-zero value
            }
        })
        .collect()
}

fn bench_zero_finding(c: &mut Criterion) {
    let sizes = [
        1_000,         // 1K
        10_000,        // 10K
        100_000,       // 100K
        256 * 256 * 2, // Our specific case
        1_000_000,     // 1M
        10_000_000,    // 10M
    ];

    let zero_probability = 0.1; // 10% zeros

    let mut group = c.benchmark_group("zero_finding");
    group.sample_size(10); // Adjust based on your needs

    for &size in &sizes {
        let data = generate_test_data(size, zero_probability);

        // Benchmark sequential version
        group.bench_function(format!("sequential_{}", size), |b| {
            b.iter(|| {
                let result = data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| match e {
                        ValType::Constant(r) | ValType::AssignedConstant(_, r) => {
                            (*r == F::ZERO).then_some(i)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                black_box(result)
            })
        });

        // Benchmark parallel version
        group.bench_function(format!("parallel_{}", size), |b| {
            b.iter(|| {
                let result = data
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, e)| match e {
                        ValType::Constant(r) | ValType::AssignedConstant(_, r) => {
                            (*r == F::ZERO).then_some(i)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                black_box(result)
            })
        });

        // Benchmark chunked parallel version
        group.bench_function(format!("chunked_parallel_{}", size), |b| {
            b.iter(|| {
                let num_cores = thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                let chunk_size = (size / num_cores).max(100);

                let result = data
                    .par_chunks(chunk_size)
                    .enumerate()
                    .flat_map(|(chunk_idx, chunk)| {
                        chunk
                            .par_iter() // Make sure we use par_iter() here
                            .enumerate()
                            .filter_map(move |(i, e)| match e {
                                ValType::Constant(r) | ValType::AssignedConstant(_, r) => {
                                    (*r == F::ZERO).then_some(chunk_idx * chunk_size + i)
                                }
                                _ => None,
                            })
                    })
                    .collect::<Vec<_>>();
                black_box(result)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_zero_finding);
criterion_main!(benches);
