use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::table::Range;
use ezkl::circuit::*;
use ezkl::pfsys::create_proof_circuit;
use ezkl::pfsys::TranscriptType;
use ezkl::pfsys::{create_keys, srs::gen_srs};
use ezkl::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::marker::PhantomData;

// Age verification specific ranges and constants
const AGE_RANGE: Range = (0, 120); // Human age range
const FACE_EMBEDDING_DIM: usize = 128; // Typical face embedding dimension

#[derive(Clone)]
struct AgeVerificationCircuit {
    face_embedding: ValTensor<Fr>,
    reference_embedding: ValTensor<Fr>,
    _marker: PhantomData<Fr>,
}

// Circuit implementation
// ...

fn run_age_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("age_verification");
    let params = gen_srs::<KZGCommitmentScheme<_>>(15); // Reduced from 17
    
    // Create face embeddings for testing
    let face_embedding = create_random_face_embedding();
    let reference_embedding = create_random_face_embedding();
    
    let circuit = AgeVerificationCircuit {
        face_embedding: ValTensor::from(face_embedding),
        reference_embedding: ValTensor::from(reference_embedding),
        _marker: PhantomData,
    };
    
    // Benchmark proving key generation
    group.bench_with_input(BenchmarkId::new("pk", "age_circuit"), &circuit, |b, circuit| {
        b.iter(|| {
            create_keys::<KZGCommitmentScheme<Bn256>, AgeVerificationCircuit>(circuit, &params, true)
                .unwrap();
        });
    });
    
    // Get proving key
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, AgeVerificationCircuit>(&circuit, &params, true).unwrap();
    
    // Benchmark proof generation with both original and optimized transcript
    for transcript_type in [TranscriptType::EVM, TranscriptType::Age].iter() {
        group.bench_with_input(
            BenchmarkId::new("prove", format!("transcript_{:?}", transcript_type)), 
            &(circuit.clone(), *transcript_type), 
            |b, (circuit, transcript_type)| {
                b.iter(|| {
                    let prover = create_proof_circuit::<
                        KZGCommitmentScheme<_>,
                        AgeVerificationCircuit,
                        ProverSHPLONK<_>,
                        VerifierSHPLONK<_>,
                        SingleStrategy<_>,
                        _,
                        EvmTranscript<_, _, _, _>,
                        EvmTranscript<_, _, _, _>,
                    >(
                        circuit.clone(),
                        vec![],
                        &params,
                        &pk,
                        CheckMode::UNSAFE,
                        ezkl::Commitments::KZG,
                        *transcript_type,
                        None,
                        None,
                    );
                    prover.unwrap();
                });
            }
        );
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_plots().sample_size(10);
    targets = run_age_benchmark
}
criterion_main!(benches); 