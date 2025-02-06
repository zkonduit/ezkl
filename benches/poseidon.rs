use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use ezkl::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use ezkl::circuit::modules::Module;
use ezkl::circuit::*;
use ezkl::pfsys::create_keys;
use ezkl::pfsys::create_proof_circuit;
use ezkl::pfsys::srs::gen_srs;
use ezkl::pfsys::TranscriptType;
use ezkl::tensor::*;
use halo2_proofs::circuit::Value;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

#[derive(Clone, Debug)]
struct MyCircuit {
    image: ValTensor<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::configure(cs, ())
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let chip: PoseidonChip<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE> =
            PoseidonChip::new(config);
        chip.layout(&mut layouter, &[self.image.clone()], 0, &mut HashMap::new())?;
        Ok(())
    }
}

fn runposeidon(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon");

    for size in [64, 784, 2352, 12288].iter() {
        let k = (PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::num_rows(*size)
            as f32)
            .log2()
            .ceil() as u32;
        let params = gen_srs::<KZGCommitmentScheme<_>>(k);

        let message = (0..*size).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
        let _output =
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::run(message.to_vec())
                .unwrap();

        let mut image = Tensor::from(message.into_iter().map(Value::known));
        image.reshape(&[1, *size]).unwrap();

        let circuit = MyCircuit {
            image: ValTensor::from(image),
        };

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("pk", size), &size, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit>(&circuit, &params, true)
                    .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit>(&circuit, &params, true).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("prove", size), &size, |b, &_| {
            b.iter(|| {
                let prover = create_proof_circuit::<
                    KZGCommitmentScheme<_>,
                    MyCircuit,
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
                    TranscriptType::EVM,
                    None,
                    None,
                );
                prover.unwrap();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots().sample_size(10);
  targets = runposeidon
}
criterion_main!(benches);
