use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use ezkl_lib::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig, NUM_INSTANCE_COLUMNS};
use ezkl_lib::circuit::modules::Module;
use ezkl_lib::circuit::*;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::graph::modules::{POSEIDOIN_FIXED_COST_ESTIMATE, POSEIDON_CONSTRAINTS_ESTIMATE};
use ezkl_lib::pfsys::create_keys;
use ezkl_lib::pfsys::srs::gen_srs;
use ezkl_lib::pfsys::TranscriptType;
use ezkl_lib::tensor::*;
use halo2_proofs::circuit::Value;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;

const L: usize = 10;

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
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, 10>::configure(cs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let chip: PoseidonChip<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, L> =
            PoseidonChip::new(config);
        chip.layout(
            &mut layouter,
            &[self.image.clone()],
            vec![0; NUM_INSTANCE_COLUMNS],
        )?;
        Ok(())
    }
}

fn runposeidon(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon");

    for size in [64, 784, 65536, 524288].iter() {
        let k = ((size * POSEIDON_CONSTRAINTS_ESTIMATE + POSEIDOIN_FIXED_COST_ESTIMATE) as f32)
            .log2()
            .ceil() as u32;
        let params = gen_srs::<KZGCommitmentScheme<_>>(k);

        let message = (0..*size).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();
        let output =
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, L>::run(message.to_vec())
                .unwrap();

        let mut image = Tensor::from(message.into_iter().map(|x| Value::known(x)));
        image.reshape(&[1, *size]);

        let circuit = MyCircuit {
            image: ValTensor::from(image),
        };

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("pk", size), &size, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(&circuit, &params)
                    .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(&circuit, &params).unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("prove", size), &size, |b, &_| {
            b.iter(|| {
                let prover = create_proof_circuit_kzg(
                    circuit.clone(),
                    &params,
                    output.clone(),
                    &pk,
                    TranscriptType::Blake,
                    SingleStrategy::new(&params),
                    CheckMode::UNSAFE,
                );
                prover.unwrap();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runposeidon
}
criterion_main!(benches);
