use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::*;
use ezkl_lib::commands::TranscriptType;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::pfsys::create_keys;
use ezkl_lib::pfsys::gen_srs;
use ezkl_lib::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;

static mut IMAGE_HEIGHT: usize = 2;
static mut IMAGE_WIDTH: usize = 2;
static mut IN_CHANNELS: usize = 3;

const K: usize = 17;

#[derive(Clone, Debug)]
struct MyCircuit {
    image: ValTensor<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = 10;

        let a = VarTensor::new_advice(cs, K, len * len, true);

        let b = VarTensor::new_advice(cs, K, len * len, true);

        let output = VarTensor::new_advice(cs, K, (len + 1) * len, true);

        Self::Config::configure(cs, &[a, b], &output, CheckMode::UNSAFE, 0)
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |mut region| {
                config
                    .layout(
                        &mut region,
                        &[self.image.clone()],
                        &mut 0,
                        Op::SumPool {
                            padding: (0, 0),
                            stride: (1, 1),
                            kernel_shape: (2, 2),
                        }
                        .into(),
                    )
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runsumpool(c: &mut Criterion) {
    let mut group = c.benchmark_group("accum_sumpool");

    let params = gen_srs::<KZGCommitmentScheme<_>>(K as u32);

    for size in [1, 2].iter() {
        unsafe {
            IMAGE_HEIGHT = size * 4;
            IMAGE_WIDTH = size * 4;

            let mut image = Tensor::from(
                (0..IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)
                    .map(|_| Value::known(Fr::random(OsRng))),
            );
            image.reshape(&[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);

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

            let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(&circuit, &params)
                .unwrap();

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("prove", size), &size, |b, &_| {
                b.iter(|| {
                    let prover = create_proof_circuit_kzg(
                        circuit.clone(),
                        &params,
                        vec![],
                        &pk,
                        TranscriptType::Blake,
                        SingleStrategy::new(&params),
                        CheckMode::UNSAFE,
                    );
                    prover.unwrap();
                });
            });
        }
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runsumpool
}
criterion_main!(benches);
