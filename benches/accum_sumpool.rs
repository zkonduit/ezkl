use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::execute::create_proof_circuit_kzg;
use ezkl::pfsys::create_keys;
use ezkl::pfsys::srs::gen_srs;
use ezkl::pfsys::TranscriptType;
use ezkl::tensor::*;
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
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = 10;

        let a = VarTensor::new_advice(cs, K, len * len);

        let b = VarTensor::new_advice(cs, K, len * len);

        let output = VarTensor::new_advice(cs, K, (len + 1) * len);

        Self::Config::configure(cs, &[a, b], &output, CheckMode::UNSAFE)
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |region| {
                let mut region = region::RegionCtx::new(region, 0);
                config
                    .layout(
                        &mut region,
                        &[self.image.clone()],
                        Box::new(PolyOp::SumPool {
                            padding: (0, 0),
                            stride: (1, 1),
                            kernel_shape: (2, 2),
                        }),
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
            image.reshape(&[1, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);

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
