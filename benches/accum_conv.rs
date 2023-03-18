use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::accumulated::*;
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

static mut KERNEL_HEIGHT: usize = 2;
static mut KERNEL_WIDTH: usize = 2;
static mut OUT_CHANNELS: usize = 1;
static mut IMAGE_HEIGHT: usize = 2;
static mut IMAGE_WIDTH: usize = 2;
static mut IN_CHANNELS: usize = 1;

const K: usize = 17;

#[derive(Clone, Debug)]
struct MyCircuit {
    image: ValTensor<Fr>,
    kernel: ValTensor<Fr>,
    bias: ValTensor<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = 10;

        let a = VarTensor::new_advice(cs, K, len * len, vec![len, len], true, 1000000);

        let b = VarTensor::new_advice(cs, K, len * len, vec![len, len], true, 1000000);

        let output = VarTensor::new_advice(
            cs,
            K,
            (len + 1) * len,
            vec![len, 1, len + 1],
            true,
            10000000,
        );

        Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        config
            .layout(
                &mut layouter,
                &[self.image.clone(), self.kernel.clone(), self.bias.clone()],
                0,
                Op::Conv {
                    padding: (0, 0),
                    stride: (1, 1),
                },
            )
            .unwrap();
        Ok(())
    }
}

fn runcnvrl(c: &mut Criterion) {
    let mut group = c.benchmark_group("cnvrl");

    let params = gen_srs::<KZGCommitmentScheme<_>>(K as u32);

    for size in [1, 2, 4].iter() {
        unsafe {
            KERNEL_HEIGHT = size * 2;
            KERNEL_WIDTH = size * 2;
            IMAGE_HEIGHT = size * 4;
            IMAGE_WIDTH = size * 4;
            IN_CHANNELS = 1;
            OUT_CHANNELS = 1;

            let mut image = Tensor::from(
                (0..IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)
                    .map(|_| Value::known(Fr::random(OsRng))),
            );
            image.reshape(&[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);
            let mut kernels = Tensor::from(
                (0..{ OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH })
                    .map(|_| Value::known(Fr::random(OsRng))),
            );
            kernels.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]);

            let bias = Tensor::from((0..{ OUT_CHANNELS }).map(|_| Value::known(Fr::random(OsRng))));

            let circuit = MyCircuit {
                image: ValTensor::from(image),
                kernel: ValTensor::from(kernels),
                bias: ValTensor::from(bias),
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
  targets = runcnvrl
}
criterion_main!(benches);
