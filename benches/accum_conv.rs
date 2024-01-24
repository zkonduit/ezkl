use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::pfsys::create_keys;
use ezkl::pfsys::create_proof_circuit_kzg;
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
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = 10;

        let a = VarTensor::new_advice(cs, K, 1, len * len);

        let b = VarTensor::new_advice(cs, K, 1, len * len);

        let output = VarTensor::new_advice(cs, K, 1, (len + 1) * len);

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
                let mut region = region::RegionCtx::new(region, 0, 1);
                config
                    .layout(
                        &mut region,
                        &[self.image.clone(), self.kernel.clone(), self.bias.clone()],
                        Box::new(PolyOp::Conv {
                            padding: [(0, 0); 2],
                            stride: (1, 1),
                        }),
                    )
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runcnvrl(c: &mut Criterion) {
    let mut group = c.benchmark_group("accum_conv");

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
            image
                .reshape(&[1, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH])
                .unwrap();
            let mut kernel = Tensor::from(
                (0..{ OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH })
                    .map(|_| Fr::random(OsRng)),
            );
            kernel
                .reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH])
                .unwrap();
            kernel.set_visibility(&ezkl::graph::Visibility::Private);

            let mut bias = Tensor::from((0..{ OUT_CHANNELS }).map(|_| Fr::random(OsRng)));
            bias.set_visibility(&ezkl::graph::Visibility::Private);

            let circuit = MyCircuit {
                image: ValTensor::from(image),
                kernel: ValTensor::try_from(kernel).unwrap(),
                bias: ValTensor::try_from(bias).unwrap(),
            };

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("pk", size), &size, |b, &_| {
                b.iter(|| {
                    create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(
                        &circuit, &params, true,
                    )
                    .unwrap();
                });
            });

            let pk =
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(&circuit, &params, true)
                    .unwrap();

            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(BenchmarkId::new("prove", size), &size, |b, &_| {
                b.iter(|| {
                    let prover = create_proof_circuit_kzg(
                        circuit.clone(),
                        &params,
                        None,
                        &pk,
                        TranscriptType::EVM,
                        SingleStrategy::new(&params),
                        CheckMode::UNSAFE,
                        None,
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
