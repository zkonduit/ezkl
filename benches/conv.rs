use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::commands::TranscriptType;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::pfsys::gen_srs;
use ezkl_lib::tensor::*;
use ezkl_lib::{circuit::polynomial::*, pfsys::create_keys};
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
const STRIDE: usize = 1;
static mut IMAGE_HEIGHT: usize = 2;
static mut IMAGE_WIDTH: usize = 2;
static mut IN_CHANNELS: usize = 1;
const PADDING: usize = 0;

const K: usize = 17;

#[derive(Clone, Debug)]
struct MyCircuit {
    image: ValTensor<Fr>,
    kernel: ValTensor<Fr>,
    bias: ValTensor<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = Config<Fr>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        unsafe {
            let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
            let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

            let input = VarTensor::new_advice(
                cs,
                K,
                IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH,
                vec![IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
                true,
                512,
            );
            let kernel = VarTensor::new_advice(
                cs,
                K,
                OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH,
                vec![OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH],
                true,
                512,
            );

            let bias = VarTensor::new_advice(cs, K, OUT_CHANNELS, vec![OUT_CHANNELS], true, 512);
            let output = VarTensor::new_advice(
                cs,
                K,
                OUT_CHANNELS * output_height * output_width,
                vec![OUT_CHANNELS, output_height, output_width],
                true,
                512,
            );

            // tells the config layer to add a conv op to a circuit gate
            let conv_node = Node {
                op: Op::Conv {
                    padding: (PADDING, PADDING),
                    stride: (STRIDE, STRIDE),
                },
                input_order: vec![
                    InputType::Input(0),
                    InputType::Input(1),
                    InputType::Input(2),
                ],
            };

            Self::Config::configure(cs, &[input, kernel, bias], &output, &[conv_node])
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let _output = config.layout(
            &mut layouter,
            &[self.image.clone(), self.kernel.clone(), self.bias.clone()],
        );
        Ok(())
    }
}

fn runcnvrl(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv");

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
