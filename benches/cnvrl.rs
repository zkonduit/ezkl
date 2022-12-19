use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::fused::*;
use ezkl::tensor::*;
use halo2_proofs::{
    arithmetic::{Field, FieldExt},
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::pallas;
use rand::rngs::OsRng;

static mut KERNEL_HEIGHT: usize = 2;
static mut KERNEL_WIDTH: usize = 2;
static mut OUT_CHANNELS: usize = 2;
const STRIDE: usize = 2;
static mut IMAGE_HEIGHT: usize = 2;
static mut IMAGE_WIDTH: usize = 2;
static mut IN_CHANNELS: usize = 2;
const PADDING: usize = 2;

const K: usize = 11;

#[derive(Clone, Debug)]
struct MyCircuit<F: FieldExt + TensorType>
where
    Value<F>: TensorType,
{
    image: ValTensor<F>,
    kernel: ValTensor<F>,
    bias: ValTensor<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F>
where
    Value<F>: TensorType,
{
    type Config = FusedConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
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
                512
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
            let conv_node = FusedNode {
                op: FusedOp::Conv((PADDING, PADDING), (STRIDE, STRIDE)),
                input_order: vec![
                    FusedInputType::Input(0),
                    FusedInputType::Input(1),
                    FusedInputType::Input(2),
                ],
            };

            Self::Config::configure(cs, &[input, kernel, bias], &output, &[conv_node])
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let _output = config.layout(
            &mut layouter,
            &[self.image.clone(), self.kernel.clone(), self.bias.clone()],
        );
        Ok(())
    }
}

fn runcnvrl(c: &mut Criterion) {
    colog::init();
    let mut group = c.benchmark_group("cnvrl");

    for size in [1, 2, 4, 8].iter() {
        unsafe {
            KERNEL_HEIGHT = size * 3;
            KERNEL_WIDTH = size * 3;
            IMAGE_HEIGHT = size * 8;
            IMAGE_WIDTH = size * 8;
            IN_CHANNELS = 3;
            OUT_CHANNELS = 3;

            let mut image = Tensor::from(
                (0..IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)
                    .map(|_| Value::known(pallas::Base::random(OsRng))),
            );
            image.reshape(&[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);
            let mut kernels = Tensor::from(
                (0..{ OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH })
                    .map(|_| Value::known(pallas::Base::random(OsRng))),
            );
            kernels.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]);

            let bias = Tensor::from(
                (0..{ OUT_CHANNELS }).map(|_| Value::known(pallas::Base::random(OsRng))),
            );

            let circuit = MyCircuit::<pallas::Base> {
                image: ValTensor::from(image),
                kernel: ValTensor::from(kernels),
                bias: ValTensor::from(bias),
            };

            group.throughput(Throughput::Elements((IMAGE_HEIGHT * IMAGE_WIDTH) as u64));
            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_size| {
                b.iter(|| {
                    let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
                    prover.assert_satisfied();
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
