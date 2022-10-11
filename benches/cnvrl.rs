use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::basic::*;
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

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        unsafe {
            let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
            let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

            let num_advices = output_height * OUT_CHANNELS
                + IMAGE_HEIGHT * IN_CHANNELS
                + OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT
                + 1;

            let advices =
                VarTensor::from(Tensor::from((0..num_advices).map(|_| meta.advice_column())));

            let input = advices.get_slice(
                &[0..IMAGE_HEIGHT * IN_CHANNELS],
                &[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            );

            let kernel = advices.get_slice(
                &[IMAGE_HEIGHT * IN_CHANNELS
                    ..IMAGE_HEIGHT * IN_CHANNELS + OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT],
                &[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH],
            );

            let bias = advices.get_slice(
                &[
                    IMAGE_HEIGHT * IN_CHANNELS + OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT
                        ..IMAGE_HEIGHT * IN_CHANNELS
                            + OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT
                            + 1,
                ],
                &[OUT_CHANNELS],
            );

            let output = advices.get_slice(
                &[
                    IMAGE_HEIGHT * IN_CHANNELS + OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT + 1
                        ..IMAGE_HEIGHT * IN_CHANNELS
                            + OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT
                            + 1
                            + output_height * OUT_CHANNELS,
                ],
                &[OUT_CHANNELS, output_height, output_width],
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

            Self::Config::configure(meta, &[input, kernel, bias, output], &[conv_node])
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
    let mut group = c.benchmark_group("cnvrl");

    let k = 8;

    for size in [1, 2, 4, 8, 16, 32].iter() {
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
                    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
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
