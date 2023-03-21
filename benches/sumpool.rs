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

static mut IMAGE_HEIGHT: usize = 2;
static mut IMAGE_WIDTH: usize = 2;
static mut IN_CHANNELS: usize = 3;

const K: usize = 17;

#[derive(Clone, Debug)]
struct MyCircuit {
    image: ValTensor<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = Config<Fr>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        unsafe {
            let output_height = (IMAGE_HEIGHT - 2) + 1;
            let output_width = (IMAGE_WIDTH - 2) + 1;

            let input = VarTensor::new_advice(
                cs,
                K,
                IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH,
                vec![IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
                true,
                512,
            );

            let output = VarTensor::new_advice(
                cs,
                K,
                3 * output_height * output_width,
                vec![3, output_height, output_width],
                true,
                512,
            );

            // tells the config layer to add a conv op to a circuit gate
            let sumpool_node = Node {
                op: Op::SumPool {
                    padding: (0, 0),
                    stride: (1, 1),
                    kernel_shape: (2, 2),
                },
                input_order: vec![InputType::Input(0)],
            };

            Self::Config::configure(cs, &[input], &output, &[sumpool_node])
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let _output = config.layout(&mut layouter, &[self.image.clone()]).unwrap();
        Ok(())
    }
}

fn runsumpool(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumpool");

    let params = gen_srs::<KZGCommitmentScheme<_>>(K as u32);

    for size in [1].iter() {
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
