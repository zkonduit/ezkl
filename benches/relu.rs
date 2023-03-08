use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::lookup::{Config, Op};
use ezkl::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp as F;
use rand::Rng;

const BITS: usize = 8;
static mut LEN: usize = 4;
const K: usize = 10;

#[derive(Clone)]
struct NLCircuit<F: FieldExt + TensorType> {
    pub input: ValTensor<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for NLCircuit<F> {
    type Config = Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        unsafe {
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512))
                .collect::<Vec<_>>();

            let nl = Op::ReLU { scale: 128 };

            Self::Config::configure(cs, &advices[0], &advices[1], BITS, &[nl])
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        config.layout(&mut layouter, &self.input).unwrap();

        Ok(())
    }
}

fn runrelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    let mut rng = rand::thread_rng();

    for &len in [4, 8, 16, 32, 64].iter() {
        unsafe {
            LEN = len;
        };

        let input: Tensor<Value<F>> =
            Tensor::<i32>::from((0..len).map(|_| rng.gen_range(0..10))).into();

        let circuit = NLCircuit::<F> {
            input: ValTensor::from(input.clone()),
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &_| {
            b.iter(|| {
                let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
                prover.assert_satisfied();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runrelu
}
criterion_main!(benches);
