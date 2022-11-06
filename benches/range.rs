use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::range::*;
use ezkl::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::{Field, FieldExt},
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::pallas;
use halo2curves::pasta::Fp as F;
use rand::rngs::OsRng;

static mut LEN: usize = 4;

#[derive(Clone)]
struct MyCircuit<F: FieldExt + TensorType, const RANGE: usize> {
    input: ValTensor<F>,
    output: ValTensor<F>,
}

impl<F: FieldExt + TensorType, const RANGE: usize> Circuit<F> for MyCircuit<F, RANGE> {
    type Config = RangeCheckConfig<F, RANGE>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let len = unsafe { LEN };
        let advices = VarTensor::from(Tensor::from((0..2).map(|_| {
            let col = cs.advice_column();
            cs.enable_equality(col);
            col
        })));
        let input = advices.get_slice(&[0..1], &[len]);
        let output = advices.get_slice(&[1..2], &[len]);
        RangeCheckConfig::configure(cs, &input, &output)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.layout(layouter.namespace(|| "Assign value"), self.input.clone(), self.output.clone());

        Ok(())
    }
}

fn runrange(c: &mut Criterion) {
    let mut group = c.benchmark_group("range");
    for &len in [4, 8, 16, 32, 64, 128].iter() {
        unsafe {
            LEN = len;
        };

        let k = 15; //2^k rows
        const RANGE: usize = 8; // 3-bit value

        let input = Tensor::from((0..len).map(|_| Value::known(pallas::Base::random(OsRng))));

        let circuit = MyCircuit::<F, RANGE> {
            input: ValTensor::from(input.clone()),
            output: ValTensor::from(input),
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &_| {
            b.iter(|| {
                let prover = MockProver::run(k, &circuit, vec![]).unwrap();
                prover.assert_satisfied();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runrange
}
criterion_main!(benches);
