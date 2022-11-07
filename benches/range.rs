use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::range::*;
use ezkl::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::pallas;
use halo2curves::pasta::Fp as F;
use itertools::Itertools;

static mut LEN: usize = 4;
const RANGE: usize = 8; // 3-bit value

#[derive(Clone)]
struct MyCircuit<F: FieldExt + TensorType> {
    input: ValTensor<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
    type Config = RangeCheckConfig<F>;
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
        let instance = {
            let l = cs.instance_column();
            cs.enable_equality(l);
            l
        };

        RangeCheckConfig::configure(cs, &input, &output, &instance, RANGE)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.layout(layouter.namespace(|| "Assign value"), self.input.clone());

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

        let input = Tensor::from((0..len).map(|_| Value::known(pallas::Base::from(1))));

        let circuit = MyCircuit::<F> {
            input: ValTensor::from(input.clone()),
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &_| {
            b.iter(|| {
                let instances = vec![(0..len).map(|_| F::from(1)).collect_vec()];
                let prover = MockProver::run(k, &circuit, instances).unwrap();
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
