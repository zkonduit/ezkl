use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::range::*;
use ezkl_lib::tensor::*;
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

const K: usize = 15; //2^k rows

#[derive(Clone)]
struct MyCircuit<F: FieldExt + TensorType> {
    input: ValTensor<F>,
    output: ValTensor<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
    type Config = RangeCheckConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let len = unsafe { LEN };
        let advices = (0..2)
            .map(|_| VarTensor::new_advice(cs, K, len, vec![len], true, 512))
            .collect_vec();

        RangeCheckConfig::configure(cs, &advices[0], &advices[1], RANGE)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config
            .layout(
                layouter.namespace(|| "Assign value"),
                self.input.clone(),
                self.output.clone(),
            )
            .unwrap();

        Ok(())
    }
}

fn runrange(c: &mut Criterion) {
    let mut group = c.benchmark_group("range");
    for &len in [4, 8, 16, 32, 64, 128].iter() {
        unsafe {
            LEN = len;
        };

        let input = Tensor::from((0..len).map(|_| Value::known(pallas::Base::from(1))));

        let circuit = MyCircuit::<F> {
            input: ValTensor::from(input.clone()),
            output: ValTensor::from(input.clone()),
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
  targets = runrange
}
criterion_main!(benches);
