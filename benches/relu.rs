use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::eltwise::{EltwiseConfig, Nonlin1d, Nonlinearity, ReLu};
use ezkl::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp as F;
use rand::Rng;
use std::marker::PhantomData;

const BITS: usize = 8;
static mut LEN: usize = 4;

#[derive(Clone)]
struct NLCircuit<F: FieldExt + TensorType, NL: Nonlinearity<F>> {
    assigned: Nonlin1d<F, NL>,
    _marker: PhantomData<NL>,
}

impl<F: FieldExt + TensorType, NL: 'static + Nonlinearity<F> + Clone> Circuit<F>
    for NLCircuit<F, NL>
{
    type Config = EltwiseConfig<F, NL>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        unsafe {
            let advices = VarTensor::Advice {
                inner: cs.advice_column(),
                dims: [LEN].to_vec(),
            };
            Self::Config::configure(cs, advices, Some(&[BITS, 128]))
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        config.layout(&mut layouter, self.assigned.input.clone());

        Ok(())
    }
}

fn runrelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");
    let k = 9;

    let mut rng = rand::thread_rng();

    for &len in [4, 8, 16, 32, 64, 128].iter() {
        unsafe {
            LEN = len;
        };

        let input: Tensor<Value<F>> =
            Tensor::<i32>::from((0..len).map(|_| rng.gen_range(0..10))).into();

        let assigned: Nonlin1d<F, ReLu<F>> = Nonlin1d {
            input: ValTensor::from(input.clone()),
            output: ValTensor::from(input),
            _marker: PhantomData,
        };

        let circuit = NLCircuit::<F, ReLu<F>> {
            assigned,
            _marker: PhantomData,
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
  targets = runrelu
}
criterion_main!(benches);
