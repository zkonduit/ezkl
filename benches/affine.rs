use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::{Field, FieldExt},
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::pallas;
use halo2curves::pasta::Fp as F;
use halo2deeplearning::nn::affine::Affine1dConfig;
use halo2deeplearning::nn::*;
use halo2deeplearning::tensor::*;
use rand::rngs::OsRng;
use std::marker::PhantomData;

static mut LEN: usize = 4;

#[derive(Clone)]
struct MyCircuit<F: FieldExt + TensorType> {
    input: ValTensor<F>,
    l0_params: [ValTensor<F>; 2],
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
    type Config = Affine1dConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let len = unsafe { LEN };
        let advices = VarTensor::from(Tensor::from((0..len + 3).map(|_| {
            let col = cs.advice_column();
            cs.enable_equality(col);
            col
        })));

        let kernel = advices.get_slice(&[0..len], &[len, len]);
        let bias = advices.get_slice(&[len + 2..len + 3], &[len]);
        let input = advices.get_slice(&[len..len + 1], &[len]);
        let output = advices.get_slice(&[len + 1..len + 2], &[len]);

        Self::Config::configure(
            cs,
            &[kernel.clone(), bias.clone(), input.clone(), output.clone()], None
        )
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.layout(
            &mut layouter,
            &[

                self.l0_params[0].clone(),
                self.l0_params[1].clone(),
                self.input.clone(),
            ],
        );
        Ok(())
    }
}

fn runaffine(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine");
    for &len in [4, 8, 16, 32, 64, 128].iter() {
        unsafe {
            LEN = len;
        };

        let k = 15; //2^k rows
                    // parameters
        let mut l0_kernel =
            Tensor::from((0..len * len).map(|_| Value::known(pallas::Base::random(OsRng))));
        l0_kernel.reshape(&[len, len]);

        let l0_bias = Tensor::from((0..len).map(|_| Value::known(pallas::Base::random(OsRng))));

        let input = Tensor::from((0..len).map(|_| Value::known(pallas::Base::random(OsRng))));

        let circuit = MyCircuit::<F> {
            input: ValTensor::from(input),
            l0_params: [ValTensor::from(l0_kernel), ValTensor::from(l0_bias)],
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
  targets = runaffine
}
criterion_main!(benches);
