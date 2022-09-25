use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::Fp as F;
use halo2deeplearning::nn::affine::Affine1dConfig;
use halo2deeplearning::nn::*;
use halo2deeplearning::tensor::*;
use rand::Rng;
use std::marker::PhantomData;

static mut LEN: usize = 4;

#[derive(Clone)]
struct MyCircuit<F: FieldExt> {
    input: Tensor<i32>,
    l0_params: [Tensor<i32>; 2],
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

        Self::Config::configure(
            cs,
            &[kernel.clone(), bias.clone()],
            advices.get_slice(&[len..len + 1], &[len]),
            advices.get_slice(&[len + 1..len + 2], &[len]),
        )
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let x: Tensor<Value<F>> = self.input.clone().into();
        config.layout(
            &mut layouter,
            ValTensor::from(x),
            &self
                .l0_params
                .iter()
                .map(|a| ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(a.clone())))
                .collect::<Vec<ValTensor<F>>>(),
        );
        Ok(())
    }
}

fn runaffine(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine");
    for size in [1, 2, 4, 8, 16, 32].iter() {
        let len = unsafe {
            LEN = size * 4;
            LEN
        };
        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &_| {
            b.iter(|| {
                let k = 15; //2^k rows
                            // parameters
                let mut rng = rand::thread_rng();

                let w = (0..len * len)
                    .map(|_| rng.gen_range(0..10))
                    .collect::<Vec<_>>();
                let l0_kernel = Tensor::<i32>::new(Some(&w), &[len, len]).unwrap();
                let b = (0..len).map(|_| rng.gen_range(0..10)).collect::<Vec<_>>();
                let l0_bias = Tensor::<i32>::new(Some(&b), &[len]).unwrap();

                let input = (0..len).map(|_| rng.gen_range(0..10)).collect::<Vec<_>>();
                // input data, with 1 padding to allow for bias
                let input = Tensor::<i32>::new(Some(&input), &[len]).unwrap();

                let circuit = MyCircuit::<F> {
                    input,
                    l0_params: [l0_kernel, l0_bias],
                    _marker: PhantomData,
                };
                let prover = MockProver::run(k, &circuit, vec![]).unwrap();
                prover.assert_satisfied();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, runaffine);
criterion_main!(benches);
