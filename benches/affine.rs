use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::polynomial::*;
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
use std::marker::PhantomData;

static mut LEN: usize = 4;
const K: usize = 16;

#[derive(Clone)]
struct MyCircuit<F: FieldExt + TensorType> {
    input: ValTensor<F>,
    l0_params: [ValTensor<F>; 2],
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
    type Config = Config<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let len = unsafe { LEN };

        let input = VarTensor::new_advice(cs, K, len, vec![len], true, 512);
        let kernel = VarTensor::new_advice(cs, K, len * len, vec![len, len], true, 512);
        let bias = VarTensor::new_advice(cs, K, len, vec![len], true, 512);
        let output = VarTensor::new_advice(cs, K, len, vec![len], true, 512);
        // tells the config layer to add an affine op to a circuit gate
        let affine_node = Node {
            op: Op::Affine,
            input_order: vec![
                InputType::Input(0),
                InputType::Input(1),
                InputType::Input(2),
            ],
        };

        Self::Config::configure(cs, &[input, kernel, bias], &output, &[affine_node])
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config
            .layout(
                &mut layouter,
                &[
                    self.input.clone(),
                    self.l0_params[0].clone(),
                    self.l0_params[1].clone(),
                ],
            )
            .unwrap();
        Ok(())
    }
}

fn runaffine(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine");
    for &len in [4, 8, 16, 32, 64].iter() {
        unsafe {
            LEN = len;
        };

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
  targets = runaffine
}
criterion_main!(benches);
