use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::base::CheckMode;
use ezkl_lib::circuit::fused::*;
use ezkl_lib::commands::TranscriptType;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::pfsys::{create_keys, gen_srs};
use ezkl_lib::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;
use std::marker::PhantomData;

static mut LEN: usize = 4;
const K: usize = 16;

#[derive(Clone)]
struct MyCircuit {
    inputs: [ValTensor<Fr>; 3],
    _marker: PhantomData<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = Config<Fr>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = unsafe { LEN };

        let input = VarTensor::new_advice(cs, K, len, vec![len], true);
        let kernel = VarTensor::new_advice(cs, K, len * len, vec![len, len], true);
        let bias = VarTensor::new_advice(cs, K, len, vec![len], true);
        let output = VarTensor::new_advice(cs, K, len, vec![len], true);
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
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        config.layout(&mut layouter, &self.inputs).unwrap();
        Ok(())
    }
}

fn runaffine(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine");
    let params = gen_srs::<KZGCommitmentScheme<_>>(17);
    for &len in [4].iter() {
        unsafe {
            LEN = len;
        };

        // parameters
        let mut kernel = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        kernel.reshape(&[len, len]);

        let bias = Tensor::from((0..len).map(|_| Value::known(Fr::random(OsRng))));

        let input = Tensor::from((0..len).map(|_| Value::known(Fr::random(OsRng))));

        let circuit = MyCircuit {
            inputs: [
                ValTensor::from(input),
                ValTensor::from(kernel),
                ValTensor::from(bias),
            ],
            _marker: PhantomData,
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", len), &len, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(&circuit, &params)
                    .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, Fr, MyCircuit>(&circuit, &params).unwrap();

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("prove", len), &len, |b, &_| {
            b.iter(|| {
                let prover = create_proof_circuit_kzg(
                    circuit.clone(),
                    &params,
                    vec![],
                    &pk,
                    TranscriptType::Blake,
                    SingleStrategy::new(&params),
                    CheckMode::UNSAFE,
                );
                prover.unwrap();
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
