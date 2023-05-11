use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::poly::PolyOp;
use ezkl_lib::circuit::*;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::pfsys::TranscriptType;
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
    inputs: [ValTensor<Fr>; 1],
    _marker: PhantomData<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = unsafe { LEN };

        let a = VarTensor::new_advice(cs, K, len);
        let b = VarTensor::new_advice(cs, K, len);
        let output = VarTensor::new_advice(cs, K, len);

        Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |mut region| {
                config
                    .layout(
                        &mut Some(&mut region),
                        &self.inputs,
                        &mut 0,
                        Box::new(PolyOp::Pow(4)),
                    )
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runpow(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_pow");
    let params = gen_srs::<KZGCommitmentScheme<_>>(17);
    for &len in [16, 512].iter() {
        unsafe {
            LEN = len;
        };

        // parameters
        let a = Tensor::from((0..len).map(|_| Value::known(Fr::random(OsRng))));

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a)],
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
                    CheckMode::SAFE,
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
  targets = runpow
}
criterion_main!(benches);
