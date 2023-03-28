use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::base::*;
use ezkl_lib::circuit::lookup::{Config as LookupConfig, Op as LookupOp};

use ezkl_lib::commands::TranscriptType;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::pfsys::{create_keys, gen_srs};
use ezkl_lib::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use std::marker::PhantomData;

const BITS: usize = 8;
static mut LEN: usize = 4;
const K: usize = 16;

#[derive(Clone)]
struct MyCircuit {
    inputs: [ValTensor<Fr>; 2],
    _marker: PhantomData<Fr>,
}

// A columnar ReLu MLP
#[derive(Clone)]
struct MyConfig {
    base_config: BaseConfig<Fr>,
    l1: LookupConfig<Fr>,
}

impl Circuit<Fr> for MyCircuit {
    type Config = MyConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = unsafe { LEN };

        let a = VarTensor::new_advice(cs, K, len, true);
        let b = VarTensor::new_advice(cs, K, len, true);
        let output = VarTensor::new_advice(cs, K, len, true);

        // sets up a new relu table
        let l1 = LookupConfig::configure(cs, &b, &output, BITS, &[LookupOp::ReLU { scale: 1 }]);

        let base_config = BaseConfig::configure(cs, &[a, b], &output, CheckMode::UNSAFE, 0);
        MyConfig { base_config, l1 }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        config.l1.layout_table(&mut layouter).unwrap();
        layouter.assign_region(
            || "",
            |mut region| {
                let op = Op::Matmul;
                let mut offset = 0;
                let output = config
                    .base_config
                    .layout(&mut region, &self.inputs, &mut offset, op.clone())
                    .unwrap();
                println!("offset: {}", offset);
                let _output = config.l1.layout(&mut region, &output, &mut offset).unwrap();
                Ok(())
            },
        )?;

        Ok(())
    }
}

fn runmatmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("accum_matmul");
    let params = gen_srs::<KZGCommitmentScheme<_>>(17);
    for &len in [4, 32].iter() {
        unsafe {
            LEN = len;
        };

        let mut a = Tensor::from((0..len * len).map(|i| Value::known(Fr::from(i as u64))));
        a.reshape(&[len, len]);

        // parameters
        let mut b = Tensor::from((0..len).map(|i| Value::known(Fr::from(i as u64))));
        b.reshape(&[len, 1]);

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
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
                println!("Prover: {:?}", prover);
                prover.unwrap();
            });
        });
    }
    group.finish();
}

criterion_group! {
  name = benches;
  config = Criterion::default().with_plots();
  targets = runmatmul
}
criterion_main!(benches);
