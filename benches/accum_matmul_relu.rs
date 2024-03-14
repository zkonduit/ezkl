use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::table::Range;
use ezkl::circuit::*;

use ezkl::circuit::lookup::LookupOp;
use ezkl::circuit::poly::PolyOp;
use ezkl::pfsys::create_proof_circuit;
use ezkl::pfsys::TranscriptType;
use ezkl::pfsys::{create_keys, srs::gen_srs};
use ezkl::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::marker::PhantomData;

const BITS: Range = (-32768, 32768);
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
}

impl Circuit<Fr> for MyCircuit {
    type Config = MyConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = unsafe { LEN };

        let a = VarTensor::new_advice(cs, K, 1, len);
        let b = VarTensor::new_advice(cs, K, 1, len);
        let output = VarTensor::new_advice(cs, K, 1, len);

        let mut base_config =
            BaseConfig::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::UNSAFE);

        // sets up a new relu table
        base_config
            .configure_lookup(cs, &b, &output, &a, BITS, K, &LookupOp::ReLU)
            .unwrap();

        MyConfig { base_config }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        config.base_config.layout_tables(&mut layouter).unwrap();
        layouter.assign_region(
            || "",
            |region| {
                let op = PolyOp::Einsum {
                    equation: "ij,jk->ik".to_string(),
                };
                let mut region = region::RegionCtx::new(region, 0, 1);
                let output = config
                    .base_config
                    .layout(&mut region, &self.inputs, Box::new(op))
                    .unwrap();
                let _output = config
                    .base_config
                    .layout(&mut region, &[output.unwrap()], Box::new(LookupOp::ReLU))
                    .unwrap();
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

        let mut a = Tensor::from((0..len * len).map(|_| Value::known(Fr::from(1))));
        a.reshape(&[len, len]).unwrap();

        // parameters
        let mut b = Tensor::from((0..len).map(|_| Value::known(Fr::from(1))));
        b.reshape(&[len, 1]).unwrap();

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", len), &len, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit>(&circuit, &params, true)
                    .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit>(&circuit, &params, true).unwrap();

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("prove", len), &len, |b, &_| {
            b.iter(|| {
                let prover = create_proof_circuit::<
                    KZGCommitmentScheme<_>,
                    MyCircuit,
                    ProverSHPLONK<_>,
                    VerifierSHPLONK<_>,
                    SingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit.clone(),
                    vec![],
                    &params,
                    &pk,
                    CheckMode::UNSAFE,
                    ezkl::Commitments::KZG,
                    TranscriptType::EVM,
                    None,
                    None,
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
  targets = runmatmul
}
criterion_main!(benches);
