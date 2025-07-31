use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::einsum::analysis::analyze_einsum_usage;
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::pfsys::create_proof_circuit;
use ezkl::pfsys::TranscriptType;
use ezkl::pfsys::{create_keys, srs::gen_srs};
use ezkl::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use rand::rngs::OsRng;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::collections::HashMap;
use std::marker::PhantomData;

static mut LEN: usize = 4;
const K: usize = 16;

#[derive(Clone)]
struct MyCircuit {
    inputs: [ValTensor<Fr>; 2],
    einsum: Einsum,
    _marker: PhantomData<Fr>,
}

struct Einsum {
    equation: String,
    input_to_dims: HashMap<char, usize>,
}

impl Einsum {
    pub fn new() -> Self {
        todo!()
    }
    
    pub fn get_challenges<F: PrimeField + TensorType + PartialOrd>(
        &self,
        config: BaseConfig<F>,
        mut layouter: impl Layouter<Fr>,
    ) -> Vec<ValTensor<F>> {
        let challenges = config.challenges.iter().map(|c| layouter.get_challenge(c));
        let powers = output.chars().map(|c| self.input_to_dims.get(c).unwrap());

        challenges.zip(powers).map(|(challenge, power)| {
            // Use `scan` to compute [challenge^i], i = 1..=power
        })
    }
}

impl Circuit<Fr> for MyCircuit {
    type Config = BaseConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(&self, cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let len = unsafe { LEN };

        let a = VarTensor::new_advice(cs, K, 1, len * len);

        let b = VarTensor::new_advice(cs, K, 1, len * len);

        let output = VarTensor::new_advice(cs, K, 1, (len + 1) * len);

        let mut config = Self::Config::default();

        let equations = HashMap::new();
        equations.insert(self.einsum.equation, self.einsum.input_to_dims);
        let analysis = analyze_einsum_usage(&equations)?;
        config.configure_einsums(cs, analysis)?;

        config
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let challenges = ();

        layouter.assign_region(
            || "",
            |region| {
                let mut region = region::RegionCtx::new(region, 0, 1, 1024, 2);
                config
                    .layout(
                        &mut region,
                        &self.inputs.iter().collect_vec(),
                        Box::new(PolyOp::Einsum {
                            equation: self.einsum.equation,
                        }),
                    )
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runmatmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("accum_einsum_matmul");
    let params = gen_srs::<KZGCommitmentScheme<_>>(17);
    for &len in [4, 32].iter() {
        unsafe {
            LEN = len;
        };

        let mut a = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        a.reshape(&[len, len]).unwrap();

        // parameters
        let mut b = Tensor::from((0..len * len).map(|_| Value::known(Fr::random(OsRng))));
        b.reshape(&[len, len]).unwrap();

        let einsum = Einsum {

        }

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
