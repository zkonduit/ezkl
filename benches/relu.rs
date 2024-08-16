use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::region::RegionCtx;
use ezkl::circuit::table::Range;
use ezkl::circuit::{ops::lookup::LookupOp, BaseConfig as Config, CheckMode};
use ezkl::fieldutils::IntegerRep;
use ezkl::pfsys::create_proof_circuit;
use ezkl::pfsys::TranscriptType;
use ezkl::pfsys::{create_keys, srs::gen_srs};
use ezkl::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::Rng;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

const BITS: Range = (-32768, 32768);
static mut LEN: usize = 4;
const K: usize = 16;

#[derive(Clone)]
struct NLCircuit {
    pub input: ValTensor<Fr>,
}

impl Circuit<Fr> for NLCircuit {
    type Config = Config<Fr>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        unsafe {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, K, 1, LEN))
                .collect::<Vec<_>>();

            let nl = LookupOp::ReLU;

            let mut config = Config::default();

            config
                .configure_lookup(cs, &advices[0], &advices[1], &advices[2], BITS, K, &nl)
                .unwrap();

            config
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        config.layout_tables(&mut layouter).unwrap();
        layouter.assign_region(
            || "",
            |region| {
                let mut region = RegionCtx::new(region, 0, 1);
                config
                    .layout(&mut region, &[self.input.clone()], Box::new(LookupOp::ReLU))
                    .unwrap();
                Ok(())
            },
        )?;
        Ok(())
    }
}

fn runrelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    let mut rng = rand::thread_rng();
    let params = gen_srs::<KZGCommitmentScheme<_>>(17);
    for &len in [4, 8].iter() {
        unsafe {
            LEN = len;
        };

        let input: Tensor<Value<Fr>> =
            Tensor::<IntegerRep>::from((0..len).map(|_| rng.gen_range(0..10))).into();

        let circuit = NLCircuit {
            input: ValTensor::from(input.clone()),
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", len), &len, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, NLCircuit>(&circuit, &params, true)
                    .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, NLCircuit>(&circuit, &params, true).unwrap();

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("prove", len), &len, |b, &_| {
            b.iter(|| {
                let prover = create_proof_circuit::<
                    KZGCommitmentScheme<_>,
                    NLCircuit,
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
  targets = runrelu
}
criterion_main!(benches);
