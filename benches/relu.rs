use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl::circuit::region::RegionCtx;
use ezkl::circuit::{ops::lookup::LookupOp, BaseConfig as Config, CheckMode};
use ezkl::execute::create_proof_circuit_kzg;
use ezkl::pfsys::TranscriptType;
use ezkl::pfsys::{create_keys, srs::gen_srs};
use ezkl::tensor::*;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::Rng;

const BITS: usize = 8;
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
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, K, LEN))
                .collect::<Vec<_>>();

            let nl = LookupOp::ReLU { scale: 128 };

            let mut config = Config::default();

            config
                .configure_lookup(cs, &advices[0], &advices[1], BITS, &nl)
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
                let mut region = RegionCtx::new(region, 0);
                config
                    .layout(
                        &mut region,
                        &[self.input.clone()],
                        Box::new(LookupOp::ReLU { scale: 128 }),
                    )
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
            Tensor::<i32>::from((0..len).map(|_| rng.gen_range(0..10))).into();

        let circuit = NLCircuit {
            input: ValTensor::from(input.clone()),
        };

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::new("pk", len), &len, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, NLCircuit>(&circuit, &params)
                    .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, Fr, NLCircuit>(&circuit, &params).unwrap();

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
  targets = runrelu
}
criterion_main!(benches);
