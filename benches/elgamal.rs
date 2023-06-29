use ark_std::test_rng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ezkl_lib::circuit::modules::elgamal::{
    ElGamalConfig, ElGamalGadget, ElGamalVariables, NUM_INSTANCE_COLUMNS,
};
use ezkl_lib::circuit::modules::Module;
use ezkl_lib::circuit::*;
use ezkl_lib::execute::create_proof_circuit_kzg;
use ezkl_lib::graph::modules::{ELGAMAL_CONSTRAINTS_ESTIMATE, ELGAMAL_FIXED_COST_ESTIMATE};
use ezkl_lib::pfsys::create_keys;
use ezkl_lib::pfsys::srs::gen_srs;
use ezkl_lib::pfsys::TranscriptType;
use ezkl_lib::tensor::*;
use halo2_proofs::circuit::Value;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::{Bn256, Fr};
use rand::rngs::OsRng;

#[derive(Clone, Debug)]
struct EncryptytionCircuit {
    message: ValTensor<Fr>,
    variables: ElGamalVariables,
}

impl Circuit<Fr> for EncryptytionCircuit {
    type Config = ElGamalConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        ElGamalGadget::configure(cs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        let mut chip = ElGamalGadget::new(config);
        chip.load_variables(self.variables.clone());
        let sk: Tensor<ValType<Fr>> =
            Tensor::new(Some(&[Value::known(self.variables.sk).into()]), &[1]).unwrap();
        chip.layout(
            &mut layouter,
            &[self.message.clone(), sk.into()],
            vec![0; NUM_INSTANCE_COLUMNS],
        )?;
        Ok(())
    }
}

fn runelgamal(c: &mut Criterion) {
    let mut group = c.benchmark_group("elgamal");

    for size in [64, 784, 65536, 4194304].iter() {
        let mut rng = test_rng();

        let k = ((size * ELGAMAL_CONSTRAINTS_ESTIMATE + ELGAMAL_FIXED_COST_ESTIMATE) as f32)
            .log2()
            .ceil() as u32;
        let params = gen_srs::<KZGCommitmentScheme<_>>(k);

        let var = ElGamalVariables::gen_random(&mut rng);

        let message = (0..*size).map(|_| Fr::random(OsRng)).collect::<Vec<_>>();

        let run_inputs = (message.clone(), var.clone());
        let public_inputs: Vec<Vec<Fr>> = ElGamalGadget::run(run_inputs).unwrap();

        let mut message: Tensor<ValType<Fr>> =
            message.into_iter().map(|m| Value::known(m).into()).into();
        message.reshape(&[1, *size]);

        let circuit = EncryptytionCircuit {
            message: message.into(),
            variables: var,
        };

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("pk", size), &size, |b, &_| {
            b.iter(|| {
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, EncryptytionCircuit>(
                    &circuit, &params,
                )
                .unwrap();
            });
        });

        let pk =
            create_keys::<KZGCommitmentScheme<Bn256>, Fr, EncryptytionCircuit>(&circuit, &params)
                .unwrap();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("prove", size), &size, |b, &_| {
            b.iter(|| {
                let prover = create_proof_circuit_kzg(
                    circuit.clone(),
                    &params,
                    public_inputs.clone(),
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
  targets = runelgamal
}
criterion_main!(benches);
