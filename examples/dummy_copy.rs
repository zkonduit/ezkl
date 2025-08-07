use ezkl::circuit::einsum::analysis::analyze_einsum_usage;
use ezkl::circuit::einsum::analysis::analyze_single_equation;
use ezkl::circuit::poly::PolyOp;
use ezkl::circuit::*;
use ezkl::pfsys::TranscriptType;
use ezkl::pfsys::{create_keys, srs::gen_srs};
use ezkl::pfsys::{create_proof_circuit, verify_proof_circuit};
use ezkl::tensor::*;
use halo2_proofs::circuit::floor_planner::V1;
use halo2_proofs::plonk::{Advice, Column, SecondPhase, Selector};
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverSHPLONK;
use halo2_proofs::poly::kzg::multiopen::VerifierSHPLONK;
use halo2_proofs::poly::kzg::strategy::SingleStrategy;
use halo2_proofs::poly::Rotation;
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, Value},
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
const K: usize = 8;

#[derive(Clone)]
struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
    value: Value<F>,
}

#[derive(Clone)]
struct MyConfig {
    phase_one: Column<Advice>,
    phase_two: Column<Advice>,
    selector: Selector,
}

impl Circuit<Fr> for MyCircuit<Fr> {
    type Config = MyConfig;
    type FloorPlanner = V1;
    type Params = ();

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<Fr>, _params: Self::Params) -> Self::Config {
        Self::configure(cs)
    }

    fn params(&self) -> Self::Params {}

    fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
        let phase_one = cs.advice_column();
        let phase_two = cs.advice_column_in(SecondPhase);
        let selector = cs.selector();

        cs.enable_equality(phase_one);
        cs.enable_equality(phase_two);

        cs.create_gate("", |cs| {
            let selector = cs.query_selector(selector);
            let phase_one = cs.query_advice(phase_one, Rotation::cur());
            let phase_two = cs.query_advice(phase_two, Rotation::cur());

            vec![selector * (phase_one - phase_two)]
        });

        MyConfig {
            phase_one,
            phase_two,
            selector,
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "",
            |mut region| {
                config.selector.enable(&mut region, 0)?;
                let cell = region.assign_advice(|| "", config.phase_one, 0, || self.value)?;
                cell.copy_advice(|| "", &mut region, config.phase_two, 0)?;
                Ok(())
            },
        )
    }
}

fn run_copy() {
    let params = gen_srs::<KZGCommitmentScheme<_>>(10);

    let circuit = MyCircuit {
        value: Value::known(Fr::random(OsRng)),
    };

    let pk =
        create_keys::<KZGCommitmentScheme<Bn256>, MyCircuit<Fr>>(&circuit, &params, true).unwrap();

    let prover = create_proof_circuit::<
        KZGCommitmentScheme<_>,
        MyCircuit<Fr>,
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
    let strategy = SingleStrategy::new(&params);

    let checkable_pf = prover.unwrap();
    let params = params.verifier_params();
    verify_proof_circuit::<
        VerifierSHPLONK<'_, Bn256>,
        KZGCommitmentScheme<Bn256>,
        SingleStrategy<_>,
        _,
        EvmTranscript<_, _, _, _>,
    >(&checkable_pf, params, pk.get_vk(), strategy, params.n())
    .unwrap();
}

pub fn main() {
    run_copy()
}
