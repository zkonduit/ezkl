/// Aggregation circuit
pub mod aggregation;

use super::prepare_circuit_and_public_input;
use super::ModelInput;
use crate::fieldutils::i32_to_felt;
#[cfg(feature = "evm")]
use aggregation::Plonk;
use aggregation::{PoseidonTranscript, Snark};
#[cfg(feature = "evm")]
use ethereum_types::Address;
#[cfg(feature = "evm")]
use foundry_evm::executor::{fork::MultiFork, Backend, ExecutorBuilder};
#[cfg(feature = "evm")]
use halo2_proofs::plonk::VerifyingKey;
use halo2_proofs::{
    dev::MockProver,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey},
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer},
};
#[cfg(feature = "evm")]
use halo2curves::bn256::Fq;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use itertools::Itertools;
use log::trace;
#[cfg(feature = "evm")]
use plonk_verifier::{
    loader::evm::{encode_calldata, EvmLoader},
    system::halo2::transcript::evm::EvmTranscript,
    verifier::PlonkVerifier,
};
use plonk_verifier::{
    loader::native::NativeLoader,
    system::halo2::{compile, Config},
};

use rand::rngs::OsRng;
use std::io::Cursor;
#[cfg(feature = "evm")]
use std::rc::Rc;

pub fn gen_application_snark(params: &ParamsKZG<Bn256>, data: &ModelInput) -> Snark {
    let (circuit, public_inputs) = prepare_circuit_and_public_input::<Fr>(data);

    let pk = gen_pk(params, &circuit);
    let number_instance = public_inputs.clone()[0].len();
    trace!("number_instance {:?}", number_instance);
    let protocol = compile(
        params,
        pk.get_vk(),
        Config::kzg().with_num_instance(vec![number_instance]),
    );
    let pi_inner: Vec<Vec<Fr>> = public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i32_to_felt::<Fr>(*e)).collect::<Vec<Fr>>())
        .collect::<Vec<Vec<Fr>>>();
    //    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[Fr]>>();
    trace!("pi_inner {:?}", pi_inner);
    let proof = gen_kzg_proof::<
        _,
        _,
        PoseidonTranscript<NativeLoader, _>,
        PoseidonTranscript<NativeLoader, _>,
    >(params, &pk, circuit.clone(), pi_inner.clone());
    Snark::new(protocol, pi_inner, proof)
}

#[cfg(feature = "evm")]
pub fn gen_aggregation_evm_verifier(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    num_instance: Vec<usize>,
    accumulator_indices: Vec<(usize, usize)>,
) -> Vec<u8> {
    let svk = params.get_g()[0].into();
    let dk = (params.g2(), params.s_g2()).into();
    let protocol = compile(
        params,
        vk,
        Config::kzg()
            .with_num_instance(num_instance.clone())
            .with_accumulator_indices(accumulator_indices),
    );

    let loader = EvmLoader::new::<Fq, Fr>();
    let mut transcript = EvmTranscript::<_, Rc<EvmLoader>, _, _>::new(loader.clone());

    let instances = transcript.load_instances(num_instance);
    let proof = Plonk::read_proof(&svk, &protocol, &instances, &mut transcript).unwrap();
    Plonk::verify(&svk, &dk, &protocol, &instances, &proof).unwrap();

    loader.deployment_code()
}

#[cfg(feature = "evm")]
pub fn evm_verify(deployment_code: Vec<u8>, instances: Vec<Vec<Fr>>, proof: Vec<u8>) {
    let calldata = encode_calldata(&instances, &proof);
    let success = {
        let mut evm = ExecutorBuilder::default()
            .with_gas_limit(u64::MAX.into())
            .build(Backend::new(MultiFork::new().0, None));

        let caller = Address::from_low_u64_be(0xfe);
        let verifier = evm
            .deploy(caller, deployment_code.into(), 0.into(), None)
            .unwrap()
            .address;
        let result = evm
            .call_raw(caller, verifier, calldata.into(), 0.into())
            .unwrap();

        dbg!(result.gas_used);

        !result.reverted
    };
    assert!(success);
}

pub fn gen_srs(k: u32) -> ParamsKZG<Bn256> {
    ParamsKZG::<Bn256>::setup(k, OsRng)
}

pub fn gen_pk<C: Circuit<Fr>>(params: &ParamsKZG<Bn256>, circuit: &C) -> ProvingKey<G1Affine> {
    let vk = keygen_vk(params, circuit).unwrap();
    keygen_pk(params, vk, circuit).unwrap()
}

/// Generates proof for either application circuit (model) or aggregation circuit.
pub fn gen_kzg_proof<
    C: Circuit<Fr>,
    E: EncodedChallenge<G1Affine>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, G1Affine, E>,
    TW: TranscriptWriterBuffer<Vec<u8>, G1Affine, E>,
>(
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
    circuit: C,
    instances: Vec<Vec<Fr>>,
) -> Vec<u8> {
    MockProver::run(params.k(), &circuit, instances.clone())
        .unwrap()
        .assert_satisfied();

    let instances = instances
        .iter()
        .map(|instances| instances.as_slice())
        .collect_vec();
    let proof = {
        let mut transcript = TW::init(Vec::new());
        create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, TW, _>(
            params,
            pk,
            &[circuit],
            &[instances.as_slice()],
            OsRng,
            &mut transcript,
        )
        .unwrap();
        transcript.finalize()
    };

    let accept = {
        let mut transcript = TR::init(Cursor::new(proof.clone()));
        VerificationStrategy::<_, VerifierGWC<_>>::finalize(
            verify_proof::<_, VerifierGWC<_>, _, TR, _>(
                params.verifier_params(),
                pk.get_vk(),
                AccumulatorStrategy::new(params.verifier_params()),
                &[instances.as_slice()],
                &mut transcript,
            )
            .unwrap(),
        )
    };
    assert!(accept);

    proof
}
