use super::prepare_circuit_and_public_input;
use super::OnnxInput;
use crate::aggregation;
use ethereum_types::Address;
use foundry_evm::executor::{fork::MultiFork, Backend, ExecutorBuilder};
use itertools::Itertools;
use plonk_verifier::{
    loader::{
        evm::{encode_calldata, EvmLoader},
        native::NativeLoader,
    },
    system::halo2::{compile, transcript::evm::EvmTranscript, Config},
    verifier::PlonkVerifier,
};
use std::{io::Cursor, rc::Rc};
//
//use crate::commands::{data_path, Cli};
use crate::fieldutils::i32_to_felt;
//use crate::onnx::{utilities::vector_to_quantized, OnnxCircuit, OnnxModel};
//use crate::tensor::Tensor;
use halo2_proofs::{
    //    arithmetic::FieldExt,
    dev::MockProver,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey},
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
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine};
use log::trace;
use rand::rngs::OsRng;

pub fn gen_application_snark(params: &ParamsKZG<Bn256>, data: &OnnxInput) -> aggregation::Snark {
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
        aggregation::PoseidonTranscript<NativeLoader, _>,
        aggregation::PoseidonTranscript<NativeLoader, _>,
    >(params, &pk, circuit.clone(), pi_inner.clone());
    aggregation::Snark::new(protocol, pi_inner, proof)
}

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
    let proof =
        aggregation::Plonk::read_proof(&svk, &protocol, &instances, &mut transcript).unwrap();
    aggregation::Plonk::verify(&svk, &dk, &protocol, &instances, &proof).unwrap();

    loader.deployment_code()
}

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
