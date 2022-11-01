use crate::abort;
use clap::Parser;
//
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
use crate::commands::{data_path, Cli};
use crate::fieldutils::i32_to_felt;
use crate::onnx::{utilities::vector_to_quantized, OnnxCircuit, OnnxModel};
use crate::tensor::Tensor;
use halo2_proofs::{
    arithmetic::FieldExt,
    dev::{MockProver, VerifyFailure},
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey},
    poly::{
        commitment::{Params, ParamsProver},
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, TranscriptReadBuffer,
        TranscriptWriterBuffer,
    },
};
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine};
use halo2curves::pasta::vesta;
use halo2curves::pasta::Fp;
use halo2curves::pasta::{EqAffine, Fp as F};
use log::{error, info, trace};
use rand::rngs::OsRng;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::ops::Deref;
use std::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OnnxInput {
    pub input_data: Vec<Vec<f32>>,
    pub input_shapes: Vec<Vec<usize>>,
    pub public_inputs: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Proof {
    pub input_shapes: Vec<Vec<usize>>,
    pub public_inputs: Vec<Vec<i32>>,
    pub proof: Vec<u8>,
}

/// Helper function to print helpful error messages after verification has failed.
pub fn parse_prover_errors(f: &VerifyFailure) {
    match f {
        VerifyFailure::Lookup {
            name,
            location,
            lookup_index,
        } => {
            error!("lookup {:?} is out of range, try increasing 'bits' or reducing 'scale' ({} and lookup index {}).",
            name, location, lookup_index);
        }
        VerifyFailure::ConstraintNotSatisfied {
            constraint,
            location,
            cell_values,
        } => {
            error!(
                "constraint {:?} was not satisfied ({} with values {:?}).",
                constraint, location, cell_values
            );
        }
        VerifyFailure::ConstraintPoisoned { constraint } => {
            error!("constraint {:?} was poisoned", constraint);
        }
        VerifyFailure::Permutation { column, location } => {
            error!(
                "permutation did not preserve column cell value (try increasing 'scale') ({} {}).",
                column, location
            );
        }
        e => {
            error!("{:?}", e);
        }
    }
}

pub fn prepare_circuit_and_public_input<F: FieldExt>(
    data: &OnnxInput,
) -> (OnnxCircuit<F>, Vec<Tensor<i32>>) {
    let onnx_model = OnnxModel::from_arg();
    let out_scales = onnx_model.get_output_scales();
    let circuit = prepare_circuit(data);

    // quantize the supplied data using the provided scale.
    let public_inputs = data
        .public_inputs
        .iter()
        .enumerate()
        .map(
            |(idx, v)| match vector_to_quantized(v, &Vec::from([v.len()]), 0.0, out_scales[idx]) {
                Ok(q) => q,
                Err(e) => {
                    abort!("failed to quantize vector {:?}", e);
                }
            },
        )
        .collect();
    trace!("{:?}", public_inputs);
    (circuit, public_inputs)
}

pub fn prepare_circuit<F: FieldExt>(data: &OnnxInput) -> OnnxCircuit<F> {
    let args = Cli::parse();

    // quantize the supplied data using the provided scale.
    let inputs = data
        .input_data
        .iter()
        .zip(data.input_shapes.clone())
        .map(|(i, s)| match vector_to_quantized(i, &s, 0.0, args.scale) {
            Ok(q) => q,
            Err(e) => {
                abort!("failed to quantize vector {:?}", e);
            }
        })
        .collect();

    OnnxCircuit::<F> {
        inputs,
        _marker: PhantomData,
    }
}

pub fn prepare_data(datapath: String) -> OnnxInput {
    let mut file = match File::open(data_path(datapath)) {
        Ok(t) => t,
        Err(e) => {
            abort!("failed to open data file {:?}", e);
        }
    };
    let mut data = String::new();
    match file.read_to_string(&mut data) {
        Ok(_) => {}
        Err(e) => {
            abort!("failed to read file {:?}", e);
        }
    };
    let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");
    info!(
        "public inputs (network outputs) lengths: {:?}",
        data.public_inputs
            .iter()
            .map(|i| i.len())
            .collect::<Vec<usize>>()
    );

    data
}

// IPA
pub fn create_ipa_proof(
    circuit: OnnxCircuit<Fp>,
    public_inputs: Vec<Tensor<i32>>,
    params: &ParamsIPA<vesta::Affine>,
) -> (ProvingKey<EqAffine>, Vec<u8>, Vec<Vec<usize>>) {
    //	Real proof
    let empty_circuit = circuit.without_witnesses();

    // Initialize the proving key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk(params, &empty_circuit).expect("keygen_vk should not fail");
    info!("VK took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let pk = keygen_pk(params, vk, &empty_circuit).expect("keygen_pk should not fail");
    info!("PK took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let mut rng = OsRng;

    let pi_inner: Vec<Vec<F>> = public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i32_to_felt::<F>(*e)).collect::<Vec<F>>())
        .collect::<Vec<Vec<F>>>();
    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[F]>>();
    let pi_for_real_prover: &[&[&[F]]] = &[&pi_inner];
    trace!("pi for real prover {:?}", pi_for_real_prover);

    let dims = circuit.inputs.iter().map(|i| i.dims().to_vec()).collect();

    create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
        params,
        &pk,
        &[circuit],
        pi_for_real_prover,
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();
    info!("Proof took {}", now.elapsed().as_secs());

    (pk, proof, dims)
}

pub fn verify_ipa_proof(proof: Proof) -> bool {
    let args = Cli::parse();
    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);

    let inputs = proof
        .input_shapes
        .iter()
        .map(
            |s| match Tensor::new(Some(&vec![0; s.iter().product()]), s) {
                Ok(t) => t,
                Err(e) => {
                    abort!("failed to initialize tensor {:?}", e);
                }
            },
        )
        .collect();
    let circuit = OnnxCircuit::<F> {
        inputs,
        _marker: PhantomData,
    };
    let empty_circuit = circuit.without_witnesses();
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");

    let pi_inner: Vec<Vec<F>> = proof
        .public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i32_to_felt::<F>(*e)).collect::<Vec<F>>())
        .collect::<Vec<Vec<F>>>();
    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[F]>>();
    let pi_for_real_prover: &[&[&[F]]] = &[&pi_inner];
    trace!("pi for real prover {:?}", pi_for_real_prover);

    let now = Instant::now();
    let strategy = SingleStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof.proof[..]);

    trace!("params computed");

    let result = verify_proof(
        &params,
        pk.get_vk(),
        strategy,
        pi_for_real_prover,
        &mut transcript,
    )
    .is_ok();
    info!("verify took {}", now.elapsed().as_secs());
    result
}

// KZG

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
