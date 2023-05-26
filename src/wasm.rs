use std::sync::Arc;

use halo2_proofs::plonk::*;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};

use halo2curves::serde::SerdeObject;
use snark_verifier::system::halo2::compile;
use wasm_bindgen::prelude::*;

use console_error_panic_hook;

pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
/// Initialize panic hook for wasm
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use crate::execute::{create_proof_circuit_kzg, verify_proof_circuit_kzg};
use crate::graph::{ModelCircuit, ModelParams};
use crate::pfsys::Snarkbytes;

#[wasm_bindgen]
/// Verify proof in browser using wasm
pub fn verify_wasm(
    proof_js: wasm_bindgen::Clamped<Vec<u8>>,
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    circuit_params_ser: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> bool {
    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).unwrap();

    let circuit_params: ModelParams = bincode::deserialize(&circuit_params_ser[..]).unwrap();

    let snark_bytes: Snarkbytes = bincode::deserialize(&proof_js[..]).unwrap();

    let instances = snark_bytes
        .instances
        .iter()
        .map(|i| {
            i.iter()
                .map(|e| Fr::from_raw_bytes_unchecked(e))
                .collect::<Vec<Fr>>()
        })
        .collect::<Vec<Vec<Fr>>>();

    let mut reader = std::io::BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, ModelCircuit<Fr>>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_params,
    )
    .unwrap();

    let protocol = compile(
        &params,
        &vk,
        snark_verifier::system::halo2::Config::kzg()
            .with_num_instance(snark_bytes.num_instance.clone()),
    );

    let snark = crate::pfsys::Snark {
        instances,
        proof: snark_bytes.proof,
        protocol: Some(protocol),
        transcript_type: snark_bytes.transcript_type,
    };

    let strategy = KZGSingleStrategy::new(params.verifier_params());

    let result = verify_proof_circuit_kzg(params.verifier_params(), snark, &vk, strategy);

    if result.is_ok() {
        true
    } else {
        false
    }
}

/// Prove proof in browser using wasm
#[wasm_bindgen]
pub fn prove_wasm(
    data: wasm_bindgen::Clamped<Vec<u8>>,
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    circuit_ser: wasm_bindgen::Clamped<Vec<u8>>,
    circuit_params_ser: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> Vec<u8> {
    // read in kzg params
    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).unwrap();

    // read in model input
    let data: crate::pfsys::ModelInput = serde_json::from_slice(&data[..]).unwrap();

    // read in circuit params
    let circuit_params: ModelParams = bincode::deserialize(&circuit_params_ser[..]).unwrap();

    // read in proving key
    let mut reader = std::io::BufReader::new(&pk[..]);
    let pk = ProvingKey::<G1Affine>::read::<_, ModelCircuit<Fr>>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_params.clone(),
    )
    .unwrap();

    // read in circuit
    let mut reader = std::io::BufReader::new(&circuit_ser[..]);
    let model = crate::graph::Model::new(
        &mut reader,
        circuit_params.run_args,
        crate::graph::Mode::Prove,
        circuit_params.visibility,
    )
    .unwrap();

    let circuit =
        ModelCircuit::<Fr>::new(&data, Arc::new(model), crate::circuit::CheckMode::UNSAFE).unwrap();

    // prep public inputs
    let public_inputs = circuit.prepare_public_inputs(&data).unwrap();

    let strategy = KZGSingleStrategy::new(&params);
    let proof = create_proof_circuit_kzg(
        circuit,
        &params,
        public_inputs,
        &pk,
        crate::pfsys::TranscriptType::EVM,
        strategy,
        crate::circuit::CheckMode::UNSAFE,
    )
    .unwrap();

    bincode::serialize(&proof.to_bytes()).unwrap()
}
