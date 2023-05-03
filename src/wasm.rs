use halo2_proofs::plonk::*;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};

use halo2curves::serde::SerdeObject;
use snark_verifier::system::halo2::compile;
use wasm_bindgen::prelude::*;
// pub use wasm_bindgen_rayon::init_thread_pool;

use console_error_panic_hook;

#[wasm_bindgen]
/// Initialize panic hook for wasm
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use crate::execute::verify_proof_circuit_kzg;
use crate::graph::ModelCircuit;
use crate::pfsys::Snarkbytes;

#[wasm_bindgen]
/// Verify proof in browser using wasm
pub fn verify_wasm(
    proof_js: JsValue,
    vk: JsValue,
    circuit_params_ser: JsValue,
    params_ser: JsValue,
) -> bool {
    let binding = params_ser.into_serde::<Vec<u8>>().unwrap();
    let mut reader = std::io::BufReader::new(&binding[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).unwrap();

    let snark_bytes = proof_js.into_serde::<Snarkbytes>().unwrap();
    let circuit_params = circuit_params_ser
        .into_serde::<crate::graph::ModelParams>()
        .unwrap();

    let instances = snark_bytes
        .instances
        .iter()
        .map(|i| {
            i.iter()
                .map(|e| Fr::from_raw_bytes_unchecked(e))
                .collect::<Vec<Fr>>()
        })
        .collect::<Vec<Vec<Fr>>>();

    let binding = vk.into_serde::<Vec<u8>>().unwrap();
    let mut reader = std::io::BufReader::new(&binding[..]);
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
    };

    let strategy = KZGSingleStrategy::new(params.verifier_params());

    let result = verify_proof_circuit_kzg(
        params.verifier_params(),
        snark,
        &vk,
        crate::commands::TranscriptType::Blake,
        strategy,
    );

    if result.is_ok() {
        true
    } else {
        false
    }
}
