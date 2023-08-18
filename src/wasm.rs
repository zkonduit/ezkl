use crate::circuit::modules::elgamal::ElGamalCipher;
use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::PoseidonChip;
use crate::circuit::modules::Module;
use crate::fieldutils::felt_to_i128;
use crate::fieldutils::i128_to_felt;
use crate::graph::modules::POSEIDON_LEN_GRAPH;
use crate::graph::quantize_float;
use crate::graph::scale_to_multiplier;
use halo2_proofs::plonk::*;
use halo2_proofs::poly::commitment::{CommitmentScheme, ParamsProver};
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::{FromUniformBytes, PrimeField};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::tensor::TensorType;
use wasm_bindgen::prelude::*;

use console_error_panic_hook;

#[cfg(feature = "web")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
/// Initialize panic hook for wasm
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

use crate::graph::{GraphCircuit, GraphSettings};
use crate::pfsys::{create_proof_circuit_kzg, verify_proof_circuit_kzg};

/// Converts 4 u64s to a field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vecu64_to_felt(array: wasm_bindgen::Clamped<Vec<u8>>) -> String {
    let felt: Fr = serde_json::from_slice(&array[..]).unwrap();
    format!("{:?}", felt)
}

/// Converts 4 u64s representing a field element directly to an integer
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vecu64_to_int(array: wasm_bindgen::Clamped<Vec<u8>>) -> wasm_bindgen::Clamped<Vec<u8>> {
    let felt: Fr = serde_json::from_slice(&array[..]).unwrap();
    wasm_bindgen::Clamped(serde_json::to_vec(&felt_to_i128(felt)).unwrap())
}

/// Converts 4 u64s representing a field element directly to a (rescaled from fixed point scaling) floating point
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vecU64ToFloat(array: wasm_bindgen::Clamped<Vec<u8>>, scale: u32) -> f64 {
    let felt: Fr = serde_json::from_slice(&array[..]).unwrap();
    let int_rep = felt_to_i128(felt);
    let multiplier = scale_to_multiplier(scale);
    int_rep as f64 / multiplier
}

/// Converts a floating point element to 4 u64s representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn floatToVecU64(input: f64, scale: u32) -> wasm_bindgen::Clamped<Vec<u8>> {
    let int_rep = quantize_float(&input, 0.0, scale).unwrap();
    let felt = i128_to_felt(int_rep);
    let vec = crate::pfsys::field_to_vecu64_montgomery::<halo2curves::bn256::Fr>(&felt);
    wasm_bindgen::Clamped(serde_json::to_vec(&vec).unwrap())
}

/// Generate a poseidon hash in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn poseidonHash(message: wasm_bindgen::Clamped<Vec<u8>>) -> wasm_bindgen::Clamped<Vec<u8>> {
    let message: Vec<Fr> = serde_json::from_slice(&message[..]).unwrap();

    let output =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
            message.clone(),
        )
        .unwrap();

    wasm_bindgen::Clamped(serde_json::to_vec(&output).unwrap())
}

/// Generates random elgamal variables from a random seed value in browser.
/// Make sure input seed comes a secure source of randomness
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn elgamalGenRandom(rng: wasm_bindgen::Clamped<Vec<u8>>) -> Vec<u8> {
    let seed: &[u8] = &rng;
    let mut rng = StdRng::from_seed(seed.try_into().unwrap());

    let output = crate::circuit::modules::elgamal::ElGamalVariables::gen_random(&mut rng);

    serde_json::to_vec(&output).unwrap()
}

/// Encrypt using elgamal in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn elgamalEncrypt(
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    message: wasm_bindgen::Clamped<Vec<u8>>,
    r: wasm_bindgen::Clamped<Vec<u8>>,
) -> Vec<u8> {
    let pk: G1Affine = serde_json::from_slice(&pk[..]).unwrap();
    let message: Vec<Fr> = serde_json::from_slice(&message[..]).unwrap();
    let r: Fr = serde_json::from_slice(&r[..]).unwrap();

    let output = crate::circuit::modules::elgamal::ElGamalGadget::encrypt(pk, message, r);

    serde_json::to_vec(&output).unwrap()
}

/// Decrypt using elgamal in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn elgamalDecrypt(
    cipher: wasm_bindgen::Clamped<Vec<u8>>,
    sk: wasm_bindgen::Clamped<Vec<u8>>,
) -> Vec<u8> {
    let sk: Fr = serde_json::from_slice(&sk[..]).unwrap();

    let cipher: ElGamalCipher = serde_json::from_slice(&cipher[..]).unwrap();

    let output = crate::circuit::modules::elgamal::ElGamalGadget::decrypt(&cipher, sk);

    serde_json::to_vec(&output).unwrap()
}

/// Verify proof in browser using wasm
#[wasm_bindgen]
pub fn verify(
    proof_js: wasm_bindgen::Clamped<Vec<u8>>,
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    circuit_settings_ser: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> bool {
    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).unwrap();

    let circuit_settings: GraphSettings =
        serde_json::from_slice(&circuit_settings_ser[..]).unwrap();

    let snark: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof_js[..]).unwrap();

    let mut reader = std::io::BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .unwrap();

    let strategy = KZGSingleStrategy::new(params.verifier_params());

    let result = verify_proof_circuit_kzg(params.verifier_params(), snark, &vk, strategy);

    if result.is_ok() {
        true
    } else {
        false
    }
}

/// Prove in browser using wasm
#[wasm_bindgen]
pub fn prove(
    witness: wasm_bindgen::Clamped<Vec<u8>>,
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    circuit_ser: wasm_bindgen::Clamped<Vec<u8>>,
    circuit_settings_ser: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> Vec<u8> {
    // read in kzg params
    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).unwrap();

    // read in model input
    let data: crate::graph::GraphWitness = serde_json::from_slice(&witness[..]).unwrap();

    // read in circuit params
    let circuit_settings: GraphSettings =
        serde_json::from_slice(&circuit_settings_ser[..]).unwrap();

    // read in proving key
    let mut reader = std::io::BufReader::new(&pk[..]);
    let pk = ProvingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings.clone(),
    )
    .unwrap();

    // read in circuit
    let model: crate::graph::Model = bincode::deserialize(&circuit_ser[..]).unwrap();

    let mut circuit = GraphCircuit::new(model, circuit_settings.run_args).unwrap();

    // prep public inputs
    circuit.load_graph_witness(&data).unwrap();
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

    serde_json::to_string(&proof).unwrap().into_bytes()
}

// HELPER FUNCTIONS

/// Creates a [VerifyingKey] and [ProvingKey] for a [GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`) for the WASM target
#[cfg(target_arch = "wasm32")]
pub fn create_keys_wasm<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
) -> Result<ProvingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
where
    C: Circuit<Scheme::Scalar>,
    <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
{
    //	Real proof
    let empty_circuit = <C as Circuit<F>>::without_witnesses(circuit);

    // Initialize the proving key
    let vk = keygen_vk(params, &empty_circuit)?;
    let pk = keygen_pk(params, vk, &empty_circuit)?;
    Ok(pk)
}
