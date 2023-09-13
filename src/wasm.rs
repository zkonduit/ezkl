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
use wasm_bindgen_console_logger::DEFAULT_LOGGER;

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
pub fn vecU64ToFelt(array: wasm_bindgen::Clamped<Vec<u8>>) -> Result<String, JsError> {
    let felt: Fr =
        serde_json::from_slice(&array[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    Ok(format!("{:?}", felt))
}

/// Converts 4 u64s representing a field element directly to an integer
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vecU64ToInt(
    array: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let felt: Fr =
        serde_json::from_slice(&array[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&felt_to_i128(felt)).map_err(|e| JsError::new(&format!("{}", e)))?,
    ))
}

/// Converts 4 u64s representing a field element directly to a (rescaled from fixed point scaling) floating point
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vecU64ToFloat(array: wasm_bindgen::Clamped<Vec<u8>>, scale: u32) -> Result<f64, JsError> {
    let felt: Fr =
        serde_json::from_slice(&array[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    let int_rep = felt_to_i128(felt);
    let multiplier = scale_to_multiplier(scale);
    Ok(int_rep as f64 / multiplier)
}

/// Converts a floating point element to 4 u64s representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn floatToVecU64(input: f64, scale: u32) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let int_rep =
        quantize_float(&input, 0.0, scale).map_err(|e| JsError::new(&format!("{}", e)))?;
    let felt = i128_to_felt(int_rep);
    let vec = crate::pfsys::field_to_vecu64_montgomery::<halo2curves::bn256::Fr>(&felt);
    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&vec).map_err(|e| JsError::new(&format!("{}", e)))?,
    ))
}

/// Converts a buffer to vector of 4 u64s representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn bufferToVecOfVecU64(
    buffer: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    // Convert the buffer to a slice
    let buffer: &[u8] = &buffer;

    // Divide the buffer into chunks of 64 bytes
    let chunks = buffer.chunks_exact(16);

    // Get the remainder
    let remainder = chunks.remainder();

    // Add 0s to the remainder to make it 64 bytes
    let mut remainder = remainder.to_vec();

    // Collect chunks into a Vec<[u8; 16]>.
    let chunks: Result<Vec<[u8; 16]>, JsError> = chunks
        .map(|slice| {
            let array: [u8; 16] = slice
                .try_into()
                .map_err(|_| JsError::new("failed to slice input chunks"))?;
            Ok(array)
        })
        .collect();

    let mut chunks = chunks?;

    if remainder.len() != 0 {
        remainder.resize(16, 0);
        // Convert the Vec<u8> to [u8; 16]
        let remainder_array: [u8; 16] = remainder
            .try_into()
            .map_err(|_| JsError::new("failed to slice remainder"))?;
        // append the remainder to the chunks
        chunks.push(remainder_array);
    }

    // Convert each chunk to a field element
    let field_elements: Vec<Fr> = chunks
        .iter()
        .map(|x| PrimeField::from_u128(u8_array_to_u128_le(*x)))
        .collect();

    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&field_elements).map_err(|e| JsError::new(&format!("{}", e)))?,
    ))
}

/// Generate a poseidon hash in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn poseidonHash(
    message: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let message: Vec<Fr> =
        serde_json::from_slice(&message[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let output =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
            message.clone(),
        )
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&output).map_err(|e| JsError::new(&format!("{}", e)))?,
    ))
}

/// Generates random elgamal variables from a random seed value in browser.
/// Make sure input seed comes a secure source of randomness
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn elgamalGenRandom(rng: wasm_bindgen::Clamped<Vec<u8>>) -> Result<Vec<u8>, JsError> {
    let seed: &[u8] = &rng;
    let mut rng = StdRng::from_seed(
        seed.try_into()
            .map_err(|e| JsError::new(&format!("{}", e)))?,
    );

    let output = crate::circuit::modules::elgamal::ElGamalVariables::gen_random(&mut rng);

    serde_json::to_vec(&output).map_err(|e| JsError::new(&format!("{}", e)))
}

/// Encrypt using elgamal in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn elgamalEncrypt(
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    message: wasm_bindgen::Clamped<Vec<u8>>,
    r: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    let pk: G1Affine =
        serde_json::from_slice(&pk[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    let message: Vec<Fr> =
        serde_json::from_slice(&message[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    let r: Fr = serde_json::from_slice(&r[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let output = crate::circuit::modules::elgamal::ElGamalGadget::encrypt(pk, message, r);

    serde_json::to_vec(&output).map_err(|e| JsError::new(&format!("{}", e)))
}

/// Decrypt using elgamal in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn elgamalDecrypt(
    cipher: wasm_bindgen::Clamped<Vec<u8>>,
    sk: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    let sk: Fr = serde_json::from_slice(&sk[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let cipher: ElGamalCipher =
        serde_json::from_slice(&cipher[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let output = crate::circuit::modules::elgamal::ElGamalGadget::decrypt(&cipher, sk);

    serde_json::to_vec(&output).map_err(|e| JsError::new(&format!("{}", e)))
}

/// Generate a witness file from input.json, compiled model and a settings.json file.
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn genWitness(
    compiled_model: wasm_bindgen::Clamped<Vec<u8>>,
    input: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    let compiled_model: crate::graph::Model =
        bincode::deserialize(&compiled_model[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    let input: crate::graph::input::GraphData =
        serde_json::from_slice(&input[..]).map_err(|e| JsError::new(&format!("{}", e)))?;
    let circuit_settings: crate::graph::GraphSettings =
        serde_json::from_slice(&settings[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    // read in circuit
    let mut circuit = GraphCircuit::new(compiled_model, &circuit_settings.run_args)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    let mut input = circuit
        .load_graph_input(&input)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    let witness = circuit
        .forward(&mut input)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    serde_json::to_vec(&witness).map_err(|e| JsError::new(&format!("{}", e)))
}

/// Verify proof in browser using wasm
#[wasm_bindgen]
pub fn verify(
    proof_js: wasm_bindgen::Clamped<Vec<u8>>,
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
    srs: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    let mut reader = std::io::BufReader::new(&srs[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("{}", e)))?;

    let circuit_settings: GraphSettings =
        serde_json::from_slice(&settings[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let snark: crate::pfsys::Snark<Fr, G1Affine> =
        serde_json::from_slice(&proof_js[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let mut reader = std::io::BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| JsError::new(&format!("{}", e)))?;

    let strategy = KZGSingleStrategy::new(params.verifier_params());

    let result = verify_proof_circuit_kzg(params.verifier_params(), snark, &vk, strategy);

    match result {
        Ok(_) => Ok(true),
        Err(e) => Err(JsError::new(&format!("{}", e))),
    }
}

/// Prove in browser using wasm
#[wasm_bindgen]
pub fn prove(
    witness: wasm_bindgen::Clamped<Vec<u8>>,
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    compiled_model: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
    srs: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    log::set_logger(&DEFAULT_LOGGER).unwrap();
    #[cfg(feature = "det-prove")]
    log::set_max_level(log::LevelFilter::Debug);
    #[cfg(not(feature = "det-prove"))]
    log::set_max_level(log::LevelFilter::Info);
    // read in kzg params
    let mut reader = std::io::BufReader::new(&srs[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("{}", e)))?;

    // read in model input
    let data: crate::graph::GraphWitness =
        serde_json::from_slice(&witness[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    // read in circuit params
    let circuit_settings: GraphSettings =
        serde_json::from_slice(&settings[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    // read in proving key
    let mut reader = std::io::BufReader::new(&pk[..]);
    let pk = ProvingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings.clone(),
    )
    .map_err(|e| JsError::new(&format!("{}", e)))?;

    // read in circuit
    let compiled_model: crate::graph::Model =
        bincode::deserialize(&compiled_model[..]).map_err(|e| JsError::new(&format!("{}", e)))?;

    let mut circuit = GraphCircuit::new(compiled_model, &circuit_settings.run_args)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    // prep public inputs
    circuit
        .load_graph_witness(&data)
        .map_err(|e| JsError::new(&format!("{}", e)))?;
    let public_inputs = circuit
        .prepare_public_inputs(&data)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

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
    .map_err(|e| JsError::new(&format!("{}", e)))?;

    Ok(serde_json::to_string(&proof)
        .map_err(|e| JsError::new(&format!("{}", e)))?
        .into_bytes())
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

///
pub fn u8_array_to_u128_le(arr: [u8; 16]) -> u128 {
    let mut n: u128 = 0;
    for &b in arr.iter().rev() {
        n <<= 8;
        n |= b as u128;
    }
    n
}
