use crate::{
    circuit::{
        modules::{
            polycommit::PolyCommitChip,
            poseidon::{
                spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
                PoseidonChip,
            },
            Module,
        },
    },
    fieldutils::{felt_to_integer_rep, integer_rep_to_felt},
    graph::{
        modules::POSEIDON_LEN_GRAPH, quantize_float, scale_to_multiplier, GraphCircuit,
        GraphSettings,
    },
};
use console_error_panic_hook;
use halo2_proofs::{
    plonk::*,
    poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG}
};
use halo2curves::{
    bn256::{Bn256, Fr, G1Affine},
    ff::PrimeField,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_console_logger::DEFAULT_LOGGER;

use crate::bindings::universal::{
    EZKLError as ExternalEZKLError,
    encode_verifier_calldata,
    gen_witness,
    gen_vk,
    gen_pk,
    verify_aggr,
    witness_validation,
    compiled_circuit_validation,
    input_validation,
    proof_validation,
    vk_validation,
    pk_validation,
    settings_validation,
    srs_validation,
};
#[cfg(feature = "web")]
pub use wasm_bindgen_rayon::init_thread_pool;

impl From<ExternalEZKLError> for JsError {
    fn from(e: ExternalEZKLError) -> Self {
        JsError::new(&format!("{}", e))
    }
}

#[wasm_bindgen]
/// Initialize logger for wasm
pub fn init_logger() {
    log::set_logger(&DEFAULT_LOGGER).unwrap();
}

#[wasm_bindgen]
/// Initialize panic hook for wasm
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Wrapper around the halo2 encode call data method
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn encodeVerifierCalldata(
    proof: wasm_bindgen::Clamped<Vec<u8>>,
    vk_address: Option<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    encode_verifier_calldata(proof.0, vk_address).map_err(JsError::from)
}

/// Converts a hex string to a byte array
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn feltToBigEndian(array: wasm_bindgen::Clamped<Vec<u8>>) -> Result<String, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    Ok(format!("{:?}", felt))
}

/// Converts a felt to a little endian string
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn feltToLittleEndian(array: wasm_bindgen::Clamped<Vec<u8>>) -> Result<String, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    let repr = serde_json::to_string(&felt).unwrap();
    let b: String = serde_json::from_str(&repr).unwrap();
    Ok(b)
}

/// Converts a hex string to a byte array
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn feltToInt(
    array: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&felt_to_integer_rep(felt))
            .map_err(|e| JsError::new(&format!("Failed to serialize integer: {}", e)))?,
    ))
}

/// Converts felts to a floating point element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn feltToFloat(
    array: wasm_bindgen::Clamped<Vec<u8>>,
    scale: crate::Scale,
) -> Result<f64, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    let int_rep = felt_to_integer_rep(felt);
    let multiplier = scale_to_multiplier(scale);
    Ok(int_rep as f64 / multiplier)
}

/// Converts a floating point number to a hex string representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn floatToFelt(
    input: f64,
    scale: crate::Scale,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let int_rep =
        quantize_float(&input, 0.0, scale).map_err(|e| JsError::new(&format!("{}", e)))?;
    let felt = integer_rep_to_felt(int_rep);
    let vec = crate::pfsys::field_to_string::<halo2curves::bn256::Fr>(&felt);
    Ok(wasm_bindgen::Clamped(serde_json::to_vec(&vec).map_err(
        |e| JsError::new(&format!("Failed to serialize a float to felt{}", e)),
    )?))
}

/// Generate a kzg commitment.
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn kzgCommit(
    message: wasm_bindgen::Clamped<Vec<u8>>,
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let message: Vec<Fr> = serde_json::from_slice(&message[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize message: {}", e)))?;

    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("Failed to deserialize params: {}", e)))?;

    let mut reader = std::io::BufReader::new(&vk[..]);
    let circuit_settings: GraphSettings = serde_json::from_slice(&settings[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize settings: {}", e)))?;
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| JsError::new(&format!("Failed to deserialize vk: {}", e)))?;

    let output = PolyCommitChip::commit::<KZGCommitmentScheme<Bn256>>(
        message,
        (vk.cs().blinding_factors() + 1) as u32,
        &params,
    );

    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&output).map_err(|e| JsError::new(&format!("{}", e)))?,
    ))
}

/// Converts a buffer to vector of 4 u64s representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn bufferToVecOfFelt(
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
        serde_json::to_vec(&field_elements)
            .map_err(|e| JsError::new(&format!("Failed to serialize field elements: {}", e)))?,
    ))
}

/// Generate a poseidon hash in browser. Input message
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn poseidonHash(
    message: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let message: Vec<Fr> = serde_json::from_slice(&message[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize message: {}", e)))?;

    let output =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
            message.clone(),
        )
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    Ok(wasm_bindgen::Clamped(serde_json::to_vec(&output).map_err(
        |e| JsError::new(&format!("Failed to serialize poseidon hash output: {}", e)),
    )?))
}

/// Generate a witness file from input.json, compiled model and a settings.json file.
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn genWitness(
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    input: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    gen_witness(compiled_circuit.0, input.0).map_err(JsError::from)
}

/// Generate verifying key in browser
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn genVk(
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
    compress_selectors: bool,
) -> Result<Vec<u8>, JsError> {
    gen_vk(compiled_circuit.0, params_ser.0, compress_selectors).map_err(JsError::from)
}

/// Generate proving key in browser
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn genPk(
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    gen_pk(vk.0, compiled_circuit.0, params_ser.0).map_err(JsError::from)
}

/// Verify proof in browser using wasm
#[wasm_bindgen]
pub fn verify(
    proof_js: wasm_bindgen::Clamped<Vec<u8>>,
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
    srs: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    super::universal::verify(proof_js.0, vk.0, settings.0, srs.0).map_err(JsError::from)
}

/// Verify aggregate proof in browser using wasm
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn verifyAggr(
    proof_js: wasm_bindgen::Clamped<Vec<u8>>,
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    logrows: u64,
    srs: wasm_bindgen::Clamped<Vec<u8>>,
    commitment: &str,
) -> Result<bool, JsError> {
    verify_aggr(proof_js.0, vk.0, logrows, srs.0, commitment).map_err(JsError::from)
}

/// Prove in browser using wasm
#[wasm_bindgen]
pub fn prove(
    witness: wasm_bindgen::Clamped<Vec<u8>>,
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    srs: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    super::universal::prove(witness.0, pk.0, compiled_circuit.0, srs.0).map_err(JsError::from)
}

// VALIDATION FUNCTIONS

/// Witness file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn witnessValidation(witness: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    witness_validation(witness.0).map_err(JsError::from)
}
/// Compiled circuit validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn compiledCircuitValidation(
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    compiled_circuit_validation(compiled_circuit.0).map_err(JsError::from)
}
/// Input file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn inputValidation(input: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    input_validation(input.0).map_err(JsError::from)
}
/// Proof file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn proofValidation(proof: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    proof_validation(proof.0).map_err(JsError::from)
}
/// Vk file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vkValidation(
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    vk_validation(vk.0, settings.0).map_err(JsError::from)
}
/// Pk file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn pkValidation(
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    pk_validation(pk.0, settings.0).map_err(JsError::from)
}
/// Settings file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn settingsValidation(settings: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    settings_validation(settings.0).map_err(JsError::from)
}
/// Srs file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn srsValidation(srs: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    srs_validation(srs.0).map_err(JsError::from)
}

/// HELPER FUNCTIONS
pub fn u8_array_to_u128_le(arr: [u8; 16]) -> u128 {
    let mut n: u128 = 0;
    for &b in arr.iter().rev() {
        n <<= 8;
        n |= b as u128;
    }
    n
}
