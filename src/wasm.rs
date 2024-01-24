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
    commitment::{KZGCommitmentScheme, ParamsKZG},
    strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_solidity_verifier::encode_calldata;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::{FromUniformBytes, PrimeField};

use crate::tensor::TensorType;
use wasm_bindgen::prelude::*;
use wasm_bindgen_console_logger::DEFAULT_LOGGER;

use console_error_panic_hook;

#[cfg(feature = "web")]
pub use wasm_bindgen_rayon::init_thread_pool;

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

use crate::graph::{GraphCircuit, GraphSettings};
use crate::pfsys::{create_proof_circuit_kzg, verify_proof_circuit_kzg};

/// Wrapper around the halo2 encode call data method
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn encodeVerifierCalldata(
    proof: wasm_bindgen::Clamped<Vec<u8>>,
    vk_address: Option<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    let snark: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize proof: {}", e)))?;

    let vk_address: Option<[u8; 20]> = if let Some(vk_address) = vk_address {
        let array: [u8; 20] = serde_json::from_slice(&vk_address[..])
            .map_err(|e| JsError::new(&format!("Failed to deserialize vk address: {}", e)))?;
        Some(array)
    } else {
        None
    };

    let flattened_instances = snark.instances.into_iter().flatten();

    let encoded = encode_calldata(
        vk_address,
        &snark.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    Ok(encoded)
}

/// Converts 4 u64s to a field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn stringToFelt(array: wasm_bindgen::Clamped<Vec<u8>>) -> Result<String, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    Ok(format!("{:?}", felt))
}

/// Converts 4 u64s representing a field element directly to an integer
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn stringToInt(
    array: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    Ok(wasm_bindgen::Clamped(
        serde_json::to_vec(&felt_to_i128(felt))
            .map_err(|e| JsError::new(&format!("Failed to serialize integer: {}", e)))?,
    ))
}

/// Converts 4 u64s representing a field element directly to a (rescaled from fixed point scaling) floating point
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn stringToFloat(
    array: wasm_bindgen::Clamped<Vec<u8>>,
    scale: crate::Scale,
) -> Result<f64, JsError> {
    let felt: Fr = serde_json::from_slice(&array[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize field element: {}", e)))?;
    let int_rep = felt_to_i128(felt);
    let multiplier = scale_to_multiplier(scale);
    Ok(int_rep as f64 / multiplier)
}

/// Converts a floating point element to 4 u64s representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn floatTostring(
    input: f64,
    scale: crate::Scale,
) -> Result<wasm_bindgen::Clamped<Vec<u8>>, JsError> {
    let int_rep =
        quantize_float(&input, 0.0, scale).map_err(|e| JsError::new(&format!("{}", e)))?;
    let felt = i128_to_felt(int_rep);
    let vec = crate::pfsys::field_to_string_montgomery::<halo2curves::bn256::Fr>(&felt);
    Ok(wasm_bindgen::Clamped(serde_json::to_vec(&vec).map_err(
        |e| JsError::new(&format!("Failed to serialize string_montgomery{}", e)),
    )?))
}

/// Converts a buffer to vector of 4 u64s representing a fixed point field element
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn bufferToVecOfstring(
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
    let mut circuit: crate::graph::GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize compiled model: {}", e)))?;
    let input: crate::graph::input::GraphData = serde_json::from_slice(&input[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize input: {}", e)))?;

    let mut input = circuit
        .load_graph_input(&input)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    let witness = circuit
        .forward(&mut input, None, None)
        .map_err(|e| JsError::new(&format!("{}", e)))?;

    serde_json::to_vec(&witness)
        .map_err(|e| JsError::new(&format!("Failed to serialize witness: {}", e)))
}

/// Generate verifying key in browser
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn genVk(
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
    compress_selectors: bool,
) -> Result<Vec<u8>, JsError> {
    // Read in kzg params
    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("Failed to deserialize params: {}", e)))?;
    // Read in compiled circuit
    let circuit: crate::graph::GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize compiled model: {}", e)))?;

    // Create verifying key
    let vk = create_vk_wasm::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(
        &circuit,
        &params,
        compress_selectors,
    )
    .map_err(Box::<dyn std::error::Error>::from)
    .map_err(|e| JsError::new(&format!("Failed to create verifying key: {}", e)))?;

    let mut serialized_vk = Vec::new();
    vk.write(&mut serialized_vk, halo2_proofs::SerdeFormat::RawBytes)
        .map_err(|e| JsError::new(&format!("Failed to serialize vk: {}", e)))?;

    Ok(serialized_vk)
}

/// Generate proving key in browser
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn genPk(
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    params_ser: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    // Read in kzg params
    let mut reader = std::io::BufReader::new(&params_ser[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("Failed to deserialize params: {}", e)))?;
    // Read in compiled circuit
    let circuit: crate::graph::GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize compiled model: {}", e)))?;

    // Read in verifying key
    let mut reader = std::io::BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit.settings().clone(),
    )
    .map_err(|e| JsError::new(&format!("Failed to deserialize verifying key: {}", e)))?;
    // Create proving key
    let pk = create_pk_wasm::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk, &circuit, &params)
        .map_err(Box::<dyn std::error::Error>::from)
        .map_err(|e| JsError::new(&format!("Failed to create proving key: {}", e)))?;

    let mut serialized_pk = Vec::new();
    pk.write(&mut serialized_pk, halo2_proofs::SerdeFormat::RawBytes)
        .map_err(|e| JsError::new(&format!("Failed to serialize pk: {}", e)))?;

    Ok(serialized_pk)
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
            .map_err(|e| JsError::new(&format!("Failed to deserialize params: {}", e)))?;

    let circuit_settings: GraphSettings = serde_json::from_slice(&settings[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize settings: {}", e)))?;

    let snark: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof_js[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize proof: {}", e)))?;

    let mut reader = std::io::BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| JsError::new(&format!("Failed to deserialize vk: {}", e)))?;

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
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
    srs: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<Vec<u8>, JsError> {
    #[cfg(feature = "det-prove")]
    log::set_max_level(log::LevelFilter::Debug);
    #[cfg(not(feature = "det-prove"))]
    log::set_max_level(log::LevelFilter::Info);
    // read in kzg params
    let mut reader = std::io::BufReader::new(&srs[..]);
    let params: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("Failed to deserialize srs: {}", e)))?;

    // read in circuit
    let mut circuit: crate::graph::GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize circuit: {}", e)))?;

    // read in model input
    let data: crate::graph::GraphWitness = serde_json::from_slice(&witness[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize witness: {}", e)))?;

    // read in proving key
    let mut reader = std::io::BufReader::new(&pk[..]);
    let pk = ProvingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit.settings().clone(),
    )
    .map_err(|e| JsError::new(&format!("Failed to deserialize proving key: {}", e)))?;

    // prep public inputs
    circuit
        .load_graph_witness(&data)
        .map_err(|e| JsError::new(&format!("{}", e)))?;
    let public_inputs = circuit
        .prepare_public_inputs(&data)
        .map_err(|e| JsError::new(&format!("{}", e)))?;
    let proof_split_commits: Option<crate::pfsys::ProofSplitCommit> = data.into();

    let strategy = KZGSingleStrategy::new(&params);
    let proof = create_proof_circuit_kzg(
        circuit,
        &params,
        Some(public_inputs),
        &pk,
        crate::pfsys::TranscriptType::EVM,
        strategy,
        crate::circuit::CheckMode::UNSAFE,
        proof_split_commits,
    )
    .map_err(|e| JsError::new(&format!("{}", e)))?;

    Ok(serde_json::to_string(&proof)
        .map_err(|e| JsError::new(&format!("{}", e)))?
        .into_bytes())
}

/// print hex representation of a proof
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn printProofHex(proof: wasm_bindgen::Clamped<Vec<u8>>) -> Result<String, JsError> {
    let proof: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize proof: {}", e)))?;
    let hex_str = hex::encode(proof.proof);
    Ok(format!("0x{}", hex_str))
}
// VALIDATION FUNCTIONS

/// Witness file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn witnessValidation(witness: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    let _: crate::graph::GraphWitness = serde_json::from_slice(&witness[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize witness: {}", e)))?;

    Ok(true)
}
/// Compiled circuit validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn compiledCircuitValidation(
    compiled_circuit: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    let _: crate::graph::GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize compiled circuit: {}", e)))?;

    Ok(true)
}
/// Input file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn inputValidation(input: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    let _: crate::graph::input::GraphData = serde_json::from_slice(&input[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize input: {}", e)))?;

    Ok(true)
}
/// Proof file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn proofValidation(proof: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    let _: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize proof: {}", e)))?;

    Ok(true)
}
/// Vk file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn vkValidation(
    vk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    let circuit_settings: GraphSettings = serde_json::from_slice(&settings[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize settings: {}", e)))?;
    let mut reader = std::io::BufReader::new(&vk[..]);
    let _ = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| JsError::new(&format!("Failed to deserialize vk: {}", e)))?;

    Ok(true)
}
/// Pk file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn pkValidation(
    pk: wasm_bindgen::Clamped<Vec<u8>>,
    settings: wasm_bindgen::Clamped<Vec<u8>>,
) -> Result<bool, JsError> {
    let circuit_settings: GraphSettings = serde_json::from_slice(&settings[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize settings: {}", e)))?;
    let mut reader = std::io::BufReader::new(&pk[..]);
    let _ = ProvingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| JsError::new(&format!("Failed to deserialize proving key: {}", e)))?;

    Ok(true)
}
/// Settings file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn settingsValidation(settings: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    let _: GraphSettings = serde_json::from_slice(&settings[..])
        .map_err(|e| JsError::new(&format!("Failed to deserialize settings: {}", e)))?;

    Ok(true)
}
/// Srs file validation
#[wasm_bindgen]
#[allow(non_snake_case)]
pub fn srsValidation(srs: wasm_bindgen::Clamped<Vec<u8>>) -> Result<bool, JsError> {
    let mut reader = std::io::BufReader::new(&srs[..]);
    let _: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader)
            .map_err(|e| JsError::new(&format!("Failed to deserialize params: {}", e)))?;

    Ok(true)
}

// HELPER FUNCTIONS

/// Creates a [ProvingKey] for a [GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`) for the WASM target
#[cfg(target_arch = "wasm32")]
pub fn create_vk_wasm<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
    compress_selectors: bool,
) -> Result<VerifyingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
where
    C: Circuit<Scheme::Scalar>,
    <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
{
    //	Real proof
    let empty_circuit = <C as Circuit<F>>::without_witnesses(circuit);

    // Initialize the verifying key
    let vk = keygen_vk_custom(params, &empty_circuit, compress_selectors)?;
    Ok(vk)
}
/// Creates a [ProvingKey] from a [VerifyingKey] for a [GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`) for the WASM target
#[cfg(target_arch = "wasm32")]
pub fn create_pk_wasm<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
    vk: VerifyingKey<Scheme::Curve>,
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
