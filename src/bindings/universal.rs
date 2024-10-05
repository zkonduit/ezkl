use halo2_proofs::{
    plonk::*,
    poly::{
        commitment::{CommitmentScheme, ParamsProver},
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::{ProverIPA, VerifierIPA},
            strategy::SingleStrategy as IPASingleStrategy,
        },
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy as KZGSingleStrategy,
        },
        VerificationStrategy,
    },
};
use std::fmt::Display;
use std::io::BufReader;
use std::str::FromStr;

use crate::{
    circuit::region::RegionSettings,
    graph::GraphSettings,
    pfsys::{
        create_proof_circuit,
        evm::aggregation_kzg::{AggregationCircuit, PoseidonTranscript},
        verify_proof_circuit, TranscriptType,
    },
    tensor::TensorType,
    CheckMode, Commitments, EZKLError as InnerEZKLError,
};

use crate::graph::{GraphCircuit, GraphWitness};
use halo2_solidity_verifier::encode_calldata;
use halo2curves::{
    bn256::{Bn256, Fr, G1Affine},
    ff::{FromUniformBytes, PrimeField},
};
use snark_verifier::{loader::native::NativeLoader, system::halo2::transcript::evm::EvmTranscript};

/// Wrapper around the Error Message
#[cfg_attr(target_os = "ios", derive(uniffi::Error))]
#[derive(Debug)]
pub enum EZKLError {
    /// Some Comment
    InternalError(String),
}

impl Display for EZKLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EZKLError::InternalError(e) => write!(f, "Internal error: {}", e),
        }
    }
}

impl From<InnerEZKLError> for EZKLError {
    fn from(e: InnerEZKLError) -> Self {
        EZKLError::InternalError(e.to_string())
    }
}

/// Encode verifier calldata from proof and ethereum vk_address
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn encode_verifier_calldata(
    // TODO - shuold it be pub(crate) or pub or pub(super)?
    proof: Vec<u8>,
    vk_address: Option<Vec<u8>>,
) -> Result<Vec<u8>, EZKLError> {
    let snark: crate::pfsys::Snark<Fr, G1Affine> =
        serde_json::from_slice(&proof[..]).map_err(InnerEZKLError::from)?;

    let vk_address: Option<[u8; 20]> = if let Some(vk_address) = vk_address {
        let array: [u8; 20] =
            serde_json::from_slice(&vk_address[..]).map_err(InnerEZKLError::from)?;
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

/// Generate witness from compiled circuit and input json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn gen_witness(compiled_circuit: Vec<u8>, input: Vec<u8>) -> Result<Vec<u8>, EZKLError> {
    let mut circuit: crate::graph::GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize compiled model: {}", e)))?;
    let input: crate::graph::input::GraphData = serde_json::from_slice(&input[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize input: {}", e)))?;

    let mut input = circuit
        .load_graph_input(&input)
        .map_err(|e| EZKLError::InternalError(format!("{}", e)))?;

    let witness = circuit
        .forward::<KZGCommitmentScheme<Bn256>>(
            &mut input,
            None,
            None,
            RegionSettings::all_true(
                circuit.settings().run_args.decomp_base,
                circuit.settings().run_args.decomp_legs,
            ),
        )
        .map_err(|e| EZKLError::InternalError(format!("{}", e)))?;

    serde_json::to_vec(&witness)
        .map_err(|e| EZKLError::InternalError(format!("Failed to serialize witness: {}", e)))
}

/// Generate verifying key from compiled circuit, and parameters srs
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn gen_vk(
    compiled_circuit: Vec<u8>,
    srs: Vec<u8>,
    compress_selectors: bool,
) -> Result<Vec<u8>, EZKLError> {
    let mut reader = BufReader::new(&srs[..]);
    let params: ParamsKZG<Bn256> = get_params(&mut reader)?;

    let circuit: GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize circuit: {}", e)))?;

    let vk = create_vk_lean::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(
        &circuit,
        &params,
        compress_selectors,
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to create verifying key: {}", e)))?;

    let mut serialized_vk = Vec::new();
    vk.write(&mut serialized_vk, halo2_proofs::SerdeFormat::RawBytes)
        .map_err(|e| {
            EZKLError::InternalError(format!("Failed to serialize verifying key: {}", e))
        })?;

    Ok(serialized_vk)
}

/// Generate proving key from vk, compiled circuit and parameters srs
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn gen_pk(
    vk: Vec<u8>,
    compiled_circuit: Vec<u8>,
    srs: Vec<u8>,
) -> Result<Vec<u8>, EZKLError> {
    let mut reader = BufReader::new(&srs[..]);
    let params: ParamsKZG<Bn256> = get_params(&mut reader)?;

    let circuit: GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize circuit: {}", e)))?;

    let mut reader = BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit.settings().clone(),
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize verifying key: {}", e)))?;

    let pk = create_pk_lean::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk, &circuit, &params)
        .map_err(|e| EZKLError::InternalError(format!("Failed to create proving key: {}", e)))?;

    let mut serialized_pk = Vec::new();
    pk.write(&mut serialized_pk, halo2_proofs::SerdeFormat::RawBytes)
        .map_err(|e| EZKLError::InternalError(format!("Failed to serialize proving key: {}", e)))?;

    Ok(serialized_pk)
}

/// Verify proof with vk, proof json, circuit settings json and srs
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn verify(
    proof: Vec<u8>,
    vk: Vec<u8>,
    settings: Vec<u8>,
    srs: Vec<u8>,
) -> Result<bool, EZKLError> {
    let circuit_settings: GraphSettings = serde_json::from_slice(&settings[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize settings: {}", e)))?;

    let proof: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize proof: {}", e)))?;

    let mut reader = BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings.clone(),
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize vk: {}", e)))?;

    let orig_n = 1 << circuit_settings.run_args.logrows;
    let commitment = circuit_settings.run_args.commitment.into();

    let mut reader = BufReader::new(&srs[..]);
    let result = match commitment {
        Commitments::KZG => {
            let params: ParamsKZG<Bn256> = get_params(&mut reader)?;
            let strategy = KZGSingleStrategy::new(params.verifier_params());
            match proof.transcript_type {
                TranscriptType::EVM => verify_proof_circuit::<
                    VerifierSHPLONK<'_, Bn256>,
                    KZGCommitmentScheme<Bn256>,
                    KZGSingleStrategy<_>,
                    _,
                    EvmTranscript<G1Affine, _, _, _>,
                >(&proof, &params, &vk, strategy, orig_n),
                TranscriptType::Poseidon => {
                    verify_proof_circuit::<
                        VerifierSHPLONK<'_, Bn256>,
                        KZGCommitmentScheme<Bn256>,
                        KZGSingleStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                    >(&proof, &params, &vk, strategy, orig_n)
                }
            }
        }
        Commitments::IPA => {
            let params: ParamsIPA<_> = get_params(&mut reader)?;
            let strategy = IPASingleStrategy::new(params.verifier_params());
            match proof.transcript_type {
                TranscriptType::EVM => verify_proof_circuit::<
                    VerifierIPA<_>,
                    IPACommitmentScheme<G1Affine>,
                    IPASingleStrategy<_>,
                    _,
                    EvmTranscript<G1Affine, _, _, _>,
                >(&proof, &params, &vk, strategy, orig_n),
                TranscriptType::Poseidon => {
                    verify_proof_circuit::<
                        VerifierIPA<_>,
                        IPACommitmentScheme<G1Affine>,
                        IPASingleStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                    >(&proof, &params, &vk, strategy, orig_n)
                }
            }
        }
    };

    match result {
        Ok(_) => Ok(true),
        Err(e) => Err(EZKLError::InternalError(format!(
            "Verification failed: {}",
            e
        ))),
    }
}

/// Verify aggregate proof with vk, proof, circuit settings and srs
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn verify_aggr(
    proof_js: Vec<u8>,
    vk: Vec<u8>,
    logrows: u64,
    srs: Vec<u8>,
    commitment: &str,
) -> Result<bool, EZKLError> {
    let proof: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof_js[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize proof: {}", e)))?;

    let mut reader = BufReader::new(&vk[..]);
    let vk = VerifyingKey::<G1Affine>::read::<_, AggregationCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        (),
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize vk: {}", e)))?;

    let commit = Commitments::from_str(commitment)
        .map_err(|e| EZKLError::InternalError(format!("Invalid commitment: {}", e)))?;

    let orig_n = 1 << logrows;

    let mut reader = BufReader::new(&srs[..]);
    let result = match commit {
        Commitments::KZG => {
            let params: ParamsKZG<Bn256> = get_params(&mut reader)?;
            let strategy = KZGSingleStrategy::new(params.verifier_params());
            match proof.transcript_type {
                TranscriptType::EVM => verify_proof_circuit::<
                    VerifierSHPLONK<'_, Bn256>,
                    KZGCommitmentScheme<Bn256>,
                    KZGSingleStrategy<_>,
                    _,
                    EvmTranscript<G1Affine, _, _, _>,
                >(&proof, &params, &vk, strategy, orig_n),

                TranscriptType::Poseidon => {
                    verify_proof_circuit::<
                        VerifierSHPLONK<'_, Bn256>,
                        KZGCommitmentScheme<Bn256>,
                        KZGSingleStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                    >(&proof, &params, &vk, strategy, orig_n)
                }
            }
        }
        Commitments::IPA => {
            let params: ParamsIPA<_> =
                halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).map_err(
                    |e| EZKLError::InternalError(format!("Failed to deserialize params: {}", e)),
                )?;
            let strategy = IPASingleStrategy::new(params.verifier_params());
            match proof.transcript_type {
                TranscriptType::EVM => verify_proof_circuit::<
                    VerifierIPA<_>,
                    IPACommitmentScheme<G1Affine>,
                    IPASingleStrategy<_>,
                    _,
                    EvmTranscript<G1Affine, _, _, _>,
                >(&proof, &params, &vk, strategy, orig_n),
                TranscriptType::Poseidon => {
                    verify_proof_circuit::<
                        VerifierIPA<_>,
                        IPACommitmentScheme<G1Affine>,
                        IPASingleStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                    >(&proof, &params, &vk, strategy, orig_n)
                }
            }
        }
    };

    result
        .map(|_| true)
        .map_err(|e| EZKLError::InternalError(format!("{}", e)))
}

/// Prove in browser with compiled circuit, witness json, proving key, and srs
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn prove(
    witness: Vec<u8>,
    pk: Vec<u8>,
    compiled_circuit: Vec<u8>,
    srs: Vec<u8>,
) -> Result<Vec<u8>, EZKLError> {
    #[cfg(feature = "det-prove")]
    log::set_max_level(log::LevelFilter::Debug);
    #[cfg(not(feature = "det-prove"))]
    log::set_max_level(log::LevelFilter::Info);

    let mut circuit: GraphCircuit = bincode::deserialize(&compiled_circuit[..])
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize circuit: {}", e)))?;

    let data: GraphWitness = serde_json::from_slice(&witness[..]).map_err(InnerEZKLError::from)?;

    let mut reader = BufReader::new(&pk[..]);
    let pk = ProvingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit.settings().clone(),
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize proving key: {}", e)))?;

    circuit
        .load_graph_witness(&data)
        .map_err(InnerEZKLError::from)?;
    let public_inputs = circuit
        .prepare_public_inputs(&data)
        .map_err(InnerEZKLError::from)?;
    let proof_split_commits: Option<crate::pfsys::ProofSplitCommit> = data.into();

    let mut reader = BufReader::new(&srs[..]);
    let commitment = circuit.settings().run_args.commitment.into();

    let proof = match commitment {
        Commitments::KZG => {
            let params: ParamsKZG<Bn256> =
                halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).map_err(
                    |e| EZKLError::InternalError(format!("Failed to deserialize srs: {}", e)),
                )?;

            create_proof_circuit::<
                KZGCommitmentScheme<Bn256>,
                _,
                ProverSHPLONK<_>,
                VerifierSHPLONK<_>,
                KZGSingleStrategy<_>,
                _,
                EvmTranscript<_, _, _, _>,
                EvmTranscript<_, _, _, _>,
            >(
                circuit,
                vec![public_inputs],
                &params,
                &pk,
                CheckMode::UNSAFE,
                Commitments::KZG,
                TranscriptType::EVM,
                proof_split_commits,
                None,
            )
        }
        Commitments::IPA => {
            let params: ParamsIPA<_> =
                halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).map_err(
                    |e| EZKLError::InternalError(format!("Failed to deserialize srs: {}", e)),
                )?;

            create_proof_circuit::<
                IPACommitmentScheme<G1Affine>,
                _,
                ProverIPA<_>,
                VerifierIPA<_>,
                IPASingleStrategy<_>,
                _,
                EvmTranscript<_, _, _, _>,
                EvmTranscript<_, _, _, _>,
            >(
                circuit,
                vec![public_inputs],
                &params,
                &pk,
                CheckMode::UNSAFE,
                Commitments::IPA,
                TranscriptType::EVM,
                proof_split_commits,
                None,
            )
        }
    }
    .map_err(InnerEZKLError::from)?;

    Ok(serde_json::to_vec(&proof).map_err(InnerEZKLError::from)?)
}

/// Validate the witness json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn witness_validation(witness: Vec<u8>) -> Result<bool, EZKLError> {
    let _: GraphWitness = serde_json::from_slice(&witness[..]).map_err(InnerEZKLError::from)?;

    Ok(true)
}

/// Validate the compiled circuit
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn compiled_circuit_validation(compiled_circuit: Vec<u8>) -> Result<bool, EZKLError> {
    let _: GraphCircuit = bincode::deserialize(&compiled_circuit[..]).map_err(|e| {
        EZKLError::InternalError(format!("Failed to deserialize compiled circuit: {}", e))
    })?;

    Ok(true)
}

/// Validate the input json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn input_validation(input: Vec<u8>) -> Result<bool, EZKLError> {
    let _: crate::graph::input::GraphData =
        serde_json::from_slice(&input[..]).map_err(InnerEZKLError::from)?;

    Ok(true)
}

/// Validate the proof json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn proof_validation(proof: Vec<u8>) -> Result<bool, EZKLError> {
    let _: crate::pfsys::Snark<Fr, G1Affine> =
        serde_json::from_slice(&proof[..]).map_err(InnerEZKLError::from)?;

    Ok(true)
}

/// Validate the verifying key given the settings json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn vk_validation(vk: Vec<u8>, settings: Vec<u8>) -> Result<bool, EZKLError> {
    let circuit_settings: GraphSettings =
        serde_json::from_slice(&settings[..]).map_err(InnerEZKLError::from)?;

    let mut reader = BufReader::new(&vk[..]);
    let _ = VerifyingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize verifying key: {}", e)))?;

    Ok(true)
}

/// Validate the proving key given the settings json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn pk_validation(pk: Vec<u8>, settings: Vec<u8>) -> Result<bool, EZKLError> {
    let circuit_settings: GraphSettings =
        serde_json::from_slice(&settings[..]).map_err(InnerEZKLError::from)?;

    let mut reader = BufReader::new(&pk[..]);
    let _ = ProvingKey::<G1Affine>::read::<_, GraphCircuit>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        circuit_settings,
    )
    .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize proving key: {}", e)))?;

    Ok(true)
}

/// Validate the settings json
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn settings_validation(settings: Vec<u8>) -> Result<bool, EZKLError> {
    let _: GraphSettings = serde_json::from_slice(&settings[..]).map_err(InnerEZKLError::from)?;

    Ok(true)
}

/// Validate the srs
#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn srs_validation(srs: Vec<u8>) -> Result<bool, EZKLError> {
    let mut reader = BufReader::new(&srs[..]);
    let _: ParamsKZG<Bn256> =
        halo2_proofs::poly::commitment::Params::<'_, G1Affine>::read(&mut reader).map_err(|e| {
            EZKLError::InternalError(format!("Failed to deserialize params: {}", e))
        })?;

    Ok(true)
}

// HELPER FUNCTIONS

fn get_params<Scheme: for<'a> halo2_proofs::poly::commitment::Params<'a, halo2curves::bn256::G1Affine>>(mut reader: &mut BufReader<&[u8]>) -> Result<Scheme, EZKLError> {    halo2_proofs::poly::commitment::Params::<G1Affine>::read(&mut reader)
        .map_err(|e| EZKLError::InternalError(format!("Failed to deserialize params: {}", e)))
}

/// Creates a [ProvingKey] for a [GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`) for the WASM target
pub fn create_vk_lean<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
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
pub fn create_pk_lean<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
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
