#![allow(unused_imports)] // TODO - remove this line after implementing the functions

use std::fmt::Display;
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

use crate::{circuit::{
    modules::{
        polycommit::PolyCommitChip,
        poseidon::{
            spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
            PoseidonChip,
        },
        Module,
    },
    region::RegionSettings,
}, fieldutils::{felt_to_integer_rep, integer_rep_to_felt}, graph::{
    modules::POSEIDON_LEN_GRAPH, quantize_float, scale_to_multiplier, GraphCircuit,
    GraphSettings,
}, pfsys::{
    create_proof_circuit,
    evm::aggregation_kzg::{AggregationCircuit, PoseidonTranscript},
    verify_proof_circuit, TranscriptType,
}, tensor::TensorType, CheckMode, Commitments, EZKLError as InnerEZKLError};

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



#[cfg_attr(target_os = "ios", uniffi::export)]
pub(crate) fn encode_verifier_calldata(
    proof: Vec<u8>,
    vk_address: Option<Vec<u8>>,
) -> Result<Vec<u8>, EZKLError> {
    let snark: crate::pfsys::Snark<Fr, G1Affine> = serde_json::from_slice(&proof[..]).map_err(InnerEZKLError::from)?;

    let vk_address: Option<[u8; 20]> = if let Some(vk_address) = vk_address {
        let array: [u8; 20] = serde_json::from_slice(&vk_address[..]).map_err(InnerEZKLError::from)?;
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