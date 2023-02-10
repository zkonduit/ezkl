/// Aggregate proof generation for EVM
pub mod aggregation;

use halo2_proofs::plonk;
use halo2_proofs::poly::kzg::commitment::ParamsKZG;
/// Simple proof generation for EVM
// pub mod simple;
use halo2curves::bn256::{Bn256, Fr, G1Affine};

use crate::pfsys::{keygen_pk, keygen_vk, Circuit, Error, OsRng, ProvingKey};

use ethereum_types::Address;
use foundry_evm::executor::{fork::MultiFork, Backend, ExecutorBuilder};

use snark_verifier::loader::evm::encode_calldata;

use thiserror::Error;

#[derive(Error, Debug)]
/// Errors related to proof aggregation
pub enum AggregationError {
    /// A KZG proof could not be verified
    #[error("failed to verify KZG proof")]
    KZGProofVerification,
    /// EVM execution errors
    #[error("EVM execution of raw code failed")]
    EVMRawExecution,
    /// proof read errors
    #[error("Failed to read proof")]
    ProofRead,
    /// proof verification errors
    #[error("Failed to verify proof")]
    ProofVerify,
    /// proof creation errors
    #[error("Failed to create proof")]
    ProofCreate,
}

/// Verify by executing bytecode with instance variables and proof as input
pub fn evm_verify(
    deployment_code: Vec<u8>,
    instances: Vec<Vec<Fr>>,
    proof: Vec<u8>,
) -> Result<bool, Box<dyn Error>> {
    let calldata = encode_calldata(&instances, &proof);
    let mut evm = ExecutorBuilder::default()
        .with_gas_limit(u64::MAX.into())
        .build(Backend::new(MultiFork::new().0, None));

    let caller = Address::from_low_u64_be(0xfe);
    let verifier = evm
        .deploy(caller, deployment_code.into(), 0.into(), None)
        .map_err(Box::new)?
        .address;
    let result = evm
        .call_raw(caller, verifier, calldata.into(), 0.into())
        .map_err(|_| Box::new(AggregationError::EVMRawExecution))?;

    dbg!(result.gas_used);

    Ok(!result.reverted)
}

/// Generate a structured reference string for testing. Not secure, do not use in production.
pub fn gen_srs(k: u32) -> ParamsKZG<Bn256> {
    ParamsKZG::<Bn256>::setup(k, OsRng)
}

/// Generate the proving key
pub fn gen_pk<C: Circuit<Fr>>(
    params: &ParamsKZG<Bn256>,
    circuit: &C,
) -> Result<ProvingKey<G1Affine>, plonk::Error> {
    let vk = keygen_vk(params, circuit)?;
    keygen_pk(params, vk, circuit)
}
