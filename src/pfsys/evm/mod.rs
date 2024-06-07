use thiserror::Error;

/// Aggregate proof generation for EVM using KZG
pub mod aggregation_kzg;

#[derive(Error, Debug)]
/// Errors related to evm verification
pub enum EvmVerificationError {
    /// If the Solidity verifier worked but returned false
    #[error("Solidity verifier found the proof invalid")]
    InvalidProof,
    /// If the Solidity verifier threw and error (e.g. OutOfGas)
    #[error("Execution of Solidity code failed: {0}")]
    SolidityExecution(String),
    /// EVM verify errors
    #[error("evm verification reverted: {0}")]
    Reverted(String),
    /// EVM verify errors
    #[error("evm deployment failed: {0}")]
    DeploymentFailed(String),
    /// Invalid Visibility
    #[error("Invalid visibility")]
    InvalidVisibility,
}
