use thiserror::Error;

/// Aggregate proof generation for EVM
pub mod aggregation;
/// Simple (single) proof generation for EVM
pub mod single;

#[derive(Error, Debug)]
/// Errors related to evm verification
pub enum EvmVerificationError {
    /// If the Solidity verifier worked but returned false
    #[error("Solidity verifier found the proof invalid")]
    InvalidProof,
    /// If the Solidity verifier threw and error (e.g. OutOfGas)
    #[error("Execution of Solidity code failed")]
    SolidityExecution,
    /// EVM execution errors
    #[error("EVM execution of raw code failed")]
    RawExecution,
    /// EVM verify errors
    #[error("evm verification reverted")]
    Reverted,
    /// EVM verify errors
    #[error("evm deployment failed")]
    Deploy,
}
/// YulCode type which is just an alias of string
pub type YulCode = String;
