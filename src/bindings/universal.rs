use std::fmt::Display;
use uniffi::export;
use crate::EZKLError as InnerEZKLError;

/// Wrapper around the Error Message
#[derive(uniffi::Error, Debug)]
pub enum EZKLError {
    /// Some Comment
    InternalError(String),
    /// Some Comment
    InvalidInput(String),
}

impl Display for EZKLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EZKLError::InternalError(e) => write!(f, "Internal error: {}", e),
            EZKLError::InvalidInput(e) => write!(f, "Invalid input: {}", e),
        }
    }
}

impl From<InnerEZKLError> for EZKLError {
    fn from(e: InnerEZKLError) -> Self {
        EZKLError::InternalError(e.to_string())
    }
}

/// Wrapper around the halo2 encode call data method
#[export]
pub fn encode_verifier_calldata(
    proof: Vec<u8>,
    vk_address: Option<Vec<u8>>,
) -> Result<Vec<u8>, EZKLError> {
    Ok(super::logic::encode_verifier_calldata(proof, vk_address).unwrap())
}