use thiserror::Error;

/// Error type for the pfsys module
#[derive(Error, Debug)]
pub enum PfsysError {
    /// Failed to save the proof
    #[error("Failed to save the proof: {0}")]
    SaveProof(String),
    /// Failed to load the proof
    #[error("Failed to load the proof: {0}")]
    LoadProof(String),
    /// Halo2 error
    #[error("Halo2 error: {0}")]
    Halo2Error(#[from] halo2_proofs::plonk::Error),
    /// Failed to write point to transcript
    #[error("Failed to write point to transcript: {0}")]
    WritePoint(String),
    /// Invalid commitment scheme
    #[error("Invalid commitment scheme")]
    InvalidCommitmentScheme,
    /// Failed to load vk from file
    #[error("Failed to load vk from file: {0}")]
    LoadVk(String),
    /// Failed to load pk from file
    #[error("Failed to load pk from file: {0}")]
    LoadPk(String),
}
