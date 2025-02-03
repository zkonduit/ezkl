use thiserror::Error;

use super::ops::DecompositionError;

/// A wrapper for tensor related errors.
#[derive(Debug, Error)]
pub enum TensorError {
    /// Shape mismatch in a operation
    #[error("dimension mismatch in tensor op: {0}")]
    DimMismatch(String),
    /// Shape when instantiating
    #[error("dimensionality error when manipulating a tensor: {0}")]
    DimError(String),
    /// wrong method was called on a tensor-like struct
    #[error("wrong method called")]
    WrongMethod,
    /// Significant bit truncation when instantiating
    #[error("significant bit truncation when instantiating, try lowering the scale")]
    SigBitTruncationError,
    /// Failed to convert to field element tensor
    #[error("failed to convert to field element tensor")]
    FeltError,
    /// Unsupported operation
    #[error("unsupported operation on a tensor type")]
    Unsupported,
    /// Overflow
    #[error("unsigned integer overflow or underflow error in op: {0}")]
    Overflow(String),
    /// Unset visibility
    #[error("unset visibility")]
    UnsetVisibility,
    /// File save error
    #[error("save error: {0}")]
    FileSaveError(String),
    /// File load error
    #[error("load error: {0}")]
    FileLoadError(String),
    /// Decomposition error
    #[error("decomposition error: {0}")]
    DecompositionError(#[from] DecompositionError),
    /// Invalid argument
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// Index out of bounds
    #[error("index {0} out of bounds for dimension {1}")]
    IndexOutOfBounds(usize, usize),
}
