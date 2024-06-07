use thiserror::Error;

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
    #[error("Significant bit truncation when instantiating, try lowering the scale")]
    SigBitTruncationError,
    /// Failed to convert to field element tensor
    #[error("Failed to convert to field element tensor")]
    FeltError,
    /// Unsupported operation
    #[error("Unsupported operation on a tensor type")]
    Unsupported,
    /// Overflow
    #[error("Unsigned integer overflow or underflow error in op: {0}")]
    Overflow(String),
    /// Unset visibility
    #[error("Unset visibility")]
    UnsetVisibility,
}
