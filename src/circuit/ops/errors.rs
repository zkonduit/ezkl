use std::convert::Infallible;

use crate::tensor::TensorError;
use halo2_proofs::plonk::Error as PlonkError;
use thiserror::Error;

/// Error type for the circuit module
#[derive(Error, Debug)]
pub enum CircuitError {
    /// Halo 2 error
    #[error("[halo2] {0}")]
    Halo2Error(#[from] PlonkError),
    /// Tensor error
    #[error("[tensor] {0}")]
    TensorError(#[from] TensorError),
    /// Shape mismatch in circuit construction
    #[error("dimension mismatch in circuit construction for op: {0}")]
    DimMismatch(String),
    /// Error when instantiating lookup tables
    #[error("failed to instantiate lookup tables")]
    LookupInstantiation,
    /// A lookup table was was already assigned
    #[error("attempting to initialize an already instantiated lookup table")]
    TableAlreadyAssigned,
    /// This operation is unsupported
    #[error("unsupported operation in graph")]
    UnsupportedOp,
    ///
    #[error("invalid einsum expression")]
    InvalidEinsum,
    /// Flush error
    #[error("failed to flush, linear coord is not aligned with the next row")]
    FlushError,
    /// Constrain error
    #[error("constrain_equal: one of the tensors is assigned and the other is not")]
    ConstrainError,
    /// Failed to get lookups
    #[error("failed to get lookups for op: {0}")]
    GetLookupsError(String),
    /// Failed to get range checks
    #[error("failed to get range checks for op: {0}")]
    GetRangeChecksError(String),
    /// Failed to get dynamic lookup
    #[error("failed to get dynamic lookup for op: {0}")]
    GetDynamicLookupError(String),
    /// Failed to get shuffle
    #[error("failed to get shuffle for op: {0}")]
    GetShuffleError(String),
    /// Failed to get constants
    #[error("failed to get constants for op: {0}")]
    GetConstantsError(String),
    /// Slice length mismatch
    #[error("slice length mismatch: {0}")]
    SliceLengthMismatch(#[from] std::array::TryFromSliceError),
    /// Bad conversion
    #[error("invalid conversion: {0}")]
    InvalidConversion(#[from] Infallible),
    /// Invalid min/max lookup range
    #[error("invalid min/max lookup range: min: {0}, max: {1}")]
    InvalidMinMaxRange(i64, i64),
    /// Missing product in einsum
    #[error("missing product in einsum")]
    MissingEinsumProduct,
    /// Mismatched lookup length
    #[error("mismatched lookup lengths: {0} and {1}")]
    MismatchedLookupLength(usize, usize),
    /// Mismatched shuffle length
    #[error("mismatched shuffle lengths: {0} and {1}")]
    MismatchedShuffleLength(usize, usize),
    /// Mismatched lookup table lengths
    #[error("mismatched lookup table lengths: {0} and {1}")]
    MismatchedLookupTableLength(usize, usize),
    /// Wrong column type for lookup
    #[error("wrong column type for lookup: {0}")]
    WrongColumnType(String),
    /// Wrong column type for dynamic lookup
    #[error("wrong column type for dynamic lookup: {0}")]
    WrongDynamicColumnType(String),
    /// Missing selectors
    #[error("missing selectors for op: {0}")]
    MissingSelectors(String),
    /// Table lookup error
    #[error("value ({0}) out of range: ({1}, {2})")]
    TableOOR(i64, i64, i64),
    /// Loookup not configured
    #[error("lookup not configured: {0}")]
    LookupNotConfigured(String),
    /// Range check not configured
    #[error("range check not configured: {0}")]
    RangeCheckNotConfigured(String),
    /// Missing layout
    #[error("missing layout for op: {0}")]
    MissingLayout(String),
}
