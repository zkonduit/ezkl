use crate::tensor::*;
/// Element-wise operations using lookup tables.
pub mod lookup;
/// Structs and methods for configuring and assigning polynomial constraints to a gate within a Halo2 circuit.
pub mod polynomial;
/// A layer for range checks using polynomials.
pub mod range;
/// Utility functions for building gates.
pub mod utils;

use thiserror::Error;

/// circuit related errors.
#[derive(Debug, Error)]
pub enum CircuitError {
    /// Shape mismatch in circuit construction
    #[error("dimension mismatch in circuit construction for op: {0}")]
    DimMismatch(String),
    /// Error when instantiating lookup tables
    #[error("failed to instantiate lookup tables")]
    LookupInstantiation,
    /// A lookup table was was already assigned
    #[error("attempting to initialize an already instantiated lookup table")]
    TableAlreadyAssigned,
}


