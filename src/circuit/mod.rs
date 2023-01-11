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

/// A wrapper for tensor related errors.
#[derive(Debug, Error)]
pub enum CircuitError {
    /// Shape mismatch in circuit construction
    DimMismatch(String),
    /// Error when instantiating lookup tables
    LookupInstantiation,
    /// A lookup table was was already assigned
    TableAlreadyAssigned,
    /// A val/var tensor combination is not yet implemented
    VariableComb,
}

impl std::fmt::Display for CircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CircuitError::VariableComb => {
                write!(f, "var/val tensor combination is not yet implemented",)
            }
            CircuitError::DimMismatch(op) => {
                write!(f, "dimension mismatch in circuit construction: {}", op)
            }
            CircuitError::LookupInstantiation => {
                write!(f, "failed to instantiate lookup tables")
            }
            CircuitError::TableAlreadyAssigned => write!(
                f,
                "attempting to initialize an already instantiated lookup table"
            ),
        }
    }
}
