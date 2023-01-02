use crate::tensor::*;
/// Element-wise operations using lookup tables.
pub mod lookup;
/// Structs and methods for configuring and assigning polynomial constraints to a gate within a Halo2 circuit.
pub mod polynomial;
/// A layer for range checks using polynomials.
pub mod range;
/// Utility functions for building gates.
pub mod utils;
