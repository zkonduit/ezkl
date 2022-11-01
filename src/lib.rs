#![feature(slice_flatten)]

/// Aggregation circuit
pub mod aggregation;
/// Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
pub mod circuit;
/// Commands
pub mod commands;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(feature = "onnx")]
pub mod onnx;
/// Tools for proofs and verification used by cli
pub mod prove_verify;
/// An implementation of multi-dimensional tensors.
pub mod tensor;

/// A macro to abort concisely.
#[macro_export]
macro_rules! abort {
    ($msg:literal $(, $ex:expr)*) => {
        error!($msg, $($ex,)*);
        panic!();
    };
}
