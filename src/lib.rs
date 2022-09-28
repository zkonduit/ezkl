#![feature(slice_flatten)]

/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for configuring neural network layers and assigning values to them in a Halo2 circuit.
pub mod nn;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(feature = "onnx")]
pub mod onnx;
/// An implementation of multi-dimensional tensors.
pub mod tensor;
/// Implementations of common operations on tensors.
pub mod tensor_ops;
