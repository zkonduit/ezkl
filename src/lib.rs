#![deny(
    bad_style,
    dead_code,
    improper_ctypes,
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    patterns_in_fns_without_body,
    private_in_public,
    unconditional_recursion,
    unused,
    unused_allocation,
    unused_comparisons,
    unused_parens,
    while_true,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    missing_debug_implementations,
    unsafe_code
)]
#![feature(lint_reasons)]
#![feature(int_roundings)]

//! A library for turning computational graphs, such as neural networks, into ZK-circuits.
//!

use circuit::Tolerance;
use clap::Args;
use graph::Visibility;
use serde::{Deserialize, Serialize};

/// Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
pub mod circuit;
/// CLI commands.
#[cfg(not(target_arch = "wasm32"))]
pub mod commands;
#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs, reason = "abigen doesn't generate docs for this module")]
/// Utility functions for contracts
pub mod eth;
/// Command execution
///
#[cfg(not(target_arch = "wasm32"))]
pub mod execute;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(feature = "onnx")]
pub mod graph;
/// beautiful logging
#[cfg(not(target_arch = "wasm32"))]
pub mod logger;
/// Tools for proofs and verification used by cli
pub mod pfsys;
/// Python bindings
#[cfg(feature = "python-bindings")]
pub mod python;
/// An implementation of multi-dimensional tensors.
pub mod tensor;
/// wasm prover and verifier
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm;

/// Parameters specific to a proving run
#[derive(Debug, Copy, Args, Deserialize, Serialize, Clone, Default, PartialEq, PartialOrd)]
pub struct RunArgs {
    /// The tolerance for error on model outputs
    #[arg(short = 'T', long, default_value = "0")]
    pub tolerance: Tolerance,
    /// The denominator in the fixed point representation used when quantizing
    #[arg(short = 'S', long, default_value = "7")]
    pub scale: u32,
    /// The number of bits used in lookup tables
    #[arg(short = 'B', long, default_value = "16")]
    pub bits: usize,
    /// The log_2 number of rows
    #[arg(short = 'K', long, default_value = "17")]
    pub logrows: u32,
    /// The number of batches to split the input data into
    #[arg(long, default_value = "1")]
    pub batch_size: usize,
    /// Flags whether inputs are public, private, hashed
    #[arg(long, default_value = "private")]
    pub input_visibility: Visibility,
    /// Flags whether outputs are public, private, hashed
    #[arg(long, default_value = "public")]
    pub output_visibility: Visibility,
    /// Flags whether params are public, private, hashed
    #[arg(long, default_value = "private")]
    pub param_visibility: Visibility,
}
