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

/// Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
pub mod circuit;
/// CLI commands.
pub mod commands;
#[cfg(not(target_arch = "wasm32"))]
#[allow(missing_docs, reason = "abigen doesn't generate docs for this module")]
/// Utility functions for contracts
pub mod eth;
/// Command execution
///
pub mod execute;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(feature = "onnx")]
pub mod graph;
/// beautiful logging
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
