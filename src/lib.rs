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
    missing_debug_implementations,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    missing_debug_implementations,
    missing_docs,
    unsafe_code,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications
)]
#![feature(slice_flatten)]
//! A library for turning computational graphs, such as neural networks, into ZK-circuits.
//!

/// Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
pub mod circuit;
/// CLI commands.
pub mod commands;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(feature = "onnx")]
pub mod graph;
/// Tools for proofs and verification used by cli
pub mod pfsys;
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
