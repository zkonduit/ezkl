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
#![feature(round_ties_even)]

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
// abigen doesn't generate docs for this module
#[allow(missing_docs)]
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
/// Methods for deploying and interacting with the ezkl hub
#[cfg(not(target_arch = "wasm32"))]
pub mod hub;
/// beautiful logging
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
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
#[derive(Debug, Args, Deserialize, Serialize, Clone, Default, PartialEq, PartialOrd)]
pub struct RunArgs {
    /// The tolerance for error on model outputs
    #[arg(short = 'T', long, default_value = "0")]
    pub tolerance: Tolerance,
    /// The denominator in the fixed point representation used when quantizing inputs
    #[arg(short = 'S', long, default_value = "7")]
    pub input_scale: u32,
    /// The denominator in the fixed point representation used when quantizing parameters
    #[arg(long, default_value = "7")]
    pub param_scale: u32,
    /// if the scale is ever > scale_rebase_multiplier * input_scale then the scale is rebased to input_scale (this a more advanced parameter, use with caution)
    #[arg(long, default_value = "1")]
    pub scale_rebase_multiplier: u32,
    /// The min and max elements in the lookup table input column
    #[arg(short = 'B', long, value_parser = parse_tuple::<i128>, default_value = "(-32768,32768)")]
    pub lookup_range: (i128, i128),
    /// The log_2 number of rows
    #[arg(short = 'K', long, default_value = "17")]
    pub logrows: u32,
    /// Hand-written parser for graph variables, eg. batch_size=1
    #[arg(short = 'V', long, value_parser = parse_key_val::<String, usize>, default_value = "batch_size=1", value_delimiter = ',')]
    pub variables: Vec<(String, usize)>,
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

impl RunArgs {
    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
    /// Parse an ezkl configuration from a json
    pub fn from_json(arg_json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(arg_json)
    }
}

/// Parse a single key-value pair
fn parse_key_val<T, U>(
    s: &str,
) -> Result<(T, U), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
    U: std::str::FromStr,
    U::Err: std::error::Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

/// Parse a tuple
fn parse_tuple<T>(s: &str) -> Result<(T, T), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr + Clone,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let res = s.trim_matches(|p| p == '(' || p == ')').split(',');

    let res = res
        .map(|x| {
            // remove blank space
            let x = x.trim();
            x.parse::<T>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if res.len() != 2 {
        return Err("invalid tuple".into());
    }
    Ok((res[0].clone(), res[1].clone()))
}
