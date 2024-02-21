#![deny(
    bad_style,
    dead_code,
    improper_ctypes,
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    patterns_in_fns_without_body,
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
// we allow this for our dynamic range based indexing scheme
#![allow(clippy::single_range_in_vec_init)]
#![feature(round_ties_even)]

//! A library for turning computational graphs, such as neural networks, into ZK-circuits.
//!

use circuit::{table::Range, CheckMode, Tolerance};
use clap::Args;
use graph::Visibility;
use serde::{Deserialize, Serialize};
use tosubcommand::ToFlags;

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
/// beautiful logging
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod logger;
/// Tools for proofs and verification used by cli
pub mod pfsys;
/// Python bindings
#[cfg(feature = "python-bindings")]
pub mod python;
/// srs sha hashes
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub mod srs_sha;
/// An implementation of multi-dimensional tensors.
pub mod tensor;
/// wasm prover and verifier
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm;

#[cfg(not(target_arch = "wasm32"))]
use lazy_static::lazy_static;

/// The denominator in the fixed point representation used when quantizing inputs
pub type Scale = i32;

#[cfg(not(target_arch = "wasm32"))]
// Buf writer capacity
lazy_static! {
    /// The capacity of the buffer used for writing to disk
    pub static ref EZKL_BUF_CAPACITY: usize = std::env::var("EZKL_BUF_CAPACITY")
        .unwrap_or("8000".to_string())
        .parse()
        .unwrap();

    /// The serialization format for the keys
    pub static ref EZKL_KEY_FORMAT: String = std::env::var("EZKL_KEY_FORMAT")
        .unwrap_or("raw-bytes".to_string());
}

#[cfg(target_arch = "wasm32")]
const EZKL_KEY_FORMAT: &str = "raw-bytes";

#[cfg(target_arch = "wasm32")]
const EZKL_BUF_CAPACITY: &usize = &8000;

/// Parameters specific to a proving run
#[derive(Debug, Args, Deserialize, Serialize, Clone, PartialEq, PartialOrd, ToFlags)]
pub struct RunArgs {
    /// The tolerance for error on model outputs
    #[arg(short = 'T', long, default_value = "0")]
    pub tolerance: Tolerance,
    /// The denominator in the fixed point representation used when quantizing inputs
    #[arg(short = 'S', long, default_value = "7", allow_hyphen_values = true)]
    pub input_scale: Scale,
    /// The denominator in the fixed point representation used when quantizing parameters
    #[arg(long, default_value = "7", allow_hyphen_values = true)]
    pub param_scale: Scale,
    /// if the scale is ever > scale_rebase_multiplier * input_scale then the scale is rebased to input_scale (this a more advanced parameter, use with caution)
    #[arg(long, default_value = "1")]
    pub scale_rebase_multiplier: u32,
    /// The min and max elements in the lookup table input column
    #[arg(short = 'B', long, value_parser = parse_key_val::<i128, i128>, default_value = "-32768->32768")]
    pub lookup_range: Range,
    /// The log_2 number of rows
    #[arg(short = 'K', long, default_value = "17")]
    pub logrows: u32,
    /// The log_2 number of rows
    #[arg(short = 'N', long, default_value = "2")]
    pub num_inner_cols: usize,
    /// Hand-written parser for graph variables, eg. batch_size=1
    #[arg(short = 'V', long, value_parser = parse_key_val::<String, usize>, default_value = "batch_size->1", value_delimiter = ',')]
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
    #[arg(long, default_value = "false")]
    /// Rebase the scale using lookup table for division instead of using a range check
    pub div_rebasing: bool,
    /// Should constants with 0.0 fraction be rebased to scale 0
    #[arg(long, default_value = "false")]
    pub rebase_frac_zero_constants: bool,
    /// check mode (safe, unsafe, etc)
    #[arg(long, default_value = "unsafe")]
    pub check_mode: CheckMode,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            tolerance: Tolerance::default(),
            input_scale: 7,
            param_scale: 7,
            scale_rebase_multiplier: 1,
            lookup_range: (-32768, 32768),
            logrows: 17,
            num_inner_cols: 2,
            variables: vec![("batch_size".to_string(), 1)],
            input_visibility: Visibility::Private,
            output_visibility: Visibility::Public,
            param_visibility: Visibility::Private,
            div_rebasing: false,
            rebase_frac_zero_constants: false,
            check_mode: CheckMode::UNSAFE,
        }
    }
}

impl RunArgs {
    ///
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.scale_rebase_multiplier < 1 {
            return Err("scale_rebase_multiplier must be >= 1".into());
        }
        if self.lookup_range.0 > self.lookup_range.1 {
            return Err("lookup_range min is greater than max".into());
        }
        if self.logrows < 1 {
            return Err("logrows must be >= 1".into());
        }
        if self.num_inner_cols < 1 {
            return Err("num_inner_cols must be >= 1".into());
        }
        Ok(())
    }

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
    T: std::str::FromStr + std::fmt::Debug,
    T::Err: std::error::Error + Send + Sync + 'static,
    U: std::str::FromStr + std::fmt::Debug,
    U::Err: std::error::Error + Send + Sync + 'static,
{
    let pos = s
        .find("->")
        .ok_or_else(|| format!("invalid x->y: no `->` found in `{s}`"))?;
    let a = s[..pos].parse()?;
    let b = s[pos + 2..].parse()?;
    Ok((a, b))
}
