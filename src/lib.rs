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
#![feature(buf_read_has_data_left)]
#![feature(stmt_expr_attributes)]

//! A library for turning computational graphs, such as neural networks, into ZK-circuits.
//!

/// Error type
// #[cfg_attr(target_os = "ios", derive(uniffi::Error))]
#[derive(thiserror::Error, Debug)]
#[allow(missing_docs)]
pub enum EZKLError {
    #[error("[aggregation] {0}")]
    AggregationError(#[from] pfsys::evm::aggregation_kzg::AggregationError),
    #[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
    #[error("[eth] {0}")]
    EthError(#[from] eth::EthError),
    #[error("[graph] {0}")]
    GraphError(#[from] graph::errors::GraphError),
    #[error("[pfsys] {0}")]
    PfsysError(#[from] pfsys::errors::PfsysError),
    #[error("[circuit] {0}")]
    CircuitError(#[from] circuit::errors::CircuitError),
    #[error("[tensor] {0}")]
    TensorError(#[from] tensor::errors::TensorError),
    #[error("[module] {0}")]
    ModuleError(#[from] circuit::modules::errors::ModuleError),
    #[error("[io] {0}")]
    IoError(#[from] std::io::Error),
    #[error("[json] {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("[utf8] {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
    #[error("[reqwest] {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("[fmt] {0}")]
    FmtError(#[from] std::fmt::Error),
    #[error("[halo2] {0}")]
    Halo2Error(#[from] halo2_proofs::plonk::Error),
    #[error("[Uncategorized] {0}")]
    UncategorizedError(String),
    #[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
    #[error("[execute] {0}")]
    ExecutionError(#[from] execute::ExecutionError),
    #[error("[srs] {0}")]
    SrsError(#[from] pfsys::srs::SrsError),
}

impl From<&str> for EZKLError {
    fn from(s: &str) -> Self {
        EZKLError::UncategorizedError(s.to_string())
    }
}

impl From<String> for EZKLError {
    fn from(s: String) -> Self {
        EZKLError::UncategorizedError(s)
    }
}

use std::str::FromStr;

use circuit::{table::Range, CheckMode, Tolerance};
use clap::Args;
use fieldutils::IntegerRep;
use graph::Visibility;
use halo2_proofs::poly::{
    ipa::commitment::IPACommitmentScheme, kzg::commitment::KZGCommitmentScheme,
};
use halo2curves::bn256::{Bn256, G1Affine};
use serde::{Deserialize, Serialize};
use tosubcommand::ToFlags;

/// Bindings managment
#[cfg(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"), feature = "python-bindings"))]
pub mod bindings;
/// Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
pub mod circuit;
/// CLI commands.
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub mod commands;
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
// abigen doesn't generate docs for this module
#[allow(missing_docs)]
/// Utility functions for contracts
pub mod eth;
/// Command execution
///
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub mod execute;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(feature = "onnx")]
pub mod graph;
/// beautiful logging
#[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
pub mod logger;
/// Tools for proofs and verification used by cli
pub mod pfsys;
/// srs sha hashes
#[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
pub mod srs_sha;
/// An implementation of multi-dimensional tensors.
pub mod tensor;
#[cfg(target_os = "ios")]
uniffi::setup_scaffolding!();

#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
use lazy_static::lazy_static;

/// The denominator in the fixed point representation used when quantizing inputs
pub type Scale = i32;

#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
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

#[cfg(any(target_os = "ios", target_arch = "wasm32"))]
const EZKL_KEY_FORMAT: &str = "raw-bytes";

#[cfg(any(target_os = "ios", target_arch = "wasm32"))]
const EZKL_BUF_CAPACITY: &usize = &8000;

#[derive(
    Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default, Copy,
)]
/// Commitment scheme
pub enum Commitments {
    #[default]
    /// KZG
    KZG,
    /// IPA
    IPA,
}

impl From<Option<Commitments>> for Commitments {
    fn from(value: Option<Commitments>) -> Self {
        value.unwrap_or(Commitments::KZG)
    }
}

impl FromStr for Commitments {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "kzg" => Ok(Commitments::KZG),
            "ipa" => Ok(Commitments::IPA),
            _ => Err("Invalid value for Commitments".to_string()),
        }
    }
}

impl From<KZGCommitmentScheme<Bn256>> for Commitments {
    fn from(_value: KZGCommitmentScheme<Bn256>) -> Self {
        Commitments::KZG
    }
}

impl From<IPACommitmentScheme<G1Affine>> for Commitments {
    fn from(_value: IPACommitmentScheme<G1Affine>) -> Self {
        Commitments::IPA
    }
}

impl std::fmt::Display for Commitments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Commitments::KZG => write!(f, "kzg"),
            Commitments::IPA => write!(f, "ipa"),
        }
    }
}

impl ToFlags for Commitments {
    /// Convert the struct to a subcommand string
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl From<String> for Commitments {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "kzg" => Commitments::KZG,
            "ipa" => Commitments::IPA,
            _ => {
                log::error!("Invalid value for Commitments");
                log::warn!("defaulting to KZG");
                Commitments::KZG
            }
        }
    }
}

/// Parameters specific to a proving run
#[derive(Debug, Args, Deserialize, Serialize, Clone, PartialEq, PartialOrd, ToFlags)]
pub struct RunArgs {
    /// The tolerance for error on model outputs
    #[arg(short = 'T', long, default_value = "0", value_hint = clap::ValueHint::Other)]
    pub tolerance: Tolerance,
    /// The denominator in the fixed point representation used when quantizing inputs
    #[arg(short = 'S', long, default_value = "7", value_hint = clap::ValueHint::Other)]
    pub input_scale: Scale,
    /// The denominator in the fixed point representation used when quantizing parameters
    #[arg(long, default_value = "7", value_hint = clap::ValueHint::Other)]
    pub param_scale: Scale,
    /// if the scale is ever > scale_rebase_multiplier * input_scale then the scale is rebased to input_scale (this a more advanced parameter, use with caution)
    #[arg(long, default_value = "1",  value_hint = clap::ValueHint::Other)]
    pub scale_rebase_multiplier: u32,
    /// The min and max elements in the lookup table input column
    #[arg(short = 'B', long, value_parser = parse_key_val::<IntegerRep, IntegerRep>, default_value = "-32768->32768")]
    pub lookup_range: Range,
    /// The log_2 number of rows
    #[arg(short = 'K', long, default_value = "17", value_hint = clap::ValueHint::Other)]
    pub logrows: u32,
    /// The log_2 number of rows
    #[arg(short = 'N', long, default_value = "2", value_hint = clap::ValueHint::Other)]
    pub num_inner_cols: usize,
    /// Hand-written parser for graph variables, eg. batch_size=1
    #[arg(short = 'V', long, value_parser = parse_key_val::<String, usize>, default_value = "batch_size->1", value_delimiter = ',', value_hint = clap::ValueHint::Other)]
    pub variables: Vec<(String, usize)>,
    /// Flags whether inputs are public, private, fixed, hashed, polycommit
    #[arg(long, default_value = "private", value_hint = clap::ValueHint::Other)]
    pub input_visibility: Visibility,
    /// Flags whether outputs are public, private, fixed, hashed, polycommit
    #[arg(long, default_value = "public", value_hint = clap::ValueHint::Other)]
    pub output_visibility: Visibility,
    /// Flags whether params are fixed, private, hashed, polycommit
    #[arg(long, default_value = "private", value_hint = clap::ValueHint::Other)]
    pub param_visibility: Visibility,
    #[arg(long, default_value = "false")]
    /// Rebase the scale using lookup table for division instead of using a range check
    pub div_rebasing: bool,
    /// Should constants with 0.0 fraction be rebased to scale 0
    #[arg(long, default_value = "false")]
    pub rebase_frac_zero_constants: bool,
    /// check mode (safe, unsafe, etc)
    #[arg(long, default_value = "unsafe", value_hint = clap::ValueHint::Other)]
    pub check_mode: CheckMode,
    /// commitment scheme
    #[arg(long, default_value = "kzg", value_hint = clap::ValueHint::Other)]
    pub commitment: Option<Commitments>,
    /// the base used for decompositions
    #[arg(long, default_value = "16384", value_hint = clap::ValueHint::Other)]
    pub decomp_base: usize,
    #[arg(long, default_value = "2", value_hint = clap::ValueHint::Other)]
    /// the number of legs used for decompositions
    pub decomp_legs: usize,
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
            commitment: None,
            decomp_base: 16384,
            decomp_legs: 2,
        }
    }
}

impl RunArgs {
    ///
    pub fn validate(&self) -> Result<(), String> {
        if self.param_visibility == Visibility::Public {
            return Err(
                "params cannot be public instances, you are probably trying to use `fixed` or `kzgcommit`"
                    .into(),
            );
        }
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
        if self.tolerance.val > 0.0 && self.output_visibility != Visibility::Public {
            return Err("tolerance > 0.0 requires output_visibility to be public".into());
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
