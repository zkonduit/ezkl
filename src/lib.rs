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
use log::warn;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use mimalloc as _;

/// Error type
// #[cfg_attr(not(feature = "ezkl"), derive(uniffi::Error))]
#[derive(thiserror::Error, Debug)]
#[allow(missing_docs)]
pub enum EZKLError {
    #[error("[aggregation] {0}")]
    AggregationError(#[from] pfsys::evm::aggregation_kzg::AggregationError),
    #[cfg(all(
        feature = "ezkl",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    ))]
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
    #[cfg(all(
        feature = "ezkl",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    ))]
    #[error("[reqwest] {0}")]
    ReqwestError(#[from] reqwest::Error),
    #[error("[fmt] {0}")]
    FmtError(#[from] std::fmt::Error),
    #[error("[halo2] {0}")]
    Halo2Error(#[from] halo2_proofs::plonk::Error),
    #[error("[Uncategorized] {0}")]
    UncategorizedError(String),
    #[cfg(all(
        feature = "ezkl",
        not(all(target_arch = "wasm32", target_os = "unknown"))
    ))]
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

use circuit::{table::Range, CheckMode};
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use clap::Args;
use fieldutils::IntegerRep;
use graph::{Visibility, MAX_PUBLIC_SRS};
use halo2_proofs::poly::{
    ipa::commitment::IPACommitmentScheme, kzg::commitment::KZGCommitmentScheme,
};
use halo2curves::bn256::{Bn256, G1Affine};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToFlags;

// if CARGO VERSION is 0.0.0 replace with "source - no compatibility guaranteed"
/// The version of the ezkl library
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get the version of the library
pub fn version() -> &'static str {
    match VERSION {
        "0.0.0" => "source - no compatibility guaranteed",
        _ => VERSION,
    }
}

/// Bindings management
#[cfg(any(
    feature = "ios-bindings",
    all(target_arch = "wasm32", target_os = "unknown"),
    feature = "python-bindings"
))]
pub mod bindings;
/// Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
pub mod circuit;
/// CLI commands.
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
pub mod commands;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
// abigen doesn't generate docs for this module
#[allow(missing_docs)]
/// Utility functions for contracts
pub mod eth;
/// Command execution
///
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
pub mod execute;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
#[cfg(any(feature = "onnx", not(feature = "ezkl")))]
pub mod graph;
/// beautiful logging
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
pub mod logger;
/// Tools for proofs and verification used by cli
pub mod pfsys;
/// srs sha hashes
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
pub mod srs_sha;
/// An implementation of multi-dimensional tensors.
pub mod tensor;
#[cfg(feature = "ios-bindings")]
uniffi::setup_scaffolding!();
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use lazy_static::lazy_static;

/// The denominator in the fixed point representation used when quantizing inputs
pub type Scale = i32;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
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

#[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
const EZKL_KEY_FORMAT: &str = "raw-bytes";

#[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
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

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
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
///
/// RunArgs contains all configuration parameters needed to control the proving process,
/// including scaling factors, visibility settings, and circuit parameters.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, PartialOrd)]
#[cfg_attr(
    all(feature = "ezkl", not(target_arch = "wasm32")),
    derive(Args, ToFlags)
)]
pub struct RunArgs {
    /// Fixed point scaling factor for quantizing inputs
    /// Higher values provide more precision but increase circuit complexity
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(short = 'S', long, default_value = "7", value_hint = clap::ValueHint::Other))]
    pub input_scale: Scale,
    /// Fixed point scaling factor for quantizing parameters
    /// Higher values provide more precision but increase circuit complexity
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "7", value_hint = clap::ValueHint::Other))]
    pub param_scale: Scale,
    /// Scale rebase threshold multiplier
    /// When scale exceeds input_scale * multiplier, it is rebased to input_scale
    /// Advanced parameter that should be used with caution
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "1", value_hint = clap::ValueHint::Other))]
    pub scale_rebase_multiplier: u32,
    /// Range for lookup table input column values
    /// Specified as (min, max) pair
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(short = 'B', long, value_parser = parse_key_val::<IntegerRep, IntegerRep>, default_value = "-32768->32768"))]
    pub lookup_range: Range,
    /// Log2 of the number of rows in the circuit
    /// Controls circuit size and proving time
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(short = 'K', long, default_value = "17", value_hint = clap::ValueHint::Other))]
    pub logrows: u32,
    /// Number of inner columns per block
    /// Affects circuit layout and efficiency
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(short = 'N', long, default_value = "2", value_hint = clap::ValueHint::Other))]
    pub num_inner_cols: usize,
    /// Graph variables for parameterizing the computation
    /// Format: "name->value", e.g. "batch_size->1"
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(short = 'V', long, value_parser = parse_key_val::<String, usize>, default_value = "batch_size->1", value_delimiter = ',', value_hint = clap::ValueHint::Other))]
    pub variables: Vec<(String, usize)>,
    /// Visibility setting for input values
    /// Controls whether inputs are public or private in the circuit
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "private", value_hint = clap::ValueHint::Other))]
    pub input_visibility: Visibility,
    /// Visibility setting for output values
    /// Controls whether outputs are public or private in the circuit
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "public", value_hint = clap::ValueHint::Other))]
    pub output_visibility: Visibility,
    /// Visibility setting for parameters
    /// Controls how parameters are handled in the circuit
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "private", value_hint = clap::ValueHint::Other))]
    pub param_visibility: Visibility,
    /// Whether to rebase constants with zero fractional part to scale 0
    /// Can improve efficiency for integer constants
    #[cfg_attr(
        all(feature = "ezkl", not(target_arch = "wasm32")),
        arg(long, default_value = "false")
    )]
    pub rebase_frac_zero_constants: bool,
    /// Circuit checking mode
    /// Controls level of constraint verification
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "unsafe", value_hint = clap::ValueHint::Other))]
    pub check_mode: CheckMode,
    /// Commitment scheme for circuit proving
    /// Affects proof size and verification time
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "kzg", value_hint = clap::ValueHint::Other))]
    pub commitment: Option<Commitments>,
    /// Base for number decomposition
    /// Must be a power of 2
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "16384", value_hint = clap::ValueHint::Other))]
    pub decomp_base: usize,
    /// Number of decomposition legs
    /// Controls decomposition granularity
    #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(long, default_value = "2", value_hint = clap::ValueHint::Other))]
    pub decomp_legs: usize,
    /// Whether to use bounded lookup for logarithm computation
    #[cfg_attr(
        all(feature = "ezkl", not(target_arch = "wasm32")),
        arg(long, default_value = "false")
    )]
    pub bounded_log_lookup: bool,
    /// Range check inputs and outputs (turn off if the inputs are felts)
    #[cfg_attr(
        all(feature = "ezkl", not(target_arch = "wasm32")),
        arg(long, default_value = "false")
    )]
    pub ignore_range_check_inputs_outputs: bool,
}

impl Default for RunArgs {
    /// Creates a new RunArgs instance with default values
    ///
    /// Default configuration is optimized for common use cases
    /// while maintaining reasonable proving time and circuit size
    fn default() -> Self {
        Self {
            bounded_log_lookup: false,
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
            rebase_frac_zero_constants: false,
            check_mode: CheckMode::UNSAFE,
            commitment: None,
            decomp_base: 16384,
            decomp_legs: 2,
            ignore_range_check_inputs_outputs: false,
        }
    }
}

impl RunArgs {
    /// Validates the RunArgs configuration
    ///
    /// Performs comprehensive validation of all parameters to ensure they are within
    /// acceptable ranges and follow required constraints. Returns accumulated errors
    /// if any validations fail.
    ///
    /// # Returns
    /// - Ok(()) if all validations pass
    /// - Err(String) with detailed error message if any validation fails
    pub fn validate(&self) -> Result<(), String> {
        let mut errors = Vec::new();

        // check if the largest represented integer in the decomposed form overflows IntegerRep
        //  try it with the largest possible value
        let max_decomp = (self.decomp_base as IntegerRep).checked_pow(self.decomp_legs as u32);
        if max_decomp.is_none() {
            errors.push(format!(
                "decomp_base^decomp_legs overflows IntegerRep: {}^{}",
                self.decomp_base, self.decomp_legs
            ));
        }

        // Visibility validations
        if self.param_visibility == Visibility::Public {
            errors.push(
                "Parameters cannot be public instances. Use 'fixed' or 'kzgcommit' instead"
                    .to_string(),
            );
        }

        // Scale validations
        if self.scale_rebase_multiplier < 1 {
            errors.push("scale_rebase_multiplier must be >= 1".to_string());
        }

        // if any of the scales are too small
        if self.input_scale < 8 || self.param_scale < 8 {
            warn!("low scale values (<8) may impact precision");
        }

        // Lookup range validations
        if self.lookup_range.0 > self.lookup_range.1 {
            errors.push(format!(
                "Invalid lookup range: min ({}) is greater than max ({})",
                self.lookup_range.0, self.lookup_range.1
            ));
        }

        // Size validations
        if self.logrows < 1 {
            errors.push("logrows must be >= 1".to_string());
        }

        if self.num_inner_cols < 1 {
            errors.push("num_inner_cols must be >= 1".to_string());
        }

        let batch_size = self.variables.iter().find(|(name, _)| name == "batch_size");
        if let Some(batch_size) = batch_size {
            if batch_size.1 == 0 {
                errors.push("'batch_size' cannot be 0".to_string());
            }
        }

        // Decomposition validations
        if self.decomp_base == 0 {
            errors.push("decomp_base cannot be 0".to_string());
        }

        if self.decomp_legs == 0 {
            errors.push("decomp_legs cannot be 0".to_string());
        }

        // Performance validations
        if self.logrows > MAX_PUBLIC_SRS {
            warn!("logrows exceeds maximum public SRS size");
        }

        // Performance warnings
        if self.input_scale > 20 || self.param_scale > 20 {
            warn!("High scale values (>20) may impact performance");
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    }

    /// Exports the configuration as JSON
    ///
    /// Serializes the RunArgs instance to a JSON string
    ///
    /// # Returns
    /// * `Ok(String)` containing JSON representation
    /// * `Err` if serialization fails
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let res = serde_json::to_string(&self)?;
        Ok(res)
    }

    /// Parses configuration from JSON
    ///
    /// Deserializes a RunArgs instance from a JSON string
    ///
    /// # Arguments
    /// * `arg_json` - JSON string containing configuration
    ///
    /// # Returns
    /// * `Ok(RunArgs)` if parsing succeeds
    /// * `Err` if parsing fails
    pub fn from_json(arg_json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(arg_json)
    }
}

// Additional helper functions for the module

/// Parses a key-value pair from a string in the format "key->value"
///
/// # Arguments
/// * `s` - Input string in the format "key->value"
///
/// # Returns
/// * `Ok((T, U))` - Parsed key and value
/// * `Err` - If parsing fails
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
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
        .ok_or_else(|| format!("invalid KEY->VALUE: no `->` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 2..].parse()?))
}

/// Verifies that a version string matches the expected artifact version
/// Logs warnings for version mismatches or unversioned artifacts
///
/// # Arguments
/// * `artifact_version` - Version string from the artifact
pub fn check_version_string_matches(artifact_version: &str) {
    if artifact_version == "0.0.0"
        || artifact_version == "source - no compatibility guaranteed"
        || artifact_version.is_empty()
    {
        log::warn!("Artifact version is 0.0.0, skipping version check");
        return;
    }

    let version = crate::version();

    if version == "source - no compatibility guaranteed" {
        log::warn!("Compiled source version is not guaranteed to match artifact version");
        return;
    }

    if version != artifact_version {
        log::warn!(
            "Version mismatch: CLI version is {} but artifact version is {}",
            version,
            artifact_version
        );
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_default_args() {
        let args = RunArgs::default();
        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_invalid_param_visibility() {
        let mut args = RunArgs::default();
        args.param_visibility = Visibility::Public;
        let err = args.validate().unwrap_err();
        assert!(err.contains("Parameters cannot be public instances"));
    }

    #[test]
    fn test_invalid_scale_rebase() {
        let mut args = RunArgs::default();
        args.scale_rebase_multiplier = 0;
        let err = args.validate().unwrap_err();
        assert!(err.contains("scale_rebase_multiplier must be >= 1"));
    }

    #[test]
    fn test_invalid_lookup_range() {
        let mut args = RunArgs::default();
        args.lookup_range = (100, -100);
        let err = args.validate().unwrap_err();
        assert!(err.contains("Invalid lookup range"));
    }

    #[test]
    fn test_invalid_logrows() {
        let mut args = RunArgs::default();
        args.logrows = 0;
        let err = args.validate().unwrap_err();
        assert!(err.contains("logrows must be >= 1"));
    }

    #[test]
    fn test_invalid_inner_cols() {
        let mut args = RunArgs::default();
        args.num_inner_cols = 0;
        let err = args.validate().unwrap_err();
        assert!(err.contains("num_inner_cols must be >= 1"));
    }

    #[test]
    fn test_zero_batch_size() {
        let mut args = RunArgs::default();
        args.variables = vec![("batch_size".to_string(), 0)];
        let err = args.validate().unwrap_err();
        assert!(err.contains("'batch_size' cannot be 0"));
    }

    #[test]
    fn test_json_serialization() {
        let args = RunArgs::default();
        let json = args.as_json().unwrap();
        let deserialized = RunArgs::from_json(&json).unwrap();
        assert_eq!(args, deserialized);
    }

    #[test]
    fn test_multiple_validation_errors() {
        let mut args = RunArgs::default();
        args.logrows = 0;
        args.lookup_range = (100, -100);
        let err = args.validate().unwrap_err();
        // Should contain multiple error messages
        assert!(err.matches("\n").count() >= 1);
    }
}
