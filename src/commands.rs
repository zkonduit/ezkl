#[cfg(not(target_arch = "wasm32"))]
use alloy::primitives::Address as H160;
use clap::{Parser, Subcommand};
use clap_complete::Shell;
#[cfg(feature = "python-bindings")]
use pyo3::{
    conversion::{FromPyObject, PyTryFrom},
    exceptions::PyValueError,
    prelude::*,
    types::PyString,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;
use tosubcommand::{ToFlags, ToSubcommand};

use crate::{pfsys::ProofType, Commitments, RunArgs};

use crate::circuit::CheckMode;
#[cfg(not(target_arch = "wasm32"))]
use crate::graph::TestDataSource;
use crate::pfsys::TranscriptType;

/// The default path to the .json data file
pub const DEFAULT_DATA: &str = "input.json";
/// The default path to the .onnx model file
pub const DEFAULT_MODEL: &str = "network.onnx";
/// The default path to the compiled model file
pub const DEFAULT_COMPILED_CIRCUIT: &str = "model.compiled";
/// The default path to the .json witness file
pub const DEFAULT_WITNESS: &str = "witness.json";
/// The default path to the circuit settings file
pub const DEFAULT_SETTINGS: &str = "settings.json";
/// The default path to the proving key file
pub const DEFAULT_PK: &str = "pk.key";
/// The default path to the verification key file
pub const DEFAULT_VK: &str = "vk.key";
/// The default path to the proving key file for aggregated proofs
pub const DEFAULT_PK_AGGREGATED: &str = "pk_aggr.key";
/// The default path to the verification key file for aggregated proofs
pub const DEFAULT_VK_AGGREGATED: &str = "vk_aggr.key";
/// The default path to the proof file
pub const DEFAULT_PROOF: &str = "proof.json";
/// The default path to the proof file for aggregated proofs
pub const DEFAULT_PROOF_AGGREGATED: &str = "proof_aggr.json";
/// Default for whether to split proofs
pub const DEFAULT_SPLIT: &str = "false";
/// Default verifier abi
pub const DEFAULT_VERIFIER_ABI: &str = "verifier_abi.json";
/// Default verifier abi for aggregated proofs
pub const DEFAULT_VERIFIER_AGGREGATED_ABI: &str = "verifier_aggr_abi.json";
/// Default verifier abi for data attestation
pub const DEFAULT_VERIFIER_DA_ABI: &str = "verifier_da_abi.json";
/// Default solidity code
pub const DEFAULT_SOL_CODE: &str = "evm_deploy.sol";
/// Default calldata path
pub const DEFAULT_CALLDATA: &str = "calldata.bytes";
/// Default solidity code for aggregated proofs
pub const DEFAULT_SOL_CODE_AGGREGATED: &str = "evm_deploy_aggr.sol";
/// Default solidity code for data attestation
pub const DEFAULT_SOL_CODE_DA: &str = "evm_deploy_da.sol";
/// Default contract address
pub const DEFAULT_CONTRACT_ADDRESS: &str = "contract.address";
/// Default contract address for data attestation
pub const DEFAULT_CONTRACT_ADDRESS_DA: &str = "contract_da.address";
/// Default contract address for vk
pub const DEFAULT_CONTRACT_ADDRESS_VK: &str = "contract_vk.address";
/// Default check mode
pub const DEFAULT_CHECKMODE: &str = "safe";
/// Default calibration target
pub const DEFAULT_CALIBRATION_TARGET: &str = "resources";
/// Default logrows for aggregated proofs
pub const DEFAULT_AGGREGATED_LOGROWS: &str = "23";
/// Default optimizer runs
pub const DEFAULT_OPTIMIZER_RUNS: &str = "1";
/// Default fuzz runs
pub const DEFAULT_FUZZ_RUNS: &str = "10";
/// Default calibration file
pub const DEFAULT_CALIBRATION_FILE: &str = "calibration.json";
/// Default lookup safety margin
pub const DEFAULT_LOOKUP_SAFETY_MARGIN: &str = "2";
/// Default Compress selectors
pub const DEFAULT_DISABLE_SELECTOR_COMPRESSION: &str = "false";
/// Default render vk separately
pub const DEFAULT_RENDER_VK_SEPERATELY: &str = "false";
/// Default VK sol path
pub const DEFAULT_VK_SOL: &str = "vk.sol";
/// Default VK abi path
pub const DEFAULT_VK_ABI: &str = "vk.abi";
/// Default scale rebase multipliers for calibration
pub const DEFAULT_SCALE_REBASE_MULTIPLIERS: &str = "1,2,10";
/// Default use reduced srs for verification
pub const DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION: &str = "false";
/// Default only check for range check rebase
pub const DEFAULT_ONLY_RANGE_CHECK_REBASE: &str = "false";
/// Default commitment
pub const DEFAULT_COMMITMENT: &str = "kzg";

#[cfg(feature = "python-bindings")]
/// Converts TranscriptType into a PyObject (Required for TranscriptType to be compatible with Python)
impl IntoPy<PyObject> for TranscriptType {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            TranscriptType::Poseidon => "poseidon".to_object(py),
            TranscriptType::EVM => "evm".to_object(py),
        }
    }
}
#[cfg(feature = "python-bindings")]
/// Obtains TranscriptType from PyObject (Required for TranscriptType to be compatible with Python)
impl<'source> FromPyObject<'source> for TranscriptType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let trystr = <PyString as PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();
        match strval.to_lowercase().as_str() {
            "poseidon" => Ok(TranscriptType::Poseidon),
            "evm" => Ok(TranscriptType::EVM),
            _ => Err(PyValueError::new_err("Invalid value for TranscriptType")),
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
/// Determines what the calibration pass should optimize for
pub enum CalibrationTarget {
    /// Optimizes for reducing cpu and memory usage
    Resources {
        /// Whether to allow for column overflow. This can reduce memory usage (eg. for a browser environment), but may result in a verifier that doesn't fit on the blockchain.
        col_overflow: bool,
    },
    /// Optimizes for numerical accuracy given the fixed point representation
    Accuracy,
}

impl Default for CalibrationTarget {
    fn default() -> Self {
        CalibrationTarget::Resources {
            col_overflow: false,
        }
    }
}

impl std::fmt::Display for CalibrationTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CalibrationTarget::Resources { col_overflow: true } => {
                    "resources/col-overflow".to_string()
                }
                CalibrationTarget::Resources {
                    col_overflow: false,
                } => "resources".to_string(),
                CalibrationTarget::Accuracy => "accuracy".to_string(),
            }
        )
    }
}

impl ToFlags for CalibrationTarget {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl From<&str> for CalibrationTarget {
    fn from(s: &str) -> Self {
        match s {
            "resources" => CalibrationTarget::Resources {
                col_overflow: false,
            },
            "resources/col-overflow" => CalibrationTarget::Resources { col_overflow: true },
            "accuracy" => CalibrationTarget::Accuracy,
            _ => {
                log::error!("Invalid value for CalibrationTarget");
                log::warn!("Defaulting to resources");
                CalibrationTarget::default()
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
/// wrapper for H160 to make it easy to parse into flag vals
pub struct H160Flag {
    inner: H160,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<H160Flag> for H160 {
    fn from(val: H160Flag) -> H160 {
        val.inner
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ToFlags for H160Flag {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{:#x}", self.inner)]
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl From<&str> for H160Flag {
    fn from(s: &str) -> Self {
        Self {
            inner: H160::from_str(s).unwrap(),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Converts CalibrationTarget into a PyObject (Required for CalibrationTarget to be compatible with Python)
impl IntoPy<PyObject> for CalibrationTarget {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            CalibrationTarget::Resources { col_overflow: true } => {
                "resources/col-overflow".to_object(py)
            }
            CalibrationTarget::Resources {
                col_overflow: false,
            } => "resources".to_object(py),
            CalibrationTarget::Accuracy => "accuracy".to_object(py),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains CalibrationTarget from PyObject (Required for CalibrationTarget to be compatible with Python)
impl<'source> FromPyObject<'source> for CalibrationTarget {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let trystr = <PyString as PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();
        match strval.to_lowercase().as_str() {
            "resources" => Ok(CalibrationTarget::Resources {
                col_overflow: false,
            }),
            "resources/col-overflow" => Ok(CalibrationTarget::Resources { col_overflow: true }),
            "accuracy" => Ok(CalibrationTarget::Accuracy),
            _ => Err(PyValueError::new_err("Invalid value for CalibrationTarget")),
        }
    }
}
// not wasm
use lazy_static::lazy_static;

// if CARGO VERSION is 0.0.0 replace with "source - no compatibility guaranteed"
lazy_static! {
    /// The version of the ezkl library
    pub static ref VERSION: &'static str =  if env!("CARGO_PKG_VERSION") == "0.0.0" {
       "source - no compatibility guaranteed"
    } else {
        env!("CARGO_PKG_VERSION")
    };
}

/// Get the styles for the CLI
pub fn get_styles() -> clap::builder::Styles {
    clap::builder::Styles::styled()
        .usage(
            clap::builder::styling::Style::new()
                .bold()
                .underline()
                .fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Cyan))),
        )
        .header(
            clap::builder::styling::Style::new()
                .bold()
                .underline()
                .fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Cyan))),
        )
        .literal(
            clap::builder::styling::Style::new().fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Magenta))),
        )
        .invalid(
            clap::builder::styling::Style::new()
                .bold()
                .fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Red))),
        )
        .error(
            clap::builder::styling::Style::new()
                .bold()
                .fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Red))),
        )
        .valid(
            clap::builder::styling::Style::new()
                .bold()
                .underline()
                .fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Green))),
        )
        .placeholder(
            clap::builder::styling::Style::new().fg_color(Some(clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::White))),
        )
}

#[allow(missing_docs)]
#[derive(Parser, Debug, Clone)]
#[command(author, about, long_about = None)]
#[clap(version = *VERSION, styles = get_styles(), trailing_var_arg = true)]
pub struct Cli {
    /// If provided, outputs the completion file for given shell
    #[clap(long = "generate", value_parser)]
    pub generator: Option<Shell>,
    #[command(subcommand)]
    #[allow(missing_docs)]
    pub command: Option<Commands>,
}


#[allow(missing_docs)]
#[derive(Debug, Subcommand, Clone, Deserialize, Serialize, PartialEq, PartialOrd, ToSubcommand)]
pub enum Commands {
    #[cfg(feature = "empty-cmd")]
    /// Creates an empty buffer
    Empty,
    /// Loads model and prints model table
    Table {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    /// Generates the witness from an input file.
    GenWitness {
        /// The path to the .json data file
        #[arg(short = 'D', long, default_value = DEFAULT_DATA, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT, value_hint = clap::ValueHint::FilePath)]
        compiled_circuit: Option<PathBuf>,
        /// Path to output the witness .json file
        #[arg(short = 'O', long, default_value = DEFAULT_WITNESS, value_hint = clap::ValueHint::FilePath)]
        output: Option<PathBuf>,
        /// Path to the verification key file (optional - solely used to generate kzg commits)
        #[arg(short = 'V', long, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// Path to the srs file (optional - solely used to generate kzg commits)
        #[arg(short = 'P', long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
    },

    /// Produces the proving hyperparameters, from run-args
    GenSettings {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
        /// The path to generate the circuit settings .json file to
        #[arg(short = 'O', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    /// Calibrates the proving scale, lookup bits and logrows from a circuit settings file.
    #[cfg(not(target_arch = "wasm32"))]
    CalibrateSettings {
        /// The path to the .json calibration data file.
        #[arg(short = 'D', long, default_value = DEFAULT_CALIBRATION_FILE, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
        /// The path to load circuit settings .json file AND overwrite (generated using the gen-settings command).
        #[arg(short = 'O', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        #[arg(long = "target", default_value = DEFAULT_CALIBRATION_TARGET, value_hint = clap::ValueHint::Other)]
        /// Target for calibration. Set to "resources" to optimize for computational resource. Otherwise, set to "accuracy" to optimize for accuracy.
        target: CalibrationTarget,
        /// the lookup safety margin to use for calibration. if the max lookup is 2^k, then the max lookup will be 2^k * lookup_safety_margin. larger = safer but slower
        #[arg(long, default_value = DEFAULT_LOOKUP_SAFETY_MARGIN, value_hint = clap::ValueHint::Other)]
        lookup_safety_margin: i64,
        /// Optional scales to specifically try for calibration. Example, --scales 0,4
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true, value_hint = clap::ValueHint::Other)]
        scales: Option<Vec<crate::Scale>>,
        /// Optional scale rebase multipliers to specifically try for calibration. This is the multiplier at which we divide to return to the input scale. Example, --scale-rebase-multipliers 0,4
        #[arg(
            long,
            value_delimiter = ',',
            allow_hyphen_values = true,
            default_value = DEFAULT_SCALE_REBASE_MULTIPLIERS,
            value_hint = clap::ValueHint::Other
        )]
        scale_rebase_multiplier: Vec<u32>,
        /// max logrows to use for calibration, 26 is the max public SRS size
        #[arg(long, value_hint = clap::ValueHint::Other)]
        max_logrows: Option<u32>,
        // whether to only range check rebases (instead of trying both range check and lookup)
        #[arg(long, default_value = DEFAULT_ONLY_RANGE_CHECK_REBASE, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        only_range_check_rebase: Option<bool>,
    },

    /// Generates a dummy SRS
    #[command(name = "gen-srs", arg_required_else_help = true)]
    GenSrs {
        /// The path to output the generated SRS
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: PathBuf,
        /// number of logrows to use for srs
        #[arg(long, value_hint = clap::ValueHint::Other)]
        logrows: usize,
        /// commitment used
        #[arg(long, default_value = DEFAULT_COMMITMENT, value_hint = clap::ValueHint::Other)]
        commitment: Option<Commitments>,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Gets an SRS from a circuit settings file.
    #[command(name = "get-srs")]
    GetSrs {
        /// The path to output the desired srs file, if set to None will save to $EZKL_REPO_PATH/srs
        #[arg(long, default_value = None, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// Path to the circuit settings .json file to read in logrows from. Overriden by logrows if specified.
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// Number of logrows to use for srs. Overrides settings_path if specified.
        #[arg(long, default_value = None, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// Commitment used
        #[arg(long, default_value = None, value_hint = clap::ValueHint::Other)]
        commitment: Option<Commitments>,
    },
    /// Loads model and input and runs mock prover (for testing)
    Mock {
        /// The path to the .json witness file (generated using the gen-witness command)
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS, value_hint = clap::ValueHint::FilePath)]
        witness: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
    },

    /// Mock aggregate proofs
    MockAggregate {
        /// The path to the snarks to aggregate over (generated using the prove command with the --proof-type=for-aggr flag)
        #[arg(long, default_value = DEFAULT_PROOF, value_delimiter = ',', allow_hyphen_values = true, value_hint = clap::ValueHint::FilePath)]
        aggregation_snarks: Vec<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = DEFAULT_SPLIT, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        split_proofs: Option<bool>,
    },

    /// Setup aggregation circuit and generate pk and vk
    SetupAggregate {
        /// The path to samples of snarks that will be aggregated over (generated using the prove command with the --proof-type=for-aggr flag)
        #[arg(long, default_value = DEFAULT_PROOF, value_delimiter = ',', allow_hyphen_values = true, value_hint = clap::ValueHint::FilePath)]
        sample_snarks: Vec<PathBuf>,
        /// The path to save the desired verification key file to
        #[arg(long, default_value = DEFAULT_VK_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to save the proving key to
        #[arg(long, default_value = DEFAULT_PK_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        pk_path: Option<PathBuf>,
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = DEFAULT_SPLIT, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        split_proofs: Option<bool>,
        /// compress selectors
        #[arg(long, default_value = DEFAULT_DISABLE_SELECTOR_COMPRESSION, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        disable_selector_compression: Option<bool>,
        /// commitment used
        #[arg(long, default_value = DEFAULT_COMMITMENT, value_hint = clap::ValueHint::Other)]
        commitment: Option<Commitments>,
    },
    /// Aggregates proofs
    Aggregate {
        /// The path to the snarks to aggregate over (generated using the prove command with the --proof-type=for-aggr flag)
        #[arg(long, default_value = DEFAULT_PROOF, value_delimiter = ',', allow_hyphen_values = true, value_hint = clap::ValueHint::FilePath)]
        aggregation_snarks: Vec<PathBuf>,
        /// The path to load the desired proving key file (generated using the setup-aggregate command)
        #[arg(long, default_value = DEFAULT_PK_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        pk_path: Option<PathBuf>,
        /// The path to output the proof file to
        #[arg(long, default_value = DEFAULT_PROOF_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long)]
        srs_path: Option<PathBuf>,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::default(),
            value_enum, 
            value_hint = clap::ValueHint::Other
        )]
        transcript: TranscriptType,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = DEFAULT_CHECKMODE, value_hint = clap::ValueHint::Other)]
        check_mode: Option<CheckMode>,
        /// whether the accumulated proofs are segments of a larger circuit
        #[arg(long, default_value = DEFAULT_SPLIT, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        split_proofs: Option<bool>,
        /// commitment used
        #[arg(long, default_value = DEFAULT_COMMITMENT, value_hint = clap::ValueHint::Other)]
        commitment: Option<Commitments>,
    },
    /// Compiles a circuit from onnx to a simplified graph (einsum + other ops) and parameters as sets of field elements
    CompileCircuit {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(long, default_value = DEFAULT_COMPILED_CIRCUIT, value_hint = clap::ValueHint::FilePath)]
        compiled_circuit: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
    },
    /// Creates pk and vk
    Setup {
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT, value_hint = clap::ValueHint::FilePath)]
        compiled_circuit: Option<PathBuf>,
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to output the verification key file to
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the proving key file to
        #[arg(long, default_value = DEFAULT_PK, value_hint = clap::ValueHint::FilePath)]
        pk_path: Option<PathBuf>,
        /// The graph witness (optional - used to override fixed values in the circuit)
        #[arg(short = 'W', long, value_hint = clap::ValueHint::FilePath)]
        witness: Option<PathBuf>,
        /// compress selectors
        #[arg(long, default_value = DEFAULT_DISABLE_SELECTOR_COMPRESSION, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        disable_selector_compression: Option<bool>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Deploys a test contact that the data attester reads from and creates a data attestation formatted input.json file that contains call data information
    #[command(arg_required_else_help = true)]
    SetupTestEvmData {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(short = 'M', long, value_hint = clap::ValueHint::FilePath)]
        compiled_circuit: Option<PathBuf>,
        /// For testing purposes only. The optional path to the .json data file that will be generated that contains the OnChain data storage information
        /// derived from the file information in the data .json file.
        /// Should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'T', long, value_hint = clap::ValueHint::FilePath)]
        test_data: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: Option<String>,
        /// where the input data come from
        #[arg(long, default_value = "on-chain", value_hint = clap::ValueHint::Other)]
        input_source: TestDataSource,
        /// where the output data come from
        #[arg(long, default_value = "on-chain", value_hint = clap::ValueHint::Other)]
        output_source: TestDataSource,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// The Data Attestation Verifier contract stores the account calls to fetch data to feed into ezkl. This call data can be updated by an admin account. This tests that admin account is able to update this call data.
    #[command(arg_required_else_help = true)]
    TestUpdateAccountCalls {
        /// The path to the verifier contract's address
        #[arg(long, value_hint = clap::ValueHint::Other)]
        addr: H160Flag,
        /// The path to the .json data file.
        #[arg(short = 'D', long, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Swaps the positions in the transcript that correspond to commitments
    SwapProofCommitments {
        /// The path to the proof file
        #[arg(short = 'P', long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the witness file
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS, value_hint = clap::ValueHint::FilePath)]
        witness_path: Option<PathBuf>,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Loads model, data, and creates proof
    Prove {
        /// The path to the .json witness file (generated using the gen-witness command)
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS, value_hint = clap::ValueHint::FilePath)]
        witness: Option<PathBuf>,
        /// The path to the compiled model file (generated using the compile-circuit command)
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT, value_hint = clap::ValueHint::FilePath)]
        compiled_circuit: Option<PathBuf>,
        /// The path to load the desired proving key file (generated using the setup command)
        #[arg(long, default_value = DEFAULT_PK, value_hint = clap::ValueHint::FilePath)]
        pk_path: Option<PathBuf>,
        /// The path to output the proof file to
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofType::Single,
            value_enum, 
            value_hint = clap::ValueHint::Other
        )]
        proof_type: ProofType,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = DEFAULT_CHECKMODE, value_hint = clap::ValueHint::Other)]
        check_mode: Option<CheckMode>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Encodes a proof into evm calldata
    #[command(name = "encode-evm-calldata")]
    EncodeEvmCalldata {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the Solidity code
        #[arg(long, default_value = DEFAULT_CALLDATA, value_hint = clap::ValueHint::FilePath)]
        calldata_path: Option<PathBuf>,
        /// The path to the verification key address (only used if the vk is rendered as a separate contract)
        #[arg(long, value_hint = clap::ValueHint::Other)]
        addr_vk: Option<H160Flag>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an Evm verifier for a single proof
    #[command(name = "create-evm-verifier")]
    CreateEvmVerifier {
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_ABI, value_hint = clap::ValueHint::FilePath)]
        abi_path: Option<PathBuf>,
        /// Whether the verifier key should be rendered as a separate contract.
        /// We recommend disabling selector compression if this is enabled.
        /// To save the verifier key as a separate contract, set this to true and then call the create-evm-vk command.        
        #[arg(long, default_value = DEFAULT_RENDER_VK_SEPERATELY, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        render_vk_seperately: Option<bool>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an Evm verifier for a single proof
    #[command(name = "create-evm-vk")]
    CreateEvmVK {
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the Solidity code
        #[arg(long, default_value = DEFAULT_VK_SOL, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VK_ABI, value_hint = clap::ValueHint::FilePath)]
        abi_path: Option<PathBuf>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an Evm verifier that attests to on-chain inputs for a single proof
    #[command(name = "create-evm-da")]
    CreateEvmDataAttestation {
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// The path to output the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE_DA, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_DA_ABI, value_hint = clap::ValueHint::FilePath)]
        abi_path: Option<PathBuf>,
        /// The path to the .json data file, which should
        /// contain the necessary calldata and account addresses
        /// needed to read from all the on-chain
        /// view functions that return the data that the network
        /// ingests as inputs.
        #[arg(short = 'D', long, default_value = DEFAULT_DATA, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an Evm verifier for an aggregate proof
    #[command(name = "create-evm-verifier-aggr")]
    CreateEvmVerifierAggr {
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_AGGREGATED_ABI, value_hint = clap::ValueHint::FilePath)]
        abi_path: Option<PathBuf>,
        // aggregated circuit settings paths, used to calculate the number of instances in the aggregate proof
        #[arg(long, default_value = DEFAULT_SETTINGS, value_delimiter = ',', allow_hyphen_values = true, value_hint = clap::ValueHint::FilePath)]
        aggregation_settings: Vec<PathBuf>,
        // logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// Whether the verifier key should be rendered as a separate contract.
        /// We recommend disabling selector compression if this is enabled.
        /// To save the verifier key as a separate contract, set this to true and then call the create-evm-vk command.        
        #[arg(long, default_value = DEFAULT_RENDER_VK_SEPERATELY, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        render_vk_seperately: Option<bool>,
    },
    /// Verifies a proof, returning accept or reject
    Verify {
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the verification key file (generated using the setup command)
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// Reduce SRS logrows to the number of instances rather than the number of logrows used for proofs (only works if the srs were generated in the same ceremony)
        #[arg(long, default_value = DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        reduced_srs: Option<bool>,
    },
    /// Verifies an aggregate proof, returning accept or reject
    VerifyAggr {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the verification key file (generated using the setup-aggregate command)
        #[arg(long, default_value = DEFAULT_VK_AGGREGATED, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// reduced srs
        #[arg(long, default_value = DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION, action = clap::ArgAction::SetTrue, value_hint = clap::ValueHint::Other)]
        reduced_srs: Option<bool>,
        /// The path to SRS, if None will use $EZKL_REPO_PATH/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// commitment
        #[arg(long, default_value = DEFAULT_COMMITMENT, value_hint = clap::ValueHint::Other)]
        commitment: Option<Commitments>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Deploys an evm verifier that is generated by ezkl
    DeployEvmVerifier {
        /// The path to the Solidity code (generated using the create-evm-verifier command)
        #[arg(long, default_value = DEFAULT_SOL_CODE, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: Option<String>,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Other)]
        /// The path to output the contract address
        addr_path: Option<PathBuf>,
        /// The optimizer runs to set on the verifier. Lower values optimize for deployment cost, while higher values optimize for gas cost.
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS, value_hint = clap::ValueHint::Other)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Deploys an evm verifier that is generated by ezkl
    DeployEvmVK {
        /// The path to the Solidity code (generated using the create-evm-verifier command)
        #[arg(long, default_value = DEFAULT_VK_SOL, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: Option<String>,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS_VK, value_hint = clap::ValueHint::Other)]
        /// The path to output the contract address
        addr_path: Option<PathBuf>,
        /// The optimizer runs to set on the verifier. Lower values optimize for deployment cost, while higher values optimize for gas cost.
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS, value_hint = clap::ValueHint::Other)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Deploys an evm verifier that allows for data attestation
    #[command(name = "deploy-evm-da")]
    DeployEvmDataAttestation {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long, default_value = DEFAULT_DATA, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// The path to the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE_DA, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: Option<String>,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS_DA, value_hint = clap::ValueHint::FilePath)]
        /// The path to output the contract address
        addr_path: Option<PathBuf>,
        /// The optimizer runs to set on the verifier. (Lower values optimize for deployment, while higher values optimize for execution)
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS, value_hint = clap::ValueHint::Other)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Verifies a proof using a local Evm executor, returning accept or reject
    #[command(name = "verify-evm")]
    VerifyEvm {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to verifier contract's address
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Other)]
        addr_verifier: H160Flag,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: Option<String>,
        /// does the verifier use data attestation ?
        #[arg(long, value_hint = clap::ValueHint::Other)]
        addr_da: Option<H160Flag>,
        // is the vk rendered seperately, if so specify an address
        #[arg(long, value_hint = clap::ValueHint::Other)]
        addr_vk: Option<H160Flag>,
    },
}


impl Commands {
    /// Converts the commands to a json string
    pub fn as_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    /// Converts a json string to a Commands struct
    pub fn from_json(json: &str) -> Self {
        serde_json::from_str(json).unwrap()
    }
}