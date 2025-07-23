#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use alloy::primitives::Address as H160;
use clap::{Command, Parser, Subcommand};
use clap_complete::{generate, Generator, Shell};
#[cfg(feature = "python-bindings")]
use pyo3::{conversion::FromPyObject, exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;
use tosubcommand::{ToFlags, ToSubcommand};

use crate::{pfsys::ProofType, Commitments, RunArgs};

use crate::circuit::CheckMode;
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
/// Default solidity code
pub const DEFAULT_SOL_CODE: &str = "evm_deploy.sol";
/// Default calldata path
pub const DEFAULT_CALLDATA: &str = "calldata.bytes";
/// Default solidity code for aggregated proofs
pub const DEFAULT_SOL_CODE_AGGREGATED: &str = "evm_deploy_aggr.sol";
/// Default contract address
pub const DEFAULT_CONTRACT_ADDRESS: &str = "contract.address";
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
/// Default render reusable verifier
pub const DEFAULT_RENDER_REUSABLE: &str = "false";
/// Default contract deployment type
pub const DEFAULT_CONTRACT_DEPLOYMENT_TYPE: &str = "verifier";
/// Default VKA calldata path
pub const DEFAULT_VKA: &str = "vka.bytes";
/// Default VK abi path
pub const DEFAULT_VK_ABI: &str = "vk.abi";
/// Default scale rebase multipliers for calibration
pub const DEFAULT_SCALE_REBASE_MULTIPLIERS: &str = "1,10";
/// Default use reduced srs for verification
pub const DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION: &str = "false";
/// Default only check for range check rebase
pub const DEFAULT_ONLY_RANGE_CHECK_REBASE: &str = "false";
/// Default commitment
pub const DEFAULT_COMMITMENT: &str = "kzg";
/// Default seed used to generate random data
pub const DEFAULT_SEED: &str = "21242";
/// Default number of decimals for instances rescaling on-chain.
pub const DEFAULT_DECIMALS: &str = "18";
/// Default path for the vka digest file
pub const DEFAULT_VKA_DIGEST: &str = "vka.digest";

#[cfg(feature = "python-bindings")]
/// Converts TranscriptType into a PyObject (Required for TranscriptType to be compatible with Python)
impl<'py> IntoPyObject<'py> for TranscriptType {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let result = match self {
            TranscriptType::Poseidon => "poseidon",
            TranscriptType::EVM => "evm",
        };
        Ok(result.into_pyobject(py)?.into_any())
    }
}
#[cfg(feature = "python-bindings")]
/// Obtains TranscriptType from PyObject (Required for TranscriptType to be compatible with Python)
impl<'source> FromPyObject<'source> for TranscriptType {
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let trystr = String::extract_bound(ob)?;
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

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
/// Determines what type of contract (verifier, verifier/reusable, vka) should be deployed
pub enum ContractType {
    /// Deploys a verifier contrat tailored to the circuit and not reusable
    Verifier {
        /// Whether to deploy a reusable verifier. This can reduce state bloat on-chain since you need only deploy a verifying key artifact (vka) for a given circuit which is significantly smaller than the verifier contract (up to 4 times smaller for large circuits)
        /// Can also be used as an alternative to aggregation for verifiers that are otherwise too large to fit on-chain.
        reusable: bool,
    },
}

impl Default for ContractType {
    fn default() -> Self {
        ContractType::Verifier { reusable: false }
    }
}

impl std::fmt::Display for ContractType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ContractType::Verifier { reusable: true } => {
                    "verifier/reusable".to_string()
                }
                ContractType::Verifier { reusable: false } => "verifier".to_string(),
            }
        )
    }
}

impl ToFlags for ContractType {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl From<&str> for ContractType {
    fn from(s: &str) -> Self {
        match s {
            "verifier" => ContractType::Verifier { reusable: false },
            "verifier/reusable" => ContractType::Verifier { reusable: true },
            _ => {
                log::error!("Invalid value for ContractType");
                log::warn!("Defaulting to verifier");
                ContractType::default()
            }
        }
    }
}

#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
/// wrapper for H160 to make it easy to parse into flag vals
pub struct H160Flag {
    inner: H160,
}
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
impl From<H160Flag> for H160 {
    fn from(val: H160Flag) -> H160 {
        val.inner
    }
}
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
impl ToFlags for H160Flag {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{:#x}", self.inner)]
    }
}
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
impl From<&str> for H160Flag {
    fn from(s: &str) -> Self {
        Self {
            inner: H160::from_str(s).unwrap(),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Converts CalibrationTarget into a PyObject (Required for CalibrationTarget to be compatible with Python)
impl<'py> IntoPyObject<'py> for CalibrationTarget {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let result = match self {
            CalibrationTarget::Resources { col_overflow: true } => {
                "resources/col-overflow"
            }
            CalibrationTarget::Resources {
                col_overflow: false,
            } => "resources",
            CalibrationTarget::Accuracy => "accuracy",
        };
        Ok(result.into_pyobject(py)?.into_any())
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains CalibrationTarget from PyObject (Required for CalibrationTarget to be compatible with Python)
impl<'source> FromPyObject<'source> for CalibrationTarget {
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let strval = String::extract_bound(ob)?;
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

#[cfg(feature = "python-bindings")]
/// Converts ContractType into a PyObject (Required for ContractType to be compatible with Python)
impl<'py> IntoPyObject<'py> for ContractType {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let result = match self {
            ContractType::Verifier { reusable: true } => "verifier/reusable",
            ContractType::Verifier { reusable: false } => "verifier",
        };
        Ok(result.into_pyobject(py)?.into_any())
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains ContractType from PyObject (Required for ContractType to be compatible with Python)
impl<'source> FromPyObject<'source> for ContractType {
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let strval = String::extract_bound(ob)?;
        match strval.to_lowercase().as_str() {
            "verifier" => Ok(ContractType::Verifier { reusable: false }),
            "verifier/reusable" => Ok(ContractType::Verifier { reusable: true }),
            _ => Err(PyValueError::new_err("Invalid value for ContractType")),
        }
    }
}

/// Get the styles for the CLI
pub fn get_styles() -> clap::builder::Styles {
    clap::builder::Styles::styled()
        .usage(
            clap::builder::styling::Style::new()
                .bold()
                .underline()
                .fg_color(Some(clap::builder::styling::Color::Ansi(
                    clap::builder::styling::AnsiColor::Cyan,
                ))),
        )
        .header(
            clap::builder::styling::Style::new()
                .bold()
                .underline()
                .fg_color(Some(clap::builder::styling::Color::Ansi(
                    clap::builder::styling::AnsiColor::Cyan,
                ))),
        )
        .literal(clap::builder::styling::Style::new().fg_color(Some(
            clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Magenta),
        )))
        .invalid(clap::builder::styling::Style::new().bold().fg_color(Some(
            clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Red),
        )))
        .error(clap::builder::styling::Style::new().bold().fg_color(Some(
            clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::Red),
        )))
        .valid(
            clap::builder::styling::Style::new()
                .bold()
                .underline()
                .fg_color(Some(clap::builder::styling::Color::Ansi(
                    clap::builder::styling::AnsiColor::Green,
                ))),
        )
        .placeholder(clap::builder::styling::Style::new().fg_color(Some(
            clap::builder::styling::Color::Ansi(clap::builder::styling::AnsiColor::White),
        )))
}

/// Print completions for the given generator
pub fn print_completions<G: Generator>(r#gen: G, cmd: &mut Command) {
    generate(
        r#gen,
        cmd,
        cmd.get_name().to_string(),
        &mut std::io::stdout(),
    );
}

#[allow(missing_docs)]
#[derive(Parser, Debug, Clone)]
#[command(author, about, long_about = None)]
#[clap(version = crate::version(), styles = get_styles(), trailing_var_arg = true)]
pub struct Cli {
    /// If provided, outputs the completion file for given shell
    #[clap(long = "generate", value_parser)]
    pub generator: Option<Shell>,
    #[command(subcommand)]
    #[allow(missing_docs)]
    pub command: Option<Commands>,
}

/// Custom parser for data field that handles both direct JSON strings and file paths with '@' prefix
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct DataField(pub String);

impl FromStr for DataField {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Check if the input starts with '@'
        if let Some(file_path) = s.strip_prefix('@') {
            // Extract the file path (remove the '@' prefix)
            // Read the file content
            let content = std::fs::read_to_string(file_path)
                .map_err(|e| format!("Failed to read data file '{}': {}", file_path, e))?;

            // Return the file content as the data field value
            Ok(DataField(content))
        } else {
            // Use the input string directly
            Ok(DataField(s.to_string()))
        }
    }
}

impl ToFlags for DataField {
    fn to_flags(&self) -> Vec<String> {
        vec![self.0.clone()]
    }
}

impl std::fmt::Display for DataField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
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
        /// The path to the .json data file (with @ prefix) or a raw data string of the form '{"input_data": [[1, 2, 3]]}'
        #[arg(short = 'D', long, default_value = DEFAULT_DATA, value_parser = DataField::from_str)]
        data: Option<DataField>,
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
    /// Generate random data for a model
    GenRandomData {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
        /// The path to the .json data file
        #[arg(short = 'D', long, default_value = DEFAULT_DATA, value_hint = clap::ValueHint::FilePath)]
        data: Option<PathBuf>,
        /// Hand-written parser for graph variables, eg. batch_size=1
        #[cfg_attr(all(feature = "ezkl", not(target_arch = "wasm32")), arg(short = 'V', long, value_parser = crate::parse_key_val::<String, usize>, default_value = "batch_size->1", value_delimiter = ',', value_hint = clap::ValueHint::Other))]
        variables: Vec<(String, usize)>,
        /// random seed for reproducibility (optional)
        #[arg(long, value_hint = clap::ValueHint::Other, default_value = DEFAULT_SEED)]
        seed: u64,
        /// min value for random data
        #[arg(long, value_hint = clap::ValueHint::Other)]
        min: Option<f32>,
        /// max value for random data
        #[arg(long, value_hint = clap::ValueHint::Other)]
        max: Option<f32>,
    },
    /// Calibrates the proving scale, lookup bits and logrows from a circuit settings file.
    CalibrateSettings {
        /// The path to the .json calibration data file.
        #[arg(short = 'D', long, default_value = DEFAULT_CALIBRATION_FILE, value_hint = clap::ValueHint::FilePath)]
        data: Option<String>,
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL, value_hint = clap::ValueHint::FilePath)]
        model: Option<PathBuf>,
        /// The path to load circuit settings .json file AND overwrite (generated using the gen-settings command).
        #[arg(short = 'O', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        #[arg(long = "target", default_value = DEFAULT_CALIBRATION_TARGET, value_hint = clap::ValueHint::Other)]
        /// Target for calibration. Set to "resources" to optimize for computational resource. Otherwise, set to "accuracy" to optimize for accuracy.
        target: CalibrationTarget,
        /// the lookup safety margin to use for calibration. if the max lookup is 2^k, then the max lookup will be ceil(2^k * lookup_safety_margin). larger = safer but slower
        #[arg(long, default_value = DEFAULT_LOOKUP_SAFETY_MARGIN, value_hint = clap::ValueHint::Other)]
        lookup_safety_margin: f64,
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

    /// Gets an SRS from a circuit settings file.
    #[command(name = "get-srs")]
    GetSrs {
        /// The path to output the desired srs file, if set to None will save to ~/.ezkl/srs
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
        #[arg(long, default_value = DEFAULT_SPLIT, action = clap::ArgAction::SetTrue)]
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
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = DEFAULT_SPLIT, action = clap::ArgAction::SetTrue)]
        split_proofs: Option<bool>,
        /// compress selectors
        #[arg(long, default_value = DEFAULT_DISABLE_SELECTOR_COMPRESSION, action = clap::ArgAction::SetTrue)]
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
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
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
        #[arg(long, default_value = DEFAULT_SPLIT, action = clap::ArgAction::SetTrue)]
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
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
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
        #[arg(long, default_value = DEFAULT_DISABLE_SELECTOR_COMPRESSION, action = clap::ArgAction::SetTrue)]
        disable_selector_compression: Option<bool>,
    },
    /// Swaps the positions in the transcript that correspond to commitments
    SwapProofCommitments {
        /// The path to the proof file
        #[arg(short = 'P', long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to the witness file
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS, value_hint = clap::ValueHint::FilePath)]
        witness_path: Option<PathBuf>,
    },

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
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
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
    /// Encodes a proof into evm calldata
    #[command(name = "encode-evm-calldata")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    EncodeEvmCalldata {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to save the calldata to
        #[arg(long, default_value = DEFAULT_CALLDATA, value_hint = clap::ValueHint::FilePath)]
        calldata_path: Option<PathBuf>,
        /// The path to the serialized VKA file
        #[cfg_attr(all(feature = "reusable-verifier", not(target_arch = "wasm32")), arg(long, value_hint = clap::ValueHint::Other))]
        vka_path: Option<PathBuf>,
    },
    /// Creates an Evm verifier for a single proof
    #[command(name = "create-evm-verifier")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    CreateEvmVerifier {
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
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
        /// Whether to render the verifier as reusable or not. If true, you will need to deploy a VK artifact, passing it as part of the calldata to the verifier.
        #[cfg_attr(all(feature = "reusable-verifier", not(target_arch = "wasm32")), arg(short = 'R', long, default_value = DEFAULT_RENDER_REUSABLE, action = clap::ArgAction::SetTrue))]
        reusable: Option<bool>,
    },
    /// Creates an evm verifier artifact to be used by the reusable verifier
    #[command(name = "create-evm-vka")]
    #[cfg(all(
        feature = "eth",
        feature = "reusable-verifier",
        not(target_arch = "wasm32")
    ))]
    CreateEvmVka {
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings .json file from (generated using the gen-settings command)
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS, value_hint = clap::ValueHint::FilePath)]
        settings_path: Option<PathBuf>,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK, value_hint = clap::ValueHint::FilePath)]
        vk_path: Option<PathBuf>,
        /// The path to output the vka calldata
        #[arg(long, default_value = DEFAULT_VKA, value_hint = clap::ValueHint::FilePath)]
        vka_path: Option<PathBuf>,
        /// The number of decimals we want to use for the rescaling of the instances into on-chain floats
        /// Default is 18, which is the number of decimals used by most ERC20 tokens
        #[arg(long, default_value = DEFAULT_DECIMALS, value_hint = clap::ValueHint::Other)]
        decimals: Option<usize>,
    },

    /// Creates an Evm verifier for an aggregate proof
    #[command(name = "create-evm-verifier-aggr")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    CreateEvmVerifierAggr {
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
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
        /// Whether to render the verifier as reusable or not. If true, you will need to deploy a VK artifact, passing it as part of the calldata to the verifier.
        #[cfg_attr(all(feature = "reusable-verifier", not(target_arch = "wasm32")), arg(short = 'R', long, action = clap::ArgAction::SetTrue))]
        reusable: Option<bool>,
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
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// Reduce SRS logrows to the number of instances rather than the number of logrows used for proofs (only works if the srs were generated in the same ceremony)
        #[arg(long, default_value = DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION, action = clap::ArgAction::SetTrue)]
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
        #[arg(long, default_value = DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION, action = clap::ArgAction::SetTrue)]
        reduced_srs: Option<bool>,
        /// The path to SRS, if None will use ~/.ezkl/srs/kzg{logrows}.srs
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        srs_path: Option<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS, value_hint = clap::ValueHint::Other)]
        logrows: Option<u32>,
        /// commitment
        #[arg(long, default_value = DEFAULT_COMMITMENT, value_hint = clap::ValueHint::Other)]
        commitment: Option<Commitments>,
    },
    /// Deploys an evm contract (verifier, reusable verifier, or vk artifact) that is generated by ezkl
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    DeployEvm {
        /// The path to the Solidity code (generated using the create-evm-verifier command)
        #[arg(long, default_value = DEFAULT_SOL_CODE, value_hint = clap::ValueHint::FilePath)]
        sol_code_path: Option<PathBuf>,
        /// RPC URL for an Ethereum node
        #[arg(short = 'U', long,  default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Url)]
        rpc_url: String,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Other)]
        /// The path to output the contract address
        addr_path: Option<PathBuf>,
        /// The optimizer runs to set on the verifier. Lower values optimize for deployment cost, while higher values optimize for gas cost.
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS, value_hint = clap::ValueHint::Other)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
        /// Contract type to be deployed
        #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
        #[arg(long = "contract-type", short = 'C', default_value = DEFAULT_CONTRACT_DEPLOYMENT_TYPE, value_hint = clap::ValueHint::Other)]
        contract: ContractType,
    },
    /// Verifies a proof using a local Evm executor, returning accept or reject
    #[command(name = "verify-evm")]
    #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
    VerifyEvm {
        /// The path to the proof file (generated using the prove command)
        #[arg(long, default_value = DEFAULT_PROOF, value_hint = clap::ValueHint::FilePath)]
        proof_path: Option<PathBuf>,
        /// The path to verifier contract's address
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Other)]
        addr_verifier: H160Flag,
        /// RPC URL for an Ethereum node
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: String,
        /// The path to the serialized vka file
        #[cfg_attr(all(feature = "reusable-verifier", not(target_arch = "wasm32")), arg(long, value_hint = clap::ValueHint::FilePath))]
        vka_path: Option<PathBuf>,
        /// The path to the serialized encoded calldata file generated via the encode_calldata command
        #[arg(long, value_hint = clap::ValueHint::FilePath)]
        encoded_calldata: Option<PathBuf>,
    },
    /// Registers a VKA, returning the its digest used to identify it on-chain.
    #[command(name = "register-vka")]
    #[cfg(feature = "reusable-verifier")]
    RegisterVka {
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long, value_hint = clap::ValueHint::Url)]
        rpc_url: String,
        /// The path to the reusable verifier contract's address
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS, value_hint = clap::ValueHint::Other)]
        addr_verifier: H160Flag,
        /// The path to the serialized VKA file
        #[arg(long, default_value = DEFAULT_VKA, value_hint = clap::ValueHint::FilePath)]
        vka_path: Option<PathBuf>,
        /// The path to output the VKA digest to
        #[arg(long, default_value = DEFAULT_VKA_DIGEST, value_hint = clap::ValueHint::FilePath)]
        vka_digest_path: Option<PathBuf>,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long, value_hint = clap::ValueHint::Other)]
        private_key: Option<String>,
    },
    #[cfg(not(feature = "no-update"))]
    /// Updates ezkl binary to version specified (or latest if not specified)
    Update {
        /// The version to update to
        #[arg(value_hint = clap::ValueHint::Other, short='v', long)]
        version: Option<String>,
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
