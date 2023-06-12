//use crate::onnx::OnnxModel;
use clap::{Args, Parser, Subcommand, ValueEnum};
use log::debug;
#[cfg(feature = "python-bindings")]
use pyo3::{
    conversion::{FromPyObject, PyTryFrom},
    exceptions::PyValueError,
    prelude::*,
    types::PyString,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use crate::circuit::{CheckMode, Tolerance};
use crate::graph::Visibility;
use crate::pfsys::TranscriptType;

impl std::fmt::Display for TranscriptType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}
#[cfg(feature = "python-bindings")]
/// Converts TranscriptType into a PyObject (Required for TranscriptType to be compatible with Python)
impl IntoPy<PyObject> for TranscriptType {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            TranscriptType::Blake => "blake".to_object(py),
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
            "blake" => Ok(TranscriptType::Blake),
            "poseidon" => Ok(TranscriptType::Poseidon),
            "evm" => Ok(TranscriptType::EVM),
            _ => Err(PyValueError::new_err("Invalid value for TranscriptType")),
        }
    }
}

#[allow(missing_docs)]
#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum StrategyType {
    Single,
    Accum,
}
impl std::fmt::Display for StrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}
#[cfg(feature = "python-bindings")]
/// Converts StrategyType into a PyObject (Required for StrategyType to be compatible with Python)
impl IntoPy<PyObject> for StrategyType {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            StrategyType::Single => "single".to_object(py),
            StrategyType::Accum => "accum".to_object(py),
        }
    }
}
#[cfg(feature = "python-bindings")]
/// Obtains StrategyType from PyObject (Required for StrategyType to be compatible with Python)
impl<'source> FromPyObject<'source> for StrategyType {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let trystr = <PyString as PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();
        match strval.to_lowercase().as_str() {
            "single" => Ok(StrategyType::Single),
            "accum" => Ok(StrategyType::Accum),
            _ => Err(PyValueError::new_err("Invalid value for StrategyType")),
        }
    }
}

/// Calibration specific parameters
#[derive(Debug, Args, Deserialize, Serialize, Clone, Default)]
pub struct CalibrationArgs {
    /// The path to the .json calibration data file. If set will finetune the selected parameters to a calibration dataset.
    #[arg(long = "calibration-data")]
    pub data: Option<PathBuf>,
    #[arg(long = "calibration-target")]
    /// Target for calibration. "Resources" will calibrate for resource usage, "Accuracy" will calibrate for numerical accuracy.
    pub target: Option<CalibrationTarget>,
}

#[derive(clap::ValueEnum, Debug, Default, Copy, Clone, Serialize, Deserialize)]
/// Determines what the calibration pass should optimize for
pub enum CalibrationTarget {
    /// Optimizes for reducing cpu and memory usage
    #[default]
    Resources,
    /// Optimizes for numerical accuracy given the fixed point representation
    Accuracy,
}

impl From<&str> for CalibrationTarget {
    fn from(s: &str) -> Self {
        match s {
            "resources" => CalibrationTarget::Resources,
            "accuracy" => CalibrationTarget::Accuracy,
            _ => panic!("invalid calibration target"),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Converts CalibrationTarget into a PyObject (Required for CalibrationTarget to be compatible with Python)
impl IntoPy<PyObject> for CalibrationTarget {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            CalibrationTarget::Resources => "resources".to_object(py),
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
            "resources" => Ok(CalibrationTarget::Resources),
            "accuracy" => Ok(CalibrationTarget::Accuracy),
            _ => Err(PyValueError::new_err("Invalid value for CalibrationTarget")),
        }
    }
}

/// Parameters specific to a proving run
#[derive(Debug, Copy, Args, Deserialize, Serialize, Clone, Default)]
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
    /// Base used to pack the public-inputs to the circuit. (value > 1) to pack instances as a single int.
    /// Useful when verifying on the EVM. Note that this will often break for very long inputs. Use with caution, still experimental.
    #[arg(long, default_value = "1")]
    pub pack_base: u32,
    /// the number of constraints the circuit might use. If not specified, this will be calculated using a 'dummy layout' pass.
    #[arg(long)]
    pub allocated_constraints: Option<usize>,
}

#[allow(missing_docs)]
impl RunArgs {
    /// Creates `RunArgs` from parsed CLI arguments
    /// # Arguments
    /// * `cli` - A [Cli] struct holding parsed CLI arguments.
    pub fn from_cli(cli: Cli) -> Result<Self, Box<dyn Error>> {
        match cli.command {
            Commands::Mock { args, .. } | Commands::GenCircuitParams { args, .. } => Ok(args),
            #[cfg(not(target_arch = "wasm32"))]
            Commands::Fuzz { args, .. } => Ok(args),
            #[cfg(feature = "render")]
            Commands::RenderCircuit { args, .. } => Ok(args),
            _ => panic!(),
        }
    }
}

const EZKLCONF: &str = "EZKLCONF";

#[allow(missing_docs)]
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    #[allow(missing_docs)]
    pub command: Commands,
}

impl Cli {
    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, Box<dyn Error>> {
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
    /// Create an ezkl configuration: if there is an EZKLCONF env variable, parse its value, else read it from the command line.
    pub fn create() -> Result<Self, Box<dyn Error>> {
        match env::var(EZKLCONF) {
            Ok(path) => {
                debug!("loading ezkl conf from {}", path);
                let mut file = File::open(path).map_err(Box::<dyn Error>::from)?;
                let mut data = String::new();
                file.read_to_string(&mut data)
                    .map_err(Box::<dyn Error>::from)?;
                Self::from_json(&data).map_err(Box::<dyn Error>::from)
            }
            Err(_e) => Ok(Cli::parse()),
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Subcommand, Clone, Deserialize, Serialize)]
pub enum Commands {
    /// Loads model and prints model table
    #[command(arg_required_else_help = true)]
    Table {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    #[cfg(feature = "render")]
    /// Renders the model circuit to a .png file. For an overview of how to interpret these plots, see https://zcash.github.io/halo2/user/dev-tools.html
    #[command(arg_required_else_help = true)]
    RenderCircuit {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// Path to save the .png circuit render
        #[arg(short = 'O', long)]
        output: PathBuf,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    /// Runs a vanilla forward pass, produces a quantized output, and saves it to a .json file
    #[command(arg_required_else_help = true)]
    Forward {
        /// The path to the .json data file
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// Path to the new .json file
        #[arg(short = 'O', long)]
        output: PathBuf,
        /// Scale to use for quantization
        #[arg(
            short = 'S',
            long,
            default_value = "7",
            conflicts_with = "circuit_params_path"
        )]
        scale: Option<u32>,
        /// The number of batches to split the input data into
        #[arg(
            short = 'B',
            long,
            default_value = "1",
            conflicts_with = "circuit_params_path"
        )]
        batch_size: Option<usize>,
        /// optional circuit params path
        #[arg(long)]
        circuit_params_path: Option<PathBuf>,
    },

    /// Calibrates the proving hyperparameters, produces a quantized output from those hyperparameters, and saves it to a .json file. The circuit parameters are also saved to a file.
    #[command(arg_required_else_help = true)]
    GenCircuitParams {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// Path to circuit_params file to output
        #[arg(short = 'O', long)]
        circuit_params_path: PathBuf,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
        /// calibration args
        #[clap(flatten)]
        calibration: CalibrationArgs,
    },

    /// Generates a dummy SRS
    #[command(name = "gen-srs", arg_required_else_help = true)]
    GenSrs {
        /// The path to output to the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// number of logrows to use for srs
        #[arg(long)]
        logrows: usize,
    },
    /// Loads model and input and runs mock prover (for testing)
    #[command(arg_required_else_help = true)]
    Mock {
        /// The path to the .json data file
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
        /// optional circuit params path (overrides any run args set)
        #[arg(long)]
        circuit_params_path: Option<PathBuf>,
    },

    /// Aggregates proofs :)
    #[command(arg_required_else_help = true)]
    Aggregate {
        /// The path to the params files.
        #[arg(long)]
        circuit_params_paths: Vec<PathBuf>,
        /// The path to the snarks to aggregate over
        #[arg(long)]
        aggregation_snarks: Vec<PathBuf>,
        /// The path to load the desired verfication key file for the snarks we're aggregating over
        #[arg(long)]
        aggregation_vk_paths: Vec<PathBuf>,
        /// The path to save the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long)]
        proof_path: PathBuf,
        /// The transcript type
        #[arg(long)]
        params_path: PathBuf,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::EVM,
            value_enum
        )]
        transcript: TranscriptType,
        /// logrows used for aggregation circuit
        #[arg(long)]
        logrows: u32,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = "safe")]
        check_mode: CheckMode,
    },

    /// Creates pk and vk and circuit params
    #[command(arg_required_else_help = true)]
    Setup {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// The parameter path
        #[arg(long)]
        params_path: PathBuf,
        /// The path to output the verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to output the proving key file
        #[arg(long)]
        pk_path: PathBuf,
        /// The path to load circuit params from
        #[arg(long)]
        circuit_params_path: PathBuf,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Fuzzes the proof pipeline with random inputs, random parameters, and random keys
    #[command(arg_required_else_help = true)]
    Fuzz {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::Blake,
            value_enum
        )]
        transcript: TranscriptType,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
        /// number of fuzz iterations
        #[arg(long)]
        num_runs: usize,
        /// optional circuit params path (overrides any run args set)
        #[arg(long)]
        circuit_params_path: Option<PathBuf>,
    },

    /// Loads model, data, and creates proof
    #[command(arg_required_else_help = true)]
    Prove {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// The path to load the desired proving key file
        #[arg(long)]
        pk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long)]
        proof_path: PathBuf,
        /// The parameter path
        #[arg(long)]
        params_path: PathBuf,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::Blake,
            value_enum
        )]
        transcript: TranscriptType,
        /// The proving strategy
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = StrategyType::Single,
            value_enum
        )]
        strategy: StrategyType,
        /// The path to load circuit params from
        #[arg(long)]
        circuit_params_path: PathBuf,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = "safe")]
        check_mode: CheckMode,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier for a single proof
    #[command(name = "create-evm-verifier", arg_required_else_help = true)]
    CreateEVMVerifier {
        /// The path to load the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// The path to save circuit params to
        #[arg(long)]
        circuit_params_path: PathBuf,
        /// The path to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to output to the desired EVM bytecode file
        #[arg(long)]
        deployment_code_path: PathBuf,
        /// The path to output the Solidity code
        #[arg(long)]
        sol_code_path: Option<PathBuf>,
        // todo, optionally allow supplying proving key
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier for an aggregate proof
    #[command(name = "create-evm-verifier-aggr", arg_required_else_help = true)]
    CreateEVMVerifierAggr {
        /// The path to load the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// The path to output to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to the deployment code
        #[arg(long)]
        deployment_code_path: Option<PathBuf>,
        /// The path to the Solidity code
        #[arg(long)]
        sol_code_path: Option<PathBuf>,
        // todo, optionally allow supplying proving key
    },

    /// Verifies a proof, returning accept or reject
    #[command(arg_required_else_help = true)]
    Verify {
        /// The path to save circuit params to
        #[arg(long)]
        circuit_params_path: PathBuf,
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to output the desired verfication key file (optional)
        #[arg(long)]
        vk_path: PathBuf,
        /// The transcript type
        #[arg(long)]
        params_path: PathBuf,
    },

    /// Verifies an aggregate proof, returning accept or reject
    #[command(arg_required_else_help = true)]
    VerifyAggr {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to output the desired verfication key file (optional)
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to load the desired verfication key file (optional)
        #[arg(long)]
        params_path: PathBuf,
        /// logrows used for aggregation circuit
        #[arg(long)]
        logrows: u32,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Verifies a proof using a local EVM executor, returning accept or reject
    #[command(name = "verify-evm", arg_required_else_help = true)]
    VerifyEVM {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to verifier contract's deployment code
        #[arg(long)]
        deployment_code_path: PathBuf,
        /// The path to the Solidity code
        #[arg(long)]
        sol_code_path: Option<PathBuf>,
        /// The number of runs set to the SOLC optimizer.
        /// Lower values optimze for deployment size while higher values optimize for execution cost.
        /// If not set will just use the default unoptimized SOLC configuration.
        #[arg(long)]
        optimizer_runs: Option<usize>,
    },

    /// Print the proof in hexadecimal
    #[command(name = "print-proof-hex", arg_required_else_help = true)]
    PrintProofHex {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
    },
}
