//use crate::onnx::OnnxModel;
use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(not(target_arch = "wasm32"))]
use ethereum_types::Address;
use log::{debug, info};
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
use std::io::{stdin, stdout, Read, Write};
use std::path::PathBuf;

use crate::circuit::CheckMode;
use crate::graph::{VarVisibility, Visibility};

#[allow(missing_docs)]
#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum TranscriptType {
    Blake,
    Poseidon,
    EVM,
}
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

/// Parameters specific to a proving run
#[derive(Debug, Args, Deserialize, Serialize, Clone, Default)]
pub struct RunArgs {
    /// The tolerance for error on model outputs
    #[arg(short = 'T', long, default_value = "0")]
    pub tolerance: usize,
    /// The denominator in the fixed point representation used when quantizing
    #[arg(short = 'S', long, default_value = "7")]
    pub scale: u32,
    /// The number of bits used in lookup tables
    #[arg(short = 'B', long, default_value = "16")]
    pub bits: usize,
    /// The log_2 number of rows
    #[arg(short = 'K', long, default_value = "17")]
    pub logrows: u32,
    /// Flags whether inputs are public
    #[arg(long, default_value = "false", action = clap::ArgAction::Set)]
    pub public_inputs: bool,
    /// Flags whether outputs are public
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub public_outputs: bool,
    /// Flags whether params are public
    #[arg(long, default_value = "false", action = clap::ArgAction::Set)]
    pub public_params: bool,
    /// Base used to pack the public-inputs to the circuit. (value > 1) to pack instances as a single int.
    /// Useful when verifying on the EVM. Note that this will often break for very long inputs. Use with caution, still experimental.
    #[arg(long, default_value = "1")]
    pub pack_base: u32,
    /// the number of constraints the circuit might use. If not specified, this will be calculated using a 'dummy layout' pass.
    #[arg(long)]
    pub allocated_constraints: Option<usize>,
    /// run sanity checks during calculations (safe or unsafe)
    #[arg(long, default_value = "safe")]
    pub check_mode: CheckMode,
}

#[allow(missing_docs)]
impl RunArgs {
    pub fn to_var_visibility(&self) -> VarVisibility {
        VarVisibility {
            input: if self.public_inputs {
                Visibility::Public
            } else {
                Visibility::Private
            },
            params: if self.public_params {
                Visibility::Public
            } else {
                Visibility::Private
            },
            output: if self.public_outputs {
                Visibility::Public
            } else {
                Visibility::Private
            },
        }
    }
}

const EZKLCONF: &str = "EZKLCONF";
const RUNARGS: &str = "RUNARGS";

#[allow(missing_docs)]
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    #[allow(missing_docs)]
    pub command: Commands,
    /// The tolerance for error on model outputs
    #[clap(flatten)]
    pub args: RunArgs,
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
            Err(_e) => match env::var(RUNARGS) {
                Ok(path) => {
                    debug!("loading run args from {}", path);
                    let mut file = File::open(path).map_err(Box::<dyn Error>::from)?;
                    let mut data = String::new();
                    file.read_to_string(&mut data)
                        .map_err(Box::<dyn Error>::from)?;
                    let args: RunArgs =
                        serde_json::from_str(&data).map_err(Box::<dyn Error>::from)?;
                    Ok(Cli {
                        command: Cli::parse().command,
                        args,
                    })
                }
                Err(_e) => Ok(Cli::parse()),
            },
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
        model: String,
    },

    #[cfg(feature = "render")]
    /// Renders the model circuit to a .png file. For an overview of how to interpret these plots, see https://zcash.github.io/halo2/user/dev-tools.html
    #[command(arg_required_else_help = true)]
    RenderCircuit {
        /// The path to the .json data file
        #[arg(short = 'D', long)]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: String,
        /// Path to save the .png circuit render
        #[arg(short = 'O', long)]
        output: String,
    },

    /// Runs a vanilla forward pass, produces a quantized output, and saves it to a .json file
    #[command(arg_required_else_help = true)]
    Forward {
        /// The path to the .json data file
        #[arg(short = 'D', long)]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: String,
        /// Path to the new .json file
        #[arg(short = 'O', long)]
        output: String,
    },

    /// Generates a dummy SRS
    #[command(name = "gen-srs", arg_required_else_help = true)]
    GenSrs {
        /// The path to output to the desired params file (optional)
        #[arg(long)]
        params_path: PathBuf,
    },
    /// Loads model and input and runs mock prover (for testing)
    #[command(arg_required_else_help = true)]
    Mock {
        /// The path to the .json data file
        #[arg(short = 'D', long)]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: String,
    },

    /// Aggregates proofs :)
    #[command(arg_required_else_help = true)]
    Aggregate {
        /// The path to the params files.
        #[arg(long)]
        circuit_params_paths: Vec<PathBuf>,
        ///the logrows used when generating the snarks we're aggregating
        #[arg(long)]
        app_logrows: u32,
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
        // todo, optionally allow supplying proving key
    },

    /// Creates pk and vk and circuit params
    #[command(arg_required_else_help = true)]
    Setup {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: String,
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
        /// The path to save circuit params to
        #[arg(long)]
        circuit_params_path: PathBuf,
    },

    /// Loads model, data, and creates proof
    #[command(arg_required_else_help = true)]
    Prove {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: String,
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
        // todo, optionally allow supplying proving key
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
        /// The path to output to the desired EVM bytecode file (optional)
        #[arg(long)]
        deployment_code_path: Option<PathBuf>,
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
        deployment_code_path: PathBuf,
        // todo, optionally allow supplying proving key
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Deploys an EVM verifier
    #[command(name = "deploy-verifier-evm", arg_required_else_help = true)]
    DeployVerifierEVM {
        /// The path to the wallet mnemonic if not set will attempt to connect to ledger
        #[arg(short = 'S', long)]
        secret: Option<PathBuf>,
        /// RPC Url
        #[arg(short = 'U', long)]
        rpc_url: String,
        /// The path to the desired EVM bytecode file (optional), either set this or sol_code_path
        #[arg(long)]
        deployment_code_path: Option<PathBuf>,
        /// The path to output the Solidity code (optional) supercedes deployment_code_path in priority
        #[arg(long)]
        sol_code_path: Option<PathBuf>,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Send a proof to be verified to an already deployed verifier
    #[command(name = "send-proof-evm", arg_required_else_help = true)]
    SendProofEVM {
        /// The path to the wallet mnemonic if not set will attempt to connect to ledger
        #[arg(short = 'S', long)]
        secret: Option<PathBuf>,
        /// RPC Url
        #[arg(short = 'U', long)]
        rpc_url: String,
        /// The deployed verifier address
        #[arg(long)]
        addr: Address,
        /// The path to the proof
        #[arg(long)]
        proof_path: PathBuf,
        /// If we have the contract abi locally (i.e adheres to format in Verifier.json)
        #[arg(long)]
        has_abi: bool,
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
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::Blake,
            value_enum
        )]
        transcript: TranscriptType,
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
        #[arg(
             long,
             require_equals = true,
             num_args = 0..=1,
             default_value_t = TranscriptType::Blake,
             value_enum
         )]
        transcript: TranscriptType,
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
    },

    /// Print the proof in hexadecimal
    #[command(name = "print-proof-hex", arg_required_else_help = true)]
    PrintProofHex {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
    },
}

/// Loads the path to a path `data` represented as a [String]. If empty queries the user for an input.
pub fn data_path(data: String) -> PathBuf {
    let mut s = String::new();
    match data.is_empty() {
        false => {
            info!("loading data from {}", data);
            PathBuf::from(data)
        }
        true => {
            info!("please enter a path to a .json file containing inputs for the model: ");
            let _ = stdout().flush();
            let _ = &stdin()
                .read_line(&mut s)
                .expect("did not enter a correct string");
            s.truncate(s.len() - 1);
            PathBuf::from(&s)
        }
    }
}
