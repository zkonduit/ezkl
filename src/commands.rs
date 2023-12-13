use clap::{Parser, Subcommand, ValueEnum};
#[cfg(not(target_arch = "wasm32"))]
use ethers::types::H160;
#[cfg(feature = "python-bindings")]
use pyo3::{
    conversion::{FromPyObject, PyTryFrom},
    exceptions::PyValueError,
    prelude::*,
    types::PyString,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;

use crate::{pfsys::ProofType, RunArgs};

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
/// Default solidity code for aggregated proofs
pub const DEFAULT_SOL_CODE_AGGREGATED: &str = "evm_deploy_aggr.sol";
/// Default solidity code for data attestation
pub const DEFAULT_SOL_CODE_DA: &str = "evm_deploy_da.sol";
/// Default contract address
pub const DEFAULT_CONTRACT_ADDRESS: &str = "contract.address";
/// Default contract address for data attestation
pub const DEFAULT_CONTRACT_ADDRESS_DA: &str = "contract_da.address";
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

impl ToString for CalibrationTarget {
    fn to_string(&self) -> String {
        match self {
            CalibrationTarget::Resources { col_overflow: true } => {
                "resources/col-overflow".to_string()
            }
            CalibrationTarget::Resources {
                col_overflow: false,
            } => "resources".to_string(),
            CalibrationTarget::Accuracy => "accuracy".to_string(),
        }
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

#[allow(missing_docs)]
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(author, about, long_about = None)]
#[clap(version = *VERSION)]
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
}

#[allow(missing_docs)]
#[derive(Debug, Subcommand, Clone, Deserialize, Serialize, PartialEq, PartialOrd)]
pub enum Commands {
    Empty,
    /// Loads model and prints model table
    Table {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL)]
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

    /// Generates the witness from an input file.
    GenWitness {
        /// The path to the .json data file
        #[arg(short = 'D', long, default_value = DEFAULT_DATA)]
        data: PathBuf,
        /// The path to the compiled model file
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT)]
        compiled_circuit: PathBuf,
        /// Path to the witness (public and private inputs) .json file
        #[arg(short = 'O', long, default_value = DEFAULT_WITNESS)]
        output: PathBuf,
        /// Path to the witness (public and private inputs) .json file (optional - solely used to generate kzg commits)
        #[arg(short = 'V', long)]
        vk_path: Option<PathBuf>,
        /// Path to the srs file (optional - solely used to generate kzg commits)
        #[arg(short = 'P', long)]
        srs_path: Option<PathBuf>,
    },

    /// Produces the proving hyperparameters, from run-args
    GenSettings {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL)]
        model: PathBuf,
        /// Path to circuit_settings file to output
        #[arg(short = 'O', long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    /// Calibrates the proving scale, lookup bits and logrows from a circuit settings file.
    #[cfg(not(target_arch = "wasm32"))]
    CalibrateSettings {
        /// The path to the .json calibration data file.
        #[arg(short = 'D', long, default_value = DEFAULT_CALIBRATION_FILE)]
        data: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL)]
        model: PathBuf,
        /// Path to circuit_settings file to read in AND overwrite.
        #[arg(short = 'O', long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
        #[arg(long = "target", default_value = DEFAULT_CALIBRATION_TARGET)]
        /// Target for calibration.
        target: CalibrationTarget,
        /// Optional scales to specifically try for calibration.
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        scales: Option<Vec<crate::Scale>>,
        /// max logrows to use for calibration, 26 is the max public SRS size
        #[arg(long)]
        max_logrows: Option<u32>,
    },

    /// Generates a dummy SRS
    #[command(name = "gen-srs", arg_required_else_help = true)]
    GenSrs {
        /// The path to output to the desired srs file
        #[arg(long)]
        srs_path: PathBuf,
        /// number of logrows to use for srs
        #[arg(long)]
        logrows: usize,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Gets an SRS from a circuit settings file.
    #[command(name = "get-srs")]
    GetSrs {
        /// The path to output to the desired srs file
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// Path to circuit_settings file to read in. Overriden by logrows if specified.
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS)]
        settings_path: Option<PathBuf>,
        /// Number of logrows to use for srs. Overrides settings_path if specified.
        #[arg(long, default_value = None)]
        logrows: Option<u32>,
        /// Check mode for srs. verifies downloaded srs is valid. set to unsafe for speed.
        #[arg(long, default_value = DEFAULT_CHECKMODE)]
        check: CheckMode,
    },
    /// Loads model and input and runs mock prover (for testing)
    Mock {
        /// The path to the .json witness file
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS)]
        witness: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL)]
        model: PathBuf,
    },

    /// Mock aggregate proofs
    MockAggregate {
        /// The path to the snarks to aggregate over
        #[arg(long, default_value = DEFAULT_PROOF, value_delimiter = ',', allow_hyphen_values = true)]
        aggregation_snarks: Vec<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS)]
        logrows: u32,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = DEFAULT_SPLIT)]
        split_proofs: bool,
    },

    /// setup aggregation circuit :)
    SetupAggregate {
        /// The path to samples of snarks that will be aggregated over
        #[arg(long, default_value = DEFAULT_PROOF, value_delimiter = ',', allow_hyphen_values = true)]
        sample_snarks: Vec<PathBuf>,
        /// The path to save the desired verification key file
        #[arg(long, default_value = DEFAULT_VK_AGGREGATED)]
        vk_path: PathBuf,
        /// The path to save the desired proving key file
        #[arg(long, default_value = DEFAULT_PK_AGGREGATED)]
        pk_path: PathBuf,
        /// The path to SRS
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS)]
        logrows: u32,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = DEFAULT_SPLIT)]
        split_proofs: bool,
    },
    /// Aggregates proofs :)
    Aggregate {
        /// The path to the snarks to aggregate over
        #[arg(long, default_value = DEFAULT_PROOF, value_delimiter = ',', allow_hyphen_values = true)]
        aggregation_snarks: Vec<PathBuf>,
        /// The path to load the desired proving key file
        #[arg(long, default_value = DEFAULT_PK_AGGREGATED)]
        pk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long, default_value = DEFAULT_PROOF_AGGREGATED)]
        proof_path: PathBuf,
        /// The path to SRS
        #[arg(long)]
        srs_path: Option<PathBuf>,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::EVM,
            value_enum
        )]
        transcript: TranscriptType,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS)]
        logrows: u32,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = DEFAULT_CHECKMODE)]
        check_mode: CheckMode,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = DEFAULT_SPLIT)]
        split_proofs: bool,
    },
    /// Compiles a circuit from onnx to a simplified graph (einsum + other ops) and parameters as sets of field elements
    CompileCircuit {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = DEFAULT_MODEL)]
        model: PathBuf,
        /// The path to output the processed model
        #[arg(long, default_value = DEFAULT_COMPILED_CIRCUIT)]
        compiled_circuit: PathBuf,
        /// The path to load circuit params from
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
    },
    /// Creates pk and vk
    Setup {
        /// The path to the compiled model file
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT)]
        compiled_circuit: PathBuf,
        /// The srs path
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// The path to output the verification key file
        #[arg(long, default_value = DEFAULT_VK)]
        vk_path: PathBuf,
        /// The path to output the proving key file
        #[arg(long, default_value = DEFAULT_PK)]
        pk_path: PathBuf,
        /// The graph witness (optional - used to override fixed values in the circuit)
        #[arg(short = 'W', long)]
        witness: Option<PathBuf>,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Fuzzes the proof pipeline with random inputs, random parameters, and random keys
    Fuzz {
        /// The path to the .json witness file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS)]
        witness: PathBuf,
        /// The path to the processed model file
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT)]
        compiled_circuit: PathBuf,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::EVM,
            value_enum
        )]
        transcript: TranscriptType,
        /// number of fuzz iterations
        #[arg(long, default_value = DEFAULT_FUZZ_RUNS)]
        num_runs: usize,
    },
    #[cfg(not(target_arch = "wasm32"))]
    #[command(arg_required_else_help = true)]
    SetupTestEVMData {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to the compiled model file
        #[arg(short = 'M', long)]
        compiled_circuit: PathBuf,
        /// For testing purposes only. The optional path to the .json data file that will be generated that contains the OnChain data storage information
        /// derived from the file information in the data .json file.
        /// Should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'T', long)]
        test_data: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        /// where does the input data come from
        #[arg(long, default_value = "on-chain")]
        input_source: TestDataSource,
        /// where does the output data come from
        #[arg(long, default_value = "on-chain")]
        output_source: TestDataSource,
    },
    #[cfg(not(target_arch = "wasm32"))]
    #[command(arg_required_else_help = true)]
    TestUpdateAccountCalls {
        /// The path to verfier contract's address
        #[arg(long)]
        addr: H160,
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Swaps the positions in the transcript that correspond to commitments
    SwapProofCommitments {
        /// The path to the proof file
        #[arg(short = 'P', long, default_value = DEFAULT_PROOF)]
        proof_path: PathBuf,
        /// The path to the witness file
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS)]
        witness_path: PathBuf,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Loads model, data, and creates proof
    Prove {
        /// The path to the .json witness file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'W', long, default_value = DEFAULT_WITNESS)]
        witness: PathBuf,
        /// The path to the compiled model file
        #[arg(short = 'M', long, default_value = DEFAULT_COMPILED_CIRCUIT)]
        compiled_circuit: PathBuf,
        /// The path to load the desired proving key file
        #[arg(long, default_value = DEFAULT_PK)]
        pk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long, default_value = DEFAULT_PROOF)]
        proof_path: PathBuf,
        /// The parameter path
        #[arg(long)]
        srs_path: Option<PathBuf>,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofType::Single,
            value_enum
        )]
        proof_type: ProofType,
        /// run sanity checks during calculations (safe or unsafe)
        #[arg(long, default_value = DEFAULT_CHECKMODE)]
        check_mode: CheckMode,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier for a single proof
    #[command(name = "create-evm-verifier")]
    CreateEVMVerifier {
        /// The path to load the desired params file
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings from
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK)]
        vk_path: PathBuf,
        /// The path to output the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE)]
        sol_code_path: PathBuf,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_ABI)]
        abi_path: PathBuf,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier that attests to on-chain inputs for a single proof
    #[command(name = "create-evm-da")]
    CreateEVMDataAttestation {
        /// The path to load the desired srs file from
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// The path to load circuit settings from
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
        /// The path to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK)]
        vk_path: PathBuf,
        /// The path to output the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE_DA)]
        sol_code_path: PathBuf,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_DA_ABI)]
        abi_path: PathBuf,
        /// The path to the .json data file, which should
        /// contain the necessary calldata and accoount addresses
        /// needed need to read from all the on-chain
        /// view functions that return the data that the network
        /// ingests as inputs.
        #[arg(short = 'D', long, default_value = DEFAULT_DATA)]
        data: PathBuf,
        // todo, optionally allow supplying proving key
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier for an aggregate proof
    #[command(name = "create-evm-verifier-aggr")]
    CreateEVMVerifierAggr {
        /// The path to load the desired srs file from
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// The path to  to load the desired verification key file
        #[arg(long, default_value = DEFAULT_VK_AGGREGATED)]
        vk_path: PathBuf,
        /// The path to the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE_AGGREGATED)]
        sol_code_path: PathBuf,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = DEFAULT_VERIFIER_AGGREGATED_ABI)]
        abi_path: PathBuf,
        // aggregated circuit settings paths, used to calculate the number of instances in the aggregate proof
        #[arg(long, default_value = DEFAULT_SETTINGS, value_delimiter = ',', allow_hyphen_values = true)]
        aggregation_settings: Vec<PathBuf>,
        // logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS)]
        logrows: u32,
    },
    /// Verifies a proof, returning accept or reject
    Verify {
        /// The path to load circuit params from
        #[arg(short = 'S', long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
        /// The path to the proof file
        #[arg(long, default_value = DEFAULT_PROOF)]
        proof_path: PathBuf,
        /// The path to output the desired verification key file (optional)
        #[arg(long, default_value = DEFAULT_VK)]
        vk_path: PathBuf,
        /// The kzg srs path
        #[arg(long)]
        srs_path: Option<PathBuf>,
    },
    /// Verifies an aggregate proof, returning accept or reject
    VerifyAggr {
        /// The path to the proof file
        #[arg(long, default_value = DEFAULT_PROOF_AGGREGATED)]
        proof_path: PathBuf,
        /// The path to output the desired verification key file (optional)
        #[arg(long, default_value = DEFAULT_VK_AGGREGATED)]
        vk_path: PathBuf,
        /// The srs path
        #[arg(long)]
        srs_path: Option<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long, default_value = DEFAULT_AGGREGATED_LOGROWS)]
        logrows: u32,
    },
    #[cfg(not(target_arch = "wasm32"))]
    DeployEvmVerifier {
        /// The path to the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE)]
        sol_code_path: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS)]
        /// The path to output the contract address
        addr_path: PathBuf,
        /// The optimizer runs to set on the verifier. (Lower values optimize for deployment, while higher values optimize for execution)
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    #[command(name = "deploy-evm-da")]
    DeployEvmDataAttestation {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long, default_value = DEFAULT_DATA)]
        data: PathBuf,
        /// The path to load circuit params from
        #[arg(long, default_value = DEFAULT_SETTINGS)]
        settings_path: PathBuf,
        /// The path to the Solidity code
        #[arg(long, default_value = DEFAULT_SOL_CODE_DA)]
        sol_code_path: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS_DA)]
        /// The path to output the contract address
        addr_path: PathBuf,
        /// The optimizer runs to set on the verifier. (Lower values optimize for deployment, while higher values optimize for execution)
        #[arg(long, default_value = DEFAULT_OPTIMIZER_RUNS)]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Verifies a proof using a local EVM executor, returning accept or reject
    #[command(name = "verify-evm")]
    VerifyEVM {
        /// The path to the proof file
        #[arg(long, default_value = DEFAULT_PROOF)]
        proof_path: PathBuf,
        /// The path to verfier contract's address
        #[arg(long, default_value = DEFAULT_CONTRACT_ADDRESS)]
        addr_verifier: H160,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        /// does the verifier use data attestation ?
        #[arg(long)]
        addr_da: Option<H160>,
    },

    /// Print the proof in hexadecimal
    #[command(name = "print-proof-hex")]
    PrintProofHex {
        /// The path to the proof file
        #[arg(long, default_value = DEFAULT_PROOF)]
        proof_path: PathBuf,
    },
}
