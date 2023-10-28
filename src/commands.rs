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

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
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
            _ => panic!("invalid calibration target"),
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

    /// Generates the witness from an input file.
    #[command(arg_required_else_help = true)]
    GenWitness {
        /// The path to the .json data file
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to the compiled model file
        #[arg(short = 'M', long)]
        compiled_circuit: PathBuf,
        /// Path to the witness (public and private inputs) .json file
        #[arg(short = 'O', long, default_value = "witness.json")]
        output: PathBuf,
        /// Path to the witness (public and private inputs) .json file (optional - solely used to generate kzg commits)
        #[arg(short = 'V', long)]
        vk_path: Option<PathBuf>,
        /// Path to the srs file (optional - solely used to generate kzg commits)
        #[arg(short = 'P', long)]
        srs_path: Option<PathBuf>,
    },

    /// Produces the proving hyperparameters, from run-args
    #[command(arg_required_else_help = true)]
    GenSettings {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// Path to circuit_settings file to output
        #[arg(short = 'O', long, default_value = "settings.json")]
        settings_path: PathBuf,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
    },

    /// Calibrates the proving scale, lookup bits and logrows from a circuit settings file.
    #[cfg(not(target_arch = "wasm32"))]
    #[command(arg_required_else_help = true)]
    CalibrateSettings {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// Path to circuit_settings file to read in AND overwrite.
        #[arg(short = 'O', long, default_value = "settings.json")]
        settings_path: PathBuf,
        /// The path to the .json calibration data file.
        #[arg(short = 'D', long = "data")]
        data: PathBuf,
        #[arg(long = "target", default_value = "resources")]
        /// Target for calibration.
        target: CalibrationTarget,
        /// Optional scales to specifically try for calibration.
        #[arg(long, value_delimiter = ',')]
        scales: Option<Vec<u32>>,
        /// max logrows to use for calibration, 26 is the max public SRS size
        #[arg(long)]
        max_logrows: Option<u32>,
    },

    /// Generates a dummy SRS
    #[command(name = "gen-srs", arg_required_else_help = true)]
    GenSrs {
        /// The path to output to the desired srs file
        #[arg(long, default_value = "kzg.srs")]
        srs_path: PathBuf,
        /// number of logrows to use for srs
        #[arg(long)]
        logrows: usize,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Gets an SRS from a circuit settings file.
    #[command(name = "get-srs", arg_required_else_help = true)]
    GetSrs {
        /// The path to output to the desired srs file
        #[arg(long, default_value = "kzg.srs")]
        srs_path: PathBuf,
        /// Path to circuit_settings file to read in. Overrides logrows if specified.
        #[arg(short = 'S', long, default_value = None)]
        settings_path: Option<PathBuf>,
        /// Number of logrows to use for srs. To manually override the logrows, omit specifying the settings_path
        #[arg(long, default_value = None)]
        logrows: Option<u32>,
        /// Check mode for srs. verifies downloaded srs is valid. set to unsafe for speed.
        #[arg(long, default_value = "safe")]
        check: CheckMode,
    },
    /// Loads model and input and runs mock prover (for testing)
    #[command(arg_required_else_help = true)]
    Mock {
        /// The path to the .json witness file
        #[arg(short = 'W', long)]
        witness: PathBuf,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
    },

    /// Mock aggregate proofs
    #[command(arg_required_else_help = true)]
    MockAggregate {
        /// The path to the snarks to aggregate over
        #[arg(long)]
        aggregation_snarks: Vec<PathBuf>,
        /// logrows used for aggregation circuit
        #[arg(long)]
        logrows: u32,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = "false")]
        split_proofs: bool,
    },

    /// setup aggregation circuit :)
    #[command(arg_required_else_help = true)]
    SetupAggregate {
        /// The path to samples of snarks that will be aggregated over
        #[arg(long)]
        sample_snarks: Vec<PathBuf>,
        /// The path to save the desired verfication key file
        #[arg(long, default_value = "vk_aggr.key")]
        vk_path: PathBuf,
        /// The path to save the desired proving key file
        #[arg(long, default_value = "pk_aggr.key")]
        pk_path: PathBuf,
        /// The path to SRS
        #[arg(long)]
        srs_path: PathBuf,
        /// logrows used for aggregation circuit
        #[arg(long)]
        logrows: u32,
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = "false")]
        split_proofs: bool,
    },
    /// Aggregates proofs :)
    #[command(arg_required_else_help = true)]
    Aggregate {
        /// The path to the snarks to aggregate over
        #[arg(long)]
        aggregation_snarks: Vec<PathBuf>,
        /// The path to load the desired proving key file
        #[arg(long)]
        pk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long, default_value = "proof_aggr.proof")]
        proof_path: PathBuf,
        /// The path to SRS
        #[arg(long)]
        srs_path: PathBuf,
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
        /// whether the accumulated are segments of a larger proof
        #[arg(long, default_value = "false")]
        split_proofs: bool,
    },
    /// Compiles a circuit from onnx to a simplified graph (einsum + other ops) and parameters as sets of field elements
    #[command(arg_required_else_help = true)]
    CompileCircuit {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// The path to output the processed model
        #[arg(long)]
        compiled_circuit: PathBuf,
        /// The path to load circuit params from
        #[arg(short = 'S', long)]
        settings_path: PathBuf,
    },
    /// Creates pk and vk
    #[command(arg_required_else_help = true)]
    Setup {
        /// The path to the compiled model file
        #[arg(short = 'M', long)]
        compiled_circuit: PathBuf,
        /// The srs path
        #[arg(long)]
        srs_path: PathBuf,
        /// The path to output the verfication key file
        #[arg(long, default_value = "vk.key")]
        vk_path: PathBuf,
        /// The path to output the proving key file
        #[arg(long, default_value = "pk.key")]
        pk_path: PathBuf,
        /// The graph witness (optional - used to override fixed values in the circuit)
        #[arg(short = 'W', long)]
        witness: Option<PathBuf>,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Fuzzes the proof pipeline with random inputs, random parameters, and random keys
    #[command(arg_required_else_help = true)]
    Fuzz {
        /// The path to the .json witness file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'W', long)]
        witness: PathBuf,
        /// The path to the processed model file
        #[arg(short = 'M', long)]
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
        #[arg(long, default_value = "10")]
        num_runs: usize,
    },
    #[cfg(not(target_arch = "wasm32"))]
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
    #[command(arg_required_else_help = true)]
    SwapProofCommitments {
        /// The path to the proof file
        #[arg(short = 'P', long)]
        proof_path: PathBuf,
        /// The path to the witness file
        #[arg(short = 'W', long)]
        witness_path: PathBuf,
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Loads model, data, and creates proof
    #[command(arg_required_else_help = true)]
    Prove {
        /// The path to the .json witness file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'W', long)]
        witness: PathBuf,
        /// The path to the compiled model file
        #[arg(short = 'M', long)]
        compiled_circuit: PathBuf,
        /// The path to load the desired proving key file
        #[arg(long)]
        pk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long, default_value = "proof.proof")]
        proof_path: PathBuf,
        /// The parameter path
        #[arg(long)]
        srs_path: PathBuf,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofType::Single,
            value_enum
        )]
        proof_type: ProofType,
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
        srs_path: PathBuf,
        /// The path to load circuit settings from
        #[arg(short = 'S', long)]
        settings_path: PathBuf,
        /// The path to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to output the Solidity code
        #[arg(long, default_value = "evm_deploy.sol")]
        sol_code_path: PathBuf,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = "verifier_abi.json")]
        abi_path: PathBuf,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier that attests to on-chain inputs for a single proof
    #[command(name = "create-evm-da", arg_required_else_help = true)]
    CreateEVMDataAttestation {
        /// The path to load the desired srs file from
        #[arg(long)]
        srs_path: PathBuf,
        /// The path to load circuit settings from
        #[arg(short = 'S', long)]
        settings_path: PathBuf,
        /// The path to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to output the Solidity code
        #[arg(long, default_value = "evm_da_deploy.sol")]
        sol_code_path: PathBuf,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = "verifier_da_abi.json")]
        abi_path: PathBuf,
        /// The path to the .json data file, which should
        /// contain the necessary calldata and accoount addresses
        /// needed need to read from all the on-chain
        /// view functions that return the data that the network
        /// ingests as inputs.
        #[arg(short = 'D', long)]
        data: PathBuf,
        // todo, optionally allow supplying proving key
    },

    #[cfg(not(target_arch = "wasm32"))]
    /// Creates an EVM verifier for an aggregate proof
    #[command(name = "create-evm-verifier-aggr", arg_required_else_help = true)]
    CreateEVMVerifierAggr {
        /// The path to load the desired srs file from
        #[arg(long)]
        srs_path: PathBuf,
        /// The path to output to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to the Solidity code
        #[arg(long, default_value = "evm_deploy_aggr.sol")]
        sol_code_path: PathBuf,
        /// The path to output the Solidity verifier ABI
        #[arg(long, default_value = "verifier_aggr_abi.json")]
        abi_path: PathBuf,
        // aggregated circuit settings paths, used to calculate the number of instances in the aggregate proof
        #[arg(long)]
        aggregation_settings: Vec<PathBuf>,
    },
    /// Verifies a proof, returning accept or reject
    #[command(arg_required_else_help = true)]
    Verify {
        /// The path to load circuit params from
        #[arg(short = 'S', long)]
        settings_path: PathBuf,
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to output the desired verfication key file (optional)
        #[arg(long)]
        vk_path: PathBuf,
        /// The kzg srs path
        #[arg(long)]
        srs_path: PathBuf,
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
        /// The srs path
        #[arg(long)]
        srs_path: PathBuf,
        /// logrows used for aggregation circuit
        #[arg(long)]
        logrows: u32,
    },
    #[cfg(not(target_arch = "wasm32"))]
    DeployEvmVerifier {
        /// The path to the Solidity code
        #[arg(long)]
        sol_code_path: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        #[arg(long, default_value = "contract.address")]
        /// The path to output the contract address
        addr_path: PathBuf,
        /// The optimizer runs to set on the verifier. (Lower values optimize for deployment, while higher values optimize for execution)
        #[arg(long, default_value = "1")]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    #[command(name = "deploy-evm-da", arg_required_else_help = true)]
    DeployEvmDataAttestation {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// The path to load circuit params from
        #[arg(long)]
        settings_path: PathBuf,
        /// The path to the Solidity code
        #[arg(long)]
        sol_code_path: PathBuf,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        #[arg(long, default_value = "contract_da.address")]
        /// The path to output the contract address
        addr_path: PathBuf,
        /// The optimizer runs to set on the verifier. (Lower values optimize for deployment, while higher values optimize for execution)
        #[arg(long, default_value = "1")]
        optimizer_runs: usize,
        /// Private secp256K1 key in hex format, 64 chars, no 0x prefix, of the account signing transactions. If None the private key will be generated by Anvil
        #[arg(short = 'P', long)]
        private_key: Option<String>,
    },
    #[cfg(not(target_arch = "wasm32"))]
    /// Verifies a proof using a local EVM executor, returning accept or reject
    #[command(name = "verify-evm", arg_required_else_help = true)]
    VerifyEVM {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to verfier contract's address
        #[arg(long)]
        addr_verifier: H160,
        /// RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
        #[arg(short = 'U', long)]
        rpc_url: Option<String>,
        /// does the verifier use data attestation ?
        #[arg(long)]
        addr_da: Option<H160>,
    },

    /// Print the proof in hexadecimal
    #[command(name = "print-proof-hex", arg_required_else_help = true)]
    PrintProofHex {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,
    },

    /// Gets credentials from the hub
    #[command(name = "get-hub-credentials", arg_required_else_help = true)]
    #[cfg(not(target_arch = "wasm32"))]
    GetHubCredentials {
        /// The path to the model file
        #[arg(short = 'N', long)]
        username: String,
        /// The path to the input json file
        #[arg(short = 'U', long)]
        url: Option<String>,
    },

    /// Create artifacts and deploys them on the hub
    #[command(name = "create-hub-artifact", arg_required_else_help = true)]
    #[cfg(not(target_arch = "wasm32"))]
    CreateHubArtifact {
        /// The path to the model file
        #[arg(short = 'M', long)]
        uncompiled_circuit: PathBuf,
        /// The path to the input json file
        #[arg(short = 'D', long)]
        data: PathBuf,
        /// the hub's url
        #[arg(short = 'O', long)]
        organization_id: String,
        ///artifact name
        #[arg(short = 'A', long)]
        artifact_name: String,
        /// the hub's url
        #[arg(short = 'U', long)]
        url: Option<String>,
        /// proving arguments
        #[clap(flatten)]
        args: RunArgs,
        /// calibration target
        #[arg(long, default_value = "resources")]
        target: CalibrationTarget,
    },

    /// Create artifacts and deploys them on the hub
    #[command(name = "prove-hub", arg_required_else_help = true)]
    #[cfg(not(target_arch = "wasm32"))]
    ProveHub {
        /// The path to the model file
        #[arg(short = 'A', long)]
        artifact_id: String,
        /// The path to the input json file
        #[arg(short = 'D', long)]
        data: PathBuf,
        #[arg(short = 'U', long)]
        url: Option<String>,
        #[arg(short = 'T', long)]
        transcript_type: Option<String>,
    },

    /// Create artifacts and deploys them on the hub
    #[command(name = "get-hub-proof", arg_required_else_help = true)]
    #[cfg(not(target_arch = "wasm32"))]
    GetHubProof {
        /// The path to the model file
        #[arg(short = 'A', long)]
        artifact_id: String,
        #[arg(short = 'U', long)]
        url: Option<String>,
    },
}
