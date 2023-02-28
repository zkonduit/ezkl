//use crate::onnx::OnnxModel;
use clap::{Args, Parser, Subcommand, ValueEnum};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{stdin, stdout, Read, Write};
use std::path::PathBuf;

#[allow(missing_docs)]
#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum ProofSystem {
    IPA,
    KZG,
}
impl std::fmt::Display for ProofSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}

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

/// Parameters specific to a proving run
#[derive(Debug, Args, Deserialize, Serialize, Clone)]
pub struct RunArgs {
    /// The tolerance for error on model outputs
    #[arg(short = 'T', long, default_value = "0")]
    pub tolerance: usize,
    /// The denominator in the fixed point representation used when quantizing
    #[arg(short = 'S', long, default_value = "7")]
    pub scale: i32,
    /// The number of bits used in lookup tables
    #[arg(short = 'B', long, default_value = "16")]
    pub bits: usize,
    /// The log_2 number of rows
    #[arg(short = 'K', long, default_value = "17")]
    pub logrows: u32,
    /// Flags whether inputs are public
    #[arg(long, default_value = "false")]
    pub public_inputs: bool,
    /// Flags whether outputs are public
    #[arg(long, default_value = "true")]
    pub public_outputs: bool,
    /// Flags whether params are public
    #[arg(long, default_value = "false")]
    pub public_params: bool,
    /// Flags to set maximum rotations
    #[arg(short = 'M', long, default_value = "512")]
    pub max_rotations: usize,
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
        /// The [ProofSystem] we'll be using.
        #[arg(
            long,
	    short = 'B',
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::KZG,
            value_enum
        )]
        pfsys: ProofSystem,
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
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// The path to the desired output file
        #[arg(long)]
        aggregation_snarks: Vec<PathBuf>,
        /// The path to load the desired verfication key file
        #[arg(long)]
        aggregation_vk_paths: Vec<PathBuf>,
        /// The path to save the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to load the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// The [ProofSystem] we'll be using.
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::KZG,
            value_enum
        )]
        pfsys: ProofSystem,
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

    /// Loads model and data, prepares vk and pk, and creates proof
    #[command(arg_required_else_help = true)]
    Prove {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// The path to output to the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to the desired output file
        #[arg(long)]
        proof_path: PathBuf,
        /// The path to load the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// The [ProofSystem] we'll be using.
        #[arg(
            long,
	    short = 'B',
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::KZG,
            value_enum
        )]
        pfsys: ProofSystem,
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = TranscriptType::Blake,
            value_enum
        )]
        transcript: TranscriptType,
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

    /// Creates an EVM verifier for a single proof
    #[command(name = "create-evm-verifier", arg_required_else_help = true)]
    CreateEVMVerifier {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long)]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
        /// The path to load the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// The path to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to output to the desired EVM bytecode file (optional)
        #[arg(long, required_if_eq("transcript", "evm"))]
        deployment_code_path: Option<PathBuf>,
        /// The path to output the Solidity code
        #[arg(long, required_if_eq("transcript", "evm"))]
        sol_code_path: Option<PathBuf>,
        /// The [ProofSystem] we'll be using.
        #[arg(
            long,
	    short = 'B',
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::KZG,
            value_enum
        )]
        pfsys: ProofSystem,
        // todo, optionally allow supplying proving key
    },

    /// Creates an EVM verifier for an aggregate proof
    #[command(name = "create-evm-verifier-aggr", arg_required_else_help = true)]
    CreateEVMVerifierAggr {
        /// The path to load the desired params file
        #[arg(long)]
        params_path: PathBuf,
        /// The path to output to load the desired verfication key file
        #[arg(long)]
        vk_path: PathBuf,
        /// The path to output to the desired verification code
        #[arg(long, required_if_eq("transcript", " "))]
        deployment_code_path: Option<PathBuf>,
        /// The [ProofSystem] we'll be using.
        #[arg(
            long,
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::KZG,
            value_enum
        )]
        pfsys: ProofSystem,
        // todo, optionally allow supplying proving key
    },

    /// Verifies a proof, returning accept or reject
    #[command(arg_required_else_help = true)]
    Verify {
        /// The path to the .onnx model file
        #[arg(short = 'M', long)]
        model: PathBuf,
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
            default_value_t = ProofSystem::KZG,
            value_enum
        )]
        pfsys: ProofSystem,
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
             default_value_t = ProofSystem::KZG,
             value_enum
         )]
        pfsys: ProofSystem,
        #[arg(
             long,
             require_equals = true,
             num_args = 0..=1,
             default_value_t = TranscriptType::Blake,
             value_enum
         )]
        transcript: TranscriptType,
    },

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
        #[arg(long, required_if_eq("transcript", "evm"))]
        sol_code_path: Option<PathBuf>,

        #[arg(
             long,
             require_equals = true,
             num_args = 0..=1,
             default_value_t = ProofSystem::KZG,
             value_enum
         )]
        pfsys: ProofSystem,
    },

    /// Print the proof in hexadecimal
    #[command(name = "print-proof-hex", arg_required_else_help = true)]
    PrintProofHex {
        /// The path to the proof file
        #[arg(long)]
        proof_path: PathBuf,

        #[arg(
             long,
             require_equals = true,
             num_args = 0..=1,
             default_value_t = ProofSystem::KZG,
             value_enum
         )]
        pfsys: ProofSystem,
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
