//use crate::onnx::OnnxModel;
use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::io::{stdin, stdout, Write};
use std::path::PathBuf;

const EZKLCONF: &str = "EZKLCONF";

#[allow(missing_docs)]
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    #[allow(missing_docs)]
    pub command: Commands,
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
    pub fn create() -> Self {
        match env::var(EZKLCONF) {
            Ok(val) => Self::from_json(&val).unwrap(),
            Err(_e) => Cli::parse(),
        }
    }
}

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
#[derive(Debug, Subcommand, Clone, Deserialize, Serialize)]
pub enum Commands {
    /// Loads model and prints model table
    #[command(arg_required_else_help = true)]
    Table {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = "")]
        model: String,
    },

    /// Loads model and input and runs mock prover (for testing)
    #[command(arg_required_else_help = true)]
    Mock {
        /// The path to the .json data file
        #[arg(short = 'D', long, default_value = "")]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = "")]
        model: String,
    },

    /// Loads model and input and runs full prover (for testing)
    #[command(arg_required_else_help = true)]
    Fullprove {
        /// The path to the .json data file
        #[arg(short = 'D', long, default_value = "")]
        data: String,
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = "")]
        model: String,
        //todo: optional Params
        #[arg(
            long,
            num_args = 0..=1,
            default_value_t = ProofSystem::IPA,
//            default_missing_value = "always",
            value_enum
        )]
        pfsys: ProofSystem,
    },

    /// Loads model and data, prepares vk and pk, and creates proof, saving proof in --output
    Prove {
        /// The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
        #[arg(short = 'D', long, default_value = "")]
        data: String,

        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = "")]
        model: PathBuf,
        /// The path to the desired output file
        #[arg(short = 'O', long, default_value = "")]
        proof_path: PathBuf,
        /// The path to output to the desired verfication key file (optional)
        #[arg(long, default_value = "")]
        vk_path: PathBuf,
        /// The path to output to the desired verfication key file (optional)
        #[arg(long, default_value = "")]
        params_path: PathBuf,

        // /// The path to the Params for the proof system
        // #[arg(short = 'P', long, default_value = "")]
        // params: PathBuf,
        #[arg(
            long,
	    short = 'B',
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::IPA,
            default_missing_value = "always",
            value_enum
        )]
        /// The [ProofSystem] we'll be using.
        pfsys: ProofSystem,
        // todo, optionally allow supplying proving key
    },
    /// Verifies a proof, returning accept or reject
    Verify {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = "")]
        model: PathBuf,

        /// The path to the proof file
        #[arg(short = 'P', long, default_value = "")]
        proof_path: PathBuf,
        /// The path to output to the desired verfication key file (optional)
        #[arg(long, default_value = "")]
        vk_path: PathBuf,
        /// The path to output to the desired verfication key file (optional)
        #[arg(long, default_value = "")]
        params_path: PathBuf,

        // /// The path to the Params for the proof system
        // #[arg(short = 'P', long, default_value = "")]
        // params: PathBuf,
        #[arg(
            long,
	    short = 'B',
            require_equals = true,
            num_args = 0..=1,
            default_value_t = ProofSystem::IPA,
            default_missing_value = "always",
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
