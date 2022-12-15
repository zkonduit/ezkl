//use crate::onnx::OnnxModel;
use clap::{Parser, Subcommand, ValueEnum};
use log::info;
use std::io::{stdin, stdout, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
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
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
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

#[derive(Debug, Subcommand)]
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
        output: PathBuf,
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
        // todo, optionally allow supplying proving key
    },
    /// Verifies a proof, returning accept or reject
    Verify {
        /// The path to the .onnx model file
        #[arg(short = 'M', long, default_value = "")]
        model: PathBuf,

        /// The path to the proof file
        #[arg(short = 'P', long, default_value = "")]
        proof: PathBuf,
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
