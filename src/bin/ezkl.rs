use clap::Parser;
use halo2_proofs::dev::MockProver;
use halo2curves::pasta::Fp as F;
use halo2deeplearning::fieldutils::i32_to_felt;
use halo2deeplearning::onnx::{utilities::vector_to_quantized, Cli, OnnxCircuit};
use log::{info, trace};
use serde::Deserialize;
use std::fs::File;
use std::io::{stdin, stdout, Read, Write};
use std::marker::PhantomData;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct OnnxInput {
    input_data: Vec<f32>,
    input_shape: Vec<usize>,
    public_input: Vec<f32>,
}

pub fn main() {
    colog::init();

    info!(
        "
        ███████╗███████╗██╗  ██╗██╗
        ██╔════╝╚══███╔╝██║ ██╔╝██║
        █████╗    ███╔╝ █████╔╝ ██║
        ██╔══╝   ███╔╝  ██╔═██╗ ██║
        ███████╗███████╗██║  ██╗███████╗
        ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝

        -----------------------------------------------------------
        Easy Zero Knowledge for Neural Networks.
        -----------------------------------------------------------
        "
    );

    let k = 16; //2^k rows
    let mut args = Cli::parse();
    // load
    let mut s = String::new();

    let data_path = match args.data.is_empty() {
        false => {
            info!("loading data from {}", args.data.clone());
            args.data = args.data.replace(' ', "");
            Path::new(&args.data)
        }
        true => {
            info!("please enter a path to a .json file containing inputs for the model: ");
            let _ = stdout().flush();
            let _ = &stdin()
                .read_line(&mut s)
                .expect("did not enter a correct string");
            s.truncate(s.len() - 1);
            Path::new(&s)
        }
    };
    assert!(data_path.exists());
    let mut file = File::open(data_path).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();

    let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");

    // quantize the supplied data using the provided scale.
    let input = vector_to_quantized(&data.input_data, &data.input_shape, 0.0, args.scale).unwrap();
    info!(
        "public input length (network output) {:?}",
        data.public_input.len()
    );
    // quantize the supplied data using the provided scale.
    let public_input = vector_to_quantized(
        &data.public_input,
        &Vec::from([data.public_input.len()]),
        0.0,
        args.scale,
    )
    .unwrap();

    trace!("{:?}", public_input);

    let circuit = OnnxCircuit::<F> {
        input,
        _marker: PhantomData,
    };

    let prover = MockProver::run(
        k,
        &circuit,
        vec![public_input.iter().map(|x| i32_to_felt::<F>(*x)).collect()],
    )
    .unwrap();
    prover.assert_satisfied();
}
