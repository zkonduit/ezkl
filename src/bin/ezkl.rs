use clap::Parser;
use colog;
use halo2_proofs::dev::MockProver;
use halo2curves::pasta::Fp as F;
use halo2deeplearning::fieldutils::i32_to_felt;
use halo2deeplearning::onnx::{Cli, OnnxCircuit};
use halo2deeplearning::tensor::Tensor;
use log::info;
use serde;
use serde::Deserialize;
use serde_json;
use std::fs::File;
use std::io::{stdin, stdout, Read, Write};
use std::marker::PhantomData;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct OnnxInput {
    input_data: Vec<i32>,
    input_shape: Vec<usize>,
    public_input: Vec<i32>,
}

pub fn main() {
    colog::init();

    let k = 16; //2^k rows
    let mut args = Cli::parse();
    // load
    let mut s = String::new();

    let data_path = match args.data.is_empty() {
        false => {
            info!("loading data from {}", args.data.clone());
            args.data = args.data.replace(" ", "");
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

    let input = Tensor::<i32>::new(Some(&data.input_data), &data.input_shape).unwrap();

    info!(
        "public input length (network output) {:?}",
        data.public_input.len()
    );

    let circuit = OnnxCircuit::<F> {
        input,
        _marker: PhantomData,
    };

    let prover = MockProver::run(
        k,
        &circuit,
        vec![data
            .public_input
            .iter()
            .map(|x| i32_to_felt::<F>(*x))
            .collect()],
    )
    .unwrap();
    prover.assert_satisfied();
}
