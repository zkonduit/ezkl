// Really these are halo2 plonkish IOP + commitment scheme, but we only support Plonkish IOP so far, so there is no ambiguity
pub mod ipa;
pub mod kzg;

use crate::abort;
use crate::commands::{data_path, Cli};
use crate::onnx::{utilities::vector_to_quantized, Model, ModelCircuit};
use crate::tensor::Tensor;
use clap::Parser;
use halo2_proofs::{arithmetic::FieldExt, dev::VerifyFailure};
use log::{error, info, trace};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelInput {
    pub input_data: Vec<Vec<f32>>,
    pub input_shapes: Vec<Vec<usize>>,
    pub public_inputs: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Proof {
    pub input_shapes: Vec<Vec<usize>>,
    pub public_inputs: Vec<Vec<i32>>,
    pub proof: Vec<u8>,
}

/// Helper function to print helpful error messages after verification has failed.
pub fn parse_prover_errors(f: &VerifyFailure) {
    match f {
        VerifyFailure::Lookup {
            name,
            location,
            lookup_index,
        } => {
            error!("lookup {:?} is out of range, try increasing 'bits' or reducing 'scale' ({} and lookup index {}).",
            name, location, lookup_index);
        }
        VerifyFailure::ConstraintNotSatisfied {
            constraint,
            location,
            cell_values: _,
        } => {
            error!("{} was not satisfied {}).", constraint, location);
        }
        VerifyFailure::ConstraintPoisoned { constraint } => {
            error!("constraint {:?} was poisoned", constraint);
        }
        VerifyFailure::Permutation { column, location } => {
            error!(
                "permutation did not preserve column cell value (try increasing 'scale') ({} {}).",
                column, location
            );
        }
        VerifyFailure::CellNotAssigned {
            gate,
            region,
            gate_offset,
            column,
            offset,
        } => {
            error!(
                "Unnassigned value in {} ({}) and {} ({:?}, {})",
                gate, region, gate_offset, column, offset
            );
        }
    }
}

pub fn prepare_circuit_and_public_input<F: FieldExt>(
    data: &ModelInput,
) -> (ModelCircuit<F>, Vec<Tensor<i32>>) {
    let onnx_model = Model::from_arg();
    let out_scales = onnx_model.get_output_scales();
    let circuit = prepare_circuit(data);

    // quantize the supplied data using the provided scale.
    let public_inputs = data
        .public_inputs
        .iter()
        .enumerate()
        .map(
            |(idx, v)| match vector_to_quantized(v, &Vec::from([v.len()]), 0.0, out_scales[idx]) {
                Ok(q) => q,
                Err(e) => {
                    abort!("failed to quantize vector {:?}", e);
                }
            },
        )
        .collect();
    trace!("{:?}", public_inputs);
    (circuit, public_inputs)
}

pub fn prepare_circuit<F: FieldExt>(data: &ModelInput) -> ModelCircuit<F> {
    let args = Cli::parse();

    // quantize the supplied data using the provided scale.
    let inputs = data
        .input_data
        .iter()
        .zip(data.input_shapes.clone())
        .map(|(i, s)| match vector_to_quantized(i, &s, 0.0, args.scale) {
            Ok(q) => q,
            Err(e) => {
                abort!("failed to quantize vector {:?}", e);
            }
        })
        .collect();

    ModelCircuit::<F> {
        inputs,
        _marker: PhantomData,
    }
}

pub fn prepare_data(datapath: String) -> ModelInput {
    let mut file = match File::open(data_path(datapath)) {
        Ok(t) => t,
        Err(e) => {
            abort!("failed to open data file {:?}", e);
        }
    };
    let mut data = String::new();
    match file.read_to_string(&mut data) {
        Ok(_) => {}
        Err(e) => {
            abort!("failed to read file {:?}", e);
        }
    };
    let data: ModelInput = serde_json::from_str(&data).expect("JSON was not well-formatted");
    info!(
        "public inputs (network outputs) lengths: {:?}",
        data.public_inputs
            .iter()
            .map(|i| i.len())
            .collect::<Vec<usize>>()
    );

    data
}
