use crate::circuit::CheckMode;
use crate::commands::RunArgs;
use crate::graph::{vector_to_quantized, Mode, Model, VarVisibility, Visibility};
use crate::pfsys::{gen_srs as ezkl_gen_srs, prepare_data, save_params};
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2curves::bn256::{Bn256, Fr};
use log::trace;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use std::fs::File;
use std::path::PathBuf;
use tabled::Table;

// See commands.rs and execute.rs
// RenderCircuit
// #[pyfunction]
// fn render_circuit(
//     data_path: &Path,
//     model: _,
//     output_path: &Path,
//     args: Vec<String>
// ) -> PyResult<()> {
//     let data = prepare_data(data_path.to_string());
//     let circuit = prepare_model_circuit::<Fr>(&data, &cli.args)?;

//     halo2_proofs::dev::CircuitLayout::default()
//         .show_labels(false)
//         .render(args.logrows, &circuit, &root)?;
// }

// Table

#[pyfunction]
fn table(model: String) -> Result<String, PyErr> {
    // use default values to initialize model
    let run_args = RunArgs {
        tolerance: 0,
        scale: 7,
        bits: 16,
        logrows: 17,
        public_inputs: true,
        public_outputs: true,
        public_params: false,
        pack_base: 1,
        check_mode: CheckMode::SAFE,
    };

    // use default values to initialize model
    let visibility = VarVisibility {
        input: Visibility::Public,
        params: Visibility::Private,
        output: Visibility::Public,
    };

    let result = Model::new::<Fr>(model, run_args, Mode::Mock, visibility);

    match result {
        Ok(m) => Ok(Table::new(m.nodes.iter()).to_string()),
        Err(_) => Err(PyIOError::new_err("Failed to import model")),
    }
}

#[pyfunction]
fn gen_srs(params_path: PathBuf, logrows: u32) -> PyResult<()> {
    let run_args = RunArgs {
        tolerance: 0,
        scale: 7,
        bits: 16,
        logrows: logrows,
        public_inputs: true,
        public_outputs: true,
        public_params: false,
        pack_base: 1,
        check_mode: CheckMode::SAFE,
    };
    let params = ezkl_gen_srs::<KZGCommitmentScheme<Bn256>>(run_args.logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
    Ok(())
}

#[pyfunction(signature = (
    data,
    model,
    output,
    tolerance=0,
    scale=7,
    bits=16,
    logrows=17,
    public_inputs=true,
    public_outputs=true,
    public_params=false,
    pack_base=1,
    check_mode="safe"
))]
fn forward(
    data: String,
    model: String,
    output: String,
    tolerance: usize,
    scale: u32,
    bits: usize,
    logrows: u32,
    public_inputs: bool,
    public_outputs: bool,
    public_params: bool,
    pack_base: u32,
    check_mode: &str,
) -> PyResult<()> {
    let data = prepare_data(data);

    match data {
        Ok(m) => {
            let run_args = RunArgs {
                tolerance: tolerance,
                scale: scale,
                bits: bits,
                logrows: logrows,
                public_inputs: public_inputs,
                public_outputs: public_outputs,
                public_params: public_params,
                pack_base: pack_base,
                check_mode: CheckMode::from(check_mode.to_string()),
            };
            let mut new_data = m;
            let mut model_inputs = vec![];
            // quantize the supplied data using the provided scale.
            for v in new_data.input_data.iter() {
                match vector_to_quantized(v, &Vec::from([v.len()]), 0.0, run_args.scale) {
                    Ok(t) => model_inputs.push(t),
                    Err(_) => return Err(PyValueError::new_err("Failed to quantize vector")),
                }
            }
            let res = Model::forward::<Fr>(model, &model_inputs, run_args);

            match res {
                Ok(r) => {
                    let float_res: Vec<Vec<f32>> = r.iter().map(|t| t.to_vec()).collect();
                    trace!("forward pass output: {:?}", float_res);
                    new_data.output_data = float_res;

                    match serde_json::to_writer(&File::create(output)?, &new_data) {
                        Ok(_) => {
                            // TODO output a dictionary
                            // obtain gil
                            // TODO: Convert to Python::with_gil() when it stabilizes
                            // let gil = Python::acquire_gil();
                            // obtain python instance
                            // let py = gil.python();
                            // return Ok(new_data.to_object(py))
                            Ok(())
                        }
                        Err(_) => return Err(PyIOError::new_err("Failed to create output file")),
                    }
                }
                Err(_) => Err(PyRuntimeError::new_err("Failed to compute forward pass")),
            }
        }
        Err(_) => Err(PyIOError::new_err("Failed to import files")),
    }
}

// TODO: Mock
// TODO: Aggregate
// TODO: Prove
// TODO: CreateEVMVerifier
// TODO: CreateEVMVerifierAggr
// TODO: DeployVerifierEVM
// TODO: SendProofEVM
// TODO: Verify
// TODO: VerifyAggr
// TODO: VerifyEVM
// TODO: PrintProofHex

// Python Module
#[pymodule]
fn ezkl_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(table, m)?)?;
    m.add_function(wrap_pyfunction!(gen_srs, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    Ok(())
}
