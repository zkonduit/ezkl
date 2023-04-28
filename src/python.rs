use crate::circuit::CheckMode;
use crate::commands::RunArgs;
use crate::graph::{vector_to_quantized, Mode, Model, ModelCircuit, VarVisibility, Visibility};
use crate::pfsys::{gen_srs as ezkl_gen_srs, prepare_data, save_params};
// use std::env;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::dev::MockProver;
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

/// Environment variable for EZKLCONF
// const EZKLCONF: &str = "EZKLCONF";

/// pyclass containing the struct used for run_args
#[pyclass]
#[derive(Clone)]
struct PyRunArgs {
    #[pyo3(get, set)]
    pub tolerance: usize,
    #[pyo3(get, set)]
    pub scale: u32,
    #[pyo3(get, set)]
    pub bits: usize,
    #[pyo3(get, set)]
    pub logrows: u32,
    #[pyo3(get, set)]
    pub public_inputs: bool,
    #[pyo3(get, set)]
    pub public_outputs: bool,
    #[pyo3(get, set)]
    pub public_params: bool,
    #[pyo3(get, set)]
    pub pack_base: u32,
    #[pyo3(get, set)]
    pub allocated_constraints: Option<usize>,
    #[pyo3(get, set)]
    pub check_mode: CheckMode,
}

/// default instantiation of PyRunArgs
#[pymethods]
impl PyRunArgs {
    #[new]
    fn new() -> Self {
        PyRunArgs {
            tolerance: 0,
            scale: 7,
            bits: 16,
            logrows: 17,
            public_inputs: true,
            public_outputs: true,
            public_params: false,
            pack_base: 1,
            allocated_constraints: None,
            check_mode: CheckMode::SAFE,
        }
    }
}

/// Conversion between PyRunArgs and RunArgs
impl From<PyRunArgs> for RunArgs {
    fn from(py_run_args: PyRunArgs) -> Self {
        RunArgs {
            tolerance: py_run_args.tolerance,
            scale: py_run_args.scale,
            bits: py_run_args.bits,
            logrows: py_run_args.logrows,
            public_inputs: py_run_args.public_inputs,
            public_outputs: py_run_args.public_outputs,
            public_params: py_run_args.public_params,
            pack_base: py_run_args.pack_base,
            allocated_constraints: py_run_args.allocated_constraints,
            check_mode: py_run_args.check_mode,
        }
    }
}

/// Displays the table as a string in python
#[pyfunction(signature = (
    model,
    py_run_args = None
))]
fn table(model: String, py_run_args: Option<PyRunArgs>) -> Result<String, PyErr> {
    let run_args = py_run_args.unwrap_or_else(PyRunArgs::new).into();

    // use default values to initialize model
    let visibility = VarVisibility {
        input: Visibility::Public,
        params: Visibility::Private,
        output: Visibility::Public,
    };

    let result = Model::<Fr>::new(model, run_args, Mode::Mock, visibility);

    match result {
        Ok(m) => Ok(Table::new(m.nodes.iter()).to_string()),
        Err(_) => Err(PyIOError::new_err("Failed to import model")),
    }
}

/// generates the srs
#[pyfunction(signature = (
    params_path,
    py_run_args = None
))]
fn gen_srs(params_path: PathBuf, py_run_args: Option<PyRunArgs>) -> PyResult<()> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let params = ezkl_gen_srs::<KZGCommitmentScheme<Bn256>>(run_args.logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
    Ok(())
}

/// runs the forward pass operation
#[pyfunction(signature = (
    data,
    model,
    output,
    py_run_args = None
))]
fn forward(
    data: String,
    model: String,
    output: String,
    py_run_args: Option<PyRunArgs>
) -> PyResult<()> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let data = prepare_data(data);

    match data {
        Ok(m) => {
            let mut new_data = m;
            let mut model_inputs = vec![];
            // quantize the supplied data using the provided scale.
            for v in new_data.input_data.iter() {
                match vector_to_quantized(v, &Vec::from([v.len()]), 0.0, run_args.scale) {
                    Ok(t) => model_inputs.push(t),
                    Err(_) => return Err(PyValueError::new_err("Failed to quantize vector")),
                }
            }
            let res = Model::<Fr>::forward(model, &model_inputs, run_args);

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

/// mocks the prover
#[pyfunction(signature = (
    data,
    model,
    py_run_args = None
))]
fn mock(
    data: String,
    model: String,
    py_run_args: Option<PyRunArgs>
) -> Result<bool, PyErr> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let logrow = run_args.logrows;

    let data = prepare_data(data);

    // set EZKL
    // env::set_var(EZKLCONF, "PYTHON");

    match data {
        Ok(d) => {
            // use default values to initialize model
            let visibility = VarVisibility {
                input: Visibility::Public,
                params: Visibility::Private,
                output: Visibility::Public,
            };

            let model_proc = Model::<Fr>::new(model, run_args, Mode::Mock, visibility);

            match model_proc {
                Ok(m) => {
                    let circuit = ModelCircuit::<Fr>::new(&d, m);

                    match circuit {
                        Ok(c) => {
                            let public_inputs = c.prepare_public_inputs(&d);

                            match public_inputs {
                                Ok(pi) => {
                                    let prover = MockProver::run(logrow, &c, pi); // this is putting messages in stdout
                                    match prover {
                                        Ok(pr) => {
                                            pr.assert_satisfied();

                                            let res = pr.verify();
                                            match res {
                                                Ok(_) => return Ok(true),
                                                Err(_) => return Ok(false),
                                            }
                                        }
                                        Err(_) => Err(PyRuntimeError::new_err("Failed to run prover")),
                                    }
                                }
                                Err(_) => Err(PyRuntimeError::new_err("Failed to prepare public inputs")),
                            }
                        }
                        Err(_) => Err(PyRuntimeError::new_err("Failed to create circuit")),
                    }
                }
                Err(_) => Err(PyIOError::new_err("Failed to process model"))
            }
        }
        Err(_) => Err(PyIOError::new_err("Failed to import files")),
    }
}

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
    m.add_class::<PyRunArgs>()?;
    m.add_function(wrap_pyfunction!(table, m)?)?;
    m.add_function(wrap_pyfunction!(gen_srs, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(mock, m)?)?;

    Ok(())
}
