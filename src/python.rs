use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyIOError;
use pyo3_log;
use tabled::Table;
use crate::graph::{Model, Visibility, VarVisibility, Mode};
use crate::commands::RunArgs;
use crate::circuit::base::CheckMode;
use crate::pfsys::{gen_srs as ezkl_gen_srs, save_params};
use std::path::PathBuf;
use halo2curves::bn256::Bn256;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;


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
fn table(
    model: String,
) -> Result<String, PyErr> {
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

    let result = Model::new(
        model,
        run_args,
        Mode::Mock,
        visibility,
    );

    match result {
        Ok(m) => {
            Ok(Table::new(m.nodes.iter()).to_string())
        },
        Err(_) => {
            Err(PyIOError::new_err("Failed to import model"))
        },
    }
}

#[pyfunction]
fn gen_srs(
    params_path: PathBuf,
    logrows: u32,
) -> PyResult<()> {
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

#[pyfunction(run_args="**")]
fn forward(
    data: String,
    model: String,
    output: String,
    run_args: Option<&PyDict>
) -> PyResult<&PyDict> {
    let mut data = prepare_data(data)?;

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

    // quantize the supplied data using the provided scale.
    let mut model_inputs = vec![];
    for v in data.input_data.iter() {
        let t = vector_to_quantized(v, &Vec::from([v.len()]), 0.0, cli.args.scale)?;
        model_inputs.push(t);
    }

    let res = Model::forward(model, &model_inputs, cli.args)?;

    let float_res: Vec<Vec<f32>> = res.iter().map(|t| t.to_vec()).collect();
    trace!("forward pass output: {:?}", float_res);
    data.output_data = float_res;

    serde_json::to_writer(&File::create(output)?, &data)?;
    Ok(())
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
    Ok(())
}