use crate::circuit::{CheckMode, Tolerance};
use crate::commands::{CalibrationArgs, CalibrationTarget, RunArgs, StrategyType};
use crate::graph::{Model, Visibility};
use crate::pfsys::{
    gen_srs as ezkl_gen_srs, save_params,
     Snark, TranscriptType,
};
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2curves::bn256::Bn256;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use std::{fs::File, path::PathBuf};
use tokio::runtime::Runtime;

/// pyclass containing the struct used for run_args
#[pyclass]
#[derive(Clone)]
struct PyRunArgs {
    #[pyo3(get, set)]
    pub tolerance: Tolerance,
    #[pyo3(get, set)]
    pub scale: u32,
    #[pyo3(get, set)]
    pub bits: usize,
    #[pyo3(get, set)]
    pub logrows: u32,
    #[pyo3(get, set)]
    pub input_visibility: Visibility,
    #[pyo3(get, set)]
    pub output_visibility: Visibility,
    #[pyo3(get, set)]
    pub param_visibility: Visibility,
    #[pyo3(get, set)]
    pub pack_base: u32,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub allocated_constraints: Option<usize>,
}

/// default instantiation of PyRunArgs
#[pymethods]
impl PyRunArgs {
    #[new]
    fn new() -> Self {
        PyRunArgs {
            tolerance: Tolerance::Abs { val: 0 },
            scale: 7,
            bits: 16,
            logrows: 17,
            input_visibility: "public".into(),
            output_visibility: "public".into(),
            param_visibility: "private".into(),
            pack_base: 1,
            batch_size: 1,
            allocated_constraints: None,
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
            input_visibility: py_run_args.input_visibility,
            output_visibility: py_run_args.output_visibility,
            param_visibility: py_run_args.param_visibility,
            pack_base: py_run_args.pack_base,
            allocated_constraints: py_run_args.allocated_constraints,
            batch_size: py_run_args.batch_size,
        }
    }
}

/// pyclass containing the struct used for calibration_args
#[pyclass]
#[derive(Clone)]
struct PyCalArgs {
    #[pyo3(get, set)]
    pub data: Option<PathBuf>,
    #[pyo3(get, set)]
    pub target: Option<CalibrationTarget>,
}

/// default instantiation of PyCalArgs
#[pymethods]
impl PyCalArgs {
    #[new]
    fn new() -> Self {
        PyCalArgs {
            data: None::<PathBuf>,
            target: None::<CalibrationTarget>,
        }
    }
}

/// Conversion between PyRunArgs and RunArgs
impl From<PyCalArgs> for CalibrationArgs {
    fn from(py_cal_args: PyCalArgs) -> Self {
        CalibrationArgs {
            data: py_cal_args.data,
            target: py_cal_args.target,
        }
    }
}

/// Displays the table as a string in python
#[pyfunction(signature = (
    model,
    py_run_args = None
))]
fn table(model: String, py_run_args: Option<PyRunArgs>) -> PyResult<String> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let mut reader = File::open(model).map_err(|_| PyIOError::new_err("Failed to open model"))?;
    let result = Model::new(&mut reader, run_args);

    match result {
        Ok(m) => Ok(m.table_nodes()),
        Err(_) => Err(PyIOError::new_err("Failed to import model")),
    }
}

/// generates the srs
#[pyfunction(signature = (
    params_path,
    logrows,
))]
fn gen_srs(params_path: PathBuf, logrows: usize) -> PyResult<()> {
    let params = ezkl_gen_srs::<KZGCommitmentScheme<Bn256>>(logrows as u32);
    save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
    Ok(())
}

/// runs the forward pass operation
#[pyfunction(signature = (
    model,
    output,
    py_run_args = None,
    py_cal_args = None
))]
fn gen_circuit_params(
    model: PathBuf,
    output: PathBuf,
    py_run_args: Option<PyRunArgs>,
    py_cal_args: Option<PyCalArgs>,
) -> Result<bool, PyErr> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let calibration_args: CalibrationArgs = py_cal_args.unwrap_or_else(PyCalArgs::new).into();

    crate::execute::gen_circuit_params(model, output, run_args, calibration_args).map_err(|e| {
        let err_str = format!("Failed to generate circuit params: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// runs the forward pass operation
#[pyfunction(signature = (
    data,
    model,
    output,
    scale = 7, 
    batch_size = 1,  
    circuit_params_path = None, 
))]
fn forward(
    data: PathBuf,
    model: PathBuf,
    output: PathBuf,
    scale: Option<u32>,
    batch_size: Option<usize>,
    circuit_params_path: Option<PathBuf>,
) -> PyResult<()> {
    crate::execute::forward(model, data, output, scale, batch_size, circuit_params_path)
        .map_err(|e| {
            let err_str = format!("Failed to run forward: {}", e);
            PyRuntimeError::new_err(err_str)})?;
    Ok(())
}

/// mocks the prover
#[pyfunction(signature = (
    data,
    model,
    circuit_params_path,
    py_run_args = None
))]
fn mock(data: PathBuf, model: PathBuf, circuit_params_path: Option<PathBuf>, py_run_args: Option<PyRunArgs>) -> PyResult<bool> {
    let run_args = py_run_args.unwrap_or_else(PyRunArgs::new).into();

    crate::execute::mock(model, data, run_args, circuit_params_path).map_err(|e| {
        let err_str = format!("Failed to run setup: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    model,
    vk_path,
    pk_path,
    params_path,
    circuit_params_path,
))]
fn setup(
    model: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    params_path: PathBuf,
    circuit_params_path: PathBuf,
) -> Result<bool, PyErr> {

    crate::execute::setup(model, params_path, circuit_params_path, vk_path, pk_path).map_err(|e| {
            let err_str = format!("Failed to run setup: {}", e);
            PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    data,
    model,
    pk_path,
    proof_path,
    params_path,
    transcript,
    strategy,
    circuit_params_path,
))]
fn prove(
    data: PathBuf,
    model: PathBuf,
    pk_path: PathBuf,
    proof_path: PathBuf,
    params_path: PathBuf,
    transcript: TranscriptType,
    strategy: StrategyType,
    circuit_params_path: PathBuf,
) -> Result<bool, PyErr> {
    
    crate::execute::prove(data, model, pk_path, proof_path, params_path, transcript, strategy, circuit_params_path, CheckMode::UNSAFE).map_err(|e| {
        let err_str = format!("Failed to run prove: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// verifies a given proof
#[pyfunction(signature = (
    proof_path,
    circuit_params_path,
    vk_path,
    params_path,
))]
fn verify(
    proof_path: PathBuf,
    circuit_params_path: PathBuf,
    vk_path: PathBuf,
    params_path: PathBuf,
) -> Result<bool, PyErr> {

    crate::execute::verify(proof_path, circuit_params_path, vk_path, params_path).map_err(|e| {
        let err_str = format!("Failed to run verify: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)

}

/// creates an aggregated proof
#[pyfunction(signature = (
    proof_path,
    aggregation_snarks,
    circuit_params_paths,
    aggregation_vk_paths,
    vk_path,
    params_path,
    transcript,
    logrows,
    check_mode,
))]
fn aggregate(
    proof_path: PathBuf,
    aggregation_snarks: Vec<PathBuf>,
    circuit_params_paths: Vec<PathBuf>,
    aggregation_vk_paths: Vec<PathBuf>,
    vk_path: PathBuf,
    params_path: PathBuf,
    transcript: TranscriptType,
    logrows: u32,
    check_mode: CheckMode,
) -> Result<bool, PyErr> {
    // the K used for the aggregation circuit
    crate::execute::aggregate(proof_path,
        aggregation_snarks,
        circuit_params_paths,
        aggregation_vk_paths,
        vk_path,
        params_path,
        transcript,
        logrows,
        check_mode).map_err(|e| {
        let err_str = format!("Failed to run aggregate: {}", e);
        PyRuntimeError::new_err(err_str)})?;
    
    Ok(true)
}

/// verifies and aggregate proof
#[pyfunction(signature = (
    proof_path,
    vk_path,
    params_path,
    logrows
))]
fn verify_aggr(
    proof_path: PathBuf,
    vk_path: PathBuf,
    params_path: PathBuf,
    logrows: u32,
) -> Result<bool, PyErr> {

    crate::execute::verify_aggr(proof_path, vk_path, params_path, logrows).map_err(|e| {
        let err_str = format!("Failed to run verify_aggr: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// creates an EVM compatible verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    vk_path,
    params_path,
    circuit_params_path,
    deployment_code_path,
    sol_code_path=None,
))]
fn create_evm_verifier(
    vk_path: PathBuf,
    params_path: PathBuf,
    circuit_params_path: PathBuf,
    deployment_code_path: PathBuf,
    sol_code_path: Option<PathBuf>,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_verifier(
        vk_path,
        params_path,
        circuit_params_path,
        deployment_code_path,
        sol_code_path,
    ).map_err(|e| {
        let err_str = format!("Failed to run create_evm_verifier: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// verifies an evm compatible proof, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    proof_path,
    deployment_code_path,
    sol_code_path=None,
    runs=None
))]
fn verify_evm(
    proof_path: PathBuf,
    deployment_code_path: PathBuf,
    sol_code_path: Option<PathBuf>,
    runs: Option<usize>,
) -> Result<bool, PyErr> {

    Runtime::new()
            .unwrap()
            .block_on(crate::execute::verify_evm(
        proof_path,
        deployment_code_path,
        sol_code_path,
        runs,
    )).map_err(|e| {
        let err_str = format!("Failed to run verify_evm: {}", e);
        PyRuntimeError::new_err(err_str)})?;

    Ok(true)
}

/// creates an evm compatible aggregate verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    vk_path,
    params_path,
    deployment_code_path,
    sol_code_path=None,
))]
fn create_evm_verifier_aggr(
    vk_path: PathBuf,
    params_path: PathBuf,
    deployment_code_path: Option<PathBuf>,
    sol_code_path: Option<PathBuf>,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_aggregate_verifier(
        vk_path,
        params_path,
        deployment_code_path,
        sol_code_path,
    ).map_err(|e| {
        let err_str = format!("Failed to run create_evm_verifier_aggr: {}", e);
        PyRuntimeError::new_err(err_str)})?;
    Ok(true)
}

/// print hex representation of a proof
#[pyfunction(signature = (proof_path))]
fn print_proof_hex(proof_path: PathBuf) -> Result<String, PyErr> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)
        .map_err(|_| PyIOError::new_err("Failed to load proof"))?;

    // let mut return_string: String = "";
    // for instance in proof.instances {
    //     return_string.push_str(instance + "\n");
    // }
    // return_string = hex::encode(proof.proof);

    // return proof for now
    Ok(hex::encode(proof.proof))
}

// Python Module
#[pymodule]
fn ezkl_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // NOTE: DeployVerifierEVM and SendProofEVM will be implemented in python in pyezkl
    pyo3_log::init();
    m.add_class::<PyRunArgs>()?;
    m.add_class::<PyCalArgs>()?;
    m.add_function(wrap_pyfunction!(table, m)?)?;
    m.add_function(wrap_pyfunction!(gen_srs, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(mock, m)?)?;
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(gen_circuit_params, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(verify_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier, m)?)?;
    m.add_function(wrap_pyfunction!(verify_evm, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(print_proof_hex, m)?)?;

    Ok(())
}
