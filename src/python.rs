use crate::circuit::{CheckMode, Tolerance};
use crate::commands::{CalibrationTarget, RunArgs, StrategyType};
use crate::fieldutils::{felt_to_i128, i128_to_felt};
use crate::graph::{quantize_float, scale_to_multiplier, Model, Visibility};
use crate::pfsys::{save_params, srs::gen_srs as ezkl_gen_srs, Snark, TranscriptType};
use ethers::types::H160;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2curves::bn256::Bn256;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use std::str::FromStr;
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
            tolerance: Tolerance::default(),
            scale: 7,
            bits: 16,
            logrows: 17,
            input_visibility: "public".into(),
            output_visibility: "public".into(),
            param_visibility: "private".into(),
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
            allocated_constraints: py_run_args.allocated_constraints,
            batch_size: py_run_args.batch_size,
        }
    }
}

/// Converts 4 u64s to a field element
#[pyfunction(signature = (
    array,
))]
fn vecu64_to_felt(array: [u64; 4]) -> PyResult<String> {
    Ok(format!(
        "{:?}",
        crate::pfsys::vecu64_to_field::<halo2curves::bn256::Fr>(&array)
    ))
}

/// Converts 4 u64s representing a field element directly to an integer
#[pyfunction(signature = (
    array,
))]
fn vecu64_to_int(array: [u64; 4]) -> PyResult<i128> {
    let felt = crate::pfsys::vecu64_to_field::<halo2curves::bn256::Fr>(&array);
    let int_rep = felt_to_i128(felt);
    Ok(int_rep)
}

/// Converts 4 u64s representing a field element directly to a (rescaled from fixed point scaling) floating point
#[pyfunction(signature = (
    array,
    scale
))]
fn vecu64_to_float(array: [u64; 4], scale: u32) -> PyResult<f64> {
    let felt = crate::pfsys::vecu64_to_field::<halo2curves::bn256::Fr>(&array);
    let int_rep = felt_to_i128(felt);
    let multiplier = scale_to_multiplier(scale);
    let float_rep = int_rep as f64 / multiplier;
    Ok(float_rep)
}

/// Converts a floating point element to 4 u64s representing a fixed point field element
#[pyfunction(signature = (
input,
scale
))]
fn float_to_vecu64(input: f64, scale: u32) -> PyResult<[u64; 4]> {
    let int_rep = quantize_float(&input, 0.0, scale)
        .map_err(|_| PyIOError::new_err("Failed to quantize input"))?;
    let felt = i128_to_felt(int_rep);
    Ok(crate::pfsys::field_to_vecu64::<halo2curves::bn256::Fr>(
        &felt,
    ))
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
    srs_path,
    logrows,
))]
fn gen_srs(srs_path: PathBuf, logrows: usize) -> PyResult<()> {
    let params = ezkl_gen_srs::<KZGCommitmentScheme<Bn256>>(logrows as u32);
    save_params::<KZGCommitmentScheme<Bn256>>(&srs_path, &params)?;
    Ok(())
}

/// gets a public srs
#[pyfunction(signature = (
    srs_path,
    settings_path=None,
    logrows=None,
))]
fn get_srs(srs_path: PathBuf, settings_path: Option<PathBuf>, logrows: Option<u32>) -> PyResult<bool> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::get_srs_cmd(
            srs_path,
            settings_path,
            logrows,
            CheckMode::SAFE,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to get srs: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;
    Ok(true)
}

/// generates the circuit settings
#[pyfunction(signature = (
    model,
    output,
    py_run_args = None,
))]
fn gen_settings(
    model: PathBuf,
    output: PathBuf,
    py_run_args: Option<PyRunArgs>,
) -> Result<bool, PyErr> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();

    crate::execute::gen_circuit_settings(model, output, run_args).map_err(|e| {
        let err_str = format!("Failed to generate settings: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// calibrates the circuit settings
#[pyfunction(signature = (
    data,
    model,
    settings,
    target,
))]
fn calibrate_settings(
    py: Python,
    data: PathBuf,
    model: PathBuf,
    settings: PathBuf,
    target: Option<CalibrationTarget>,
) -> PyResult<&pyo3::PyAny> {
    let target = target.unwrap_or(CalibrationTarget::Resources);
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::calibrate(model, data, settings, target)
            .await
            .map_err(|e| {
                let err_str = format!("Failed to calibrate settings: {}", e);
                PyRuntimeError::new_err(err_str)
            })?;
        Ok(true)
    })
}

/// runs the forward pass operation
#[pyfunction(signature = (
    data,
    model,
    output,
    settings_path,
))]
fn gen_witness(
    data: PathBuf,
    model: PathBuf,
    output: Option<PathBuf>,
    settings_path: PathBuf,
) -> PyResult<PyObject> {
    let output = Runtime::new()
        .unwrap()
        .block_on(crate::execute::gen_witness(
            model,
            data,
            output,
            settings_path,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run generate witness: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;
    Python::with_gil(|py| Ok(output.to_object(py)))
}

/// mocks the prover
#[pyfunction(signature = (
    witness,
    model,
    settings_path,
))]
fn mock(witness: PathBuf, model: PathBuf, settings_path: PathBuf) -> PyResult<bool> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::mock(model, witness, settings_path))
        .map_err(|e| {
            let err_str = format!("Failed to run mock: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

/// mocks the aggregate prover
#[pyfunction(signature = (
    aggregation_snarks,
    logrows,
))]
fn mock_aggregate(aggregation_snarks: Vec<PathBuf>, logrows: u32) -> PyResult<bool> {
    crate::execute::mock_aggregate(aggregation_snarks, logrows).map_err(|e| {
        let err_str = format!("Failed to run mock: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    model,
    vk_path,
    pk_path,
    srs_path,
    settings_path,
))]
fn setup(
    model: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    srs_path: PathBuf,
    settings_path: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::setup(model, srs_path, settings_path, vk_path, pk_path).map_err(|e| {
        let err_str = format!("Failed to run setup: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    witness,
    model,
    pk_path,
    proof_path,
    srs_path,
    transcript,
    strategy,
    settings_path,
))]
fn prove(
    witness: PathBuf,
    model: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    srs_path: PathBuf,
    transcript: TranscriptType,
    strategy: StrategyType,
    settings_path: PathBuf,
) -> PyResult<PyObject> {
    let snark = Runtime::new()
        .unwrap()
        .block_on(crate::execute::prove(
            witness,
            model,
            pk_path,
            proof_path,
            srs_path,
            transcript,
            strategy,
            settings_path,
            CheckMode::UNSAFE,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run prove: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Python::with_gil(|py| Ok(snark.to_object(py)))
}

/// verifies a given proof
#[pyfunction(signature = (
    proof_path,
    settings_path,
    vk_path,
    srs_path,
))]
fn verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::verify(proof_path, settings_path, vk_path, srs_path).map_err(|e| {
        let err_str = format!("Failed to run verify: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

#[pyfunction(signature = (
    sample_snarks,
    vk_path,
    pk_path,
    srs_path,
    logrows,
))]
fn setup_aggregate(
    sample_snarks: Vec<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    srs_path: PathBuf,
    logrows: u32,
) -> Result<bool, PyErr> {
    crate::execute::setup_aggregate(sample_snarks, vk_path, pk_path, srs_path, logrows).map_err(
        |e| {
            let err_str = format!("Failed to setup aggregate: {}", e);
            PyRuntimeError::new_err(err_str)
        },
    )?;

    Ok(true)
}

#[pyfunction(signature = (
    model,
    compiled_model,
    settings_path,
))]
fn compile_model(
    model: PathBuf,
    compiled_model: PathBuf,
    settings_path: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::compile_model(model, compiled_model, settings_path).map_err(|e| {
        let err_str = format!("Failed to setup aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// creates an aggregated proof
#[pyfunction(signature = (
    proof_path,
    aggregation_snarks,
    vk_path,
    srs_path,
    transcript,
    logrows,
    check_mode,
))]
fn aggregate(
    proof_path: PathBuf,
    aggregation_snarks: Vec<PathBuf>,
    vk_path: PathBuf,
    srs_path: PathBuf,
    transcript: TranscriptType,
    logrows: u32,
    check_mode: CheckMode,
) -> Result<bool, PyErr> {
    // the K used for the aggregation circuit
    crate::execute::aggregate(
        proof_path,
        aggregation_snarks,
        vk_path,
        srs_path,
        transcript,
        logrows,
        check_mode,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// verifies and aggregate proof
#[pyfunction(signature = (
    proof_path,
    vk_path,
    srs_path,
    logrows
))]
fn verify_aggr(
    proof_path: PathBuf,
    vk_path: PathBuf,
    srs_path: PathBuf,
    logrows: u32,
) -> Result<bool, PyErr> {
    crate::execute::verify_aggr(proof_path, vk_path, srs_path, logrows).map_err(|e| {
        let err_str = format!("Failed to run verify_aggr: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// creates an EVM compatible verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    vk_path,
    srs_path,
    settings_path,
    sol_code_path,
    abi_path
))]
fn create_evm_verifier(
    vk_path: PathBuf,
    srs_path: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_verifier(vk_path, srs_path, settings_path, sol_code_path, abi_path)
        .map_err(|e| {
            let err_str = format!("Failed to run create_evm_verifier: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

// creates an EVM compatible data attestation verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    vk_path,
    srs_path,
    settings_path,
    sol_code_path,
    abi_path,
    input_data
))]
fn create_evm_data_attestation_verifier(
    vk_path: PathBuf,
    srs_path: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    input_data: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_data_attestation_verifier(
        vk_path,
        srs_path,
        settings_path,
        sol_code_path,
        abi_path,
        input_data,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run create_evm_data_attestation_verifier: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

#[pyfunction(signature = (
    addr_path,
    sol_code_path,
    rpc_url=None,
))]
fn deploy_evm(
    addr_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::deploy_evm(
            sol_code_path,
            rpc_url,
            addr_path,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

#[pyfunction(signature = (
    addr_path,
    input_data,
    settings_path,
    sol_code_path,
    rpc_url=None,
))]
fn deploy_da_evm(
    addr_path: PathBuf,
    input_data: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::deploy_da_evm(
            input_data,
            settings_path,
            sol_code_path,
            rpc_url,
            addr_path,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_da_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}
/// verifies an evm compatible proof, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    proof_path,
    addr,
    rpc_url=None,
    data_attestation = false,
))]
fn verify_evm(
    proof_path: PathBuf,
    addr: &str,
    rpc_url: Option<String>,
    data_attestation: bool,
) -> Result<bool, PyErr> {
    let addr = H160::from_str(addr).map_err(|e| {
        let err_str = format!("address is invalid: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Runtime::new()
        .unwrap()
        .block_on(crate::execute::verify_evm(
            proof_path,
            addr,
            rpc_url,
            data_attestation,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run verify_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

/// creates an evm compatible aggregate verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    vk_path,
    srs_path,
    sol_code_path,
    abi_path,
    aggregation_settings
))]
fn create_evm_verifier_aggr(
    vk_path: PathBuf,
    srs_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    aggregation_settings: Vec<PathBuf>,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_aggregate_verifier(
        vk_path,
        srs_path,
        sol_code_path,
        abi_path,
        aggregation_settings,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run create_evm_verifier_aggr: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;
    Ok(true)
}

/// print hex representation of a proof
#[pyfunction(signature = (proof_path))]
fn print_proof_hex(proof_path: PathBuf) -> Result<String, PyErr> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)
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
fn ezkl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // NOTE: DeployVerifierEVM and SendProofEVM will be implemented in python in pyezkl
    pyo3_log::init();
    m.add_class::<PyRunArgs>()?;
    m.add_function(wrap_pyfunction!(vecu64_to_felt, m)?)?;
    m.add_function(wrap_pyfunction!(vecu64_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(vecu64_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(float_to_vecu64, m)?)?;
    m.add_function(wrap_pyfunction!(table, m)?)?;
    m.add_function(wrap_pyfunction!(mock, m)?)?;
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(gen_srs, m)?)?;
    m.add_function(wrap_pyfunction!(get_srs, m)?)?;
    m.add_function(wrap_pyfunction!(gen_witness, m)?)?;
    m.add_function(wrap_pyfunction!(gen_settings, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_settings, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(mock_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(setup_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(compile_model, m)?)?;
    m.add_function(wrap_pyfunction!(verify_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_evm, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_da_evm, m)?)?;
    m.add_function(wrap_pyfunction!(verify_evm, m)?)?;
    m.add_function(wrap_pyfunction!(print_proof_hex, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_data_attestation_verifier, m)?)?;

    Ok(())
}
