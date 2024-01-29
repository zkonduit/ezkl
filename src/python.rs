use crate::circuit::modules::kzg::KZGChip;
use crate::circuit::modules::poseidon::{
    spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
    PoseidonChip,
};
use crate::circuit::modules::Module;
use crate::circuit::{CheckMode, Tolerance};
use crate::commands::*;
use crate::fieldutils::{felt_to_i128, i128_to_felt};
use crate::graph::modules::POSEIDON_LEN_GRAPH;
use crate::graph::TestDataSource;
use crate::graph::{
    quantize_float, scale_to_multiplier, GraphCircuit, GraphSettings, Model, Visibility,
};
use crate::pfsys::evm::aggregation::AggregationCircuit;
use crate::pfsys::{
    load_pk, load_vk, save_params, save_vk, srs::gen_srs as ezkl_gen_srs, srs::load_srs, ProofType,
    Snark, TranscriptType,
};
use crate::RunArgs;
use ethers::types::H160;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine, G1};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use snark_verifier::util::arithmetic::PrimeField;
use std::str::FromStr;
use std::{fs::File, path::PathBuf};
use tokio::runtime::Runtime;

type PyFelt = String;

#[pyclass]
#[derive(Debug, Clone)]
enum PyTestDataSource {
    /// The data is loaded from a file
    File,
    /// The data is loaded from the chain
    OnChain,
}

impl From<PyTestDataSource> for TestDataSource {
    fn from(py_test_data_source: PyTestDataSource) -> Self {
        match py_test_data_source {
            PyTestDataSource::File => TestDataSource::File,
            PyTestDataSource::OnChain => TestDataSource::OnChain,
        }
    }
}

/// pyclass containing the struct used for G1
#[pyclass]
#[derive(Debug, Clone)]
struct PyG1 {
    #[pyo3(get, set)]
    x: PyFelt,
    #[pyo3(get, set)]
    y: PyFelt,
    #[pyo3(get, set)]
    z: PyFelt,
}

impl From<G1> for PyG1 {
    fn from(g1: G1) -> Self {
        PyG1 {
            x: crate::pfsys::field_to_string_montgomery::<Fq>(&g1.x),
            y: crate::pfsys::field_to_string_montgomery::<Fq>(&g1.y),
            z: crate::pfsys::field_to_string_montgomery::<Fq>(&g1.z),
        }
    }
}

impl From<PyG1> for G1 {
    fn from(val: PyG1) -> Self {
        G1 {
            x: crate::pfsys::string_to_field_montgomery::<Fq>(&val.x),
            y: crate::pfsys::string_to_field_montgomery::<Fq>(&val.y),
            z: crate::pfsys::string_to_field_montgomery::<Fq>(&val.z),
        }
    }
}

impl pyo3::ToPyObject for PyG1 {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let g1_dict = pyo3::types::PyDict::new(py);

        g1_dict.set_item("x", self.x.to_object(py)).unwrap();
        g1_dict.set_item("y", self.y.to_object(py)).unwrap();
        g1_dict.set_item("z", self.z.to_object(py)).unwrap();
        g1_dict.into()
    }
}

/// pyclass containing the struct used for G1
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyG1Affine {
    #[pyo3(get, set)]
    ///
    pub x: PyFelt,
    #[pyo3(get, set)]
    ///
    pub y: PyFelt,
}

impl From<G1Affine> for PyG1Affine {
    fn from(g1: G1Affine) -> Self {
        PyG1Affine {
            x: crate::pfsys::field_to_string_montgomery::<Fq>(&g1.x),
            y: crate::pfsys::field_to_string_montgomery::<Fq>(&g1.y),
        }
    }
}

impl From<PyG1Affine> for G1Affine {
    fn from(val: PyG1Affine) -> Self {
        G1Affine {
            x: crate::pfsys::string_to_field_montgomery::<Fq>(&val.x),
            y: crate::pfsys::string_to_field_montgomery::<Fq>(&val.y),
        }
    }
}

impl pyo3::ToPyObject for PyG1Affine {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let g1_dict = pyo3::types::PyDict::new(py);

        g1_dict.set_item("x", self.x.to_object(py)).unwrap();
        g1_dict.set_item("y", self.y.to_object(py)).unwrap();
        g1_dict.into()
    }
}

/// pyclass containing the struct used for run_args
#[pyclass]
#[derive(Clone)]
struct PyRunArgs {
    #[pyo3(get, set)]
    pub tolerance: f32,
    #[pyo3(get, set)]
    pub input_scale: crate::Scale,
    #[pyo3(get, set)]
    pub param_scale: crate::Scale,
    #[pyo3(get, set)]
    pub scale_rebase_multiplier: u32,
    #[pyo3(get, set)]
    pub lookup_range: (i128, i128),
    #[pyo3(get, set)]
    pub logrows: u32,
    #[pyo3(get, set)]
    pub num_inner_cols: usize,
    #[pyo3(get, set)]
    pub input_visibility: Visibility,
    #[pyo3(get, set)]
    pub output_visibility: Visibility,
    #[pyo3(get, set)]
    pub param_visibility: Visibility,
    #[pyo3(get, set)]
    pub variables: Vec<(String, usize)>,
}

/// default instantiation of PyRunArgs
#[pymethods]
impl PyRunArgs {
    #[new]
    fn new() -> Self {
        RunArgs::default().into()
    }
}

/// Conversion between PyRunArgs and RunArgs
impl From<PyRunArgs> for RunArgs {
    fn from(py_run_args: PyRunArgs) -> Self {
        RunArgs {
            tolerance: Tolerance::from(py_run_args.tolerance),
            input_scale: py_run_args.input_scale,
            param_scale: py_run_args.param_scale,
            num_inner_cols: py_run_args.num_inner_cols,
            scale_rebase_multiplier: py_run_args.scale_rebase_multiplier,
            lookup_range: py_run_args.lookup_range,
            logrows: py_run_args.logrows,
            input_visibility: py_run_args.input_visibility,
            output_visibility: py_run_args.output_visibility,
            param_visibility: py_run_args.param_visibility,
            variables: py_run_args.variables,
        }
    }
}

impl Into<PyRunArgs> for RunArgs {
    fn into(self) -> PyRunArgs {
        PyRunArgs {
            tolerance: self.tolerance.val,
            input_scale: self.input_scale,
            param_scale: self.param_scale,
            num_inner_cols: self.num_inner_cols,
            scale_rebase_multiplier: self.scale_rebase_multiplier,
            lookup_range: self.lookup_range,
            logrows: self.logrows,
            input_visibility: self.input_visibility,
            output_visibility: self.output_visibility,
            param_visibility: self.param_visibility,
            variables: self.variables,
        }
    }
}

/// Converts 4 u64s to a field element
#[pyfunction(signature = (
    array,
))]
fn string_to_felt(array: PyFelt) -> PyResult<String> {
    Ok(format!(
        "{:?}",
        crate::pfsys::string_to_field_montgomery::<Fr>(&array)
    ))
}

/// Converts 4 u64s representing a field element directly to an integer
#[pyfunction(signature = (
    array,
))]
fn string_to_int(array: PyFelt) -> PyResult<i128> {
    let felt = crate::pfsys::string_to_field_montgomery::<Fr>(&array);
    let int_rep = felt_to_i128(felt);
    Ok(int_rep)
}

/// Converts 4 u64s representing a field element directly to a (rescaled from fixed point scaling) floating point
#[pyfunction(signature = (
    array,
    scale
))]
fn string_to_float(array: PyFelt, scale: crate::Scale) -> PyResult<f64> {
    let felt = crate::pfsys::string_to_field_montgomery::<Fr>(&array);
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
fn float_to_string(input: f64, scale: crate::Scale) -> PyResult<PyFelt> {
    let int_rep = quantize_float(&input, 0.0, scale)
        .map_err(|_| PyIOError::new_err("Failed to quantize input"))?;
    let felt = i128_to_felt(int_rep);
    Ok(crate::pfsys::field_to_string_montgomery::<Fr>(&felt))
}

/// Converts a buffer to vector of 4 u64s representing a fixed point field element
#[pyfunction(signature = (
    buffer
    ))]
fn buffer_to_felts(buffer: Vec<u8>) -> PyResult<Vec<String>> {
    fn u8_array_to_u128_le(arr: [u8; 16]) -> u128 {
        let mut n: u128 = 0;
        for &b in arr.iter().rev() {
            n <<= 8;
            n |= b as u128;
        }
        n
    }

    let buffer = &buffer[..];

    // Divide the buffer into chunks of 64 bytes
    let chunks = buffer.chunks_exact(16);

    // Get the remainder
    let remainder = chunks.remainder();

    // Add 0s to the remainder to make it 64 bytes
    let mut remainder = remainder.to_vec();

    // Collect chunks into a Vec<[u8; 16]>.
    let chunks: Result<Vec<[u8; 16]>, PyErr> = chunks
        .map(|slice| {
            let array: [u8; 16] = slice
                .try_into()
                .map_err(|_| PyIOError::new_err("Failed to slice input buffer"))?;
            Ok(array)
        })
        .collect();

    let mut chunks = chunks?;

    if !remainder.is_empty() {
        remainder.resize(16, 0);
        // Convert the Vec<u8> to [u8; 16]
        let remainder_array: [u8; 16] = remainder
            .try_into()
            .map_err(|_| PyIOError::new_err("Failed to slice remainder"))?;
        // append the remainder to the chunks
        chunks.push(remainder_array);
    }

    // Convert each chunk to a field element
    let field_elements: Vec<Fr> = chunks
        .iter()
        .map(|x| PrimeField::from_u128(u8_array_to_u128_le(*x)))
        .collect();

    let field_elements: Vec<String> = field_elements.iter().map(|x| format!("{:?}", x)).collect();

    Ok(field_elements)
}

/// Generate a poseidon hash.
#[pyfunction(signature = (
    message,
    ))]
fn poseidon_hash(message: Vec<PyFelt>) -> PyResult<Vec<PyFelt>> {
    let message: Vec<Fr> = message
        .iter()
        .map(crate::pfsys::string_to_field_montgomery::<Fr>)
        .collect::<Vec<_>>();

    let output =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
            message.clone(),
        )
        .map_err(|_| PyIOError::new_err("Failed to run poseidon"))?;

    let hash = output[0]
        .iter()
        .map(crate::pfsys::field_to_string_montgomery::<Fr>)
        .collect::<Vec<_>>();
    Ok(hash)
}

/// Generate a kzg commitment.
#[pyfunction(signature = (
    message,
    vk_path=PathBuf::from(DEFAULT_VK),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    srs_path=None
    ))]
fn kzg_commit(
    message: Vec<PyFelt>,
    vk_path: PathBuf,
    settings_path: PathBuf,
    srs_path: Option<PathBuf>,
) -> PyResult<Vec<PyG1Affine>> {
    let message: Vec<Fr> = message
        .iter()
        .map(crate::pfsys::string_to_field_montgomery::<Fr>)
        .collect::<Vec<_>>();

    let settings = GraphSettings::load(&settings_path)
        .map_err(|_| PyIOError::new_err("Failed to load circuit settings"))?;

    let srs_path = crate::execute::get_srs_path(settings.run_args.logrows, srs_path);

    let srs = load_srs::<KZGCommitmentScheme<Bn256>>(srs_path)
        .map_err(|_| PyIOError::new_err("Failed to load srs"))?;

    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk_path, settings)
        .map_err(|_| PyIOError::new_err("Failed to load vk"))?;

    let output = KZGChip::commit(
        message,
        vk.cs().degree() as u32,
        (vk.cs().blinding_factors() + 1) as u32,
        &srs,
    );

    Ok(output.iter().map(|x| (*x).into()).collect::<Vec<_>>())
}

/// Swap the commitments in a proof
#[pyfunction(signature = (
    proof_path=PathBuf::from(DEFAULT_PROOF),
    witness_path=PathBuf::from(DEFAULT_WITNESS),
    ))]
fn swap_proof_commitments(proof_path: PathBuf, witness_path: PathBuf) -> PyResult<()> {
    crate::execute::swap_proof_commitments(proof_path, witness_path)
        .map_err(|_| PyIOError::new_err("Failed to swap commitments"))?;

    Ok(())
}

/// Generates a vk from a pk for a model circuit and saves it to a file
#[pyfunction(signature = (
    path_to_pk=PathBuf::from(DEFAULT_PK),
    circuit_settings_path=PathBuf::from(DEFAULT_SETTINGS),
    vk_output_path=PathBuf::from(DEFAULT_VK),
    ))]
fn gen_vk_from_pk_single(
    path_to_pk: PathBuf,
    circuit_settings_path: PathBuf,
    vk_output_path: PathBuf,
) -> PyResult<bool> {
    let settings = GraphSettings::load(&circuit_settings_path)
        .map_err(|_| PyIOError::new_err("Failed to load circuit settings"))?;

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(path_to_pk, settings)
        .map_err(|_| PyIOError::new_err("Failed to load pk"))?;

    let vk = pk.get_vk();

    // now save
    save_vk::<KZGCommitmentScheme<Bn256>>(&vk_output_path, vk)
        .map_err(|_| PyIOError::new_err("Failed to save vk"))?;

    Ok(true)
}

/// Generates a vk from a pk for an aggregate circuit and saves it to a file
#[pyfunction(signature = (
    path_to_pk=PathBuf::from(DEFAULT_PK_AGGREGATED),
    vk_output_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
    ))]
fn gen_vk_from_pk_aggr(path_to_pk: PathBuf, vk_output_path: PathBuf) -> PyResult<bool> {
    let pk = load_pk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(path_to_pk, ())
        .map_err(|_| PyIOError::new_err("Failed to load pk"))?;

    let vk = pk.get_vk();

    // now save
    save_vk::<KZGCommitmentScheme<Bn256>>(&vk_output_path, vk)
        .map_err(|_| PyIOError::new_err("Failed to save vk"))?;

    Ok(true)
}

/// Displays the table as a string in python
#[pyfunction(signature = (
    model = PathBuf::from(DEFAULT_MODEL),
    py_run_args = None
))]
fn table(model: PathBuf, py_run_args: Option<PyRunArgs>) -> PyResult<String> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let mut reader = File::open(model).map_err(|_| PyIOError::new_err("Failed to open model"))?;
    let result = Model::new(&mut reader, &run_args);

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
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    logrows=None,
    srs_path=None
))]
fn get_srs(
    settings_path: Option<PathBuf>,
    logrows: Option<u32>,
    srs_path: Option<PathBuf>,
) -> PyResult<bool> {
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
    model=PathBuf::from(DEFAULT_MODEL),
    output=PathBuf::from(DEFAULT_SETTINGS),
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
    data = PathBuf::from(DEFAULT_CALIBRATION_FILE),
    model = PathBuf::from(DEFAULT_MODEL),
    settings = PathBuf::from(DEFAULT_SETTINGS),
    target = CalibrationTarget::default(), // default is "resources
    lookup_safety_margin = DEFAULT_LOOKUP_SAFETY_MARGIN.parse().unwrap(),
    scales = None,
    scale_rebase_multiplier = DEFAULT_SCALE_REBASE_MULTIPLIERS.split(",").map(|x| x.parse().unwrap()).collect(),
    max_logrows = None,
))]
fn calibrate_settings(
    data: PathBuf,
    model: PathBuf,
    settings: PathBuf,
    target: CalibrationTarget,
    lookup_safety_margin: i128,
    scales: Option<Vec<crate::Scale>>,
    scale_rebase_multiplier: Vec<u32>,
    max_logrows: Option<u32>,
) -> Result<bool, PyErr> {
    crate::execute::calibrate(
        model,
        data,
        settings,
        target,
        lookup_safety_margin,
        scales,
        scale_rebase_multiplier,
        max_logrows,
    )
    .map_err(|e| {
        let err_str = format!("Failed to calibrate settings: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// runs the forward pass operation
#[pyfunction(signature = (
    data=PathBuf::from(DEFAULT_DATA),
    model=PathBuf::from(DEFAULT_MODEL),
    output=PathBuf::from(DEFAULT_WITNESS),
    vk_path=None,
    srs_path=None,
))]
fn gen_witness(
    data: PathBuf,
    model: PathBuf,
    output: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
) -> PyResult<PyObject> {
    let output = Runtime::new()
        .unwrap()
        .block_on(crate::execute::gen_witness(
            model, data, output, vk_path, srs_path,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run generate witness: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;
    Python::with_gil(|py| Ok(output.to_object(py)))
}

/// mocks the prover
#[pyfunction(signature = (
    witness=PathBuf::from(DEFAULT_WITNESS),
    model=PathBuf::from(DEFAULT_MODEL),
))]
fn mock(witness: PathBuf, model: PathBuf) -> PyResult<bool> {
    crate::execute::mock(model, witness).map_err(|e| {
        let err_str = format!("Failed to run mock: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;
    Ok(true)
}

/// mocks the aggregate prover
#[pyfunction(signature = (
    aggregation_snarks=vec![PathBuf::from(DEFAULT_PROOF)],
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    split_proofs = false,
))]
fn mock_aggregate(
    aggregation_snarks: Vec<PathBuf>,
    logrows: u32,
    split_proofs: bool,
) -> PyResult<bool> {
    crate::execute::mock_aggregate(aggregation_snarks, logrows, split_proofs).map_err(|e| {
        let err_str = format!("Failed to run mock: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    model=PathBuf::from(DEFAULT_MODEL),
    vk_path=PathBuf::from(DEFAULT_VK),
    pk_path=PathBuf::from(DEFAULT_PK),
    srs_path=None,
    witness_path = None,
    compress_selectors=DEFAULT_COMPRESS_SELECTORS.parse().unwrap(),
))]
fn setup(
    model: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    srs_path: Option<PathBuf>,
    witness_path: Option<PathBuf>,
    compress_selectors: bool,
) -> Result<bool, PyErr> {
    crate::execute::setup(
        model,
        srs_path,
        vk_path,
        pk_path,
        witness_path,
        compress_selectors,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run setup: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    witness=PathBuf::from(DEFAULT_WITNESS),
    model=PathBuf::from(DEFAULT_MODEL),
    pk_path=PathBuf::from(DEFAULT_PK),
    proof_path=None,
    proof_type=ProofType::default(),
    srs_path=None,
))]
fn prove(
    witness: PathBuf,
    model: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    proof_type: ProofType,
    srs_path: Option<PathBuf>,
) -> PyResult<PyObject> {
    let snark = crate::execute::prove(
        witness,
        model,
        pk_path,
        proof_path,
        srs_path,
        proof_type,
        CheckMode::UNSAFE,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run prove: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Python::with_gil(|py| Ok(snark.to_object(py)))
}

/// verifies a given proof
#[pyfunction(signature = (
    proof_path=PathBuf::from(DEFAULT_PROOF),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    vk_path=PathBuf::from(DEFAULT_VK),
    srs_path=None,
))]
fn verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
) -> Result<bool, PyErr> {
    crate::execute::verify(proof_path, settings_path, vk_path, srs_path).map_err(|e| {
        let err_str = format!("Failed to run verify: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

#[pyfunction(signature = (
    sample_snarks=vec![PathBuf::from(DEFAULT_PROOF)],
    vk_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
    pk_path=PathBuf::from(DEFAULT_PK_AGGREGATED),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    split_proofs = false,
    srs_path = None,
    compress_selectors=DEFAULT_COMPRESS_SELECTORS.parse().unwrap(),
))]
fn setup_aggregate(
    sample_snarks: Vec<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    logrows: u32,
    split_proofs: bool,
    srs_path: Option<PathBuf>,
    compress_selectors: bool,
) -> Result<bool, PyErr> {
    crate::execute::setup_aggregate(
        sample_snarks,
        vk_path,
        pk_path,
        srs_path,
        logrows,
        split_proofs,
        compress_selectors,
    )
    .map_err(|e| {
        let err_str = format!("Failed to setup aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

#[pyfunction(signature = (
    model=PathBuf::from(DEFAULT_MODEL),
    compiled_circuit=PathBuf::from(DEFAULT_COMPILED_CIRCUIT),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
))]
fn compile_circuit(
    model: PathBuf,
    compiled_circuit: PathBuf,
    settings_path: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::compile_circuit(model, compiled_circuit, settings_path).map_err(|e| {
        let err_str = format!("Failed to setup aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// creates an aggregated proof
#[pyfunction(signature = (
    aggregation_snarks=vec![PathBuf::from(DEFAULT_PROOF)],
    proof_path=PathBuf::from(DEFAULT_PROOF_AGGREGATED),
    vk_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
    transcript=TranscriptType::default(),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    check_mode=CheckMode::UNSAFE,
    split_proofs = false,
    srs_path=None,
))]
fn aggregate(
    aggregation_snarks: Vec<PathBuf>,
    proof_path: PathBuf,
    vk_path: PathBuf,
    transcript: TranscriptType,
    logrows: u32,
    check_mode: CheckMode,
    split_proofs: bool,
    srs_path: Option<PathBuf>,
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
        split_proofs,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// verifies and aggregate proof
#[pyfunction(signature = (
    proof_path=PathBuf::from(DEFAULT_PROOF_AGGREGATED),
    vk_path=PathBuf::from(DEFAULT_VK),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    srs_path=None,
))]
fn verify_aggr(
    proof_path: PathBuf,
    vk_path: PathBuf,
    logrows: u32,
    srs_path: Option<PathBuf>,
) -> Result<bool, PyErr> {
    crate::execute::verify_aggr(proof_path, vk_path, srs_path, logrows).map_err(|e| {
        let err_str = format!("Failed to run verify_aggr: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// creates an EVM compatible verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    vk_path=PathBuf::from(DEFAULT_VK),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE),
    abi_path=PathBuf::from(DEFAULT_VERIFIER_ABI),
    srs_path=None,
    render_vk_seperately = DEFAULT_RENDER_VK_SEPERATELY.parse().unwrap(),
))]
fn create_evm_verifier(
    vk_path: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    srs_path: Option<PathBuf>,
    render_vk_seperately: bool,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_verifier(
        vk_path,
        srs_path,
        settings_path,
        sol_code_path,
        abi_path,
        render_vk_seperately,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run create_evm_verifier: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

// creates an EVM compatible data attestation verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    input_data=PathBuf::from(DEFAULT_DATA),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE_DA),
    abi_path=PathBuf::from(DEFAULT_VERIFIER_DA_ABI),
))]
fn create_evm_data_attestation(
    input_data: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_data_attestation(settings_path, sol_code_path, abi_path, input_data)
        .map_err(|e| {
            let err_str = format!("Failed to run create_evm_data_attestation: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

#[pyfunction(signature = (
    data_path,
    compiled_circuit_path,
    test_data,
    input_source,
    output_source,
    rpc_url=None,
))]
fn setup_test_evm_witness(
    data_path: PathBuf,
    compiled_circuit_path: PathBuf,
    test_data: PathBuf,
    input_source: PyTestDataSource,
    output_source: PyTestDataSource,
    rpc_url: Option<String>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::setup_test_evm_witness(
            data_path,
            compiled_circuit_path,
            test_data,
            rpc_url,
            input_source.into(),
            output_source.into(),
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run setup_test_evm_witness: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

#[pyfunction(signature = (
    addr_path,
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE),
    rpc_url=None,
    optimizer_runs=DEFAULT_OPTIMIZER_RUNS.parse().unwrap(),
    private_key=None,
))]
fn deploy_evm(
    addr_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: usize,
    private_key: Option<String>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::deploy_evm(
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            "Halo2Verifier",
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

#[pyfunction(signature = (
    addr_path,
    sol_code_path=PathBuf::from(DEFAULT_VK_SOL),
    rpc_url=None,
    optimizer_runs=DEFAULT_OPTIMIZER_RUNS.parse().unwrap(),
    private_key=None,
))]
fn deploy_vk_evm(
    addr_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: usize,
    private_key: Option<String>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::deploy_evm(
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            "Halo2VerifyingKey",
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
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE_DA),
    rpc_url=None,
    optimizer_runs=DEFAULT_OPTIMIZER_RUNS.parse().unwrap(),
    private_key=None
))]
fn deploy_da_evm(
    addr_path: PathBuf,
    input_data: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: usize,
    private_key: Option<String>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::deploy_da_evm(
            input_data,
            settings_path,
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_da_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}
/// verifies an evm compatible proof, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    addr_verifier,
    proof_path=PathBuf::from(DEFAULT_PROOF),
    rpc_url=None,
    addr_da = None,
    addr_vk = None,
))]
fn verify_evm(
    addr_verifier: &str,
    proof_path: PathBuf,
    rpc_url: Option<String>,
    addr_da: Option<&str>,
    addr_vk: Option<&str>,
) -> Result<bool, PyErr> {
    let addr_verifier = H160::from_str(addr_verifier).map_err(|e| {
        let err_str = format!("address is invalid: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;
    let addr_da = if let Some(addr_da) = addr_da {
        let addr_da = H160::from_str(addr_da).map_err(|e| {
            let err_str = format!("address is invalid: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;
        Some(addr_da)
    } else {
        None
    };
    let addr_vk = if let Some(addr_vk) = addr_vk {
        let addr_vk = H160::from_str(addr_vk).map_err(|e| {
            let err_str = format!("address is invalid: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;
        Some(addr_vk)
    } else {
        None
    };

    Runtime::new()
        .unwrap()
        .block_on(crate::execute::verify_evm(
            proof_path,
            addr_verifier,
            rpc_url,
            addr_da,
            addr_vk,
        ))
        .map_err(|e| {
            let err_str = format!("Failed to run verify_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

    Ok(true)
}

/// creates an evm compatible aggregate verifier, you will need solc installed in your environment to run this
#[pyfunction(signature = (
    aggregation_settings=vec![PathBuf::from(DEFAULT_PROOF)],
    vk_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE),
    abi_path=PathBuf::from(DEFAULT_VERIFIER_ABI),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    srs_path=None,
    render_vk_seperately = DEFAULT_RENDER_VK_SEPERATELY.parse().unwrap(),
))]
fn create_evm_verifier_aggr(
    aggregation_settings: Vec<PathBuf>,
    vk_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    logrows: u32,
    srs_path: Option<PathBuf>,
    render_vk_seperately: bool,
) -> Result<bool, PyErr> {
    crate::execute::create_evm_aggregate_verifier(
        vk_path,
        srs_path,
        sol_code_path,
        abi_path,
        aggregation_settings,
        logrows,
        render_vk_seperately,
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

    let hex_str = hex::encode(proof.proof);
    Ok(format!("0x{}", hex_str))
}

// Python Module
#[pymodule]
fn ezkl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PyRunArgs>()?;
    m.add_class::<PyG1Affine>()?;
    m.add_class::<PyG1>()?;
    m.add_class::<PyTestDataSource>()?;
    m.add_function(wrap_pyfunction!(string_to_felt, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(string_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(kzg_commit, m)?)?;
    m.add_function(wrap_pyfunction!(swap_proof_commitments, m)?)?;
    m.add_function(wrap_pyfunction!(poseidon_hash, m)?)?;
    m.add_function(wrap_pyfunction!(float_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(buffer_to_felts, m)?)?;
    m.add_function(wrap_pyfunction!(gen_vk_from_pk_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(gen_vk_from_pk_single, m)?)?;
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
    m.add_function(wrap_pyfunction!(compile_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(verify_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_evm, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_vk_evm, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_da_evm, m)?)?;
    m.add_function(wrap_pyfunction!(verify_evm, m)?)?;
    m.add_function(wrap_pyfunction!(print_proof_hex, m)?)?;
    m.add_function(wrap_pyfunction!(setup_test_evm_witness, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_data_attestation, m)?)?;

    Ok(())
}
