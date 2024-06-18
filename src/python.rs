use crate::circuit::modules::polycommit::PolyCommitChip;
use crate::circuit::modules::poseidon::{
    spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
    PoseidonChip,
};
use crate::circuit::modules::Module;
use crate::circuit::{CheckMode, Tolerance};
use crate::commands::*;
use crate::fieldutils::{felt_to_i64, i64_to_felt};
use crate::graph::modules::POSEIDON_LEN_GRAPH;
use crate::graph::TestDataSource;
use crate::graph::{
    quantize_float, scale_to_multiplier, GraphCircuit, GraphSettings, Model, Visibility,
};
use crate::pfsys::evm::aggregation_kzg::AggregationCircuit;
use crate::pfsys::{
    load_pk, load_vk, save_params, save_vk, srs::gen_srs as ezkl_gen_srs, srs::load_srs_prover,
    ProofType, TranscriptType,
};
use crate::Commitments;
use crate::RunArgs;
use halo2_proofs::poly::ipa::commitment::IPACommitmentScheme;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine, G1};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use snark_verifier::util::arithmetic::PrimeField;
use std::str::FromStr;
use std::{fs::File, path::PathBuf};

type PyFelt = String;

/// pyclass representing an enum
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

/// pyclass containing the struct used for G1, this is mostly a helper class
#[pyclass]
#[derive(Debug, Clone)]
struct PyG1 {
    #[pyo3(get, set)]
    /// Field Element representing x
    x: PyFelt,
    #[pyo3(get, set)]
    /// Field Element representing y
    y: PyFelt,
    /// Field Element representing y
    #[pyo3(get, set)]
    z: PyFelt,
}

impl From<G1> for PyG1 {
    fn from(g1: G1) -> Self {
        PyG1 {
            x: crate::pfsys::field_to_string::<Fq>(&g1.x),
            y: crate::pfsys::field_to_string::<Fq>(&g1.y),
            z: crate::pfsys::field_to_string::<Fq>(&g1.z),
        }
    }
}

impl From<PyG1> for G1 {
    fn from(val: PyG1) -> Self {
        G1 {
            x: crate::pfsys::string_to_field::<Fq>(&val.x),
            y: crate::pfsys::string_to_field::<Fq>(&val.y),
            z: crate::pfsys::string_to_field::<Fq>(&val.z),
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
            x: crate::pfsys::field_to_string::<Fq>(&g1.x),
            y: crate::pfsys::field_to_string::<Fq>(&g1.y),
        }
    }
}

impl From<PyG1Affine> for G1Affine {
    fn from(val: PyG1Affine) -> Self {
        G1Affine {
            x: crate::pfsys::string_to_field::<Fq>(&val.x),
            y: crate::pfsys::string_to_field::<Fq>(&val.y),
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

/// Python class containing the struct used for run_args
///
/// Returns
/// -------
/// PyRunArgs
///
#[pyclass]
#[derive(Clone)]
struct PyRunArgs {
    #[pyo3(get, set)]
    /// float: The tolerance for error on model outputs
    pub tolerance: f32,
    #[pyo3(get, set)]
    /// int: The denominator in the fixed point representation used when quantizing inputs
    pub input_scale: crate::Scale,
    #[pyo3(get, set)]
    /// int:  The denominator in the fixed point representation used when quantizing parameters
    pub param_scale: crate::Scale,
    #[pyo3(get, set)]
    /// int: If the scale is ever > scale_rebase_multiplier * input_scale then the scale is rebased to input_scale (this a more advanced parameter, use with caution)
    pub scale_rebase_multiplier: u32,
    #[pyo3(get, set)]
    /// list[int]: The min and max elements in the lookup table input column
    pub lookup_range: crate::circuit::table::Range,
    #[pyo3(get, set)]
    /// int: The log_2 number of rows
    pub logrows: u32,
    #[pyo3(get, set)]
    /// int: The number of inner columns used for the lookup table
    pub num_inner_cols: usize,
    #[pyo3(get, set)]
    /// string: accepts `public`, `private`, `fixed`, `hashed/public`, `hashed/private`, `polycommit`
    pub input_visibility: Visibility,
    #[pyo3(get, set)]
    /// string: accepts `public`, `private`, `fixed`, `hashed/public`, `hashed/private`, `polycommit`
    pub output_visibility: Visibility,
    #[pyo3(get, set)]
    /// string: accepts `public`, `private`, `fixed`, `hashed/public`, `hashed/private`, `polycommit`
    pub param_visibility: Visibility,
    #[pyo3(get, set)]
    /// list[tuple[str, int]]: Hand-written parser for graph variables, eg. batch_size=1
    pub variables: Vec<(String, usize)>,
    #[pyo3(get, set)]
    /// bool: Rebase the scale using lookup table for division instead of using a range check
    pub div_rebasing: bool,
    #[pyo3(get, set)]
    /// bool: Should constants with 0.0 fraction be rebased to scale 0
    pub rebase_frac_zero_constants: bool,
    #[pyo3(get, set)]
    /// str: check mode, accepts `safe`, `unsafe`
    pub check_mode: CheckMode,
    #[pyo3(get, set)]
    /// str: commitment type, accepts `kzg`, `ipa`
    pub commitment: PyCommitments,
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
            div_rebasing: py_run_args.div_rebasing,
            rebase_frac_zero_constants: py_run_args.rebase_frac_zero_constants,
            check_mode: py_run_args.check_mode,
            commitment: Some(py_run_args.commitment.into()),
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
            div_rebasing: self.div_rebasing,
            rebase_frac_zero_constants: self.rebase_frac_zero_constants,
            check_mode: self.check_mode,
            commitment: self.commitment.into(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
/// pyclass representing an enum, denoting the type of commitment
pub enum PyCommitments {
    /// KZG commitment
    KZG,
    /// IPA commitment
    IPA,
}

impl From<Option<Commitments>> for PyCommitments {
    fn from(commitment: Option<Commitments>) -> Self {
        match commitment {
            Some(Commitments::KZG) => PyCommitments::KZG,
            Some(Commitments::IPA) => PyCommitments::IPA,
            None => PyCommitments::KZG,
        }
    }
}

impl From<PyCommitments> for Commitments {
    fn from(py_commitments: PyCommitments) -> Self {
        match py_commitments {
            PyCommitments::KZG => Commitments::KZG,
            PyCommitments::IPA => Commitments::IPA,
        }
    }
}

impl Into<PyCommitments> for Commitments {
    fn into(self) -> PyCommitments {
        match self {
            Commitments::KZG => PyCommitments::KZG,
            Commitments::IPA => PyCommitments::IPA,
        }
    }
}

impl FromStr for PyCommitments {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "kzg" => Ok(PyCommitments::KZG),
            "ipa" => Ok(PyCommitments::IPA),
            _ => Err("Invalid value for Commitments".to_string()),
        }
    }
}

/// Converts a field element hex string to big endian
///
/// Arguments
/// -------
/// felt: str
///     The field element represented as a string
///
///
/// Returns
/// -------
/// str
///     field element represented as a string
///
#[pyfunction(signature = (
    felt,
))]
fn felt_to_big_endian(felt: PyFelt) -> PyResult<String> {
    let felt = crate::pfsys::string_to_field::<Fr>(&felt);
    Ok(format!("{:?}", felt))
}

/// Converts a field element hex string to an integer
///
/// Arguments
/// -------
/// felt: str
///     The field element represented as a string
///
/// Returns
/// -------
/// int
///
#[pyfunction(signature = (
    felt,
))]
fn felt_to_int(felt: PyFelt) -> PyResult<i64> {
    let felt = crate::pfsys::string_to_field::<Fr>(&felt);
    let int_rep = felt_to_i64(felt);
    Ok(int_rep)
}

/// Converts a field element hex string to a floating point number
///
/// Arguments
/// -------
/// felt: str
///    The field element represented as a string
///
/// scale: float
///     The scaling factor used to convert the field element into a floating point representation
///
/// Returns
/// -------
/// float
///
#[pyfunction(signature = (
    felt,
    scale
))]
fn felt_to_float(felt: PyFelt, scale: crate::Scale) -> PyResult<f64> {
    let felt = crate::pfsys::string_to_field::<Fr>(&felt);
    let int_rep = felt_to_i64(felt);
    let multiplier = scale_to_multiplier(scale);
    let float_rep = int_rep as f64 / multiplier;
    Ok(float_rep)
}

/// Converts a floating point element to a field element hex string
///
/// Arguments
/// -------
/// input: float
///    The field element represented as a string
///
/// scale: float
///     The scaling factor used to quantize the float into a field element
///
/// Returns
/// -------
/// str
///     The field element represented as a string
///
#[pyfunction(signature = (
    input,
    scale
))]
fn float_to_felt(input: f64, scale: crate::Scale) -> PyResult<PyFelt> {
    let int_rep = quantize_float(&input, 0.0, scale)
        .map_err(|_| PyIOError::new_err("Failed to quantize input"))?;
    let felt = i64_to_felt(int_rep);
    Ok(crate::pfsys::field_to_string::<Fr>(&felt))
}

/// Converts a buffer to vector of field elements
///
/// Arguments
/// -------
/// buffer: list[int]
///     List of integers representing a buffer
///
/// Returns
/// -------
/// list[str]
///     List of field elements represented as strings
///
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

    let field_elements: Vec<String> = field_elements
        .iter()
        .map(|x| crate::pfsys::field_to_string::<Fr>(x))
        .collect();

    Ok(field_elements)
}

/// Generate a poseidon hash.
///
/// Arguments
/// -------
/// message: list[str]
///     List of field elements represented as strings
///
/// Returns
/// -------
/// list[str]
///     List of field elements represented as strings
///
#[pyfunction(signature = (
    message,
))]
fn poseidon_hash(message: Vec<PyFelt>) -> PyResult<Vec<PyFelt>> {
    let message: Vec<Fr> = message
        .iter()
        .map(crate::pfsys::string_to_field::<Fr>)
        .collect::<Vec<_>>();

    let output =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
            message.clone(),
        )
        .map_err(|_| PyIOError::new_err("Failed to run poseidon"))?;

    let hash = output[0]
        .iter()
        .map(crate::pfsys::field_to_string::<Fr>)
        .collect::<Vec<_>>();
    Ok(hash)
}

/// Generate a kzg commitment.
///
/// Arguments
/// -------
/// message: list[str]
///     List of field elements represnted as strings
///
/// vk_path: str
///     Path to the verification key
///
/// settings_path: str
///     Path to the settings file
///
/// srs_path: str
///     Path to the Structure Reference String (SRS) file
///
/// Returns
/// -------
/// list[PyG1Affine]
///
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
        .map(crate::pfsys::string_to_field::<Fr>)
        .collect::<Vec<_>>();

    let settings = GraphSettings::load(&settings_path)
        .map_err(|_| PyIOError::new_err("Failed to load circuit settings"))?;

    let srs_path =
        crate::execute::get_srs_path(settings.run_args.logrows, srs_path, Commitments::KZG);

    let srs = load_srs_prover::<KZGCommitmentScheme<Bn256>>(srs_path)
        .map_err(|_| PyIOError::new_err("Failed to load srs"))?;

    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)
        .map_err(|_| PyIOError::new_err("Failed to load vk"))?;

    let output = PolyCommitChip::commit::<KZGCommitmentScheme<Bn256>>(
        message,
        (vk.cs().blinding_factors() + 1) as u32,
        &srs,
    );

    Ok(output.iter().map(|x| (*x).into()).collect::<Vec<_>>())
}

/// Generate an ipa commitment.
///
/// Arguments
/// -------
/// message: list[str]
///     List of field elements represnted as strings
///
/// vk_path: str
///     Path to the verification key
///
/// settings_path: str
///     Path to the settings file
///
/// srs_path: str
///     Path to the Structure Reference String (SRS) file
///
/// Returns
/// -------
/// list[PyG1Affine]
///
#[pyfunction(signature = (
    message,
    vk_path=PathBuf::from(DEFAULT_VK),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    srs_path=None
))]
fn ipa_commit(
    message: Vec<PyFelt>,
    vk_path: PathBuf,
    settings_path: PathBuf,
    srs_path: Option<PathBuf>,
) -> PyResult<Vec<PyG1Affine>> {
    let message: Vec<Fr> = message
        .iter()
        .map(crate::pfsys::string_to_field::<Fr>)
        .collect::<Vec<_>>();

    let settings = GraphSettings::load(&settings_path)
        .map_err(|_| PyIOError::new_err("Failed to load circuit settings"))?;

    let srs_path =
        crate::execute::get_srs_path(settings.run_args.logrows, srs_path, Commitments::KZG);

    let srs = load_srs_prover::<IPACommitmentScheme<G1Affine>>(srs_path)
        .map_err(|_| PyIOError::new_err("Failed to load srs"))?;

    let vk = load_vk::<IPACommitmentScheme<G1Affine>, GraphCircuit>(vk_path, settings)
        .map_err(|_| PyIOError::new_err("Failed to load vk"))?;

    let output = PolyCommitChip::commit::<IPACommitmentScheme<G1Affine>>(
        message,
        (vk.cs().blinding_factors() + 1) as u32,
        &srs,
    );

    Ok(output.iter().map(|x| (*x).into()).collect::<Vec<_>>())
}

/// Swap the commitments in a proof
///
/// Arguments
/// -------
/// proof_path: str
///     Path to the proof file
///
/// witness_path: str
///     Path to the witness file
///
#[pyfunction(signature = (
    proof_path=PathBuf::from(DEFAULT_PROOF),
    witness_path=PathBuf::from(DEFAULT_WITNESS),
))]
fn swap_proof_commitments(proof_path: PathBuf, witness_path: PathBuf) -> PyResult<()> {
    crate::execute::swap_proof_commitments_cmd(proof_path, witness_path)
        .map_err(|_| PyIOError::new_err("Failed to swap commitments"))?;

    Ok(())
}

/// Generates a vk from a pk for a model circuit and saves it to a file
///
/// Arguments
/// -------
/// path_to_pk: str
///     Path to the proving key
///
/// circuit_settings_path: str
///     Path to the witness file
///
/// vk_output_path: str
///     Path to create the vk file
///
/// Returns
/// -------
/// bool
///
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

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(path_to_pk, settings)
        .map_err(|_| PyIOError::new_err("Failed to load pk"))?;

    let vk = pk.get_vk();

    // now save
    save_vk::<G1Affine>(&vk_output_path, vk)
        .map_err(|_| PyIOError::new_err("Failed to save vk"))?;

    Ok(true)
}

/// Generates a vk from a pk for an aggregate circuit and saves it to a file
///
/// Arguments
/// -------
/// path_to_pk: str
///     Path to the proving key
///
/// vk_output_path: str
///     Path to create the vk file
///
/// Returns
/// -------
/// bool
#[pyfunction(signature = (
    path_to_pk=PathBuf::from(DEFAULT_PK_AGGREGATED),
    vk_output_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
))]
fn gen_vk_from_pk_aggr(path_to_pk: PathBuf, vk_output_path: PathBuf) -> PyResult<bool> {
    let pk = load_pk::<KZGCommitmentScheme<Bn256>, AggregationCircuit>(path_to_pk, ())
        .map_err(|_| PyIOError::new_err("Failed to load pk"))?;

    let vk = pk.get_vk();

    // now save
    save_vk::<G1Affine>(&vk_output_path, vk)
        .map_err(|_| PyIOError::new_err("Failed to save vk"))?;

    Ok(true)
}

/// Displays the table as a string in python
///
/// Arguments
/// ---------
/// model: str
///     Path to the onnx file
///
/// Returns
/// ---------
/// str
///     Table of the nodes in the onnx file
///
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

/// Generates the Structured Reference String (SRS), use this only for testing purposes
///
/// Arguments
/// ---------
/// srs_path: str
///     Path to the create the SRS file
///
/// logrows: int
///     The number of logrows for the SRS file
///
#[pyfunction(signature = (
    srs_path,
    logrows,
))]
fn gen_srs(srs_path: PathBuf, logrows: usize) -> PyResult<()> {
    let params = ezkl_gen_srs::<KZGCommitmentScheme<Bn256>>(logrows as u32);
    save_params::<KZGCommitmentScheme<Bn256>>(&srs_path, &params)?;
    Ok(())
}

/// Gets a public srs
///
/// Arguments
/// ---------
/// settings_path: str
///     Path to the settings file
///
/// logrows: int
///     The number of logrows for the SRS file
///
/// srs_path: str
///     Path to the create the SRS file
///
/// commitment: str
///     Specify the commitment used ("kzg", "ipa")
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    logrows=None,
    srs_path=None,
    commitment=None,
))]
fn get_srs(
    py: Python,
    settings_path: Option<PathBuf>,
    logrows: Option<u32>,
    srs_path: Option<PathBuf>,
    commitment: Option<PyCommitments>,
) -> PyResult<Bound<'_, PyAny>> {
    let commitment: Option<Commitments> = match commitment {
        Some(c) => Some(c.into()),
        None => None,
    };

    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::get_srs_cmd(srs_path, settings_path, logrows, commitment)
            .await
            .map_err(|e| {
                let err_str = format!("Failed to get srs: {}", e);
                PyRuntimeError::new_err(err_str)
            })?;

        Ok(true)
    })
}

/// Generates the circuit settings
///
/// Arguments
/// ---------
/// model: str
///     Path to the onnx file
///
/// output: str
///     Path to create the settings file
///
/// py_run_args: PyRunArgs
///     PyRunArgs object to initialize the settings
///
/// Returns
/// -------
/// bool
///
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

/// Calibrates the circuit settings
///
/// Arguments
/// ---------
/// data: str
///     Path to the calibration data
///
/// model: str
///     Path to the onnx file
///
/// settings: str
///     Path to the settings file
///
/// lookup_safety_margin: int
///      the lookup safety margin to use for calibration. if the max lookup is 2^k, then the max lookup will be 2^k * lookup_safety_margin. larger = safer but slower
///
/// scales: list[int]
///     Optional scales to specifically try for calibration
///
/// scale_rebase_multiplier: list[int]
///     Optional scale rebase multipliers to specifically try for calibration. This is the multiplier at which we divide to return to the input scale.
///
/// max_logrows: int
///     Optional max logrows to use for calibration
///
/// only_range_check_rebase: bool
///     Check ranges when rebasing
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    data = PathBuf::from(DEFAULT_CALIBRATION_FILE),
    model = PathBuf::from(DEFAULT_MODEL),
    settings = PathBuf::from(DEFAULT_SETTINGS),
    target = CalibrationTarget::default(), // default is "resources
    lookup_safety_margin = DEFAULT_LOOKUP_SAFETY_MARGIN.parse().unwrap(),
    scales = None,
    scale_rebase_multiplier = DEFAULT_SCALE_REBASE_MULTIPLIERS.split(",").map(|x| x.parse().unwrap()).collect(),
    max_logrows = None,
    only_range_check_rebase = DEFAULT_ONLY_RANGE_CHECK_REBASE.parse().unwrap(),
))]
fn calibrate_settings(
    py: Python,
    data: PathBuf,
    model: PathBuf,
    settings: PathBuf,
    target: CalibrationTarget,
    lookup_safety_margin: i64,
    scales: Option<Vec<crate::Scale>>,
    scale_rebase_multiplier: Vec<u32>,
    max_logrows: Option<u32>,
    only_range_check_rebase: bool,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::calibrate(
            model,
            data,
            settings,
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            only_range_check_rebase,
            max_logrows,
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to calibrate settings: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

/// Runs the forward pass operation to generate a witness
///
/// Arguments
/// ---------
/// data: str
///     Path to the data file
///
/// model: str
///     Path to the compiled model file
///
/// output: str
///     Path to create the witness file
///
/// vk_path: str
///     Path to the verification key
///
/// srs_path: str
///     Path to the SRS file
///
/// Returns
/// -------
/// dict
///     Python object containing the witness values
///
#[pyfunction(signature = (
    data=PathBuf::from(DEFAULT_DATA),
    model=PathBuf::from(DEFAULT_COMPILED_CIRCUIT),
    output=PathBuf::from(DEFAULT_WITNESS),
    vk_path=None,
    srs_path=None,
))]
fn gen_witness(
    py: Python,
    data: PathBuf,
    model: PathBuf,
    output: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let output = crate::execute::gen_witness(model, data, output, vk_path, srs_path)
            .await
            .map_err(|e| {
                let err_str = format!("Failed to generate witness: {}", e);
                PyRuntimeError::new_err(err_str)
            })?;
        Python::with_gil(|py| Ok(output.to_object(py)))
    })
}

/// Mocks the prover
///
/// Arguments
/// ---------
/// witness: str
///     Path to the witness file
///
/// model: str
///     Path to the compiled model file
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    witness=PathBuf::from(DEFAULT_WITNESS),
    model=PathBuf::from(DEFAULT_COMPILED_CIRCUIT),
))]
fn mock(witness: PathBuf, model: PathBuf) -> PyResult<bool> {
    crate::execute::mock(model, witness).map_err(|e| {
        let err_str = format!("Failed to run mock: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;
    Ok(true)
}

/// Mocks the aggregate prover
///
/// Arguments
/// ---------
/// aggregation_snarks: list[str]
///     List of paths to the relevant proof files
///
/// logrows: int
///     Number of logrows to use for the aggregation circuit
///
/// split_proofs: bool
///     Indicates whether the accumulated are segments of a larger proof
///
/// Returns
/// -------
/// bool
///
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

/// Runs the setup process
///
/// Arguments
/// ---------
/// model: str
///     Path to the compiled model file
///
/// vk_path: str
///     Path to create the verification key file
///
/// pk_path: str
///     Path to create the proving key file
///
/// srs_path: str
///     Path to the SRS file
///
/// witness_path: str
///     Path to the witness file
///
/// disable_selector_compression: bool
///     Whether to compress the selectors or not
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    model=PathBuf::from(DEFAULT_COMPILED_CIRCUIT),
    vk_path=PathBuf::from(DEFAULT_VK),
    pk_path=PathBuf::from(DEFAULT_PK),
    srs_path=None,
    witness_path = None,
    disable_selector_compression=DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap(),
))]
fn setup(
    model: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    srs_path: Option<PathBuf>,
    witness_path: Option<PathBuf>,
    disable_selector_compression: bool,
) -> Result<bool, PyErr> {
    crate::execute::setup(
        model,
        srs_path,
        vk_path,
        pk_path,
        witness_path,
        disable_selector_compression,
    )
    .map_err(|e| {
        let err_str = format!("Failed to run setup: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// Runs the prover on a set of inputs
///
/// Arguments
/// ---------
/// witness: str
///     Path to the witness file
///
/// model: str
///     Path to the compiled model file
///
/// pk_path: str
///     Path to the proving key file
///
/// proof_path: str
///     Path to create the proof file
///
/// proof_type: str
///     Accepts `single`, `for-aggr`
///
/// srs_path: str
///     Path to the SRS file
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    witness=PathBuf::from(DEFAULT_WITNESS),
    model=PathBuf::from(DEFAULT_COMPILED_CIRCUIT),
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

/// Verifies a given proof
///
/// Arguments
/// ---------
/// proof_path: str
///     Path to create the proof file
///
/// settings_path: str
///     Path to the settings file
///
/// vk_path: str
///     Path to the verification key file
///
/// srs_path: str
///     Path to the SRS file
///
/// non_reduced_srs: bool
///     Whether to reduce the number of SRS logrows to the number of instances rather than the number of logrows used for proofs (only works if the srs were generated in the same ceremony)
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    proof_path=PathBuf::from(DEFAULT_PROOF),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    vk_path=PathBuf::from(DEFAULT_VK),
    srs_path=None,
    reduced_srs=DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse::<bool>().unwrap(),
))]
fn verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    reduced_srs: bool,
) -> Result<bool, PyErr> {
    crate::execute::verify(proof_path, settings_path, vk_path, srs_path, reduced_srs).map_err(
        |e| {
            let err_str = format!("Failed to run verify: {}", e);
            PyRuntimeError::new_err(err_str)
        },
    )?;

    Ok(true)
}

///  Runs the setup process for an aggregate setup
///
/// Arguments
/// ---------
/// sample_snarks: list[str]
///     List of paths to the various proofs
///
/// vk_path: str
///     Path to create the aggregated VK
///
/// pk_path: str
///     Path to create the aggregated PK
///
/// logrows: int
///     Number of logrows to use
///
/// split_proofs: bool
///     Whether the accumulated are segments of a larger proof
///
/// srs_path: str
///     Path to the SRS file
///
/// disable_selector_compression: bool
///     Whether to compress selectors
///
/// commitment: str
///     Accepts `kzg`, `ipa`
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    sample_snarks=vec![PathBuf::from(DEFAULT_PROOF)],
    vk_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
    pk_path=PathBuf::from(DEFAULT_PK_AGGREGATED),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    split_proofs = false,
    srs_path = None,
    disable_selector_compression=DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap(),
    commitment=DEFAULT_COMMITMENT.parse().unwrap(),
))]
fn setup_aggregate(
    sample_snarks: Vec<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    logrows: u32,
    split_proofs: bool,
    srs_path: Option<PathBuf>,
    disable_selector_compression: bool,
    commitment: PyCommitments,
) -> Result<bool, PyErr> {
    crate::execute::setup_aggregate(
        sample_snarks,
        vk_path,
        pk_path,
        srs_path,
        logrows,
        split_proofs,
        disable_selector_compression,
        commitment.into(),
    )
    .map_err(|e| {
        let err_str = format!("Failed to setup aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// Compiles the circuit for use in other steps
///
/// Arguments
/// ---------
/// model: str
///     Path to the onnx model file
///
/// compiled_circuit: str
///     Path to output the compiled circuit
///
/// settings_path: str
///     Path to the settings files
///
/// Returns
/// -------
/// bool
///
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

/// Creates an aggregated proof
///
/// Arguments
/// ---------
/// aggregation_snarks: list[str]
///     List of paths to the various proofs
///
/// proof_path: str
///     Path to output the aggregated proof
///
/// vk_path: str
///     Path to the VK file
///
/// transcript:
///     Proof transcript type to be used. `evm` used by default. `poseidon` is also supported
///
/// logrows:
///     Logrows used for aggregation circuit
///
/// check_mode: str
///     Run sanity checks during calculations. Accepts `safe` or `unsafe`
///
/// split-proofs: bool
///      Whether the accumulated proofs are segments of a larger circuit
///
/// srs_path: str
///     Path to the SRS used
///
/// commitment: str
///     Accepts "kzg" or "ipa"
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    aggregation_snarks=vec![PathBuf::from(DEFAULT_PROOF)],
    proof_path=PathBuf::from(DEFAULT_PROOF_AGGREGATED),
    vk_path=PathBuf::from(DEFAULT_VK_AGGREGATED),
    transcript=TranscriptType::default(),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    check_mode=CheckMode::UNSAFE,
    split_proofs = false,
    srs_path=None,
    commitment=DEFAULT_COMMITMENT.parse().unwrap(),
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
    commitment: PyCommitments,
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
        commitment.into(),
    )
    .map_err(|e| {
        let err_str = format!("Failed to run aggregate: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// Verifies and aggregate proof
///
/// Arguments
/// ---------
/// proof_path: str
///      The path to the proof file
///
/// vk_path: str
///     The path to the verification key file
///
/// logrows: int
///     logrows used for aggregation circuit
///
/// commitment: str
///     Accepts "kzg" or "ipa"
///
/// reduced_srs: bool
///     Whether to reduce the number of SRS logrows to the number of instances rather than the number of logrows used for proofs (only works if the srs were generated in the same ceremony)
///
/// srs_path: str
///     The path to the SRS file
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    proof_path=PathBuf::from(DEFAULT_PROOF_AGGREGATED),
    vk_path=PathBuf::from(DEFAULT_VK),
    logrows=DEFAULT_AGGREGATED_LOGROWS.parse().unwrap(),
    commitment=DEFAULT_COMMITMENT.parse().unwrap(),
    reduced_srs=DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse().unwrap(),
    srs_path=None,
))]
fn verify_aggr(
    proof_path: PathBuf,
    vk_path: PathBuf,
    logrows: u32,
    commitment: PyCommitments,
    reduced_srs: bool,
    srs_path: Option<PathBuf>,
) -> Result<bool, PyErr> {
    crate::execute::verify_aggr(
        proof_path,
        vk_path,
        srs_path,
        logrows,
        reduced_srs,
        commitment.into(),
    )
    .map_err(|e| {
        let err_str = format!("Failed to run verify_aggr: {}", e);
        PyRuntimeError::new_err(err_str)
    })?;

    Ok(true)
}

/// Creates encoded evm calldata from a proof file
///
/// Arguments
/// ---------
/// proof: str
///     Path to the proof file
///
/// calldata: str
///    Path to the calldata file to save
///
/// addr_vk: str
///    The address of the verification key contract (if the verifier key is to be rendered as a separate contract)
///
/// Returns
/// -------
/// vec[u8]
///    The encoded calldata
///
#[pyfunction(signature = (
    proof=PathBuf::from(DEFAULT_PROOF),
    calldata=PathBuf::from(DEFAULT_CALLDATA),
    addr_vk=None,
))]
fn encode_evm_calldata<'a>(
    proof: PathBuf,
    calldata: PathBuf,
    addr_vk: Option<&'a str>,
) -> Result<Vec<u8>, PyErr> {
    let addr_vk = if let Some(addr_vk) = addr_vk {
        let addr_vk = H160Flag::from(addr_vk);
        Some(addr_vk)
    } else {
        None
    };

    crate::execute::encode_evm_calldata(proof, calldata, addr_vk).map_err(|e| {
        let err_str = format!("Failed to generate calldata: {}", e);
        PyRuntimeError::new_err(err_str)
    })
}

/// Creates an EVM compatible verifier, you will need solc installed in your environment to run this
///
/// Arguments
/// ---------
/// vk_path: str
///     The path to the verification key file
///
/// settings_path: str
///     The path to the settings file
///
/// sol_code_path: str
///     The path to the create the solidity verifier
///
/// abi_path: str
///     The path to create the ABI for the solidity verifier
///
/// srs_path: str
///     The path to the SRS file
///
/// render_vk_separately: bool
///     Whether the verifier key should be rendered as a separate contract. We recommend disabling selector compression if this is enabled. To save the verifier key as a separate contract, set this to true and then call the create-evm-vk command
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    vk_path=PathBuf::from(DEFAULT_VK),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE),
    abi_path=PathBuf::from(DEFAULT_VERIFIER_ABI),
    srs_path=None,
    render_vk_seperately = DEFAULT_RENDER_VK_SEPERATELY.parse().unwrap(),
))]
fn create_evm_verifier(
    py: Python,
    vk_path: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    srs_path: Option<PathBuf>,
    render_vk_seperately: bool,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::create_evm_verifier(
            vk_path,
            srs_path,
            settings_path,
            sol_code_path,
            abi_path,
            render_vk_seperately,
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run create_evm_verifier: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

/// Creates an Evm VK
///
/// Arguments
/// ---------
/// vk_path: str
///     The path to the verification key file
///
/// settings_path: str
///     The path to the settings file
///
/// sol_code_path: str
///     The path to the create the solidity VK
///
/// abi_path: str
///     The path to create the ABI for the solidity verifier
///
/// srs_path: str
///     The path to the SRS file
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    vk_path=PathBuf::from(DEFAULT_VK),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE),
    abi_path=PathBuf::from(DEFAULT_VERIFIER_ABI),
    srs_path=None
))]
fn create_evm_vk(
    py: Python,
    vk_path: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    srs_path: Option<PathBuf>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::create_evm_vk(vk_path, srs_path, settings_path, sol_code_path, abi_path)
            .await
            .map_err(|e| {
                let err_str = format!("Failed to run create_evm_verifier: {}", e);
                PyRuntimeError::new_err(err_str)
            })?;

        Ok(true)
    })
}

/// Creates an EVM compatible data attestation verifier, you will need solc installed in your environment to run this
///
/// Arguments
/// ---------
/// input_data: str
///     The path to the .json data file, which should contain the necessary calldata and account addresses needed to read from all the on-chain view functions that return the data that the network ingests as inputs
///
/// settings_path: str
///     The path to the settings file
///
/// sol_code_path: str
///     The path to the create the solidity verifier
///
/// abi_path: str
///     The path to create the ABI for the solidity verifier
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    input_data=PathBuf::from(DEFAULT_DATA),
    settings_path=PathBuf::from(DEFAULT_SETTINGS),
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE_DA),
    abi_path=PathBuf::from(DEFAULT_VERIFIER_DA_ABI),
    witness_path=None,
))]
fn create_evm_data_attestation(
    py: Python,
    input_data: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    witness_path: Option<PathBuf>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::create_evm_data_attestation(
            settings_path,
            sol_code_path,
            abi_path,
            input_data,
            witness_path,
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run create_evm_data_attestation: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

/// Setup test evm witness
///
/// Arguments
/// ---------
/// data_path: str
///     The path to the .json data file, which should include both the network input (possibly private) and the network output (public input to the proof)
///
/// compiled_circuit_path: str
///     The path to the compiled model file (generated using the compile-circuit command)
///
/// test_data: str
///     For testing purposes only. The optional path to the .json data file that will be generated that contains the OnChain data storage information derived from the file information in the data .json file. Should include both the network input (possibly private) and the network output (public input to the proof)
///
/// input_sources: str
///     Where the input data comes from
///
/// output_source: str
///     Where the output data comes from
///
/// rpc_url: str
///     RPC URL for an EVM compatible node, if None, uses Anvil as a local RPC node
///
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    data_path,
    compiled_circuit_path,
    test_data,
    input_source,
    output_source,
    rpc_url=None,
))]
fn setup_test_evm_witness(
    py: Python,
    data_path: PathBuf,
    compiled_circuit_path: PathBuf,
    test_data: PathBuf,
    input_source: PyTestDataSource,
    output_source: PyTestDataSource,
    rpc_url: Option<String>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::setup_test_evm_witness(
            data_path,
            compiled_circuit_path,
            test_data,
            rpc_url,
            input_source.into(),
            output_source.into(),
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run setup_test_evm_witness: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

/// deploys the solidity verifier
#[pyfunction(signature = (
    addr_path,
    sol_code_path=PathBuf::from(DEFAULT_SOL_CODE),
    rpc_url=None,
    optimizer_runs=DEFAULT_OPTIMIZER_RUNS.parse().unwrap(),
    private_key=None,
))]
fn deploy_evm(
    py: Python,
    addr_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: usize,
    private_key: Option<String>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::deploy_evm(
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            "Halo2Verifier",
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

/// deploys the solidity vk verifier
#[pyfunction(signature = (
    addr_path,
    sol_code_path=PathBuf::from(DEFAULT_VK_SOL),
    rpc_url=None,
    optimizer_runs=DEFAULT_OPTIMIZER_RUNS.parse().unwrap(),
    private_key=None,
))]
fn deploy_vk_evm(
    py: Python,
    addr_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: usize,
    private_key: Option<String>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::deploy_evm(
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            "Halo2VerifyingKey",
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

/// deploys the solidity da verifier
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
    py: Python,
    addr_path: PathBuf,
    input_data: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: usize,
    private_key: Option<String>,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::deploy_da_evm(
            input_data,
            settings_path,
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run deploy_da_evm: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}
/// verifies an evm compatible proof, you will need solc installed in your environment to run this
///
/// Arguments
/// ---------
/// addr_verifier: str
///     The verifier contract's address as a hex string
///
/// proof_path: str
///     The path to the proof file (generated using the prove command)
///
/// rpc_url: str
///     RPC URL for an Ethereum node, if None will use Anvil but WON'T persist state
///
/// addr_da: str
///     does the verifier use data attestation ?
///
/// addr_vk: str
///     The addess of the separate VK contract (if the verifier key is rendered as a separate contract)
/// Returns
/// -------
/// bool
///
#[pyfunction(signature = (
    addr_verifier,
    proof_path=PathBuf::from(DEFAULT_PROOF),
    rpc_url=None,
    addr_da = None,
    addr_vk = None,
))]
fn verify_evm<'a>(
    py: Python<'a>,
    addr_verifier: &'a str,
    proof_path: PathBuf,
    rpc_url: Option<String>,
    addr_da: Option<&'a str>,
    addr_vk: Option<&'a str>,
) -> PyResult<Bound<'a, PyAny>> {
    let addr_verifier = H160Flag::from(addr_verifier);
    let addr_da = if let Some(addr_da) = addr_da {
        let addr_da = H160Flag::from(addr_da);
        Some(addr_da)
    } else {
        None
    };
    let addr_vk = if let Some(addr_vk) = addr_vk {
        let addr_vk = H160Flag::from(addr_vk);
        Some(addr_vk)
    } else {
        None
    };

    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::verify_evm(proof_path, addr_verifier, rpc_url, addr_da, addr_vk)
            .await
            .map_err(|e| {
                let err_str = format!("Failed to run verify_evm: {}", e);
                PyRuntimeError::new_err(err_str)
            })?;

        Ok(true)
    })
}

/// Creates an evm compatible aggregate verifier, you will need solc installed in your environment to run this
///
/// Arguments
/// ---------
/// aggregation_settings: str
///     path to the settings file
///
/// vk_path: str
///     The path to load the desired verification key file
///
/// sol_code_path: str
///     The path to the Solidity code
///
/// abi_path: str
///     The path to output the Solidity verifier ABI
///
/// logrows: int
///     Number of logrows used during aggregated setup
///
/// srs_path: str
///     The path to the SRS file
///
/// render_vk_separately: bool
///     Whether the verifier key should be rendered as a separate contract. We recommend disabling selector compression if this is enabled. To save the verifier key as a separate contract, set this to true and then call the create-evm-vk command
///
/// Returns
/// -------
/// bool
///
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
    py: Python,
    aggregation_settings: Vec<PathBuf>,
    vk_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    logrows: u32,
    srs_path: Option<PathBuf>,
    render_vk_seperately: bool,
) -> PyResult<Bound<'_, PyAny>> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::create_evm_aggregate_verifier(
            vk_path,
            srs_path,
            sol_code_path,
            abi_path,
            aggregation_settings,
            logrows,
            render_vk_seperately,
        )
        .await
        .map_err(|e| {
            let err_str = format!("Failed to run create_evm_verifier_aggr: {}", e);
            PyRuntimeError::new_err(err_str)
        })?;

        Ok(true)
    })
}

// Python Module
#[pymodule]
fn ezkl(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PyRunArgs>()?;
    m.add_class::<PyG1Affine>()?;
    m.add_class::<PyG1>()?;
    m.add_class::<PyTestDataSource>()?;
    m.add_class::<PyCommitments>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(felt_to_big_endian, m)?)?;
    m.add_function(wrap_pyfunction!(felt_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(felt_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(kzg_commit, m)?)?;
    m.add_function(wrap_pyfunction!(ipa_commit, m)?)?;
    m.add_function(wrap_pyfunction!(swap_proof_commitments, m)?)?;
    m.add_function(wrap_pyfunction!(poseidon_hash, m)?)?;
    m.add_function(wrap_pyfunction!(float_to_felt, m)?)?;
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
    m.add_function(wrap_pyfunction!(create_evm_vk, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_evm, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_vk_evm, m)?)?;
    m.add_function(wrap_pyfunction!(deploy_da_evm, m)?)?;
    m.add_function(wrap_pyfunction!(verify_evm, m)?)?;
    m.add_function(wrap_pyfunction!(setup_test_evm_witness, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_data_attestation, m)?)?;
    m.add_function(wrap_pyfunction!(encode_evm_calldata, m)?)?;
    Ok(())
}
