use crate::circuit::modules::elgamal::{ElGamalCipher, ElGamalVariables};
use crate::circuit::modules::poseidon::{
    spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
    PoseidonChip,
};
use crate::circuit::modules::Module;
use crate::circuit::{CheckMode, Tolerance};
use crate::commands::{CalibrationTarget, StrategyType};
use crate::fieldutils::{felt_to_i128, i128_to_felt};
use crate::graph::modules::POSEIDON_LEN_GRAPH;
use crate::graph::{
    quantize_float, scale_to_multiplier, GraphCircuit, GraphSettings, Model, Visibility,
};
use crate::pfsys::evm::aggregation::AggregationCircuit;
use crate::pfsys::{
    load_pk, save_params, save_vk, srs::gen_srs as ezkl_gen_srs, Snark, TranscriptType,
};
use crate::RunArgs;
use ethers::types::H160;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine, G1};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use rand::rngs::StdRng;
use rand::SeedableRng;
use snark_verifier::util::arithmetic::PrimeField;
use std::str::FromStr;
use std::{fs::File, path::PathBuf};
use tokio::runtime::Runtime;

type PyFelt = [u64; 4];

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
            x: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&g1.x),
            y: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&g1.y),
            z: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&g1.z),
        }
    }
}

impl Into<G1> for PyG1 {
    fn into(self) -> G1 {
        G1 {
            x: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&self.x),
            y: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&self.y),
            z: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&self.z),
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
            x: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&g1.x),
            y: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&g1.y),
        }
    }
}

impl Into<G1Affine> for PyG1Affine {
    fn into(self) -> G1Affine {
        G1Affine {
            x: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&self.x),
            y: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&self.y),
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

/// pyclass containing the struct used for ElgamalCipher
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyElGamalCipher {
    #[pyo3(get, set)]
    ///
    c1: PyG1,
    #[pyo3(get, set)]
    ///
    c2: Vec<PyFelt>,
}

impl From<PyElGamalCipher> for ElGamalCipher {
    fn from(py_elgamal_cipher: PyElGamalCipher) -> Self {
        ElGamalCipher {
            c1: py_elgamal_cipher.c1.into(),
            c2: py_elgamal_cipher
                .c2
                .iter()
                .map(|x| crate::pfsys::vecu64_to_field_montgomery::<Fr>(&x))
                .collect::<Vec<_>>(),
        }
    }
}

impl From<ElGamalCipher> for PyElGamalCipher {
    fn from(elgamal_cipher: ElGamalCipher) -> Self {
        PyElGamalCipher {
            c1: elgamal_cipher.c1.into(),
            c2: elgamal_cipher
                .c2
                .iter()
                .map(|x| crate::pfsys::field_to_vecu64_montgomery::<Fr>(&x))
                .collect::<Vec<_>>(),
        }
    }
}

/// pyclass containing the struct used for ElgamalVariables
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyElGamalVariables {
    #[pyo3(get, set)]
    r: PyFelt,
    #[pyo3(get, set)]
    pk: PyG1Affine,
    #[pyo3(get, set)]
    sk: PyFelt,
    #[pyo3(get, set)]
    window_size: usize,
    #[pyo3(get, set)]
    aux_generator: PyG1Affine,
}

impl From<PyElGamalVariables> for ElGamalVariables {
    fn from(py_elgamal_variables: PyElGamalVariables) -> Self {
        ElGamalVariables {
            r: crate::pfsys::vecu64_to_field_montgomery::<Fr>(&py_elgamal_variables.r),
            pk: G1Affine {
                x: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&py_elgamal_variables.pk.x),
                y: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&py_elgamal_variables.pk.y),
            },
            sk: crate::pfsys::vecu64_to_field_montgomery::<Fr>(&py_elgamal_variables.sk),
            window_size: py_elgamal_variables.window_size,
            aux_generator: G1Affine {
                x: crate::pfsys::vecu64_to_field_montgomery::<Fq>(
                    &py_elgamal_variables.aux_generator.x,
                ),
                y: crate::pfsys::vecu64_to_field_montgomery::<Fq>(
                    &py_elgamal_variables.aux_generator.y,
                ),
            },
        }
    }
}

impl From<ElGamalVariables> for PyElGamalVariables {
    fn from(elgamal_variables: ElGamalVariables) -> Self {
        PyElGamalVariables {
            r: crate::pfsys::field_to_vecu64_montgomery::<Fr>(&elgamal_variables.r),
            pk: PyG1Affine {
                x: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&elgamal_variables.pk.x),
                y: crate::pfsys::field_to_vecu64_montgomery::<Fq>(&elgamal_variables.pk.y),
            },
            sk: crate::pfsys::field_to_vecu64_montgomery::<Fr>(&elgamal_variables.sk),
            window_size: elgamal_variables.window_size,
            aux_generator: PyG1Affine {
                x: crate::pfsys::field_to_vecu64_montgomery::<Fq>(
                    &elgamal_variables.aux_generator.x,
                ),
                y: crate::pfsys::field_to_vecu64_montgomery::<Fq>(
                    &elgamal_variables.aux_generator.y,
                ),
            },
        }
    }
}

impl pyo3::ToPyObject for PyElGamalVariables {
    fn to_object(&self, py: pyo3::Python) -> pyo3::PyObject {
        let variables_dict = pyo3::types::PyDict::new(py);

        variables_dict.set_item("r", self.r.to_object(py)).unwrap();
        variables_dict
            .set_item("pk", self.pk.to_object(py))
            .unwrap();
        variables_dict
            .set_item("sk", self.sk.to_object(py))
            .unwrap();
        variables_dict
            .set_item("window_size", self.window_size.to_object(py))
            .unwrap();
        variables_dict
            .set_item("aux_generator", self.aux_generator.to_object(py))
            .unwrap();
        variables_dict.into()
    }
}

/// pyclass containing the struct used for run_args
#[pyclass]
#[derive(Clone)]
struct PyRunArgs {
    #[pyo3(get, set)]
    pub tolerance: f32,
    #[pyo3(get, set)]
    pub input_scale: u32,
    #[pyo3(get, set)]
    pub param_scale: u32,
    #[pyo3(get, set)]
    pub scale_rebase_multiplier: u32,
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
    pub variables: Vec<(String, usize)>,
    #[pyo3(get, set)]
    pub allocated_constraints: Option<usize>,
}

/// default instantiation of PyRunArgs
#[pymethods]
impl PyRunArgs {
    #[new]
    fn new() -> Self {
        PyRunArgs {
            tolerance: 0.0,
            input_scale: 7,
            param_scale: 7,
            scale_rebase_multiplier: 2,
            bits: 16,
            logrows: 17,
            input_visibility: "public".into(),
            output_visibility: "public".into(),
            param_visibility: "private".into(),
            variables: vec![("batch_size".to_string(), 1)],
            allocated_constraints: None,
        }
    }
}

/// Conversion between PyRunArgs and RunArgs
impl From<PyRunArgs> for RunArgs {
    fn from(py_run_args: PyRunArgs) -> Self {
        RunArgs {
            tolerance: Tolerance::from(py_run_args.tolerance),
            input_scale: py_run_args.input_scale,
            param_scale: py_run_args.param_scale,
            scale_rebase_multiplier: py_run_args.scale_rebase_multiplier,
            bits: py_run_args.bits,
            logrows: py_run_args.logrows,
            input_visibility: py_run_args.input_visibility,
            output_visibility: py_run_args.output_visibility,
            param_visibility: py_run_args.param_visibility,
            variables: py_run_args.variables,
        }
    }
}

/// Converts 4 u64s to a field element
#[pyfunction(signature = (
    array,
))]
fn vecu64_to_felt(array: PyFelt) -> PyResult<String> {
    Ok(format!(
        "{:?}",
        crate::pfsys::vecu64_to_field_montgomery::<Fr>(&array)
    ))
}

/// Converts 4 u64s representing a field element directly to an integer
#[pyfunction(signature = (
    array,
))]
fn vecu64_to_int(array: PyFelt) -> PyResult<i128> {
    let felt = crate::pfsys::vecu64_to_field_montgomery::<Fr>(&array);
    let int_rep = felt_to_i128(felt);
    Ok(int_rep)
}

/// Converts 4 u64s representing a field element directly to a (rescaled from fixed point scaling) floating point
#[pyfunction(signature = (
    array,
    scale
))]
fn vecu64_to_float(array: PyFelt, scale: u32) -> PyResult<f64> {
    let felt = crate::pfsys::vecu64_to_field_montgomery::<Fr>(&array);
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
fn float_to_vecu64(input: f64, scale: u32) -> PyResult<PyFelt> {
    let int_rep = quantize_float(&input, 0.0, scale)
        .map_err(|_| PyIOError::new_err("Failed to quantize input"))?;
    let felt = i128_to_felt(int_rep);
    Ok(crate::pfsys::field_to_vecu64_montgomery::<Fr>(&felt))
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

    if remainder.len() != 0 {
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
        .map(|x| crate::pfsys::vecu64_to_field_montgomery::<Fr>(&x))
        .collect::<Vec<_>>();

    let output =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE, POSEIDON_LEN_GRAPH>::run(
            message.clone(),
        )
        .map_err(|_| PyIOError::new_err("Failed to run poseidon"))?;

    let hash = output[0]
        .iter()
        .map(|x| crate::pfsys::field_to_vecu64_montgomery::<Fr>(&x))
        .collect::<Vec<_>>();
    Ok(hash)
}

/// Encrypt using elgamal
#[pyfunction(signature = (
    pk, message, r
    ))]
pub fn elgamal_encrypt(
    pk: PyG1Affine,
    message: Vec<PyFelt>,
    r: PyFelt,
) -> PyResult<PyElGamalCipher> {
    let pk = G1Affine {
        x: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&pk.x),
        y: crate::pfsys::vecu64_to_field_montgomery::<Fq>(&pk.y),
    };
    let message = message
        .iter()
        .map(|x| crate::pfsys::vecu64_to_field_montgomery::<Fr>(&x))
        .collect::<Vec<_>>();
    let r = crate::pfsys::vecu64_to_field_montgomery::<Fr>(&r);

    let output = crate::circuit::modules::elgamal::ElGamalGadget::encrypt(pk, message, r);
    Ok(output.into())
}

/// Decrypt using elgamal
#[pyfunction(signature = (
    cipher, sk
    ))]
pub fn elgamal_decrypt(cipher: PyElGamalCipher, sk: PyFelt) -> PyResult<Vec<PyFelt>> {
    let sk: Fr = crate::pfsys::vecu64_to_field_montgomery::<Fr>(&sk);

    let output = crate::circuit::modules::elgamal::ElGamalGadget::decrypt(&cipher.into(), sk);

    let output = output
        .iter()
        .map(|x| crate::pfsys::field_to_vecu64_montgomery::<Fr>(&x))
        .collect::<Vec<_>>();

    Ok(output)
}

/// Generates random elgamal variables from a random seed value in browser.
/// Make sure input seed comes a secure source of randomness
#[pyfunction(signature = (
    rng
    ))]
pub fn elgamal_gen_random(rng: Vec<u8>) -> PyResult<PyElGamalVariables> {
    let seed: &[u8] = &rng;
    let mut rng = StdRng::from_seed(
        seed.try_into()
            .map_err(|_| PyIOError::new_err("Failed to create random seed"))?,
    );

    let output = crate::circuit::modules::elgamal::ElGamalVariables::gen_random(&mut rng);

    Ok(output.into())
}

/// Generates a vk from a pk for a model circuit and saves it to a file
#[pyfunction(signature = (
    path_to_pk,
    circuit_settings_path,
    vk_output_path
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
    path_to_pk,
    vk_output_path
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
    model,
    py_run_args = None
))]
fn table(model: String, py_run_args: Option<PyRunArgs>) -> PyResult<String> {
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
    srs_path,
    settings_path=None,
    logrows=None,
))]
fn get_srs(
    srs_path: PathBuf,
    settings_path: Option<PathBuf>,
    logrows: Option<u32>,
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
    scales = None,
))]
fn calibrate_settings(
    py: Python,
    data: PathBuf,
    model: PathBuf,
    settings: PathBuf,
    target: Option<CalibrationTarget>,
    scales: Option<Vec<u32>>,
) -> PyResult<&pyo3::PyAny> {
    let target = target.unwrap_or(CalibrationTarget::Resources {
        col_overflow: false,
    });
    pyo3_asyncio::tokio::future_into_py(py, async move {
        crate::execute::calibrate(model, data, settings, target, scales)
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
    optimizer_runs=None
))]
fn deploy_evm(
    addr_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: Option<usize>,
) -> Result<bool, PyErr> {
    Runtime::new()
        .unwrap()
        .block_on(crate::execute::deploy_evm(
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
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
    optimizer_runs=None
))]
fn deploy_da_evm(
    addr_path: PathBuf,
    input_data: PathBuf,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<String>,
    optimizer_runs: Option<usize>,
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
    m.add_class::<PyElGamalCipher>()?;
    m.add_class::<PyElGamalVariables>()?;
    m.add_class::<PyG1Affine>()?;
    m.add_class::<PyG1>()?;
    m.add_function(wrap_pyfunction!(vecu64_to_felt, m)?)?;
    m.add_function(wrap_pyfunction!(vecu64_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(vecu64_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(poseidon_hash, m)?)?;
    m.add_function(wrap_pyfunction!(elgamal_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(elgamal_decrypt, m)?)?;
    m.add_function(wrap_pyfunction!(elgamal_gen_random, m)?)?;
    m.add_function(wrap_pyfunction!(float_to_vecu64, m)?)?;
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
