use crate::circuit::{CheckMode, Tolerance};
use crate::commands::{RunArgs, StrategyType};
use crate::eth::{fix_verifier_sol, verify_proof_via_solidity};
use crate::execute::{
    create_proof_circuit_kzg, gen_deployment_code, load_params_cmd, verify_proof_circuit_kzg,
};
use crate::graph::{
    quantize_float, GraphCircuit, GraphInput, Model, ModelParams, VarVisibility, Visibility,
};
use crate::pfsys::evm::{
    aggregation::{gen_aggregation_evm_verifier, AggregationCircuit},
    evm_verify,
    single::gen_evm_verifier,
    DeploymentCode,
};
use crate::pfsys::{
    create_keys, gen_srs as ezkl_gen_srs, load_params, load_pk, load_vk, save_params, save_pk,
    save_vk, Snark, TranscriptType,
};
use halo2_proofs::poly::kzg::{
    commitment::{KZGCommitmentScheme, ParamsKZG},
    strategy::{AccumulatorStrategy, SingleStrategy as KZGSingleStrategy},
};
use halo2_proofs::{dev::MockProver, poly::commitment::ParamsProver};
use halo2curves::bn256::{Bn256, Fr};
use log::trace;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_log;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{fs::File, io::Write, path::PathBuf, sync::Arc};
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
    pub batch_size: u32,
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

/// Displays the table as a string in python
#[pyfunction(signature = (
    model,
    py_run_args = None
))]
fn table(model: String, py_run_args: Option<PyRunArgs>) -> Result<String, PyErr> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let visibility: VarVisibility = run_args.to_var_visibility();
    let mut reader = File::open(model).map_err(|_| PyIOError::new_err("Failed to open model"))?;
    let result = Model::new(&mut reader, run_args, visibility);

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
    data,
    model,
    output,
    py_run_args = None
))]
fn forward(
    data: PathBuf,
    model: PathBuf,
    output: PathBuf,
    py_run_args: Option<PyRunArgs>,
) -> PyResult<()> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let mut data =
        GraphInput::from_path(data).map_err(|_| PyIOError::new_err("Failed to import data"))?;

    let mut model_inputs = vec![];
    // quantize the supplied data using the provided scale.
    // for v in new_data.input_data.iter() {
    //     match vector_to_quantized(v, &Vec::from([v.len()]), 0.0, run_args.scale) {
    //         Ok(t) => model_inputs.push(t),
    //         Err(_) => return Err(PyValueError::new_err("Failed to quantize vector")),
    //     }
    // }
    for v in data.input_data.iter() {
        let t: Vec<i128> = v
            .par_iter()
            .map(|x| quantize_float(x, 0.0, run_args.scale).unwrap())
            .collect();
        model_inputs.push(t.into_iter().into());
    }
    let mut reader = File::open(model).map_err(|_| PyIOError::new_err("Failed to open model"))?;

    let model = Model::new(
        &mut reader,
        run_args,
        crate::graph::VarVisibility::default(),
    )
    .map_err(|_| PyIOError::new_err("Failed to create new model"))?;

    let res = model
        .forward(&model_inputs)
        .map_err(|_| PyIOError::new_err("Failed to run forward pass"))?;

    let output_scales = model.graph.get_output_scales();
    let output_scales = output_scales
        .iter()
        .map(|scale| crate::graph::scale_to_multiplier(*scale));

    let float_res: Vec<Vec<f32>> = res
        .iter()
        .zip(output_scales)
        .map(|(t, scale)| {
            t.iter()
                .map(|e| ((*e as f64) / scale) as f32)
                .collect::<Vec<f32>>()
        })
        .collect();
    trace!("forward pass output: {:?}", float_res);
    data.output_data = float_res;

    match serde_json::to_writer(&File::create(output)?, &data) {
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

/// mocks the prover
#[pyfunction(signature = (
    data,
    model,
    py_run_args = None
))]
fn mock(data: PathBuf, model: PathBuf, py_run_args: Option<PyRunArgs>) -> Result<bool, PyErr> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let logrows = run_args.logrows;
    let data =
        GraphInput::from_path(data).map_err(|_| PyIOError::new_err("Failed to import data"))?;
    let visibility = run_args.to_var_visibility();
    let mut reader = File::open(model).map_err(|_| PyIOError::new_err("Failed to open model"))?;
    let procmodel = Model::new(&mut reader, run_args, visibility)
        .map_err(|_| PyIOError::new_err("Failed to process model"))?;

    let arcmodel: Arc<Model> = Arc::new(procmodel);
    let mut circuit = GraphCircuit::new(arcmodel, CheckMode::SAFE)
        .map_err(|_| PyRuntimeError::new_err("Failed to create circuit"))?;

    let public_inputs = circuit
        .prepare_public_inputs(&data)
        .map_err(|_| PyRuntimeError::new_err("Failed to prepare public inputs"))?;
    let prover = MockProver::run(logrows, &circuit, public_inputs)
        .map_err(|_| PyRuntimeError::new_err("Failed to run prover"))?;

    prover.assert_satisfied();

    let res = prover.verify();
    match res {
        Ok(_) => return Ok(true),
        Err(_) => return Ok(false),
    }
}

/// runs the prover on a set of inputs
#[pyfunction(signature = (
    model,
    vk_path,
    pk_path,
    params_path,
    circuit_params_path,
    py_run_args = None
))]
fn setup(
    model: String,
    vk_path: PathBuf,
    pk_path: PathBuf,
    params_path: PathBuf,
    circuit_params_path: PathBuf,
    py_run_args: Option<PyRunArgs>,
) -> Result<bool, PyErr> {
    let run_args: RunArgs = py_run_args.unwrap_or_else(PyRunArgs::new).into();
    let logrows = run_args.logrows;
    let visibility = run_args.to_var_visibility();

    let mut reader = File::open(model).map_err(|_| PyIOError::new_err("Failed to open model"))?;
    let procmodel = Model::new(&mut reader, run_args, visibility)
        .map_err(|_| PyIOError::new_err("Failed to process model"))?;

    let arcmodel: Arc<Model> = Arc::new(procmodel);
    let circuit = GraphCircuit::new(arcmodel, CheckMode::UNSAFE)
        .map_err(|_| PyRuntimeError::new_err("Failed to create circuit"))?;

    let params = load_params_cmd(params_path, logrows)
        .map_err(|_| PyIOError::new_err("Failed to load params"))?;

    let proving_key =
        create_keys::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(&circuit, &params)
            .map_err(|_| PyRuntimeError::new_err("Failed to create proving key"))?;

    let circuit_params = circuit.params.clone();

    // save the verifier key
    save_vk::<KZGCommitmentScheme<Bn256>>(&vk_path, proving_key.get_vk())
        .map_err(|_| PyIOError::new_err("Failed to save verifier key to vk_path"))?;

    // save the prover key
    save_pk::<KZGCommitmentScheme<Bn256>>(&pk_path, &proving_key)
        .map_err(|_| PyIOError::new_err("Failed to save proving key to pk_path"))?;

    // save the circuit
    circuit_params.save(&circuit_params_path);

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
    let data =
        GraphInput::from_path(data).map_err(|_| PyIOError::new_err("Failed to import data"))?;

    let model_circuit_params = ModelParams::load(&circuit_params_path);

    let mut circuit =
        GraphCircuit::from_model_params(&model_circuit_params, &model.into(), CheckMode::SAFE)
            .map_err(|_| PyRuntimeError::new_err("Failed to create circuit"))?;

    let public_inputs = circuit
        .prepare_public_inputs(&data)
        .map_err(|_| PyRuntimeError::new_err("Failed to prepare public inputs"))?;

    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)
        .map_err(|_| PyIOError::new_err("Failed to load params"))?;

    let proving_key =
        load_pk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(pk_path, circuit.params.clone())
            .map_err(|_| PyRuntimeError::new_err("Failed to create proving key"))?;

    let snark = match strategy {
        StrategyType::Single => {
            let strategy = KZGSingleStrategy::new(&params);
            match create_proof_circuit_kzg(
                circuit,
                &params,
                public_inputs,
                &proving_key,
                transcript,
                strategy,
                CheckMode::SAFE,
            ) {
                Ok(snark) => Ok(snark),
                Err(_) => Err(PyRuntimeError::new_err(
                    "Failed to create proof circuit single strategy",
                )),
            }
        }
        StrategyType::Accum => {
            let strategy = AccumulatorStrategy::new(&params);
            match create_proof_circuit_kzg(
                circuit,
                &params,
                public_inputs,
                &proving_key,
                transcript,
                strategy,
                CheckMode::SAFE,
            ) {
                Ok(snark) => Ok(snark),
                Err(_) => Err(PyRuntimeError::new_err(
                    "Failed to create proof circuit using accumulator strategy",
                )),
            }
        }
    };

    // save the snark proof
    snark?
        .save(&proof_path)
        .map_err(|_| PyIOError::new_err("Failed to save proof to proof path"))?;

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
    let model_circuit_params = ModelParams::load(&circuit_params_path);
    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)
        .map_err(|_| PyIOError::new_err("Failed to load params"))?;
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)
        .map_err(|_| PyIOError::new_err("Failed to load proof"))?;

    let strategy = KZGSingleStrategy::new(params.verifier_params());
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk_path, model_circuit_params)
        .map_err(|_| PyIOError::new_err("Failed to load verifier key"))?;
    let result = verify_proof_circuit_kzg(params.verifier_params(), proof, &vk, strategy);
    match result {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
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
    let params = load_params_cmd(params_path.clone(), logrows)
        .map_err(|_| PyIOError::new_err("Failed to load params"))?;

    let mut snarks = vec![];

    for ((proof_path, vk_path), circuit_params_path) in aggregation_snarks
        .iter()
        .zip(aggregation_vk_paths)
        .zip(circuit_params_paths)
    {
        let model_circuit_params = ModelParams::load(&circuit_params_path);
        let params_app =
            load_params_cmd(params_path.clone(), model_circuit_params.run_args.logrows)
                .map_err(|_| PyIOError::new_err("Failed to load model circuit params"))?;
        let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(
            vk_path.to_path_buf(),
            // safe to clone as the inner model is wrapped in an Arc
            model_circuit_params.clone(),
        )
        .map_err(|_| PyIOError::new_err("Failed to load vk_path"))?;
        snarks.push(
            Snark::load::<KZGCommitmentScheme<Bn256>>(proof_path, Some(&params_app), Some(&vk))
                .map_err(|_| PyIOError::new_err("Failed to load proof_path"))?,
        );
    }
    // proof aggregation
    {
        let agg_circuit = AggregationCircuit::new(&params, snarks)
            .map_err(|_| PyRuntimeError::new_err("Failed to create aggreggation circuit"))?;
        let agg_pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(
            &agg_circuit,
            &params,
        )
        .map_err(|_| PyRuntimeError::new_err("Failed to create keys"))?;

        let snark = create_proof_circuit_kzg(
            agg_circuit.clone(),
            &params,
            agg_circuit.instances(),
            &agg_pk,
            transcript,
            AccumulatorStrategy::new(&params),
            check_mode,
        )
        .map_err(|_| PyRuntimeError::new_err("Failed to create proof circuit"))?;

        snark
            .save(&proof_path)
            .map_err(|_| PyIOError::new_err("Failed to save to proof_path"))?;
        save_vk::<KZGCommitmentScheme<Bn256>>(&vk_path, agg_pk.get_vk())
            .map_err(|_| PyIOError::new_err("Failed to save to vk_path"))?;
    }
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
    let params = load_params_cmd(params_path, logrows)
        .map_err(|_| PyIOError::new_err("Failed to load params"))?;

    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)
        .map_err(|_| PyIOError::new_err("Failed to load proof"))?;

    let strategy = AccumulatorStrategy::new(params.verifier_params());
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path, ())
        .map_err(|_| PyIOError::new_err("Failed to load vk"))?;
    let result = verify_proof_circuit_kzg(&params, proof, &vk, strategy);
    Ok(result.is_ok())
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
    let model_circuit_params = ModelParams::load(&circuit_params_path);
    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)
        .map_err(|_| PyIOError::new_err("Failed to load model circuit parameters"))?;

    let num_instance = model_circuit_params
        .instance_shapes
        .iter()
        .map(|x| x.iter().product())
        .collect();

    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk_path, model_circuit_params)
        .map_err(|_| PyIOError::new_err("Failed to load verifier key"))?;

    let yul_code = gen_evm_verifier(&params, &vk, num_instance)
        .map_err(|_| PyRuntimeError::new_err("Failed to generatee evm verifier"))?;

    let deployment_code = gen_deployment_code(yul_code.clone()).unwrap();

    deployment_code
        .save(&deployment_code_path)
        .map_err(|_| PyIOError::new_err("Failed to save deployment code"))?;

    if sol_code_path.is_some() {
        let mut f = File::create(sol_code_path.as_ref().unwrap())
            .map_err(|_| PyIOError::new_err("Failed to create file"))?;
        let _ = f.write(yul_code.as_bytes());

        let output = fix_verifier_sol(sol_code_path.as_ref().unwrap().clone())
            .map_err(|_| PyRuntimeError::new_err("Failed to fix solidity verifier"))?;

        let mut f = File::create(sol_code_path.as_ref().unwrap())
            .map_err(|_| PyIOError::new_err("Failed to write solidity code into file"))?;
        let _ = f.write(output.as_bytes());
    }
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
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)
        .map_err(|_| PyIOError::new_err("Failed to load proof"))?;
    let code = DeploymentCode::load(&deployment_code_path)
        .map_err(|_| PyIOError::new_err("Failed to load deployment code path"))?;
    evm_verify(code, proof.clone())
        .map_err(|_| PyRuntimeError::new_err("Failed to verify with evm"))?;

    if sol_code_path.is_some() {
        let result = Runtime::new()
            .unwrap()
            .block_on(verify_proof_via_solidity(
                proof,
                sol_code_path.unwrap(),
                runs,
            ))
            .map_err(|_| PyRuntimeError::new_err("Failed to verify proof via solidity"))?;

        trace!("Solidity verification result: {}", result);

        assert!(result);
    }
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
    deployment_code_path: PathBuf,
    sol_code_path: Option<PathBuf>,
) -> Result<bool, PyErr> {
    let params: ParamsKZG<Bn256> = load_params::<KZGCommitmentScheme<Bn256>>(params_path)
        .map_err(|_| PyIOError::new_err("Failed to load params"))?;

    let agg_vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path, ())
        .map_err(|_| PyIOError::new_err("Failed to load vk"))?;

    let yul_code = gen_aggregation_evm_verifier(
        &params,
        &agg_vk,
        AggregationCircuit::num_instance(),
        AggregationCircuit::accumulator_indices(),
    )
    .map_err(|_| PyRuntimeError::new_err("Failed to create aggregation evm verifier"))?;

    let deployment_code = gen_deployment_code(yul_code.clone()).unwrap();

    deployment_code
        .save(&deployment_code_path)
        .map_err(|_| PyIOError::new_err("Failed to save to deployment code path"))?;
    if sol_code_path.is_some() {
        let mut f = File::create(sol_code_path.as_ref().unwrap())
            .map_err(|_| PyIOError::new_err("Failed to create file"))?;
        let _ = f.write(yul_code.as_bytes());

        let output = fix_verifier_sol(sol_code_path.as_ref().unwrap().clone())
            .map_err(|_| PyRuntimeError::new_err("Failed to fix solidity verifier"))?;

        let mut f = File::create(sol_code_path.as_ref().unwrap())
            .map_err(|_| PyIOError::new_err("Failed to write solidity code into file"))?;
        let _ = f.write(output.as_bytes());
    }
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
    m.add_function(wrap_pyfunction!(table, m)?)?;
    m.add_function(wrap_pyfunction!(gen_srs, m)?)?;
    m.add_function(wrap_pyfunction!(forward, m)?)?;
    m.add_function(wrap_pyfunction!(mock, m)?)?;
    m.add_function(wrap_pyfunction!(setup, m)?)?;
    m.add_function(wrap_pyfunction!(prove, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(verify_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier, m)?)?;
    m.add_function(wrap_pyfunction!(verify_evm, m)?)?;
    m.add_function(wrap_pyfunction!(create_evm_verifier_aggr, m)?)?;
    m.add_function(wrap_pyfunction!(print_proof_hex, m)?)?;

    Ok(())
}
