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


// use crate::commands::{Cli, Commands, StrategyType, TranscriptType};
// #[cfg(not(target_arch = "wasm32"))]
// use crate::eth::{deploy_verifier, fix_verifier_sol, send_proof, verify_proof_via_solidity};
// use crate::graph::{vector_to_quantized, Model, ModelCircuit};
// use crate::pfsys::evm::aggregation::{AggregationCircuit, PoseidonTranscript};
// #[cfg(not(target_arch = "wasm32"))]
// use crate::pfsys::evm::{aggregation::gen_aggregation_evm_verifier, single::gen_evm_verifier};
// #[cfg(not(target_arch = "wasm32"))]
// use crate::pfsys::evm::{evm_verify, DeploymentCode};
// #[cfg(feature = "render")]
// use crate::pfsys::prepare_model_circuit;
// use crate::pfsys::{create_keys, load_params, load_vk, save_params, Snark};
// use crate::pfsys::{
//     create_proof_circuit, gen_srs, prepare_data, prepare_model_circuit_and_public_input, save_vk,
//     verify_proof_circuit,
// };
// use halo2_proofs::dev::VerifyFailure;
// use halo2_proofs::plonk::{Circuit, ProvingKey, VerifyingKey};
// use halo2_proofs::poly::commitment::Params;
// use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
// use halo2_proofs::poly::kzg::multiopen::ProverGWC;
// use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
// use halo2_proofs::poly::kzg::{
//     commitment::ParamsKZG, multiopen::VerifierGWC, strategy::SingleStrategy as KZGSingleStrategy,
// };
// use halo2_proofs::poly::VerificationStrategy;
// use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
// use halo2_proofs::{dev::MockProver, poly::commitment::ParamsProver};
// use halo2curves::bn256::{Bn256, Fr, G1Affine};
// use log::{info, trace};
// #[cfg(feature = "render")]
// use plotters::prelude::*;
// use snark_verifier::loader::native::NativeLoader;
// use snark_verifier::system::halo2::transcript::evm::EvmTranscript; // use std::error::Error;
// use std::fs::File;
// #[cfg(not(target_arch = "wasm32"))]
// use std::io::Write;
// use std::time::Instant;
// use tabled::Table;
// use thiserror::Error;


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

// TODO: Forward
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