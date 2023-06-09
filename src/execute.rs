use crate::circuit::CheckMode;
use crate::commands::{Cli, Commands, StrategyType};
#[cfg(not(target_arch = "wasm32"))]
use crate::eth::{fix_verifier_sol, verify_proof_via_solidity, get_provider, read_on_chain_inputs};
use crate::graph::{scale_to_multiplier, GraphCircuit, GraphInput, Model, ModelParams};
use crate::pfsys::evm::aggregation::{AggregationCircuit, PoseidonTranscript};
#[cfg(not(target_arch = "wasm32"))]
use crate::pfsys::evm::evm_verify;
#[cfg(not(target_arch = "wasm32"))]
use crate::pfsys::evm::{aggregation::gen_aggregation_evm_verifier, single::gen_evm_verifier};
use crate::pfsys::evm::{DeploymentCode, YulCode};
use crate::pfsys::{
    create_keys, load_params, load_pk, load_vk, save_params, save_pk, Snark, TranscriptType,
};
use crate::pfsys::{create_proof_circuit, gen_srs, save_vk, verify_proof_circuit};
use ethers::providers::Middleware;
#[cfg(not(target_arch = "wasm32"))]
use gag::Gag;
use halo2_proofs::dev::VerifyFailure;
use halo2_proofs::plonk::{Circuit, ProvingKey, VerifyingKey};
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverGWC;
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, multiopen::VerifierGWC, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use halo2_proofs::{dev::MockProver, poly::commitment::ParamsProver};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
#[cfg(not(target_arch = "wasm32"))]
use halo2curves::ff::Field;
#[cfg(not(target_arch = "wasm32"))]
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::{info, trace};
#[cfg(feature = "render")]
use plotters::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use rand::Rng;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use snark_verifier::loader::evm;
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::error::Error;
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Instant;
use thiserror::Error;

/// A wrapper for tensor related errors.
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// Shape mismatch in a operation
    #[error("verification failed")]
    VerifyError(Vec<VerifyFailure>),
}

/// Run an ezkl command with given args
pub async fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        #[cfg(not(target_arch = "wasm32"))]
        Commands::Fuzz {
            data,
            model: _,
            transcript,
            args,
            num_runs,
        } => fuzz(args.logrows, data, transcript, num_runs),
        Commands::GenSrs {
            params_path,
            logrows,
        } => gen_srs_cmd(params_path, logrows as u32),
        Commands::Table { model: _, .. } => table(cli),
        #[cfg(feature = "render")]
        Commands::RenderCircuit { output, .. } => render(output),
        Commands::Forward {
            data,
            model: _,
            output,
            args: _,
        } => forward(data, output),
        Commands::Mock { data, .. } => mock(data),
        #[cfg(not(target_arch = "wasm32"))]
        Commands::CreateEVMVerifier {
            vk_path,
            params_path,
            circuit_params_path,
            deployment_code_path,
            sol_code_path,
        } => create_evm_verifier(
            vk_path,
            params_path,
            circuit_params_path,
            deployment_code_path,
            sol_code_path,
        ),
        #[cfg(not(target_arch = "wasm32"))]
        Commands::CreateEVMVerifierAggr {
            vk_path,
            params_path,
            deployment_code_path,
            sol_code_path,
        } => {
            create_evm_aggregate_verifier(vk_path, params_path, deployment_code_path, sol_code_path)
        }
        Commands::Setup {
            params_path,
            circuit_params_path,
            vk_path,
            pk_path,
            ..
        } => create_keys_kzg(params_path, vk_path, pk_path, circuit_params_path),
        Commands::Prove {
            data,
            model,
            pk_path,
            proof_path,
            params_path,
            transcript,
            strategy,
            circuit_params_path,
            check_mode,
            rpc_url
        } => prove(
            data,
            model,
            pk_path,
            proof_path,
            params_path,
            transcript,
            strategy,
            circuit_params_path,
            check_mode,
            rpc_url
        ).await,
        Commands::Aggregate {
            circuit_params_paths,
            proof_path,
            aggregation_snarks,
            aggregation_vk_paths,
            vk_path,
            params_path,
            transcript,
            logrows,
            check_mode,
        } => aggregate(
            proof_path,
            aggregation_snarks,
            circuit_params_paths,
            aggregation_vk_paths,
            vk_path,
            params_path,
            transcript,
            logrows,
            check_mode,
        ),
        Commands::Verify {
            proof_path,
            circuit_params_path,
            vk_path,
            params_path,
        } => verify(proof_path, circuit_params_path, vk_path, params_path),
        Commands::VerifyAggr {
            proof_path,
            vk_path,
            params_path,
            logrows,
        } => verify_aggr(proof_path, vk_path, params_path, logrows),
        #[cfg(not(target_arch = "wasm32"))]
        Commands::VerifyEVM {
            proof_path,
            deployment_code_path,
            sol_code_path,
            optimizer_runs,
        } => {
            verify_evm(
                proof_path,
                deployment_code_path,
                sol_code_path,
                optimizer_runs,
            )
            .await
        }
        Commands::PrintProofHex { proof_path } => print_proof_hex(proof_path),
    }
}

/// helper function
pub fn verify_proof_circuit_kzg<
    'params,
    Strategy: VerificationStrategy<'params, KZGCommitmentScheme<Bn256>, VerifierGWC<'params, Bn256>>,
>(
    params: &'params ParamsKZG<Bn256>,
    proof: Snark<Fr, G1Affine>,
    vk: &VerifyingKey<G1Affine>,
    strategy: Strategy,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error> {
    match proof.transcript_type {
        TranscriptType::Blake => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            Challenge255<_>,
            Blake2bRead<_, _, _>,
        >(&proof, params, vk, strategy),
        TranscriptType::EVM => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            _,
            EvmTranscript<G1Affine, _, _, _>,
        >(&proof, params, vk, strategy),
        TranscriptType::Poseidon => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            _,
            PoseidonTranscript<NativeLoader, _>,
        >(&proof, params, vk, strategy),
    }
}

/// helper function
pub fn create_proof_circuit_kzg<
    'params,
    C: Circuit<Fr>,
    Strategy: VerificationStrategy<'params, KZGCommitmentScheme<Bn256>, VerifierGWC<'params, Bn256>>,
>(
    circuit: C,
    params: &'params ParamsKZG<Bn256>,
    public_inputs: Vec<Vec<Fr>>,
    pk: &ProvingKey<G1Affine>,
    transcript: TranscriptType,
    strategy: Strategy,
    check_mode: CheckMode,
) -> Result<Snark<Fr, G1Affine>, Box<dyn Error>> {
    match transcript {
        TranscriptType::EVM => create_proof_circuit::<
            KZGCommitmentScheme<_>,
            Fr,
            _,
            ProverGWC<_>,
            VerifierGWC<_>,
            _,
            _,
            EvmTranscript<G1Affine, _, _, _>,
            EvmTranscript<G1Affine, _, _, _>,
        >(
            circuit,
            public_inputs,
            params,
            pk,
            strategy,
            check_mode,
            transcript,
        )
        .map_err(Box::<dyn Error>::from),
        TranscriptType::Poseidon => create_proof_circuit::<
            KZGCommitmentScheme<_>,
            Fr,
            _,
            ProverGWC<_>,
            VerifierGWC<_>,
            _,
            _,
            PoseidonTranscript<NativeLoader, _>,
            PoseidonTranscript<NativeLoader, _>,
        >(
            circuit,
            public_inputs,
            params,
            pk,
            strategy,
            check_mode,
            transcript,
        )
        .map_err(Box::<dyn Error>::from),
        TranscriptType::Blake => create_proof_circuit::<
            KZGCommitmentScheme<_>,
            Fr,
            _,
            ProverGWC<_>,
            VerifierGWC<'_, Bn256>,
            _,
            Challenge255<_>,
            Blake2bWrite<_, _, _>,
            Blake2bRead<_, _, _>,
        >(
            circuit,
            public_inputs,
            params,
            pk,
            strategy,
            check_mode,
            transcript,
        )
        .map_err(Box::<dyn Error>::from),
    }
}

fn gen_srs_cmd(params_path: PathBuf, logrows: u32) -> Result<(), Box<dyn Error>> {
    let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
    Ok(())
}

fn table(cli: Cli) -> Result<(), Box<dyn Error>> {
    let om = Model::from_ezkl_conf(cli)?;
    info!("\n {}", om.table_nodes());
    Ok(())
}

fn forward(data: PathBuf, output: PathBuf) -> Result<(), Box<dyn Error>> {
    let mut data = GraphInput::from_path(data)?;
    let mut circuit = GraphCircuit::from_arg(CheckMode::SAFE)?;
    circuit.load_inputs(&data);

    let res = circuit.forward()?;

    trace!(
        "forward pass output shapes: {:?}",
        res.outputs.iter().map(|t| t.dims()).collect_vec()
    );

    let output_scales = circuit.model.graph.get_output_scales();
    let output_scales = output_scales
        .iter()
        .map(|scale| scale_to_multiplier(*scale));

    let float_res: Vec<Vec<f32>> = res
        .outputs
        .iter()
        .zip(output_scales)
        .map(|(t, scale)| t.iter().map(|e| ((*e as f64 / scale) as f32)).collect_vec())
        .collect();
    trace!("forward pass output: {:?}", float_res);
    data.output_data = float_res;
    data.input_hashes = Some(res.input_hashes);
    data.output_hashes = Some(res.output_hashes);

    serde_json::to_writer(&File::create(output)?, &data)?;
    Ok(())
}

fn mock(data: PathBuf) -> Result<(), Box<dyn Error>> {
    let data = GraphInput::from_path(data)?;
    // mock should catch any issues by default so we set it to safe
    let mut circuit = GraphCircuit::from_arg(CheckMode::SAFE)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;

    info!("Mock proof");

    let prover = MockProver::run(circuit.model.run_args.logrows, &circuit, public_inputs)
        .map_err(Box::<dyn Error>::from)?;
    prover.assert_satisfied();
    prover
        .verify()
        .map_err(|e| Box::<dyn Error>::from(ExecutionError::VerifyError(e)))?;
    Ok(())
}

fn print_proof_hex(proof_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
    for instance in proof.instances {
        println!("{:?}", instance);
    }
    info!("{}", hex::encode(proof.proof));
    Ok(())
}
/// helper function to generate the deployment code from yul code
pub fn gen_deployment_code(yul_code: YulCode) -> Result<DeploymentCode, Box<dyn Error>> {
    Ok(DeploymentCode {
        code: evm::compile_yul(&yul_code),
    })
}

#[cfg(feature = "render")]
fn render(output: PathBuf) -> Result<(), Box<dyn Error>> {
    let circuit = GraphCircuit::from_arg(CheckMode::UNSAFE)?;
    info!("Rendering circuit");

    // Create the area we want to draw on.
    // We could use SVGBackend if we want to render to .svg instead.
    // for an overview of how to interpret these plots, see https://zcash.github.io/halo2/user/dev-tools.html
    let root = BitMapBackend::new(&output, (512, 512)).into_drawing_area();
    root.fill(&TRANSPARENT).unwrap();
    let root = root.titled("Layout", ("sans-serif", 20))?;

    halo2_proofs::dev::CircuitLayout::default()
        // We hide labels, else most circuits become impossible to decipher because of overlaid text
        .show_labels(false)
        .render(circuit.model.run_args.logrows, &circuit, &root)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn create_evm_verifier(
    vk_path: PathBuf,
    params_path: PathBuf,
    circuit_params_path: PathBuf,
    deployment_code_path: PathBuf,
    sol_code_path: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let model_circuit_params = ModelParams::load(&circuit_params_path);
    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)?;

    let num_instance = model_circuit_params
        .instance_shapes
        .iter()
        .map(|x| x.iter().product())
        .collect();

    let vk =
        load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk_path, model_circuit_params)?;
    trace!("params computed");

    let yul_code: YulCode = gen_evm_verifier(&params, &vk, num_instance)?;
    let deployment_code = gen_deployment_code(yul_code.clone()).unwrap();
    deployment_code.save(&deployment_code_path)?;

    if sol_code_path.is_some() {
        let mut f = File::create(sol_code_path.as_ref().unwrap())?;
        let _ = f.write(yul_code.as_bytes());

        let output = fix_verifier_sol(sol_code_path.as_ref().unwrap().clone())?;

        let mut f = File::create(sol_code_path.as_ref().unwrap())?;
        let _ = f.write(output.as_bytes());
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn verify_evm(
    proof_path: PathBuf,
    deployment_code_path: PathBuf,
    sol_code_path: Option<PathBuf>,
    runs: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
    let code = DeploymentCode::load(&deployment_code_path)?;
    evm_verify(code, proof.clone())?;

    if sol_code_path.is_some() {
        let result = verify_proof_via_solidity(proof, sol_code_path.unwrap(), runs).await?;

        info!("Solidity verification result: {}", result);

        assert!(result);
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn create_evm_aggregate_verifier(
    vk_path: PathBuf,
    params_path: PathBuf,
    deployment_code_path: Option<PathBuf>,
    sol_code_path: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let params: ParamsKZG<Bn256> = load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;

    let agg_vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path, ())?;

    let yul_code = gen_aggregation_evm_verifier(
        &params,
        &agg_vk,
        AggregationCircuit::num_instance(),
        AggregationCircuit::accumulator_indices(),
    )?;
    let deployment_code = gen_deployment_code(yul_code.clone()).unwrap();
    deployment_code.save(deployment_code_path.as_ref().unwrap())?;

    if sol_code_path.is_some() {
        let mut f = File::create(sol_code_path.as_ref().unwrap())?;
        let _ = f.write(yul_code.as_bytes());

        let output = fix_verifier_sol(sol_code_path.as_ref().unwrap().clone())?;

        let mut f = File::create(sol_code_path.as_ref().unwrap())?;
        let _ = f.write(output.as_bytes());
    }
    Ok(())
}

fn create_keys_kzg(
    params_path: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    circuit_params_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    // these aren't real values so the sanity checks are mostly meaningless
    let circuit = GraphCircuit::from_arg(CheckMode::UNSAFE)?;
    let params = load_params_cmd(params_path, circuit.model.run_args.logrows)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(&circuit, &params)
        .map_err(Box::<dyn Error>::from)?;
    let circuit_params = circuit.params;
    trace!("params computed");
    circuit_params.save(&circuit_params_path);

    save_vk::<KZGCommitmentScheme<Bn256>>(&vk_path, pk.get_vk())?;
    save_pk::<KZGCommitmentScheme<Bn256>>(&pk_path, &pk)?;
    Ok(())
}

async fn prove(
    data: PathBuf,
    model_path: PathBuf,
    pk_path: PathBuf,
    proof_path: PathBuf,
    params_path: PathBuf,
    transcript: TranscriptType,
    strategy: StrategyType,
    circuit_params_path: PathBuf,
    check_mode: CheckMode,
    rpc_url: Option<String>,
) -> Result<(), Box<dyn Error>> {
    let mut data = GraphInput::from_path(data)?;
    let model_circuit_params = ModelParams::load(&circuit_params_path);
    let mut circuit =
        GraphCircuit::from_model_params(&model_circuit_params, &model_path, check_mode)?;
    if circuit.model.run_args.on_chain_inputs {
        let provider = get_provider(rpc_url.unwrap().as_str())?;
        let chain_id = provider.get_chainid().await?;
        info!("using chain {}", chain_id);
        data = read_on_chain_inputs(&provider, &mut data).await?;
    }
    let public_inputs = circuit.prepare_public_inputs(&data)?;
    let circuit_params = circuit.params.clone();

    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)?;

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(pk_path, circuit_params)
        .map_err(Box::<dyn Error>::from)?;

    trace!("params computed");

    let now = Instant::now();

    // creates and verifies the proof
    let snark = match strategy {
        StrategyType::Single => {
            let strategy = KZGSingleStrategy::new(&params);
            create_proof_circuit_kzg(
                circuit,
                &params,
                public_inputs,
                &pk,
                transcript,
                strategy,
                check_mode,
            )?
        }
        StrategyType::Accum => {
            let strategy = AccumulatorStrategy::new(&params);
            create_proof_circuit_kzg(
                circuit,
                &params,
                public_inputs,
                &pk,
                transcript,
                strategy,
                check_mode,
            )?
        }
    };

    info!("proof took {}", now.elapsed().as_secs());

    snark.save(&proof_path)?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn fuzz(
    logrows: u32,
    data: PathBuf,
    transcript: TranscriptType,
    num_runs: usize,
) -> Result<(), Box<dyn Error>> {
    let passed = AtomicBool::new(true);

    info!("setting up tests");

    let _r = Gag::stdout().unwrap();
    let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);

    let data = GraphInput::from_path(data)?;
    // these aren't real values so the sanity checks are mostly meaningless
    let mut circuit = GraphCircuit::from_arg(CheckMode::UNSAFE)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(&circuit, &params)
        .map_err(Box::<dyn Error>::from)?;

    let public_inputs = circuit.prepare_public_inputs(&data)?;

    let strategy = KZGSingleStrategy::new(&params);
    std::mem::drop(_r);

    info!("starting fuzzing");

    info!("fuzzing pk");

    let fuzz_pk = || {
        let new_params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);

        let bad_pk =
            create_keys::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(&circuit, &new_params)
                .unwrap();

        let bad_proof = create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            public_inputs.clone(),
            &bad_pk,
            transcript,
            strategy.clone(),
            CheckMode::UNSAFE,
        )
        .unwrap();

        verify_proof_circuit_kzg(
            params.verifier_params(),
            bad_proof,
            pk.get_vk(),
            strategy.clone(),
        )
        .map_err(|_| ())
    };

    run_fuzz_fn(num_runs, fuzz_pk, &passed);

    info!("fuzzing public inputs");

    let fuzz_public_inputs = || {
        let mut bad_inputs = vec![];
        for l in &public_inputs {
            bad_inputs.push(vec![Fr::random(rand::rngs::OsRng); l.len()]);
        }

        let bad_proof = create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            bad_inputs.clone(),
            &pk,
            transcript,
            strategy.clone(),
            CheckMode::UNSAFE,
        )
        .unwrap();

        verify_proof_circuit_kzg(
            params.verifier_params(),
            bad_proof,
            pk.get_vk(),
            strategy.clone(),
        )
        .map_err(|_| ())
    };

    run_fuzz_fn(num_runs, fuzz_public_inputs, &passed);

    info!("fuzzing vk");

    let proof = create_proof_circuit_kzg(
        circuit.clone(),
        &params,
        public_inputs.clone(),
        &pk,
        transcript,
        strategy.clone(),
        CheckMode::SAFE,
    )?;

    let fuzz_vk = || {
        let new_params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);

        let bad_pk =
            create_keys::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(&circuit, &new_params)
                .unwrap();

        let bad_vk = bad_pk.get_vk();

        verify_proof_circuit_kzg(
            params.verifier_params(),
            proof.clone(),
            bad_vk,
            strategy.clone(),
        )
        .map_err(|_| ())
    };

    run_fuzz_fn(num_runs, fuzz_vk, &passed);

    info!("fuzzing proof bytes");

    let fuzz_proof_bytes = || {
        let mut rng = rand::thread_rng();

        let bad_proof_bytes: Vec<u8> = (0..proof.proof.len())
            .map(|_| rng.gen_range(0..20))
            .collect();

        let bad_proof = Snark::<_, _> {
            instances: proof.instances.clone(),
            proof: bad_proof_bytes,
            protocol: proof.protocol.clone(),
            transcript_type: transcript,
        };

        verify_proof_circuit_kzg(
            params.verifier_params(),
            bad_proof,
            pk.get_vk(),
            strategy.clone(),
        )
        .map_err(|_| ())
    };

    run_fuzz_fn(num_runs, fuzz_proof_bytes, &passed);

    info!("fuzzing proof instances");

    let fuzz_proof_instances = || {
        let mut bad_inputs = vec![];
        for l in &proof.instances {
            bad_inputs.push(vec![Fr::random(rand::rngs::OsRng); l.len()]);
        }

        let bad_proof = Snark::<_, _> {
            instances: bad_inputs.clone(),
            proof: proof.proof.clone(),
            protocol: proof.protocol.clone(),
            transcript_type: transcript,
        };

        verify_proof_circuit_kzg(
            params.verifier_params(),
            bad_proof,
            pk.get_vk(),
            strategy.clone(),
        )
        .map_err(|_| ())
    };

    run_fuzz_fn(num_runs, fuzz_proof_instances, &passed);

    if matches!(transcript, TranscriptType::EVM) {
        let num_instance = circuit
            .params
            .instance_shapes
            .iter()
            .map(|x| x.iter().product())
            .collect();

        let yul_code = gen_evm_verifier(&params, pk.get_vk(), num_instance)?;
        let deployment_code = gen_deployment_code(yul_code).unwrap();

        info!("fuzzing proof bytes for evm verifier");

        let fuzz_evm_proof_bytes = || {
            let mut rng = rand::thread_rng();

            let bad_proof_bytes: Vec<u8> = (0..proof.proof.len())
                .map(|_| rng.gen_range(0..20))
                .collect();

            let bad_proof = Snark::<_, _> {
                instances: proof.instances.clone(),
                proof: bad_proof_bytes,
                protocol: proof.protocol.clone(),
                transcript_type: transcript,
            };

            let res = evm_verify(deployment_code.clone(), bad_proof);

            match res {
                Ok(_) => Ok(()),
                Err(_) => Err(()),
            }
        };

        run_fuzz_fn(num_runs, fuzz_evm_proof_bytes, &passed);

        info!("fuzzing proof instances for evm verifier");

        let fuzz_evm_instances = || {
            let mut bad_inputs = vec![];
            for l in &proof.instances {
                bad_inputs.push(vec![Fr::random(rand::rngs::OsRng); l.len()]);
            }

            let bad_proof = Snark::<_, _> {
                instances: bad_inputs.clone(),
                proof: proof.proof.clone(),
                protocol: proof.protocol.clone(),
                transcript_type: transcript,
            };

            let res = evm_verify(deployment_code.clone(), bad_proof);

            match res {
                Ok(_) => Ok(()),
                Err(_) => Err(()),
            }
        };

        run_fuzz_fn(num_runs, fuzz_evm_instances, &passed);
    }

    if !passed.into_inner() {
        Err("fuzzing failed".into())
    } else {
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_fuzz_fn(
    num_runs: usize,
    f: impl Fn() -> Result<(), ()> + std::marker::Sync + std::marker::Send,
    passed: &AtomicBool,
) {
    let num_failures = AtomicI64::new(0);
    let _r = Gag::stdout().unwrap();

    (0..num_runs).into_par_iter().progress().for_each(|_| {
        let result = f();
        if result.is_ok() {
            passed.swap(false, Ordering::Relaxed);
            num_failures.fetch_add(1, Ordering::Relaxed);
        }
    });

    std::mem::drop(_r);
    info!(
        "num failures: {} out of {}",
        num_failures.load(Ordering::Relaxed),
        num_runs
    );
}

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
) -> Result<(), Box<dyn Error>> {
    // the K used for the aggregation circuit
    let params = load_params_cmd(params_path.clone(), logrows)?;

    let mut snarks = vec![];

    for ((proof_path, vk_path), circuit_params_path) in aggregation_snarks
        .iter()
        .zip(aggregation_vk_paths)
        .zip(circuit_params_paths)
    {
        let model_circuit_params = ModelParams::load(&circuit_params_path);
        let params_app =
            load_params_cmd(params_path.clone(), model_circuit_params.run_args.logrows)?;
        let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(
            vk_path.to_path_buf(),
            // safe to clone as the inner model is wrapped in an Arc
            model_circuit_params.clone(),
        )?;
        snarks.push(Snark::load::<KZGCommitmentScheme<Bn256>>(
            proof_path,
            Some(&params_app),
            Some(&vk),
        )?);
    }
    // proof aggregation
    {
        let agg_circuit = AggregationCircuit::new(&params, snarks)?;
        let agg_pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(
            &agg_circuit,
            &params,
        )?;

        let now = Instant::now();
        let snark = create_proof_circuit_kzg(
            agg_circuit.clone(),
            &params,
            agg_circuit.instances(),
            &agg_pk,
            transcript,
            AccumulatorStrategy::new(&params),
            check_mode,
        )?;

        info!("Aggregation proof took {}", now.elapsed().as_secs());
        snark.save(&proof_path)?;
        save_vk::<KZGCommitmentScheme<Bn256>>(&vk_path, agg_pk.get_vk())?;
    }
    Ok(())
}

fn verify(
    proof_path: PathBuf,
    circuit_params_path: PathBuf,
    vk_path: PathBuf,
    params_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let model_circuit_params = ModelParams::load(&circuit_params_path);

    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)?;

    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
    let model_circuit_params = ModelParams::load(&circuit_params_path);

    let strategy = KZGSingleStrategy::new(params.verifier_params());
    let vk =
        load_vk::<KZGCommitmentScheme<Bn256>, Fr, GraphCircuit>(vk_path, model_circuit_params)?;
    let now = Instant::now();
    let result = verify_proof_circuit_kzg(params.verifier_params(), proof, &vk, strategy);
    info!("verify took {}", now.elapsed().as_secs());
    info!("verified: {}", result.is_ok());
    Ok(())
}

fn verify_aggr(
    proof_path: PathBuf,
    vk_path: PathBuf,
    params_path: PathBuf,
    logrows: u32,
) -> Result<(), Box<dyn Error>> {
    let params = load_params_cmd(params_path, logrows)?;

    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;

    let strategy = AccumulatorStrategy::new(params.verifier_params());
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path, ())?;
    let now = Instant::now();
    let result = verify_proof_circuit_kzg(&params, proof, &vk, strategy);
    info!("verify took {}", now.elapsed().as_secs());
    info!("verified: {}", result.is_ok());
    Ok(())
}

/// helper function for load_params
pub fn load_params_cmd(
    params_path: PathBuf,
    logrows: u32,
) -> Result<ParamsKZG<Bn256>, Box<dyn Error>> {
    let mut params: ParamsKZG<Bn256> = load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;
    info!("downsizing params to {} logrows", logrows);
    if logrows < params.k() {
        params.downsize(logrows);
    }
    Ok(params)
}
