use crate::circuit::CheckMode;
use crate::commands::{Cli, Commands, RunArgs, StrategyType};
#[cfg(not(target_arch = "wasm32"))]
use crate::eth::{
    deploy_verifier, fix_verifier_sol, get_ledger_signing_provider, get_provider,
    get_wallet_signing_provider, send_proof, verify_proof_via_solidity,
};
use crate::graph::{quantize_float, Model, ModelCircuit, ModelParams};
use crate::pfsys::evm::aggregation::{AggregationCircuit, PoseidonTranscript};
#[cfg(not(target_arch = "wasm32"))]
use crate::pfsys::evm::{aggregation::gen_aggregation_evm_verifier, single::gen_evm_verifier};
#[cfg(not(target_arch = "wasm32"))]
use crate::pfsys::evm::{evm_verify, DeploymentCode};
use crate::pfsys::{
    create_keys, load_params, load_pk, load_vk, save_params, save_pk, Snark, TranscriptType,
};
use crate::pfsys::{create_proof_circuit, gen_srs, prepare_data, save_vk, verify_proof_circuit};
#[cfg(not(target_arch = "wasm32"))]
use ethers::providers::Middleware;
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
use log::warn;
use log::{info, trace};
#[cfg(feature = "render")]
use plotters::prelude::*;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::read_to_string;
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
use std::time::Instant;
use tabled::Table;
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
        Commands::SendProofEVM {
            secret,
            rpc_url,
            addr,
            proof_path,
            has_abi,
        } => send_proof_evm(secret, rpc_url, addr, proof_path, has_abi).await,
        #[cfg(not(target_arch = "wasm32"))]
        Commands::DeployVerifierEVM {
            secret,
            rpc_url,
            deployment_code_path,
            sol_code_path,
        } => deploy_verifier_evm(secret, rpc_url, deployment_code_path, sol_code_path).await,
        Commands::GenSrs { params_path } => gen_srs_cmd(params_path, cli.args.logrows),
        Commands::Table { model: _ } => table(cli),
        #[cfg(feature = "render")]
        Commands::RenderCircuit {
            data,
            model: _,
            output,
        } => render(data, output, cli.args.logrows),
        Commands::Forward {
            data,
            model,
            output,
        } => forward(data, model, output, cli.args),
        Commands::Mock { data, model: _ } => mock(data, cli.args.logrows),
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
            cli.args.logrows,
        ),
        #[cfg(not(target_arch = "wasm32"))]
        Commands::CreateEVMVerifierAggr {
            params_path,
            deployment_code_path,
            vk_path,
        } => create_evm_aggregate_verifier(params_path, deployment_code_path, vk_path),
        Commands::Setup {
            model: _,
            data,
            params_path,
            circuit_params_path,
            vk_path,
            pk_path,
        } => create_keys_kzg(
            data,
            params_path,
            vk_path,
            pk_path,
            circuit_params_path,
            cli.args.logrows,
        ),
        Commands::Prove {
            data,
            model,
            pk_path,
            proof_path,
            params_path,
            transcript,
            strategy,
            circuit_params_path,
        } => prove(
            data,
            model,
            pk_path,
            proof_path,
            params_path,
            transcript,
            strategy,
            circuit_params_path,
            cli.args.check_mode,
        ),
        Commands::Aggregate {
            circuit_params_paths,
            proof_path,
            aggregation_snarks,
            aggregation_vk_paths,
            vk_path,
            params_path,
            transcript,
            logrows,
        } => aggregate(
            proof_path,
            aggregation_snarks,
            circuit_params_paths,
            aggregation_vk_paths,
            vk_path,
            params_path,
            transcript,
            logrows,
            cli.args.check_mode,
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
        } => verify_evm(proof_path, deployment_code_path, sol_code_path).await,
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

#[cfg(not(target_arch = "wasm32"))]
async fn send_proof_evm(
    secret: Option<PathBuf>,
    rpc_url: String,
    addr: ethereum_types::Address,
    proof_path: PathBuf,
    has_abi: bool,
) -> Result<(), Box<dyn Error>> {
    let provider = get_provider(&rpc_url)?;
    let chain_id = provider.get_chainid().await?;
    info!("using chain {}", chain_id);
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
    if let Some(secret) = secret {
        let mnemonic = read_to_string(secret)?;
        let client = Arc::new(get_wallet_signing_provider(provider, &mnemonic).await?);
        send_proof(client.clone(), addr, client.address(), proof, has_abi).await?;
    } else {
        warn!("connect your Ledger and open the Ethereum app");
        let client = Arc::new(get_ledger_signing_provider(provider, chain_id.as_u64()).await?);
        send_proof(client.clone(), addr, client.address(), proof, has_abi).await?;
    };
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
async fn deploy_verifier_evm(
    secret: Option<PathBuf>,
    rpc_url: String,
    deployment_code_path: Option<PathBuf>,
    sol_code_path: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let provider = get_provider(&rpc_url)?;
    let chain_id = provider.get_chainid().await?;
    info!("using chain {}", chain_id);
    if let Some(secret) = secret {
        let mnemonic = read_to_string(secret)?;
        let client = Arc::new(get_wallet_signing_provider(provider, &mnemonic).await?);
        deploy_verifier(client, deployment_code_path, sol_code_path).await?;
    } else {
        warn!("connect your Ledger and open the Ethereum app");
        let client = Arc::new(get_ledger_signing_provider(provider, chain_id.as_u64()).await?);
        deploy_verifier(client, deployment_code_path, sol_code_path).await?;
    };
    Ok(())
}

fn gen_srs_cmd(params_path: PathBuf, logrows: u32) -> Result<(), Box<dyn Error>> {
    let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
    Ok(())
}

fn table(cli: Cli) -> Result<(), Box<dyn Error>> {
    let om = Model::<Fr>::from_ezkl_conf(cli)?;
    info!("\n {}", Table::new(om.nodes.iter()));
    Ok(())
}

fn forward(
    data: String,
    model: String,
    output: String,
    args: RunArgs,
) -> Result<(), Box<dyn Error>> {
    let mut data = prepare_data(data)?;

    // quantize the supplied data using the provided scale.
    let mut model_inputs = vec![];
    for v in data.input_data.iter() {
        let t: Vec<i128> = v
            .par_iter()
            .map(|x| quantize_float(x, 0.0, args.scale).unwrap())
            .collect();
        model_inputs.push(t.into_iter().into());
    }

    let res = Model::<Fr>::forward(&mut std::fs::File::open(model)?, &model_inputs, args)?;

    let float_res: Vec<Vec<f32>> = res.iter().map(|t| t.to_vec()).collect();
    trace!("forward pass output: {:?}", float_res);
    data.output_data = float_res;

    serde_json::to_writer(&File::create(output)?, &data)?;
    Ok(())
}

fn mock(data: String, logrows: u32) -> Result<(), Box<dyn Error>> {
    let data = prepare_data(data)?;
    let circuit = ModelCircuit::<Fr>::from_arg(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;

    info!("Mock proof");

    let prover =
        MockProver::run(logrows, &circuit, public_inputs).map_err(Box::<dyn Error>::from)?;
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

#[cfg(feature = "render")]
fn render(data: String, output: String, logrows: u32) -> Result<(), Box<dyn Error>> {
    let data = prepare_data(data.to_string())?;
    let circuit = ModelCircuit::<Fr>::from_arg(&data)?;
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
        .render(logrows, &circuit, &root)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn create_evm_verifier(
    vk_path: PathBuf,
    params_path: PathBuf,
    circuit_params_path: PathBuf,
    deployment_code_path: Option<PathBuf>,
    sol_code_path: Option<PathBuf>,
    logrows: u32,
) -> Result<(), Box<dyn Error>> {
    let params = load_params_cmd(params_path, logrows)?;
    let model_circuit_params = ModelParams::load(&circuit_params_path);
    let num_instance = model_circuit_params
        .instance_shapes
        .iter()
        .map(|x| x.iter().product())
        .collect();

    let vk =
        load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(vk_path, model_circuit_params)?;
    trace!("params computed");

    let (deployment_code, yul_code) = gen_evm_verifier(&params, &vk, num_instance)?;
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

#[cfg(not(target_arch = "wasm32"))]
async fn verify_evm(
    proof_path: PathBuf,
    deployment_code_path: PathBuf,
    sol_code_path: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
    let code = DeploymentCode::load(&deployment_code_path)?;
    evm_verify(code, proof.clone())?;

    if sol_code_path.is_some() {
        let result = verify_proof_via_solidity(proof, sol_code_path.unwrap())
            .await
            .unwrap();

        info!("Solidity verification result: {}", result);

        assert!(result);
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn create_evm_aggregate_verifier(
    params_path: PathBuf,
    deployment_code_path: PathBuf,
    vk_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let params: ParamsKZG<Bn256> = load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;

    let agg_vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path, ())?;

    let deployment_code = gen_aggregation_evm_verifier(
        &params,
        &agg_vk,
        AggregationCircuit::num_instance(),
        AggregationCircuit::accumulator_indices(),
    )?;
    deployment_code.save(&deployment_code_path)?;
    Ok(())
}

fn create_keys_kzg(
    data: String,
    params_path: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    circuit_params_path: PathBuf,
    logrows: u32,
) -> Result<(), Box<dyn Error>> {
    let data = prepare_data(data)?;
    let circuit = ModelCircuit::<Fr>::from_arg(&data)?;
    let params = load_params_cmd(params_path, logrows)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(&circuit, &params)
        .map_err(Box::<dyn Error>::from)?;
    let circuit_params = circuit.params.clone();
    trace!("params computed");
    circuit_params.save(&circuit_params_path);

    save_vk::<KZGCommitmentScheme<Bn256>>(&vk_path, pk.get_vk())?;
    save_pk::<KZGCommitmentScheme<Bn256>>(&pk_path, &pk)?;
    Ok(())
}

fn prove(
    data: String,
    model_path: PathBuf,
    pk_path: PathBuf,
    proof_path: PathBuf,
    params_path: PathBuf,
    transcript: TranscriptType,
    strategy: StrategyType,
    circuit_params_path: PathBuf,
    check_mode: CheckMode,
) -> Result<(), Box<dyn Error>> {
    let data = prepare_data(data)?;
    let model_circuit_params = ModelParams::load(&circuit_params_path);
    let circuit = ModelCircuit::<Fr>::from_model_params(&data, &model_circuit_params, &model_path)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;
    let circuit_params = circuit.params.clone();

    let params = load_params_cmd(params_path, model_circuit_params.run_args.logrows)?;

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(
        pk_path,
        circuit_params.clone(),
    )
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
                model_circuit_params.run_args.check_mode,
            )?
        }
    };

    info!("proof took {}", now.elapsed().as_secs());

    snark.save(&proof_path)?;

    Ok(())
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
        let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(
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
        load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(vk_path, model_circuit_params)?;
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
