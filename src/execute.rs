use crate::circuit::CheckMode;
use crate::commands::{Cli, Commands, StrategyType, TranscriptType};
#[cfg(not(target_arch = "wasm32"))]
use crate::eth::{
    deploy_verifier, fix_verifier_sol, get_ledger_signing_provider, get_provider,
    get_wallet_signing_provider, send_proof, verify_proof_via_solidity,
};
use crate::graph::{vector_to_quantized, Model, ModelCircuit};
use crate::pfsys::evm::aggregation::{AggregationCircuit, PoseidonTranscript};
#[cfg(not(target_arch = "wasm32"))]
use crate::pfsys::evm::{aggregation::gen_aggregation_evm_verifier, single::gen_evm_verifier};
#[cfg(not(target_arch = "wasm32"))]
use crate::pfsys::evm::{evm_verify, DeploymentCode};
use crate::pfsys::{create_keys, load_params, load_vk, save_params, Snark};
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
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::read_to_string;
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
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

/// helper function
fn verify_proof_circuit_kzg<
    'params,
    Strategy: VerificationStrategy<'params, KZGCommitmentScheme<Bn256>, VerifierGWC<'params, Bn256>>,
>(
    params: &'params ParamsKZG<Bn256>,
    proof: Snark<Fr, G1Affine>,
    vk: &VerifyingKey<G1Affine>,
    transcript: TranscriptType,
    strategy: Strategy,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error> {
    match transcript {
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
        >(circuit, public_inputs, params, pk, strategy, check_mode)
        .map_err(Box::<dyn Error>::from),
        TranscriptType::Poseidon => {
            create_proof_circuit::<
                KZGCommitmentScheme<_>,
                Fr,
                _,
                ProverGWC<_>,
                VerifierGWC<_>,
                _,
                _,
                PoseidonTranscript<NativeLoader, _>,
                PoseidonTranscript<NativeLoader, _>,
            >(circuit, public_inputs, params, pk, strategy, check_mode)
            .map_err(Box::<dyn Error>::from)
        }
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
        >(circuit, public_inputs, params, pk, strategy, check_mode)
        .map_err(Box::<dyn Error>::from),
    }
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
        } => {
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
                let client =
                    Arc::new(get_ledger_signing_provider(provider, chain_id.as_u64()).await?);
                send_proof(client.clone(), addr, client.address(), proof, has_abi).await?;
            };
        }
        #[cfg(not(target_arch = "wasm32"))]
        Commands::DeployVerifierEVM {
            secret,
            rpc_url,
            deployment_code_path,
            sol_code_path,
        } => {
            let provider = get_provider(&rpc_url)?;
            let chain_id = provider.get_chainid().await?;
            info!("using chain {}", chain_id);
            if let Some(secret) = secret {
                let mnemonic = read_to_string(secret)?;
                let client = Arc::new(get_wallet_signing_provider(provider, &mnemonic).await?);
                deploy_verifier(client, deployment_code_path, sol_code_path).await?;
            } else {
                warn!("connect your Ledger and open the Ethereum app");
                let client =
                    Arc::new(get_ledger_signing_provider(provider, chain_id.as_u64()).await?);
                deploy_verifier(client, deployment_code_path, sol_code_path).await?;
            };
        }
        Commands::GenSrs { params_path } => {
            let params = gen_srs::<KZGCommitmentScheme<Bn256>>(cli.args.logrows);
            save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
        }
        Commands::Table { model: _ } => {
            let om = Model::from_ezkl_conf(cli)?;
            info!("{}", Table::new(om.nodes.iter()));
        }
        #[cfg(feature = "render")]
        Commands::RenderCircuit {
            ref data,
            model: _,
            ref output,
        } => {
            let data = prepare_data(data.to_string())?;
            let circuit = prepare_model_circuit::<Fr>(&data, &cli.args)?;
            info!("Rendering circuit");

            // Create the area we want to draw on.
            // We could use SVGBackend if we want to render to .svg instead.
            // for an overview of how to interpret these plots, see https://zcash.github.io/halo2/user/dev-tools.html
            let root = BitMapBackend::new(output, (512, 512)).into_drawing_area();
            root.fill(&TRANSPARENT).unwrap();
            let root = root.titled("Layout", ("sans-serif", 20))?;

            halo2_proofs::dev::CircuitLayout::default()
                // We hide labels, else most circuits become impossible to decipher because of overlaid text
                .show_labels(false)
                .render(cli.args.logrows, &circuit, &root)?;
        }
        Commands::Forward {
            ref data,
            model,
            output,
        } => {
            let mut data = prepare_data(data.to_string())?;

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
        }
        Commands::Mock { ref data, model } => {
            let data = prepare_data(data.to_string())?;
            let model = Model::read_from_file(model.into())?;
            let circuit = ModelCircuit::<Fr>::new(&data, model)?;
            let public_inputs = circuit.prepare_public_inputs(&data)?;

            info!("Mock proof");

            let prover = MockProver::run(cli.args.logrows, &circuit, public_inputs)
                .map_err(Box::<dyn Error>::from)?;
            prover.assert_satisfied();
            prover
                .verify()
                .map_err(|e| Box::<dyn Error>::from(ExecutionError::VerifyError(e)))?;
        }
        #[cfg(not(target_arch = "wasm32"))]
        Commands::CreateEVMVerifier {
            ref data,
            model,
            ref vk_path,
            ref params_path,
            ref deployment_code_path,
            ref sol_code_path,
        } => {
            let data = prepare_data(data.to_string())?;

            let model = Model::read_from_file(model)?;
            let circuit = ModelCircuit::<Fr>::new(&data, model)?;
            let public_inputs = circuit.prepare_public_inputs(&data)?;
            let num_instance = public_inputs.iter().map(|x| x.len()).collect();
            let mut params: ParamsKZG<Bn256> =
                load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;
            info!("downsizing params to {} logrows", cli.args.logrows);
            if cli.args.logrows < params.k() {
                params.downsize(cli.args.logrows);
            }

            let vk =
                load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(vk_path.to_path_buf())?;
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
        }
        #[cfg(not(target_arch = "wasm32"))]
        Commands::CreateEVMVerifierAggr {
            params_path,
            deployment_code_path,
            vk_path,
        } => {
            let params: ParamsKZG<Bn256> = load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;

            let agg_vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path)?;

            let deployment_code = gen_aggregation_evm_verifier(
                &params,
                &agg_vk,
                AggregationCircuit::num_instance(),
                AggregationCircuit::accumulator_indices(),
            )?;
            deployment_code.save(deployment_code_path.as_ref().unwrap())?;
        }
        Commands::Prove {
            ref data,
            model,
            ref vk_path,
            ref proof_path,
            ref params_path,
            transcript,
            strategy,
        } => {
            let data = prepare_data(data.to_string())?;

            let model = Model::read_from_file(model)?;
            let circuit = ModelCircuit::<Fr>::new(&data, model)?;
            let public_inputs = circuit.prepare_public_inputs(&data)?;

            let mut params: ParamsKZG<Bn256> =
                load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;
            info!("downsizing params to {} logrows", cli.args.logrows);
            if cli.args.logrows < params.k() {
                params.downsize(cli.args.logrows);
            }
            let pk =
                create_keys::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(&circuit, &params)
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
                        cli.args.check_mode,
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
                        cli.args.check_mode,
                    )?
                }
            };

            info!("proof took {}", now.elapsed().as_secs());

            snark.save(proof_path)?;
            save_vk::<KZGCommitmentScheme<Bn256>>(vk_path, pk.get_vk())?;
        }
        Commands::Aggregate {
            model: _,
            proof_path,
            aggregation_snarks,
            ref aggregation_vk_paths,
            ref vk_path,
            ref params_path,
            app_logrows,
            transcript,
        } => {
            // the K used for the aggregation circuit
            let mut params: ParamsKZG<Bn256> =
                load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;
            info!("downsizing params to {} logrows", cli.args.logrows);
            if cli.args.logrows < params.k() {
                params.downsize(cli.args.logrows);
            }

            let mut snarks = vec![];
            // the K used when generating the application snark proof. we assume K is homogenous across snarks to aggregate
            let mut params_app = params.clone();
            info!("downsizing app params to {} logrows", app_logrows);
            if app_logrows < params.k() {
                params_app.downsize(app_logrows);
            }

            for (proof_path, vk_path) in aggregation_snarks.iter().zip(aggregation_vk_paths) {
                let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(
                    vk_path.to_path_buf(),
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
                    cli.args.check_mode,
                )?;

                info!("Aggregation proof took {}", now.elapsed().as_secs());
                snark.save(&proof_path)?;
                save_vk::<KZGCommitmentScheme<Bn256>>(vk_path, agg_pk.get_vk())?;
            }
        }
        Commands::Verify {
            model: _,
            proof_path,
            vk_path,
            params_path,
            transcript,
        } => {
            let mut params: ParamsKZG<Bn256> =
                load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;
            info!("downsizing params to {} logrows", cli.args.logrows);
            if cli.args.logrows < params.k() {
                params.downsize(cli.args.logrows);
            }

            let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;

            let strategy = KZGSingleStrategy::new(params.verifier_params());
            let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(vk_path)?;
            let result = verify_proof_circuit_kzg(
                params.verifier_params(),
                proof,
                &vk,
                transcript,
                strategy,
            );
            info!("verified: {}", result.is_ok());
        }

        Commands::VerifyAggr {
            proof_path,
            vk_path,
            params_path,
            transcript,
        } => {
            let mut params: ParamsKZG<Bn256> =
                load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;
            info!("downsizing params to {} logrows", cli.args.logrows);
            if cli.args.logrows < params.k() {
                params.downsize(cli.args.logrows);
            }

            let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;

            let strategy = AccumulatorStrategy::new(params.verifier_params());
            let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path)?;
            let result = verify_proof_circuit_kzg(&params, proof, &vk, transcript, strategy);
            info!("verified: {}", result.is_ok());
        }
        #[cfg(not(target_arch = "wasm32"))]
        Commands::VerifyEVM {
            proof_path,
            deployment_code_path,
            sol_code_path,
        } => {
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
        }
        Commands::PrintProofHex { proof_path } => {
            let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
            for instance in proof.instances {
                println!("{:?}", instance);
            }
            info!("{}", hex::encode(proof.proof))
        }
    }
    Ok(())
}
