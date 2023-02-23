use crate::commands::{Cli, Commands, ProofSystem, StrategyType, TranscriptType};
use crate::graph::{Model, ModelCircuit};
use crate::pfsys::evm::aggregation::{
    gen_aggregation_evm_verifier, AggregationCircuit, PoseidonTranscript,
};
use crate::pfsys::evm::single::gen_evm_verifier;
use crate::pfsys::evm::{evm_verify, DeploymentCode};
use crate::pfsys::{create_keys, load_params, load_vk, Snark};
use crate::pfsys::{
    create_proof_circuit, gen_srs, prepare_data, prepare_model_circuit_and_public_input,
    save_params, save_vk, verify_proof_circuit,
};
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
use log::{info, trace};
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::process::Command;
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
        >(&proof, &params, &vk, strategy),
        TranscriptType::EVM => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            _,
            EvmTranscript<G1Affine, _, _, _>,
        >(&proof, &params, &vk, strategy),
        TranscriptType::Poseidon => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            _,
            PoseidonTranscript<NativeLoader, _>,
        >(&proof, &params, &vk, strategy),
    }
}

/// helper function
fn create_proof_circuit_kzg<
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
        >(circuit, public_inputs, &params, &pk, strategy)
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
        >(circuit, public_inputs, &params, &pk, strategy)
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
        >(circuit, public_inputs, &params, &pk, strategy)
        .map_err(Box::<dyn Error>::from),
    }
}

/// Run an ezkl command with given args
pub fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        Commands::GenSrs { params_path, pfsys } => match pfsys {
            ProofSystem::IPA => {
                unimplemented!()
            }
            ProofSystem::KZG => {
                let params = gen_srs::<KZGCommitmentScheme<Bn256>>(cli.args.logrows);
                save_params::<KZGCommitmentScheme<Bn256>>(&params_path, &params)?;
            }
        },
        Commands::Table { model: _ } => {
            let om = Model::from_ezkl_conf(cli)?;
            println!("{}", Table::new(om.nodes.flatten()));
        }
        Commands::Mock { ref data, model: _ } => {
            let data = prepare_data(data.to_string())?;
            let (circuit, public_inputs) =
                prepare_model_circuit_and_public_input::<Fr>(&data, &cli)?;
            info!("Mock proof");

            let prover = MockProver::run(cli.args.logrows, &circuit, public_inputs)
                .map_err(Box::<dyn Error>::from)?;
            prover
                .verify()
                .map_err(|e| Box::<dyn Error>::from(ExecutionError::VerifyError(e)))?;
        }
        Commands::CreateEVMVerifier {
            ref data,
            model: _,
            ref vk_path,
            ref params_path,
            ref deployment_code_path,
            ref sol_code_path,
            pfsys,
        } => {
            let data = prepare_data(data.to_string())?;
            match pfsys {
                ProofSystem::IPA => {
                    unimplemented!()
                }
                ProofSystem::KZG => {
                    //let _ = (data, vk_path, params_path, deployment_code_path);
                    let (_, public_inputs) =
                        prepare_model_circuit_and_public_input::<Fr>(&data, &cli)?;
                    let num_instance = public_inputs.iter().map(|x| x.len()).collect();
                    let mut params: ParamsKZG<Bn256> =
                        load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;
                    params.downsize(cli.args.logrows);
                    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(
                        vk_path.to_path_buf(),
                    )?;
                    trace!("params computed");

                    let (deployment_code, yul_code) = gen_evm_verifier(&params, &vk, num_instance)?;
                    deployment_code.save(&deployment_code_path.as_ref().unwrap())?;

                    let mut f = File::create(sol_code_path.as_ref().unwrap()).unwrap();
                    let _ = f.write(yul_code.as_bytes());

                    let cmd = Command::new("python3")
                        .arg("fix_verifier_sol.py")
                        .arg(sol_code_path.as_ref().unwrap())
                        .output()
                        .unwrap();
                    let output = cmd.stdout;

                    let mut f = File::create(sol_code_path.as_ref().unwrap()).unwrap();
                    let _ = f.write(output.as_slice());
                }
            }
        }
        Commands::CreateEVMVerifierAggr {
            params_path,
            deployment_code_path,
            pfsys,
            vk_path,
        } => match pfsys {
            ProofSystem::IPA => {
                unimplemented!()
            }
            ProofSystem::KZG => {
                let params: ParamsKZG<Bn256> =
                    load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;

                let agg_vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(
                    vk_path.to_path_buf(),
                )?;

                let deployment_code = gen_aggregation_evm_verifier(
                    &params,
                    &agg_vk,
                    AggregationCircuit::num_instance(),
                    AggregationCircuit::accumulator_indices(),
                )?;
                deployment_code.save(&deployment_code_path.as_ref().unwrap())?;
            }
        },
        Commands::Prove {
            ref data,
            model: _,
            ref vk_path,
            ref proof_path,
            ref params_path,
            pfsys,
            transcript,
            strategy,
        } => {
            let data = prepare_data(data.to_string())?;

            match pfsys {
                ProofSystem::IPA => {
                    unimplemented!()
                }
                ProofSystem::KZG => {
                    info!("proof with {}", pfsys);
                    let (circuit, public_inputs) =
                        prepare_model_circuit_and_public_input(&data, &cli)?;
                    let mut params: ParamsKZG<Bn256> =
                        load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;
                    params.downsize(cli.args.logrows);
                    let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(
                        &circuit, &params,
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
                            )?
                        }
                    };

                    info!("proof took {}", now.elapsed().as_secs());

                    snark.save(proof_path)?;
                    save_vk::<KZGCommitmentScheme<Bn256>>(vk_path, pk.get_vk())?;
                }
            };
        }
        Commands::Aggregate {
            model: _,
            proof_path,
            aggregation_snarks,
            ref aggregation_vk_paths,
            ref vk_path,
            ref params_path,
            pfsys,
            transcript,
        } => {
            match pfsys {
                ProofSystem::IPA => {
                    unimplemented!()
                }
                ProofSystem::KZG => {
                    let params: ParamsKZG<Bn256> =
                        load_params::<KZGCommitmentScheme<Bn256>>(params_path.to_path_buf())?;

                    let mut snarks = vec![];
                    for (proof_path, vk_path) in aggregation_snarks.iter().zip(aggregation_vk_paths)
                    {
                        let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(
                            vk_path.to_path_buf(),
                        )?;
                        let params_app = {
                            let mut params_app = params.clone();
                            params_app.downsize(cli.args.logrows);
                            params_app
                        };
                        snarks.push(Snark::load::<KZGCommitmentScheme<Bn256>>(
                            &proof_path,
                            Some(&params_app),
                            Some(&vk),
                        )?);
                    }
                    // proof aggregation
                    {
                        let agg_circuit = AggregationCircuit::new(&params, snarks)?;
                        let agg_pk = create_keys::<
                            KZGCommitmentScheme<Bn256>,
                            Fr,
                            AggregationCircuit,
                        >(&agg_circuit, &params)?;

                        let now = Instant::now();
                        let snark = create_proof_circuit_kzg(
                            agg_circuit.clone(),
                            &params,
                            agg_circuit.instances(),
                            &agg_pk,
                            transcript,
                            AccumulatorStrategy::new(&params),
                        )?;

                        info!("Aggregation proof took {}", now.elapsed().as_secs());
                        snark.save(&proof_path)?;
                        save_vk::<KZGCommitmentScheme<Bn256>>(vk_path, agg_pk.get_vk())?;
                    }
                }
            }
        }
        Commands::Verify {
            model: _,
            proof_path,
            vk_path,
            params_path,
            pfsys,
            transcript,
        } => match pfsys {
            ProofSystem::IPA => {
                unimplemented!()
            }
            ProofSystem::KZG => {
                let mut params: ParamsKZG<Bn256> =
                    load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;
                params.downsize(cli.args.logrows);

                let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;

                let strategy = KZGSingleStrategy::new(&params.verifier_params());
                let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, ModelCircuit<Fr>>(vk_path)?;
                let result = verify_proof_circuit_kzg(
                    &params.verifier_params(),
                    proof,
                    &vk,
                    transcript,
                    strategy,
                );
                info!("verified: {}", result.is_ok());
            }
        },

        Commands::VerifyAggr {
            proof_path,
            vk_path,
            params_path,
            pfsys,
            transcript,
        } => match pfsys {
            ProofSystem::IPA => {
                unimplemented!()
            }
            ProofSystem::KZG => {
                let mut params: ParamsKZG<Bn256> =
                    load_params::<KZGCommitmentScheme<Bn256>>(params_path)?;
                params.downsize(cli.args.logrows);

                let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;

                let strategy = AccumulatorStrategy::new(&params.verifier_params());
                let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr, AggregationCircuit>(vk_path)?;
                let result = verify_proof_circuit_kzg(&params, proof, &vk, transcript, strategy);
                info!("verified: {}", result.is_ok());
            }
        },
        Commands::VerifyEVM {
            proof_path,
            deployment_code_path,
            pfsys,
        } => match pfsys {
            ProofSystem::IPA => {
                unimplemented!()
            }
            ProofSystem::KZG => {
                let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
                let code = DeploymentCode::load(&deployment_code_path)?;
                evm_verify(code, proof)?;
            }
        },
        Commands::PrintProofHex { proof_path, pfsys } => match pfsys {
            ProofSystem::IPA => {
                unimplemented!()
            }
            ProofSystem::KZG => {
                let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path, None, None)?;
                for instance in proof.instances {
                    println!("{:?}", instance);
                }
                println!("{}", hex::encode(proof.proof))
            }
        },
    }
    Ok(())
}
