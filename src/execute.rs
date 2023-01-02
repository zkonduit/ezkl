use crate::abort;
use crate::commands::{Cli, Commands, ProofSystem};
use crate::fieldutils::i32_to_felt;
use crate::graph::Model;
#[cfg(feature = "evm")]
use crate::pfsys::aggregation::{
    evm_verify, gen_aggregation_evm_verifier, gen_application_snark, gen_kzg_proof, gen_pk,
    gen_srs, AggregationCircuit,
};
use crate::pfsys::{create_keys, load_params, load_vk, Proof};
use crate::pfsys::{
    create_proof_model, parse_prover_errors, prepare_circuit_and_public_input, prepare_data,
    save_params, save_vk, verify_proof_model,
};
#[cfg(feature = "evm")]
use halo2_proofs::poly::commitment::Params;
use halo2_proofs::poly::ipa::commitment::IPACommitmentScheme;
use halo2_proofs::poly::ipa::multiopen::ProverIPA;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::ProverGWC;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, multiopen::VerifierGWC, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::{
    dev::MockProver,
    poly::{
        commitment::ParamsProver,
        ipa::{commitment::ParamsIPA, strategy::SingleStrategy as IPASingleStrategy},
        VerificationStrategy,
    },
};
#[cfg(feature = "evm")]
use halo2curves::bn256::G1Affine;
use halo2curves::bn256::{Bn256, Fr};
use halo2curves::pasta::vesta;
use halo2curves::pasta::Fp;
use log::{error, info, trace};
#[cfg(feature = "evm")]
use plonk_verifier::system::halo2::transcript::evm::EvmTranscript;
#[cfg(feature = "evm")]
use std::time::Instant;
use tabled::Table;

/// Run an ezkl command with given args
pub fn run(args: Cli) {
    match args.command {
        Commands::Table { model: _ } => {
            let om = Model::from_ezkl_conf(args);
            println!("{}", Table::new(om.nodes.flatten()));
        }
        Commands::Mock { ref data, model: _ } => {
            let data = prepare_data(data.to_string());
            let (circuit, public_inputs) = prepare_circuit_and_public_input(&data, &args);
            info!("Mock proof");
            let pi: Vec<Vec<Fp>> = public_inputs
                .into_iter()
                .map(|i| i.into_iter().map(i32_to_felt::<Fp>).collect())
                .collect();

            let prover = match MockProver::run(args.logrows, &circuit, pi) {
                Ok(p) => p,
                Err(e) => {
                    abort!("mock prover failed to run {:?}", e);
                }
            };
            match prover.verify() {
                Ok(_) => {
                    info!("verify succeeded")
                }
                Err(v) => {
                    for e in v.iter() {
                        parse_prover_errors(e)
                    }
                    panic!()
                }
            }
        }

        Commands::Fullprove {
            ref data,
            model: _,
            pfsys,
        } => {
            // A direct proof

            let data = prepare_data(data.to_string());

            match pfsys {
                ProofSystem::IPA => {
                    let (circuit, public_inputs) =
                        prepare_circuit_and_public_input::<Fp>(&data, &args);
                    info!("full proof with {}", pfsys);

                    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
                    let pk = create_keys::<IPACommitmentScheme<_>, Fp>(&circuit, &params);
                    let strategy = IPASingleStrategy::new(&params);
                    trace!("params computed");

                    let (proof, _dims) = create_proof_model::<
                        IPACommitmentScheme<_>,
                        Fp,
                        ProverIPA<_>,
                    >(
                        &circuit, &public_inputs, &params, &pk
                    );

                    assert!(verify_proof_model(proof, &params, pk.get_vk(), strategy));
                }
                #[cfg(not(feature = "evm"))]
                ProofSystem::KZG => {
                    // A direct proof
                    let (circuit, public_inputs) =
                        prepare_circuit_and_public_input::<Fr>(&data, &args);
                    let params: ParamsKZG<Bn256> = ParamsKZG::new(args.logrows);
                    let pk = create_keys::<KZGCommitmentScheme<_>, Fr>(&circuit, &params);
                    let strategy = KZGSingleStrategy::new(&params);
                    trace!("params computed");

                    let (proof, _dims) = create_proof_model::<
                        KZGCommitmentScheme<_>,
                        Fr,
                        ProverGWC<_>,
                    >(
                        &circuit, &public_inputs, &params, &pk
                    );

                    assert!(verify_proof_model::<_, VerifierGWC<'_, Bn256>, _, _>(
                        proof,
                        &params,
                        pk.get_vk(),
                        strategy
                    ));
                }
                #[cfg(feature = "evm")]
                ProofSystem::KZG => {
                    // We will need aggregator k > application k > bits
                    //		    let application_logrows = args.logrows; //bits + 1;
                    let aggregation_logrows = args.logrows + 6;

                    let params = gen_srs(aggregation_logrows);
                    let params_app = {
                        let mut params = params.clone();
                        params.downsize(args.logrows);
                        params
                    };
                    let now = Instant::now();
                    let snarks = [(); 1].map(|_| gen_application_snark(&params_app, &data));
                    info!("Application proof took {}", now.elapsed().as_secs());
                    let agg_circuit = AggregationCircuit::new(&params, snarks);
                    let pk = gen_pk(&params, &agg_circuit);
                    let deployment_code = gen_aggregation_evm_verifier(
                        &params,
                        pk.get_vk(),
                        AggregationCircuit::num_instance(),
                        AggregationCircuit::accumulator_indices(),
                    );
                    let now = Instant::now();
                    let proof = gen_kzg_proof::<
                        _,
                        _,
                        EvmTranscript<G1Affine, _, _, _>,
                        EvmTranscript<G1Affine, _, _, _>,
                    >(
                        &params, &pk, agg_circuit.clone(), agg_circuit.instances()
                    );
                    info!("Aggregation proof took {}", now.elapsed().as_secs());
                    let now = Instant::now();
                    evm_verify(deployment_code, agg_circuit.instances(), proof);
                    info!("verify took {}", now.elapsed().as_secs());
                }
            }
        }
        Commands::Prove {
            ref data,
            model: _,
            ref proof_path,
            ref vk_path,
            ref params_path,
            pfsys,
        } => {
            let data = prepare_data(data.to_string());

            match pfsys {
                ProofSystem::IPA => {
                    info!("proof with {}", pfsys);
                    let (circuit, public_inputs) =
                        prepare_circuit_and_public_input::<Fp>(&data, &args);
                    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
                    let pk = create_keys::<IPACommitmentScheme<_>, Fp>(&circuit, &params);
                    trace!("params computed");

                    let (proof, _) = create_proof_model::<IPACommitmentScheme<_>, Fp, ProverIPA<_>>(
                        &circuit,
                        &public_inputs,
                        &params,
                        &pk,
                    );

                    proof.save(proof_path);
                    save_params::<IPACommitmentScheme<_>>(params_path, &params);
                    save_vk::<IPACommitmentScheme<_>>(vk_path, pk.get_vk());
                }
                ProofSystem::KZG => {
                    info!("proof with {}", pfsys);
                    let (circuit, public_inputs) = prepare_circuit_and_public_input(&data, &args);
                    let params: ParamsKZG<Bn256> = ParamsKZG::new(args.logrows);
                    let pk = create_keys::<KZGCommitmentScheme<Bn256>, Fr>(&circuit, &params);
                    trace!("params computed");

                    let (proof, _input_dims) = create_proof_model::<
                        KZGCommitmentScheme<Bn256>,
                        Fr,
                        ProverGWC<'_, Bn256>,
                    >(
                        &circuit, &public_inputs, &params, &pk
                    );

                    proof.save(proof_path);
                    save_params::<KZGCommitmentScheme<Bn256>>(params_path, &params);
                    save_vk::<KZGCommitmentScheme<Bn256>>(vk_path, pk.get_vk());
                }
            };
        }
        Commands::Verify {
            model: _,
            proof_path,
            vk_path,
            params_path,
            pfsys,
        } => {
            let proof = Proof::load(&proof_path);
            match pfsys {
                ProofSystem::IPA => {
                    let params: ParamsIPA<vesta::Affine> =
                        load_params::<IPACommitmentScheme<_>>(params_path);
                    let strategy = IPASingleStrategy::new(&params);
                    let vk = load_vk::<IPACommitmentScheme<_>, Fp>(vk_path, &params);
                    let result = verify_proof_model(proof, &params, &vk, strategy);
                    info!("verified: {}", result);
                    assert!(result);
                }
                ProofSystem::KZG => {
                    let params: ParamsKZG<Bn256> =
                        load_params::<KZGCommitmentScheme<Bn256>>(params_path);
                    let strategy = KZGSingleStrategy::new(&params);
                    let vk = load_vk::<KZGCommitmentScheme<Bn256>, Fr>(vk_path, &params);
                    let result = verify_proof_model::<_, VerifierGWC<'_, Bn256>, _, _>(
                        proof, &params, &vk, strategy,
                    );
                    info!("verified: {}", result);
                    assert!(result);
                }
            }
        }
    }
}
