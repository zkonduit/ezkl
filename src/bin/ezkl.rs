use clap::Parser;
use ezkl::abort;
use ezkl::commands::{Cli, Commands, ProofSystem};
use ezkl::fieldutils::i32_to_felt;
use ezkl::graph::Model;
use ezkl::pfsys::ipa::{create_ipa_proof, verify_ipa_proof};
#[cfg(feature = "evm")]
use ezkl::pfsys::kzg::aggregation::{
    aggregation::AggregationCircuit, evm_verify, gen_aggregation_evm_verifier,
    gen_application_snark, gen_kzg_proof, gen_pk, gen_srs,
};
#[cfg(not(feature = "evm"))]
use ezkl::pfsys::kzg::single::{create_kzg_proof, verify_kzg_proof};
use ezkl::pfsys::Proof;
use ezkl::pfsys::{parse_prover_errors, prepare_circuit_and_public_input, prepare_data};
#[cfg(feature = "evm")]
use halo2_proofs::poly::commitment::Params;
#[cfg(not(feature = "evm"))]
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, multiopen::VerifierGWC, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::{
    dev::MockProver,
    plonk::verify_proof,
    poly::{
        commitment::ParamsProver,
        ipa::{commitment::ParamsIPA, strategy::SingleStrategy as IPASingleStrategy},
        VerificationStrategy,
    },
    transcript::{Blake2bRead, Challenge255, TranscriptReadBuffer},
};
use halo2curves::bn256::{Bn256, Fr};
use halo2curves::pasta::vesta;
use halo2curves::pasta::Fp;
use log::{error, info, trace};
#[cfg(feature = "evm")]
use plonk_verifier::system::halo2::transcript::evm::EvmTranscript;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{Read, Write};
use std::ops::Deref;
use std::time::Instant;
use tabled::Table;

pub fn main() {
    let args = Cli::parse();
    colog::init();
    banner();

    match args.command {
        Commands::Table { model: _ } => {
            let om = Model::from_arg();
            println!("{}", Table::new(om.nodes.flatten()));
        }
        Commands::Mock { data, model: _ } => {
            let args = Cli::parse();
            let data = prepare_data(data);
            let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
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
            data,
            model: _,
            pfsys,
        } => {
            // A direct proof
            let args = Cli::parse();
            let data = prepare_data(data);
            match pfsys {
                ProofSystem::IPA => {
                    let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
                    info!("full proof with {}", pfsys);

                    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
                    trace!("params computed");

                    let (pk, proof, _dims) =
                        create_ipa_proof(circuit, public_inputs.clone(), &params);

                    let pi_inner: Vec<Vec<Fp>> = public_inputs
                        .iter()
                        .map(|i| i.iter().map(|e| i32_to_felt::<Fp>(*e)).collect::<Vec<Fp>>())
                        .collect::<Vec<Vec<Fp>>>();
                    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[Fp]>>();
                    let pi_for_real_prover: &[&[&[Fp]]] = &[&pi_inner];

                    let now = Instant::now();
                    let strategy = IPASingleStrategy::new(&params);
                    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
                    assert!(verify_proof(
                        &params,
                        pk.get_vk(),
                        strategy,
                        pi_for_real_prover,
                        &mut transcript
                    )
                    .is_ok());
                    info!("verify took {}", now.elapsed().as_secs());
                }
                #[cfg(not(feature = "evm"))]
                ProofSystem::KZG => {
                    // A direct proof
                    let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
                    let params: ParamsKZG<Bn256> = ParamsKZG::new(args.logrows);
                    trace!("params computed");

                    let (pk, proof, _dims) =
                        create_kzg_proof(circuit, public_inputs.clone(), &params);

                    let pi_inner: Vec<Vec<Fr>> = public_inputs
                        .iter()
                        .map(|i| i.iter().map(|e| i32_to_felt::<Fr>(*e)).collect::<Vec<Fr>>())
                        .collect::<Vec<Vec<Fr>>>();
                    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[Fr]>>();
                    let pi_for_real_prover: &[&[&[Fr]]] = &[&pi_inner];

                    let now = Instant::now();
                    let strategy = KZGSingleStrategy::new(&params);
                    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
                    assert!(verify_proof::<_, VerifierGWC<_>, _, _, _>(
                        &params,
                        pk.get_vk(),
                        strategy,
                        pi_for_real_prover,
                        &mut transcript
                    )
                    .is_ok());
                    info!("verify took {}", now.elapsed().as_secs());
                }
                #[cfg(feature = "evm")]
                ProofSystem::KZG => {
                    // We will need aggregator k > application k > bits
                    //		    let application_logrows = args.logrows; //bits + 1;
                    let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
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
            data,
            model: _,
            output,
            pfsys,
        } => {
            let args = Cli::parse();
            let data = prepare_data(data);
            let checkable_pf = match pfsys {
                ProofSystem::IPA => {
                    let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
                    info!("proof with {}", pfsys);
                    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
                    trace!("params computed");

                    let (_pk, proof, _input_dims) =
                        create_ipa_proof(circuit.clone(), public_inputs.clone(), &params);

                    let pi: Vec<_> = public_inputs
                        .into_iter()
                        .map(|i| i.into_iter().collect())
                        .collect();

                    Proof {
                        input_shapes: circuit.inputs.iter().map(|i| i.dims().to_vec()).collect(),
                        public_inputs: pi,
                        proof,
                    }
                }
                ProofSystem::KZG => {
                    let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
                    info!("proof with {}", pfsys);
                    let params: ParamsKZG<Bn256> = ParamsKZG::new(args.logrows);
                    trace!("params computed");

                    let (_pk, proof, _input_dims) =
                        create_kzg_proof(circuit.clone(), public_inputs.clone(), &params);

                    let pi: Vec<_> = public_inputs
                        .into_iter()
                        .map(|i| i.into_iter().collect())
                        .collect();

                    Proof {
                        input_shapes: circuit.inputs.iter().map(|i| i.dims().to_vec()).collect(),
                        public_inputs: pi,
                        proof,
                    }
                }
            };
            let serialized = match serde_json::to_string(&checkable_pf) {
                Ok(s) => s,
                Err(e) => {
                    abort!("failed to convert proof json to string {:?}", e);
                }
            };

            let mut file = std::fs::File::create(output).expect("create failed");
            file.write_all(serialized.as_bytes()).expect("write failed");
        }
        Commands::Verify {
            model: _,
            proof,
            pfsys,
        } => {
            let mut file = match File::open(proof) {
                Ok(f) => f,
                Err(e) => {
                    abort!("failed to open proof file {:?}", e);
                }
            };
            let mut data = String::new();
            match file.read_to_string(&mut data) {
                Ok(_) => {}
                Err(e) => {
                    abort!("failed to read file {:?}", e);
                }
            };
            let proof: Proof = serde_json::from_str(&data).expect("JSON was not well-formatted");
            match pfsys {
                ProofSystem::IPA => {
                    let result = verify_ipa_proof(proof);
                    info!("verified: {}", result);
                    assert!(result);
                }
                ProofSystem::KZG => {
                    let result = verify_kzg_proof(proof);
                    info!("verified: {}", result);
                    assert!(result);
                }
            }
        }
    }
}

fn banner() {
    let ell: Vec<&str> = vec![
        "for Neural Networks",
        "Linear Algebra",
        "for Layers",
        "for the Laconic",
        "Learning",
        "for Liberty",
        "for the Lyrical",
    ];
    info!(
        "{}",
        format!(
            "
        ███████╗███████╗██╗  ██╗██╗
        ██╔════╝╚══███╔╝██║ ██╔╝██║
        █████╗    ███╔╝ █████╔╝ ██║
        ██╔══╝   ███╔╝  ██╔═██╗ ██║
        ███████╗███████╗██║  ██╗███████╗
        ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝

        -----------------------------------------------------------
        Easy Zero Knowledge {}.
        -----------------------------------------------------------
        ",
            ell.choose(&mut rand::thread_rng()).unwrap()
        )
    );
}
