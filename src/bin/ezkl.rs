use clap::Parser;
use ezkl::commands::{data_path, Cli, Commands};
use ezkl::fieldutils::i32_to_felt;
use ezkl::onnx::{utilities::vector_to_quantized, OnnxCircuit, OnnxModel};
use ezkl::tensor::Tensor;
use halo2_proofs::dev::{MockProver, VerifyFailure};
use halo2_proofs::plonk::ProvingKey;
use halo2_proofs::{
    arithmetic::FieldExt,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Circuit},
    poly::{
        commitment::ParamsProver,
        ipa::{
            commitment::{IPACommitmentScheme, ParamsIPA},
            multiopen::ProverIPA,
            strategy::SingleStrategy,
        },
        VerificationStrategy,
    },
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
    },
};
use halo2curves::pasta::vesta;
use halo2curves::pasta::Fp;
use halo2curves::pasta::{EqAffine, Fp as F};
use log::{debug, error, info, trace};
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::time::Instant;
use tabled::Table;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OnnxInput {
    input_data: Vec<Vec<f32>>,
    input_shapes: Vec<Vec<usize>>,
    public_inputs: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Proof {
    input_shapes: Vec<Vec<usize>>,
    public_inputs: Vec<Vec<i32>>,
    proof: Vec<u8>,
}

/// Helper function for printing helpful error messages after verification failed.
fn format_verify_errors(f: &VerifyFailure) {
    match f {
        VerifyFailure::Lookup {
            name,
            location,
            lookup_index,
        } => {
            error!(
                "lookup {:?} is out of range, try increasing 'bits' or reducing 'scale'",
                name
            );
            debug!("location {:?} at lookup index {:?}", location, lookup_index);
        }
        VerifyFailure::ConstraintNotSatisfied {
            constraint,
            location,
            cell_values,
        } => {
            error!("constraint {:?} was not satisfied", constraint);
            debug!("location {:?} with values {:?}", location, cell_values);
        }
        VerifyFailure::ConstraintPoisoned { constraint } => {
            error!("constraint {:?} was poisoned", constraint)
        }
        VerifyFailure::Permutation { column, location } => {
            error!("permutation did not preserve column cell value (try increasing 'scale')");
            debug!("column {:?}, at location {:?}", column, location);
        }
        e => error!("{:?}", e),
    }
}

pub fn main() {
    let args = Cli::parse();
    banner();

    match args.command {
        Commands::Table { model: _ } => {
            colog::init();
            let om = OnnxModel::from_arg();
            println!("{}", Table::new(om.onnx_nodes.flatten()));
        }
        Commands::Mock { data, model: _ } => {
            let args = Cli::parse();
            let data = prepare_data(data);
            let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
            info!("Mock proof");
            let pi: Vec<Vec<F>> = public_inputs
                .into_iter()
                .map(|i| i.into_iter().map(i32_to_felt::<F>).collect())
                .collect();

            let prover = MockProver::run(args.logrows, &circuit, pi).unwrap();
            match prover.verify() {
                Ok(_) => {
                    info!("verify succeeded")
                }
                Err(v) => {
                    for e in v.iter() {
                        format_verify_errors(e)
                    }
                }
            }
        }

        Commands::Fullprove {
            data,
            model: _,
            pfsys,
        } => {
            let args = Cli::parse();
            let data = prepare_data(data);
            let (circuit, public_inputs) = prepare_circuit_and_public_input(&data);
            info!("full proof with {}", pfsys);
            let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
            trace!("params computed");

            let (pk, proof, _dims) = create_ipa_proof(circuit, public_inputs.clone(), &params);

            let pi_inner: Vec<Vec<F>> = public_inputs
                .iter()
                .map(|i| i.iter().map(|e| i32_to_felt::<F>(*e)).collect::<Vec<F>>())
                .collect::<Vec<Vec<F>>>();
            let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[F]>>();
            let pi_for_real_prover: &[&[&[F]]] = &[&pi_inner];

            let now = Instant::now();
            let strategy = SingleStrategy::new(&params);
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
        Commands::Prove {
            data,
            model: _,
            output,
            pfsys,
        } => {
            let args = Cli::parse();
            let data = prepare_data(data);
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

            let checkable_pf = Proof {
                input_shapes: circuit.inputs.iter().map(|i| i.dims().to_vec()).collect(),
                public_inputs: pi,
                proof,
            };

            let serialized = serde_json::to_string(&checkable_pf).unwrap();

            let mut file = std::fs::File::create(output).expect("create failed");
            file.write_all(serialized.as_bytes()).expect("write failed");
        }
        Commands::Verify {
            model: _,
            proof,
            pfsys: _,
        } => {
            colog::init();
            let mut file = File::open(proof).unwrap();
            let mut data = String::new();
            file.read_to_string(&mut data).unwrap();
            let proof: Proof = serde_json::from_str(&data).expect("JSON was not well-formatted");

            let result = verify_ipa_proof(proof);
            info!("verified: {}", result);
            assert!(result);
        }
    }
}

fn prepare_circuit_and_public_input<F: FieldExt>(
    data: &OnnxInput,
) -> (OnnxCircuit<F>, Vec<Tensor<i32>>) {
    let onnx_model = OnnxModel::from_arg();
    let out_scales = onnx_model.get_output_scales();
    colog::init();
    let circuit = prepare_circuit(data);

    // quantize the supplied data using the provided scale.
    let public_inputs = data
        .public_inputs
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            vector_to_quantized(v, &Vec::from([v.len()]), 0.0, out_scales[idx]).unwrap()
        })
        .collect();
    trace!("{:?}", public_inputs);
    (circuit, public_inputs)
}

fn prepare_circuit<F: FieldExt>(data: &OnnxInput) -> OnnxCircuit<F> {
    let args = Cli::parse();

    // quantize the supplied data using the provided scale.
    let inputs = data
        .input_data
        .iter()
        .zip(data.input_shapes.clone())
        .map(|(i, s)| vector_to_quantized(i, &s, 0.0, args.scale).unwrap())
        .collect();

    OnnxCircuit::<F> {
        inputs,
        _marker: PhantomData,
    }
}

fn prepare_data(datapath: String) -> OnnxInput {
    let mut file = File::open(data_path(datapath)).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();
    let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");
    info!(
        "public inputs (network outputs) lengths: {:?}",
        data.public_inputs
            .iter()
            .map(|i| i.len())
            .collect::<Vec<usize>>()
    );

    data
}

fn create_ipa_proof(
    circuit: OnnxCircuit<Fp>,
    public_inputs: Vec<Tensor<i32>>,
    params: &ParamsIPA<vesta::Affine>,
) -> (ProvingKey<EqAffine>, Vec<u8>, Vec<Vec<usize>>) {
    //	Real proof
    let empty_circuit = circuit.without_witnesses();

    // Initialize the proving key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk(params, &empty_circuit).expect("keygen_vk should not fail");
    info!("VK took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let pk = keygen_pk(params, vk, &empty_circuit).expect("keygen_pk should not fail");
    info!("PK took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    let mut rng = OsRng;

    let pi_inner: Vec<Vec<F>> = public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i32_to_felt::<F>(*e)).collect::<Vec<F>>())
        .collect::<Vec<Vec<F>>>();
    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[F]>>();
    let pi_for_real_prover: &[&[&[F]]] = &[&pi_inner];
    trace!("pi for real prover {:?}", pi_for_real_prover);

    let dims = circuit.inputs.iter().map(|i| i.dims().to_vec()).collect();

    create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
        params,
        &pk,
        &[circuit],
        pi_for_real_prover,
        &mut rng,
        &mut transcript,
    )
    .expect("proof generation should not fail");
    let proof = transcript.finalize();
    info!("Proof took {}", now.elapsed().as_secs());

    (pk, proof, dims)
}

fn verify_ipa_proof(proof: Proof) -> bool {
    let args = Cli::parse();
    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);

    let inputs = proof
        .input_shapes
        .iter()
        .map(|s| Tensor::new(Some(&vec![0; s.iter().product()]), s).unwrap())
        .collect();
    let circuit = OnnxCircuit::<F> {
        inputs,
        _marker: PhantomData,
    };
    let empty_circuit = circuit.without_witnesses();
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");

    let pi_inner: Vec<Vec<F>> = proof
        .public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i32_to_felt::<F>(*e)).collect::<Vec<F>>())
        .collect::<Vec<Vec<F>>>();
    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[F]>>();
    let pi_for_real_prover: &[&[&[F]]] = &[&pi_inner];
    trace!("pi for real prover {:?}", pi_for_real_prover);

    let now = Instant::now();
    let strategy = SingleStrategy::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof.proof[..]);

    trace!("params computed");

    let result = verify_proof(
        &params,
        pk.get_vk(),
        strategy,
        pi_for_real_prover,
        &mut transcript,
    )
    .is_ok();
    info!("verify took {}", now.elapsed().as_secs());
    result
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
