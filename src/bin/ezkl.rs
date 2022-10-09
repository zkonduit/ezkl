use clap::Parser;
use colog;
use ezkl::commands::{data_path, Cli, Commands};
use ezkl::fieldutils::i32_to_felt;
use ezkl::onnx::{utilities::vector_to_quantized, OnnxCircuit, OnnxModel};
use ezkl::tensor::Tensor;
use halo2_proofs::dev::MockProver;
use halo2curves::pasta::Fp as F;
use log::{debug, info, trace};
use serde;
use serde::Deserialize;
use serde_json;
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
//use std::path::Path;

//use ezkl::fieldutils;
//use ezkl::tensor::*;
use halo2_proofs::{
    //    arithmetic::FieldExt,
    //    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof,
        keygen_pk,
        keygen_vk,
        verify_proof,
        Circuit, //Column, ConstraintSystem, Error,
                 // Fixed,
                 // Instance,
    },
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
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use std::time::Instant;
use tabled::Table;

#[derive(Debug, Deserialize)]
struct OnnxInput {
    input_data: Vec<f32>,
    input_shape: Vec<usize>,
    public_input: Vec<f32>,
}

pub fn main() {
    colog::init();
    let args = Cli::parse();
    banner();

    match args.command {
        Commands::Table { model: _ } => {
            let om = OnnxModel::from_arg();
            println!("{}", Table::new(om.onnx_nodes.clone()).to_string());
        }
        Commands::Mock { data, model: _ } => {
            let args = Cli::parse();
            let k = args.logrows;
            let mut file = File::open(data_path(data)).unwrap();
            let mut data = String::new();
            file.read_to_string(&mut data).unwrap();
            let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");

            // quantize the supplied data using the provided scale.
            let input =
                vector_to_quantized(&data.input_data, &data.input_shape, 0.0, args.scale).unwrap();
            info!(
                "public input length (network output) {:?}",
                data.public_input.len()
            );
            // quantize the supplied data using the provided scale.
            let public_input = vector_to_quantized(
                &data.public_input,
                &Vec::from([data.public_input.len()]),
                0.0,
                args.scale,
            )
            .unwrap();

            trace!("{:?}", public_input);

            let circuit = OnnxCircuit::<F> {
                input,
                _marker: PhantomData,
            };

            let prover = MockProver::run(
                k,
                &circuit,
                vec![public_input.iter().map(|x| i32_to_felt::<F>(*x)).collect()],
            )
            .unwrap();
            prover.assert_satisfied();
        }
        Commands::Fullprove {
            data,
            model: _,
            pfsys,
        } => {
            info!("Full proof with {}", pfsys);
            //            let proof = create_ipa_proof();
            let args = Cli::parse();
            // load
            let k = args.logrows;
            let mut file = File::open(data_path(data)).unwrap();
            let mut data = String::new();
            file.read_to_string(&mut data).unwrap();

            let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");

            // quantize the supplied data using the provided scale.
            let input =
                vector_to_quantized(&data.input_data, &data.input_shape, 0.0, args.scale).unwrap();
            info!(
                "public input length (network output) {:?}",
                data.public_input.len()
            );
            // quantize the supplied data using the provided scale.
            let public_input = vector_to_quantized(
                &data.public_input,
                &Vec::from([data.public_input.len()]),
                0.0,
                args.scale,
            )
            .unwrap();

            trace!("{:?}", public_input);

            let circuit = OnnxCircuit::<F> {
                input,
                _marker: PhantomData,
            };

            //	Real proof
            let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(k);
            trace!("params computed");
            let empty_circuit = circuit.without_witnesses();
            trace!("without witnesses done:");
            trace!("{:?}", empty_circuit);
            // Initialize the proving key
            let now = Instant::now();
            trace!("Preparing VK");
            let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
            println!("VK took {}", now.elapsed().as_secs());
            let now = Instant::now();
            let pk =
                keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
            println!("PK took {}", now.elapsed().as_secs());
            let now = Instant::now();
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
            let mut rng = OsRng;

            let pi_inner: Tensor<F> = public_input.map(|x| i32_to_felt::<F>(x).into());
            trace!("filling {:?}", pi_inner);
            let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner.into_iter().collect::<Vec<F>>()]];
            trace!("pi for real prover {:?}", pi_for_real_prover);

            create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
                &params,
                &pk,
                &[circuit],
                pi_for_real_prover,
                &mut rng,
                &mut transcript,
            )
            .expect("proof generation should not fail");
            let proof = transcript.finalize();
            //println!("{:?}", proof);
            println!("Proof took {}", now.elapsed().as_secs());

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
            println!("Verify took {}", now.elapsed().as_secs());
        }
        _ => todo!(),
    }
}

// fn create_ipa_proof() -> Vec<u8> {
//     let args = Cli::parse();
//     // load
//     let k = args.logrows;
//     let mut file = File::open(data_path(data)).unwrap();
//     let mut data = String::new();
//     file.read_to_string(&mut data).unwrap();

//     let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");

//     // quantize the supplied data using the provided scale.
//     let input = vector_to_quantized(&data.input_data, &data.input_shape, 0.0, args.scale).unwrap();
//     info!(
//         "public input length (network output) {:?}",
//         data.public_input.len()
//     );
//     // quantize the supplied data using the provided scale.
//     let public_input = vector_to_quantized(
//         &data.public_input,
//         &Vec::from([data.public_input.len()]),
//         0.0,
//         args.scale,
//     )
//     .unwrap();

//     trace!("{:?}", public_input);

//     let circuit = OnnxCircuit::<F> {
//         input,
//         _marker: PhantomData,
//     };

//     //	Real proof
//     let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(k);
//     trace!("params computed");
//     let empty_circuit = circuit.without_witnesses();
//     trace!("without witnesses done:");
//     trace!("{:?}", empty_circuit);
//     // Initialize the proving key
//     let now = Instant::now();
//     trace!("Preparing VK");
//     let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
//     println!("VK took {}", now.elapsed().as_secs());
//     let now = Instant::now();
//     let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
//     println!("PK took {}", now.elapsed().as_secs());
//     let now = Instant::now();
//     let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
//     let mut rng = OsRng;

//     let pi_inner: Tensor<F> = public_input.map(|x| i32_to_felt::<F>(x).into());
//     trace!("filling {:?}", pi_inner);
//     let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner.into_iter().collect::<Vec<F>>()]];
//     trace!("pi for real prover {:?}", pi_for_real_prover);

//     create_proof::<IPACommitmentScheme<_>, ProverIPA<_>, _, _, _, _>(
//         &params,
//         &pk,
//         &[circuit],
//         pi_for_real_prover,
//         &mut rng,
//         &mut transcript,
//     )
//     .expect("proof generation should not fail");
//     let proof = transcript.finalize();
//     //println!("{:?}", proof);
//     println!("Proof took {}", now.elapsed().as_secs());
//     proof
// }

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
