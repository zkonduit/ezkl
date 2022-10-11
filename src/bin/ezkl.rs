use clap::Parser;
use ezkl::commands::{data_path, Cli, Commands};
use ezkl::fieldutils::i32_to_felt;
use ezkl::onnx::{utilities::vector_to_quantized, OnnxCircuit, OnnxModel};
use ezkl::tensor::Tensor;
use halo2_proofs::dev::MockProver;
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
use log::{info, trace};
use rand::rngs::OsRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::time::Instant;
use tabled::Table;

#[derive(Debug, Deserialize, Serialize)]
struct OnnxInput {
    input_data: Vec<f32>,
    input_shape: Vec<usize>,
    public_input: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Proof {
    input_shape: Vec<usize>,
    public_input: Vec<i32>,
    proof: Vec<u8>,
}

pub fn main() {
    colog::init();
    let args = Cli::parse();
    banner();

    match args.command {
        Commands::Table { model: _ } => {
            let om = OnnxModel::from_arg();
            println!("{}", Table::new(om.onnx_nodes.flatten()));
        }
        Commands::Mock { data, model: _ } => {
            info!("Mock proof");
            let args = Cli::parse();
            let (circuit, public_input) = prepare_circuit_and_public_input(data);
            let prover = MockProver::run(
                args.logrows,
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
            info!("full proof with {}", pfsys);
            let args = Cli::parse();
            let (circuit, public_input) = prepare_circuit_and_public_input(data);
            let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
            trace!("params computed");

            let (pk, proof, _dims) = create_ipa_proof(circuit, public_input.clone(), &params);

            let pi_inner: Tensor<F> = public_input.map(i32_to_felt::<F>);
            let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner.into_iter().collect::<Vec<F>>()]];

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
            info!("proof with {}", pfsys);
            let args = Cli::parse();
            let (circuit, public_input) = prepare_circuit_and_public_input(data);
            let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);
            trace!("params computed");

            let (_pk, proof, _input_dims) =
                create_ipa_proof(circuit.clone(), public_input.clone(), &params);

            let pi: Vec<_> = public_input.into_iter().collect();

            let checkable_pf = Proof {
                input_shape: circuit.input.dims().to_vec(),
                public_input: pi,
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
            let mut file = File::open(proof).unwrap();
            let mut data = String::new();
            file.read_to_string(&mut data).unwrap();
            let proof: Proof = serde_json::from_str(&data).expect("JSON was not well-formatted");

            let result = verify_ipa_proof(proof);
            info!("verified: {}", result);
            println!("Verified: {}", result)
        }
    }
}

fn prepare_circuit_and_public_input<F: FieldExt>(data: String) -> (OnnxCircuit<F>, Tensor<i32>) {
    let args = Cli::parse();
    let data = prepare_data(data);

    // quantize the supplied data using the provided scale.
    let public_input = vector_to_quantized(
        &data.public_input,
        &Vec::from([data.public_input.len()]),
        0.0,
        args.scale,
    )
    .unwrap();

    trace!("{:?}", public_input);
    let circuit = prepare_circuit(data);
    (circuit, public_input)
}

fn prepare_circuit<F: FieldExt>(data: OnnxInput) -> OnnxCircuit<F> {
    let args = Cli::parse();

    // quantize the supplied data using the provided scale.
    let input = vector_to_quantized(&data.input_data, &data.input_shape, 0.0, args.scale).unwrap();
    OnnxCircuit::<F> {
        input,
        _marker: PhantomData,
    }
}

fn prepare_data(datapath: String) -> OnnxInput {
    let mut file = File::open(data_path(datapath)).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();
    let data: OnnxInput = serde_json::from_str(&data).expect("JSON was not well-formatted");
    info!(
        "public input length (network output) {:?}",
        data.public_input.len()
    );

    data
}

fn create_ipa_proof(
    circuit: OnnxCircuit<Fp>,
    public_input: Tensor<i32>,
    params: &ParamsIPA<vesta::Affine>,
) -> (ProvingKey<EqAffine>, Vec<u8>, Vec<usize>) {
    //let args = Cli::parse();
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

    let pi_inner: Tensor<F> = public_input.map(i32_to_felt::<F>);
    trace!("filling {:?}", pi_inner);
    let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner.into_iter().collect::<Vec<F>>()]];
    trace!("pi for real prover {:?}", pi_for_real_prover);

    let dims = circuit.input.dims().to_vec();

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
    //println!("{:?}", proof);
    info!("Proof took {}", now.elapsed().as_secs());

    (pk, proof, dims)
}

fn verify_ipa_proof(proof: Proof) -> bool {
    let args = Cli::parse();
    let params: ParamsIPA<vesta::Affine> = ParamsIPA::new(args.logrows);

    let input = Tensor::new(
        Some(&vec![0; proof.input_shape.iter().product()]),
        &proof.input_shape,
    )
    .unwrap();
    let circuit = OnnxCircuit::<F> {
        input,
        _marker: PhantomData,
    };
    let empty_circuit = circuit.without_witnesses();
    let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    let pk = keygen_pk(&params, vk, &empty_circuit).expect("keygen_pk should not fail");

    let pi_inner = proof
        .public_input
        .into_iter()
        .map(i32_to_felt::<F>)
        .collect::<Vec<F>>();
    let pi_for_real_prover: &[&[&[F]]] = &[&[&pi_inner]];

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
