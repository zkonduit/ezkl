use crate::fieldutils::i32_to_felt;
use crate::graph::ModelCircuit;
use crate::tensor::Tensor;
use halo2_proofs::{
    plonk::{create_proof, keygen_pk, keygen_vk, Circuit, ProvingKey},
    poly::kzg::{
        commitment::{KZGCommitmentScheme, ParamsKZG},
        multiopen::ProverGWC,
    },
    transcript::{Blake2bWrite, Challenge255, TranscriptWriterBuffer},
};
use halo2curves::bn256::{Bn256, Fr as F, G1Affine};
use log::{info, trace};
use rand::rngs::OsRng;
use std::ops::Deref;
use std::time::Instant;

pub fn create_kzg_proof(
    circuit: ModelCircuit<F>,
    public_inputs: Vec<Tensor<i32>>,
    params: &ParamsKZG<Bn256>,
) -> (ProvingKey<G1Affine>, Vec<u8>, Vec<Vec<usize>>) {
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

    create_proof::<KZGCommitmentScheme<_>, ProverGWC<_>, _, _, _, _>(
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
