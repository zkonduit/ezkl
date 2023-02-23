use crate::pfsys::evm::DeploymentCode;
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::{plonk::VerifyingKey, poly::kzg::commitment::ParamsKZG};
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine};
use snark_verifier::{
    loader::evm::{self, EvmLoader},
    pcs::kzg::{Gwc19, KzgAs},
    system::halo2::{compile, transcript::evm::EvmTranscript, Config},
    verifier::{self, SnarkVerifier},
};
use std::rc::Rc;
use thiserror::Error;

type PlonkVerifier = verifier::plonk::PlonkVerifier<KzgAs<Bn256, Gwc19>>;

#[derive(Error, Debug)]
/// Errors related to simple evm verifier generation
pub enum SimpleError {
    /// proof read errors
    #[error("Failed to read proof")]
    ProofRead,
    /// proof verification errors
    #[error("Failed to verify proof")]
    ProofVerify,
}

/// Create EVM verifier bytecode
pub fn gen_evm_verifier(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    num_instance: Vec<usize>,
) -> Result<(DeploymentCode, String), SimpleError> {
    let protocol = compile(
        params,
        vk,
        Config::kzg().with_num_instance(num_instance.clone()),
    );
    let vk = (params.get_g()[0], params.g2(), params.s_g2()).into();

    let loader = EvmLoader::new::<Fq, Fr>();
    let protocol = protocol.loaded(&loader);
    let mut transcript = EvmTranscript::<_, Rc<EvmLoader>, _, _>::new(&loader);

    let instances = transcript.load_instances(num_instance);
    let proof = PlonkVerifier::read_proof(&vk, &protocol, &instances, &mut transcript)
        .map_err(|_| SimpleError::ProofRead)?;
    PlonkVerifier::verify(&vk, &protocol, &instances, &proof)
        .map_err(|_| SimpleError::ProofVerify)?;

    let yul_code = &loader.yul_code();

    Ok((DeploymentCode {
        code: evm::compile_yul(yul_code),
    }, yul_code.clone()))
}
