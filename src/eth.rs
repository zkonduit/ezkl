use ethers::core::k256::ecdsa::SigningKey;
use ethers::middleware::SignerMiddleware;
use ethers::providers::{Http, Provider};
use ethers::signers::Signer;
use ethers::utils::AnvilInstance;
use ethers::{
    prelude::{LocalWallet, Wallet},
    utils::Anvil,
};
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use std::{convert::TryFrom, sync::Arc, time::Duration};
use ethers_solc::Solc;
use ethers::contract::ContractFactory;
use ethers::contract::abigen;
use ethers::types::U256;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;

/// A local ethers-rs based client
pub type EthersClient = Arc<SignerMiddleware<Provider<Http>, Wallet<SigningKey>>>;

/// Return an instance of Anvil and a local client
pub async fn setup_eth_backend() -> (AnvilInstance, EthersClient) {
    // Launch anvil
    let anvil = Anvil::new().spawn();

    // Instantiate the wallet
    let wallet: LocalWallet = anvil.keys()[0].clone().into();

    // Connect to the network
    let provider = Provider::<Http>::try_from(anvil.endpoint())
        .unwrap()
        .interval(Duration::from_millis(10u64));

    // Instantiate the client with the wallet
    let client = Arc::new(SignerMiddleware::new(
        provider,
        wallet.with_chain_id(anvil.chain_id()),
    ));

    (anvil, client)
}

/// Verify a proof using a Solidity verifier contract
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    sol_code_path: std::path::PathBuf,
) -> Result<bool, Box<EvmVerificationError>> {
    let (anvil, client) = setup_eth_backend().await;

    let compiled = Solc::default().compile_source(sol_code_path).unwrap();
    let (abi, bytecode, _runtime_bytecode) =
        compiled.find("Verifier").expect("could not find contract").into_parts_or_default();
    let factory = ContractFactory::new(abi, bytecode, client.clone());
    let contract = factory.deploy(()).unwrap().send().await.unwrap();
    let addr = contract.address();

    abigen!(Verifier, "./Verifier.json");
    let contract = Verifier::new(addr, client.clone());

    let mut public_inputs = vec![];
    for val in &proof.instances[0] {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.as_slice());
        public_inputs.push(u);
    }

    let result = contract.verify(
        public_inputs,
        ethers::types::Bytes::from(proof.proof.to_vec()),
        ).call().await;

    if result.is_err() {
        return Err(Box::new(EvmVerificationError::SolidityExecution));
    } 
    let result = result.unwrap();
    if !result {
        return Err(Box::new(EvmVerificationError::InvalidProof));
    }

    drop(anvil);
    Ok(result)
}
