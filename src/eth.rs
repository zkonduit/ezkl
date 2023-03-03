use crate::pfsys::evm::DeploymentCode;
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use ethereum_types::Address;
use ethers::abi::ethabi::Bytes;
use ethers::abi::Abi;
use ethers::abi::AbiEncode;
use ethers::contract::abigen;
use ethers::contract::ContractFactory;
use ethers::core::k256::ecdsa::SigningKey;
use ethers::middleware::SignerMiddleware;
use ethers::providers::Middleware;
use ethers::providers::{Http, Provider};
use ethers::signers::coins_bip39::English;
use ethers::signers::MnemonicBuilder;
use ethers::signers::Signer;
use ethers::types::transaction::eip2718::TypedTransaction;
use ethers::types::TransactionRequest;
use ethers::types::U256;
use ethers::utils::AnvilInstance;
use ethers::{
    prelude::{LocalWallet, Wallet},
    utils::Anvil,
};
use ethers_solc::Solc;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use log::{debug, info};
use snark_verifier::loader::evm::encode_calldata;
use std::error::Error;
use std::fs::read_to_string;
use std::path::PathBuf;
use std::{convert::TryFrom, sync::Arc, time::Duration};

const DEFAULT_DERIVATION_PATH_PREFIX: &str = "m/44'/60'/0'/0/";

/// A local ethers-rs based client
pub type EthersClient = Arc<SignerMiddleware<Provider<Http>, Wallet<SigningKey>>>;

/// Return an instance of Anvil and a local client
pub async fn setup_eth_backend() -> Result<(AnvilInstance, EthersClient), Box<dyn Error>> {
    // Launch anvil
    let anvil = Anvil::new().spawn();

    // Instantiate the wallet
    let wallet: LocalWallet = anvil.keys()[0].clone().into();

    // Connect to the network
    let provider =
        Provider::<Http>::try_from(anvil.endpoint())?.interval(Duration::from_millis(10u64));

    // Instantiate the client with the wallet
    let client = Arc::new(SignerMiddleware::new(
        provider,
        wallet.with_chain_id(anvil.chain_id()),
    ));

    Ok((anvil, client))
}

/// Verify a proof using a Solidity verifier contract
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    sol_code_path: PathBuf,
) -> Result<bool, Box<dyn Error>> {
    let (anvil, client) = setup_eth_backend().await?;

    let compiled = Solc::default().compile_source(sol_code_path)?;
    let (abi, bytecode, _runtime_bytecode) = compiled
        .find("Verifier")
        .expect("could not find contract")
        .into_parts_or_default();
    let factory = ContractFactory::new(abi, bytecode, client.clone());
    let contract = factory.deploy(())?.send().await?;
    let addr = contract.address();

    abigen!(Verifier, "./Verifier.json");
    let contract = Verifier::new(addr, client.clone());

    let mut public_inputs = vec![];
    for val in &proof.instances[0] {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.as_slice());
        public_inputs.push(u);
    }

    let result = contract
        .verify(
            public_inputs,
            ethers::types::Bytes::from(proof.proof.to_vec()),
        )
        .call()
        .await;

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

/// Parses a private key into a [SigningKey]  
fn parse_private_key(private_key: U256) -> Result<SigningKey, Bytes> {
    if private_key.is_zero() {
        return Err("Private key cannot be 0.".to_string().encode());
    }
    let mut bytes: [u8; 32] = [0; 32];
    private_key.to_big_endian(&mut bytes);
    SigningKey::from_bytes(&bytes).map_err(|err| err.to_string().encode())
}

/// Parses a private key into a [Wallet]  
fn get_signing_wallet(
    private_key: U256,
    chain_id: u64,
) -> Result<Wallet<SigningKey>, Box<dyn Error>> {
    let private_key = parse_private_key(private_key).unwrap();
    let wallet: Wallet<SigningKey> = private_key.into();

    Ok(wallet.with_chain_id(chain_id))
}

/// Derives a key from a mnemonic phrase
fn derive_key(mnemonic: &str, path: &str, index: u32) -> Result<U256, Bytes> {
    let derivation_path = if path.ends_with('/') {
        format!("{path}{index}")
    } else {
        format!("{path}/{index}")
    };

    let wallet = MnemonicBuilder::<English>::default()
        .phrase(mnemonic)
        .derivation_path(&derivation_path)
        .map_err(|err| err.to_string().encode())?
        .build()
        .map_err(|err| err.to_string().encode())?;

    info!("Wallet key we use: {:#?}", wallet);

    let private_key = U256::from_big_endian(wallet.signer().to_bytes().as_slice());

    info!("Private key we use: {:#?}", private_key);

    Ok(private_key)
}

/// From a mnemonic and an rpc url returns a provider that can sign transaction via HTTP
pub async fn get_signing_provider(
    mnemonic: &str,
    rpc_url: &str,
) -> SignerMiddleware<Arc<Provider<Http>>, Wallet<SigningKey>> {
    let provider =
        Provider::<Http>::try_from(rpc_url).expect("could not instantiate HTTP Provider");
    debug!("{:#?}", provider);
    // provider.for_chain(Chain::try_from(3141));
    let chain_id = provider.get_chainid().await.unwrap();
    let private_key = derive_key(mnemonic, DEFAULT_DERIVATION_PATH_PREFIX, 0).unwrap();
    let signing_wallet = get_signing_wallet(private_key, chain_id.as_u64()).unwrap();

    let provider = Arc::new(provider);

    SignerMiddleware::new(provider, signing_wallet)
}

/// Deploys a verifier contract  
pub async fn deploy_verifier(
    secret: PathBuf,
    rpc_url: String,
    deployment_code_path: Option<PathBuf>,
    sol_code_path: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    // comment the following two lines if want to deploy to anvil
    let mnemonic = read_to_string(secret)?;
    let client = Arc::new(get_signing_provider(&mnemonic, &rpc_url).await);
    // uncomment if want to test on local anvil
    // let (anvil, client) = setup_eth_backend().await;

    let gas = client.provider().get_gas_price().await?;
    info!("gas price: {:#?}", gas);

    // sol code supercedes deployment code
    let factory = match sol_code_path {
        Some(path) => {
            let compiled = Solc::default().compile_source(path).unwrap();
            let (abi, bytecode, _runtime_bytecode) = compiled
                .find("Verifier")
                .expect("could not find contract")
                .into_parts_or_default();
            ContractFactory::new(abi, bytecode, client.clone())
        }
        None => match deployment_code_path {
            Some(path) => {
                let bytecode = DeploymentCode::load(&path)?;
                ContractFactory::new(
                    // our constructor is empty and ContractFactory only uses the abi constructor -- so this should be safe
                    Abi::default(),
                    (bytecode.code().clone()).into(),
                    client.clone(),
                )
            }
            None => {
                panic!("at least one path should be set");
            }
        },
    };

    let deployer = factory.deploy(())?;

    let tx = &deployer.tx;

    debug!("transaction {:#?}", tx);
    info!(
        "estimated deployment gas cost: {:#?}",
        client.estimate_gas(tx, None).await?
    );
    let (contract, deploy_receipt) = deployer.send_with_receipt().await?;
    debug!("deploy receipt: {:#?}", deploy_receipt);
    info!("contract address: {}", contract.address());

    // uncomment if want to test on local anvil
    // drop(anvil);

    Ok(())
}

/// Sends a proof to an already deployed verifier contract
pub async fn send_proof(
    secret: PathBuf,
    rpc_url: String,
    addr: Address,
    snark: Snark<Fr, G1Affine>,
    has_abi: bool,
) -> Result<(), Box<dyn Error>> {
    info!("contract address: {}", addr);
    // comment the following two lines if want to deploy to anvil
    let mnemonic = read_to_string(secret)?;
    let client = Arc::new(get_signing_provider(&mnemonic, &rpc_url).await);
    // uncomment if want to test on local anvil
    // let (anvil, client) = setup_eth_backend().await;

    let gas = client.provider().get_gas_price().await?;
    info!("gas price: {:#?}", gas);

    let mut verify_tx: TypedTransaction = if has_abi {
        info!("using contract abi");
        abigen!(Verifier, "./Verifier.json");
        let contract = Verifier::new(addr, client.clone());

        let mut public_inputs = vec![];
        for val in &snark.instances[0] {
            let bytes = val.to_repr();
            let u = U256::from_little_endian(bytes.as_slice());
            public_inputs.push(u);
        }

        contract
            .verify(
                public_inputs,
                ethers::types::Bytes::from(snark.proof.to_vec()),
            )
            .tx
    } else {
        info!("not using contract abi");
        let calldata = encode_calldata(&snark.instances, &snark.proof);
        TransactionRequest::default()
            .to(addr)
            .from(client.address())
            .data(calldata)
            .into()
    };

    info!("created tx");
    debug!("transaction {:#?}", verify_tx);

    let gas = client.estimate_gas(&verify_tx, None).await?;
    info!("estimated deployment gas cost: {:#?}", gas);

    verify_tx.set_gas_price(gas);

    let result = client.send_transaction(verify_tx, None).await?.await;

    if result.is_err() {
        return Err(Box::new(EvmVerificationError::SolidityExecution));
    }
    let result = result.unwrap();

    debug!("transaction {:#?}", result);

    // uncomment if want to test on local anvil
    // drop(anvil);

    Ok(())
}
