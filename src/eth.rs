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
#[cfg(target_arch = "wasm32")]
use ethers::prelude::Wallet;
use ethers::providers::Middleware;
use ethers::providers::{Http, Provider};
use ethers::signers::coins_bip39::English;
use ethers::signers::MnemonicBuilder;
use ethers::signers::Signer;
use ethers::types::transaction::eip2718::TypedTransaction;
use ethers::types::TransactionRequest;
use ethers::types::U256;
#[cfg(not(target_arch = "wasm32"))]
use ethers::{
    prelude::{LocalWallet, Wallet},
    utils::{Anvil, AnvilInstance},
};
use ethers_solc::Solc;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use log::{debug, info};
use snark_verifier::loader::evm::encode_calldata;
use std::error::Error;
use std::fmt::Write;
use std::fs::read_to_string;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use std::{convert::TryFrom, sync::Arc};

const DEFAULT_DERIVATION_PATH_PREFIX: &str = "m/44'/60'/0'/0/";

/// A local ethers-rs based client
pub type EthersClient = Arc<SignerMiddleware<Provider<Http>, Wallet<SigningKey>>>;

/// Return an instance of Anvil and a local client
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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

    info!("wallet address: {:#?}", wallet.address());

    let private_key = U256::from_big_endian(wallet.signer().to_bytes().as_slice());

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

    let gas = client.provider().get_gas_price().await.unwrap();
    info!("gas price: {:#?}", gas);

    let gas_estimate = client.estimate_gas(&verify_tx, None).await?;
    info!("estimated function call gas cost: {:#?}", gas_estimate);

    client.fill_transaction(&mut verify_tx, None).await?;
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

use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Reads in raw bytes code and generates equivalent .sol file
pub fn fix_verifier_sol(input_file: PathBuf) -> Result<String, Box<dyn Error>> {
    let file = File::open(input_file.clone())?;
    let reader = BufReader::new(file);

    let mut transcript_addrs: Vec<u32> = Vec::new();
    let mut modified_lines: Vec<String> = Vec::new();

    // convert calldataload 0x0 to 0x40 to read from pubInputs, and the rest
    // from proof
    let calldata_pattern = Regex::new(r"^.*(calldataload\((0x[a-f0-9]+)\)).*$")?;
    let mstore_pattern = Regex::new(r"^\s*(mstore\(0x([0-9a-fA-F]+)+),.+\)")?;
    let mstore8_pattern = Regex::new(r"^\s*(mstore8\((\d+)+),.+\)")?;
    let mstoren_pattern = Regex::new(r"^\s*(mstore\((\d+)+),.+\)")?;
    let mload_pattern = Regex::new(r"(mload\((0x[0-9a-fA-F]+))\)")?;
    let keccak_pattern = Regex::new(r"(keccak256\((0x[0-9a-fA-F]+))")?;
    let modexp_pattern =
        Regex::new(r"(staticcall\(gas\(\), 0x5, (0x[0-9a-fA-F]+), 0xc0, (0x[0-9a-fA-F]+), 0x20)")?;
    let ecmul_pattern =
        Regex::new(r"(staticcall\(gas\(\), 0x7, (0x[0-9a-fA-F]+), 0x60, (0x[0-9a-fA-F]+), 0x40)")?;
    let ecadd_pattern =
        Regex::new(r"(staticcall\(gas\(\), 0x6, (0x[0-9a-fA-F]+), 0x80, (0x[0-9a-fA-F]+), 0x40)")?;
    let ecpairing_pattern =
        Regex::new(r"(staticcall\(gas\(\), 0x8, (0x[0-9a-fA-F]+), 0x180, (0x[0-9a-fA-F]+), 0x20)")?;
    let bool_pattern = Regex::new(r":bool")?;

    // Count the number of pub inputs
    let mut start = None;
    let mut end = None;
    let mut i = 0;
    for line in reader.lines() {
        let line = line?;
        if line.trim().starts_with("mstore(0x20") {
            start = Some(i);
        }

        if line.trim().starts_with("mstore(0x0") {
            end = Some(i);
            break;
        }
        i += 1;
    }

    let num_pubinputs = if start.is_none() {
        0
    } else {
        end.unwrap() - start.unwrap()
    };

    let mut max_pubinputs_addr = 0;
    if num_pubinputs > 0 {
        max_pubinputs_addr = num_pubinputs * 32 - 32;
    }

    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let mut line = line?;
        let m = bool_pattern.captures(&line);
        if m.is_some() {
            line = line.replace(":bool", "");
        }

        let m = calldata_pattern.captures(&line);
        if m.is_some() {
            let calldata_and_addr = m.as_ref().unwrap().get(1).unwrap().as_str();
            let addr = m.unwrap().get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr.strip_prefix("0x").unwrap(), 16)?;

            if addr_as_num <= max_pubinputs_addr {
                let pub_addr = format!("{:#x}", addr_as_num + 32);
                line = line.replace(
                    calldata_and_addr,
                    &format!("mload(add(pubInputs, {}))", pub_addr),
                );
            } else {
                let proof_addr = format!("{:#x}", addr_as_num - max_pubinputs_addr);
                line = line.replace(
                    calldata_and_addr,
                    &format!("mload(add(proof, {}))", proof_addr),
                );
            }
        }

        let m = mstore8_pattern.captures(&line);
        if m.is_some() {
            let mstore = m.as_ref().unwrap().get(1).unwrap().as_str();
            let addr = m.unwrap().get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr, 10)?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mstore,
                &format!("mstore8(add(transcript, {})", transcript_addr),
            );
        }

        let m = mstoren_pattern.captures(&line);
        if m.is_some() {
            let mstore = m.as_ref().unwrap().get(1).unwrap().as_str();
            let addr = m.unwrap().get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr, 10)?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mstore,
                &format!("mstore(add(transcript, {})", transcript_addr),
            );
        }

        let m = modexp_pattern.captures(&line);
        if m.is_some() {
            let modexp = m.as_ref().unwrap().get(1).unwrap().as_str();
            let start_addr = m.as_ref().unwrap().get(2).unwrap().as_str();
            let result_addr = m.unwrap().get(3).unwrap().as_str();
            let start_addr_as_num =
                u32::from_str_radix(start_addr.strip_prefix("0x").unwrap(), 16)?;
            let result_addr_as_num =
                u32::from_str_radix(result_addr.strip_prefix("0x").unwrap(), 16)?;

            let transcript_addr = format!("{:#x}", start_addr_as_num);
            transcript_addrs.push(start_addr_as_num);
            let result_addr = format!("{:#x}", result_addr_as_num);
            line = line.replace(
                modexp,
                &format!(
                    "staticcall(gas(), 0x5, add(transcript, {}), 0xc0, add(transcript, {}), 0x20",
                    transcript_addr, result_addr
                ),
            );
        }

        let m = ecmul_pattern.captures(&line);
        if m.is_some() {
            let ecmul = m.as_ref().unwrap().get(1).unwrap().as_str();
            let start_addr = m.as_ref().as_ref().unwrap().get(2).unwrap().as_str();
            let result_addr = m.unwrap().get(3).unwrap().as_str();
            let start_addr_as_num =
                u32::from_str_radix(start_addr.strip_prefix("0x").unwrap(), 16)?;
            let result_addr_as_num =
                u32::from_str_radix(result_addr.strip_prefix("0x").unwrap(), 16)?;

            let transcript_addr = format!("{:#x}", start_addr_as_num);
            let result_addr = format!("{:#x}", result_addr_as_num);
            transcript_addrs.push(start_addr_as_num);
            transcript_addrs.push(result_addr_as_num);
            line = line.replace(
                ecmul,
                &format!(
                    "staticcall(gas(), 0x7, add(transcript, {}), 0x60, add(transcript, {}), 0x40",
                    transcript_addr, result_addr
                ),
            );
        }

        let m = ecadd_pattern.captures(&line);
        if m.is_some() {
            let ecadd = m.as_ref().unwrap().get(1).unwrap().as_str();
            let start_addr = m.as_ref().unwrap().get(2).unwrap().as_str();
            let result_addr = m.unwrap().get(3).unwrap().as_str();
            let start_addr_as_num =
                u32::from_str_radix(start_addr.strip_prefix("0x").unwrap(), 16)?;
            let result_addr_as_num =
                u32::from_str_radix(result_addr.strip_prefix("0x").unwrap(), 16)?;

            let transcript_addr = format!("{:#x}", start_addr_as_num);
            let result_addr = format!("{:#x}", result_addr_as_num);
            transcript_addrs.push(start_addr_as_num);
            transcript_addrs.push(result_addr_as_num);
            line = line.replace(
                ecadd,
                &format!(
                    "staticcall(gas(), 0x6, add(transcript, {}), 0x80, add(transcript, {}), 0x40",
                    transcript_addr, result_addr
                ),
            );
        }

        let m = ecpairing_pattern.captures(&line);
        if m.is_some() {
            let ecpairing = m.as_ref().unwrap().get(1).unwrap().as_str();
            let start_addr = m.as_ref().unwrap().get(2).unwrap().as_str();
            let result_addr = m.unwrap().get(3).unwrap().as_str();
            let start_addr_as_num =
                u32::from_str_radix(start_addr.strip_prefix("0x").unwrap(), 16)?;
            let result_addr_as_num =
                u32::from_str_radix(result_addr.strip_prefix("0x").unwrap(), 16)?;

            let transcript_addr = format!("{:#x}", start_addr_as_num);
            let result_addr = format!("{:#x}", result_addr_as_num);
            transcript_addrs.push(start_addr_as_num);
            transcript_addrs.push(result_addr_as_num);
            line = line.replace(
                ecpairing,
                &format!(
                    "staticcall(gas(), 0x8, add(transcript, {}), 0x180, add(transcript, {}), 0x20",
                    transcript_addr, result_addr
                ),
            );
        }

        let m = mstore_pattern.captures(&line);
        if m.is_some() {
            let mstore = m.as_ref().unwrap().get(1).unwrap().as_str();
            let addr = m.as_ref().unwrap().get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr, 16)?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mstore,
                &format!("mstore(add(transcript, {})", transcript_addr),
            );
        }

        let m = keccak_pattern.captures(&line);
        if m.is_some() {
            let keccak = m.as_ref().unwrap().get(1).unwrap().as_str();
            let addr = m.as_ref().unwrap().get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr.strip_prefix("0x").unwrap(), 16)?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                keccak,
                &format!("keccak256(add(transcript, {})", transcript_addr),
            );
        }

        // mload can show up multiple times per line
        loop {
            let m = mload_pattern.captures(&line);
            if m.is_none() {
                break;
            }
            let mload = m.as_ref().unwrap().get(1).unwrap().as_str();
            let addr = m.as_ref().unwrap().get(2).unwrap().as_str();

            let addr_as_num = u32::from_str_radix(addr.strip_prefix("0x").unwrap(), 16)?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mload,
                &format!("mload(add(transcript, {})", transcript_addr),
            );
        }

        modified_lines.push(line);
    }

    // get the max transcript addr
    let max_transcript_addr = transcript_addrs.iter().max().unwrap() / 32;
    let mut contract = format!(
        "// SPDX-License-Identifier: MIT
    pragma solidity ^0.8.17;
    
    contract Verifier {{
        function verify(
            uint256[] memory pubInputs,
            bytes memory proof
        ) public view returns (bool) {{
            bool success = true;
            bytes32[{}] memory transcript;
            assembly {{
        ",
        max_transcript_addr
    )
    .trim()
    .to_string();

    // using a boxed Write trait object here to show it works for any Struct impl'ing Write
    // you may also use a std::fs::File here
    let write: Box<&mut dyn Write> = Box::new(&mut contract);

    for line in modified_lines[16..modified_lines.len() - 7].iter() {
        write!(write, "{}", line).unwrap();
    }
    writeln!(write, "}} return success; }} }}")?;
    return Ok(contract);
}
