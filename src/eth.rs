use crate::graph::input::{CallsToAccount, GraphData};
use crate::graph::DataSource;
#[cfg(not(target_arch = "wasm32"))]
use crate::graph::GraphSettings;
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use ethers::abi::Contract;
use ethers::contract::abigen;
use ethers::contract::ContractFactory;
use ethers::core::k256::ecdsa::SigningKey;
use ethers::middleware::SignerMiddleware;
use ethers::prelude::ContractInstance;
#[cfg(target_arch = "wasm32")]
use ethers::prelude::Wallet;
use ethers::providers::Middleware;
use ethers::providers::{Http, Provider};
use ethers::signers::Signer;
use ethers::solc::{CompilerInput, Solc};
use ethers::types::transaction::eip2718::TypedTransaction;
use ethers::types::Bytes;
use ethers::types::TransactionRequest;
use ethers::types::H160;
use ethers::types::U256;
#[cfg(not(target_arch = "wasm32"))]
use ethers::{
    prelude::{LocalWallet, Wallet},
    utils::{Anvil, AnvilInstance},
};
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use log::{debug, info};
use std::error::Error;
use std::fmt::Write;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use std::{convert::TryFrom, sync::Arc};

/// A local ethers-rs based client
pub type EthersClient = Arc<SignerMiddleware<Provider<Http>, Wallet<SigningKey>>>;

// Generate contract bindings OUTSIDE the functions so they are part of library
abigen!(TestReads, "./abis/TestReads.json");
abigen!(Verifier, "./abis/Verifier.json");
abigen!(
    DataAttestationVerifier,
    "./abis/DataAttestationVerifier.json"
);
abigen!(QuantizeData, "./abis/QuantizeData.json");

const TESTREADS_SOL: &str = include_str!("../contracts/TestReads.sol");
const QUANTIZE_DATA_SOL: &str = include_str!("../contracts/QuantizeData.sol");
const ATTESTDATA_SOL: &str = include_str!("../contracts/AttestData.sol");
const VERIFIERBASE_SOL: &str = include_str!("../contracts/VerifierBase.sol");

/// Return an instance of Anvil and a client for the given RPC URL. If none is provided, a local client is used.
#[cfg(not(target_arch = "wasm32"))]
pub async fn setup_eth_backend(
    rpc_url: Option<&str>,
) -> Result<(AnvilInstance, EthersClient), Box<dyn Error>> {
    // Launch anvil
    let anvil = Anvil::new().spawn();

    // Instantiate the wallet
    let wallet: LocalWallet = anvil.keys()[0].clone().into();

    let endpoint = if let Some(rpc_url) = rpc_url {
        rpc_url.to_string()
    } else {
        anvil.endpoint()
    };

    // Connect to the network
    let provider = Provider::<Http>::try_from(endpoint)?.interval(Duration::from_millis(10u64));

    let chain_id = provider.get_chainid().await?;
    info!("using chain {}", chain_id);

    // Instantiate the client with the wallet
    let client = Arc::new(SignerMiddleware::new(
        provider,
        wallet.with_chain_id(anvil.chain_id()),
    ));

    Ok((anvil, client))
}

///
pub async fn deploy_verifier_via_solidity(
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: Option<usize>,
) -> Result<ethers::types::Address, Box<dyn Error>> {
    let (_, client) = setup_eth_backend(rpc_url).await?;

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "Verifier", runs)?;
    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone())?;

    let contract = factory.deploy(())?.send().await?;
    let addr = contract.address();
    Ok(addr)
}

///
pub async fn deploy_da_verifier_via_solidity(
    settings_path: PathBuf,
    input: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: Option<usize>,
) -> Result<ethers::types::Address, Box<dyn Error>> {
    let (_, client) = setup_eth_backend(rpc_url).await?;

    let input = GraphData::from_path(input)?;

    let settings = GraphSettings::load(&settings_path)?;

    let mut scales = vec![];
    // The data that will be stored in the test contracts that will eventually be read from.
    let mut calls_to_accounts = vec![];

    let instance_shapes = settings.model_instance_shapes;

    let mut instance_idx = 0;
    let mut contract_instance_offset = 0;

    if let DataSource::OnChain(source) = input.input_data {
        for call in source.calls {
            calls_to_accounts.push(call);
            instance_idx += 1;
        }
    } else if let DataSource::File(source) = input.input_data {
        if settings.run_args.input_visibility.is_public() {
            instance_idx += source.len();
            for s in source {
                contract_instance_offset += s.len();
            }
        }
    }

    if let Some(DataSource::OnChain(source)) = input.output_data {
        let output_scales = settings.model_output_scales;
        for call in source.calls {
            calls_to_accounts.push(call);
        }

        // give each input a scale
        for scale in output_scales {
            scales.extend(vec![
                scale;
                instance_shapes[instance_idx].iter().product::<usize>()
            ]);
            instance_idx += 1;
        }
    }

    let (contract_addresses, call_data, decimals) = if !calls_to_accounts.is_empty() {
        let mut contract_addresses = vec![];
        let mut call_data = vec![];
        let mut decimals: Vec<Vec<u8>> = vec![];
        for (i, val) in calls_to_accounts.iter().enumerate() {
            let contract_address_bytes = hex::decode(val.address.clone())?;
            let contract_address = H160::from_slice(&contract_address_bytes);
            contract_addresses.push(contract_address);
            call_data.push(vec![]);
            decimals.push(vec![]);
            for (call, decimal) in &val.call_data {
                let call_data_bytes = hex::decode(call)?;
                call_data[i].push(ethers::types::Bytes::from(call_data_bytes));
                decimals[i].push(*decimal);
            }
        }
        (contract_addresses, call_data, decimals)
    } else {
        panic!("Data source for either input_data or output_data must be OnChain")
    };

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "DataAttestationVerifier", runs)?;
    let factory =
        get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone()).unwrap();

    info!("call_data: {:#?}", call_data);
    info!("contract_addresses: {:#?}", contract_addresses);
    info!("decimals: {:#?}", decimals);

    let contract = factory
        .deploy((
            contract_addresses,
            call_data,
            decimals,
            scales,
            contract_instance_offset as u32,
        ))?
        .send()
        .await?;

    Ok(contract.address())
}

/// Verify a proof using a Solidity verifier contract
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: ethers::types::Address,
    rpc_url: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    use ethers::abi::{Function, Param, ParamType, StateMutability, Token};

    let mut public_inputs: Vec<U256> = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.as_slice());
        public_inputs.push(u);
    }

    info!("public_inputs: {:#?}", public_inputs);
    info!(
        "proof: {:#?}",
        ethers::types::Bytes::from(proof.proof.to_vec())
    );

    #[allow(deprecated)]
    let func = Function {
        name: "verify".to_owned(),
        inputs: vec![
            Param {
                name: "pubInputs".to_owned(),
                kind: ParamType::FixedArray(Box::new(ParamType::Uint(256)), public_inputs.len()),
                internal_type: None,
            },
            Param {
                name: "proof".to_owned(),
                kind: ParamType::Bytes,
                internal_type: None,
            },
        ],
        outputs: vec![Param {
            name: "success".to_owned(),
            kind: ParamType::Bool,
            internal_type: None,
        }],
        constant: None,
        state_mutability: StateMutability::View,
    };

    let encoded = func.encode_input(&[
        Token::FixedArray(public_inputs.into_iter().map(Token::Uint).collect()),
        Token::Bytes(proof.proof),
    ])?;

    info!("encoded: {:#?}", hex::encode(&encoded));
    let (anvil, client) = setup_eth_backend(rpc_url).await?;
    let tx: TypedTransaction = TransactionRequest::default()
        .to(addr)
        .from(client.address())
        .data(encoded)
        .into();
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx, None).await;

    if result.is_err() {
        return Err(Box::new(EvmVerificationError::SolidityExecution));
    }
    let result = result.unwrap();
    info!("result: {:#?}", result);
    // decode return bytes value into uint8
    let result = result.to_vec().last().unwrap() == &1u8;
    if !result {
        return Err(Box::new(EvmVerificationError::InvalidProof));
    }

    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx, None).await?
    );

    drop(anvil);
    Ok(true)
}

fn count_decimal_places(num: f32) -> usize {
    // Convert the number to a string
    let s = num.to_string();

    // Find the decimal point
    match s.find('.') {
        Some(index) => {
            // Count the number of characters after the decimal point
            s[index + 1..].len()
        }
        None => 0,
    }
}

///
pub async fn setup_test_contract<M: 'static + Middleware>(
    client: Arc<M>,
    data: &[Vec<f32>],
) -> Result<(ContractInstance<Arc<M>, M>, Vec<u8>), Box<dyn Error>> {
    // save the abi to a tmp file
    let mut sol_path = std::env::temp_dir();
    sol_path.push("testreads.sol");
    std::fs::write(&sol_path, TESTREADS_SOL)?;

    // Compile the contract
    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_path, "TestReads", None).unwrap();

    let factory =
        get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone()).unwrap();

    let mut decimals = vec![];
    let mut scaled_by_decimals_data = vec![];
    for input in &data[0] {
        let decimal_places = count_decimal_places(*input) as u8;
        let scaled_by_decimals = input * f32::powf(10., decimal_places.into());
        scaled_by_decimals_data.push(scaled_by_decimals as u128);
        decimals.push(decimal_places);
    }

    let contract = factory.deploy(scaled_by_decimals_data)?.send().await?;
    Ok((contract, decimals))
}

/// Verify a proof using a Solidity DataAttestationVerifier contract.
/// Used for testing purposes.
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_with_data_attestation(
    proof: Snark<Fr, G1Affine>,
    addr: ethers::types::Address,
    rpc_url: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    use ethers::abi::{Function, Param, ParamType, StateMutability, Token};

    let mut public_inputs: Vec<U256> = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.as_slice());
        public_inputs.push(u);
    }

    info!("public_inputs: {:#?}", public_inputs);
    info!(
        "proof: {:#?}",
        ethers::types::Bytes::from(proof.proof.to_vec())
    );

    #[allow(deprecated)]
    let func = Function {
        name: "verifyWithDataAttestation".to_owned(),
        inputs: vec![
            Param {
                name: "pubInputs".to_owned(),
                kind: ParamType::FixedArray(Box::new(ParamType::Uint(256)), public_inputs.len()),
                internal_type: None,
            },
            Param {
                name: "proof".to_owned(),
                kind: ParamType::Bytes,
                internal_type: None,
            },
        ],
        outputs: vec![Param {
            name: "success".to_owned(),
            kind: ParamType::Bool,
            internal_type: None,
        }],
        constant: None,
        state_mutability: StateMutability::View,
    };

    let encoded = func.encode_input(&[
        Token::FixedArray(public_inputs.into_iter().map(Token::Uint).collect()),
        Token::Bytes(proof.proof),
    ])?;

    info!("encoded: {:#?}", hex::encode(&encoded));
    let (anvil, client) = setup_eth_backend(rpc_url).await?;
    let tx: TypedTransaction = TransactionRequest::default()
        .to(addr)
        .from(client.address())
        .data(encoded)
        .into();
    debug!("transaction {:#?}", tx);
    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx, None).await?
    );

    let result = client.call(&tx, None).await;
    if result.is_err() {
        return Err(Box::new(EvmVerificationError::SolidityExecution));
    }
    let result = result.unwrap();
    info!("result: {:#?}", result);
    // decode return bytes value into uint8
    let result = result.to_vec().last().unwrap() == &1u8;
    if !result {
        return Err(Box::new(EvmVerificationError::InvalidProof));
    }
    drop(anvil);
    Ok(true)
}

/// get_provider returns a JSON RPC HTTP Provider
pub fn get_provider(rpc_url: &str) -> Result<Provider<Http>, Box<dyn Error>> {
    let provider = Provider::<Http>::try_from(rpc_url)?;
    debug!("{:#?}", provider);
    Ok(provider)
}

/// Tests on-chain data storage by deploying a contract that stores the network input and or output
/// data in its storage. It does this by converting the floating point values to integers and storing the
/// the number of decimals of the floating point value on chain.
pub async fn test_on_chain_data<M: 'static + Middleware>(
    client: Arc<M>,
    data: &[Vec<f32>],
) -> Result<Vec<CallsToAccount>, Box<dyn Error>> {
    let (contract, decimals) = setup_test_contract(client.clone(), data).await?;

    let contract = TestReads::new(contract.address(), client.clone());

    // Get the encoded call data for each input
    let mut calldata = vec![];
    for (i, _) in data.iter().flatten().enumerate() {
        let function = contract.method::<_, U256>("arr", i as u32).unwrap();
        let call = function.calldata().unwrap();
        // Push (call, decimals) to the calldata vector, and set the decimals to 0.
        calldata.push((hex::encode(call), decimals[i]));
    }
    // Instantiate a new CallsToAccount struct
    let calls_to_account = CallsToAccount {
        call_data: calldata,
        address: hex::encode(contract.address().as_bytes()),
    };
    info!("calls_to_account: {:#?}", calls_to_account);
    Ok(vec![calls_to_account])
}

/// Reads on-chain inputs, returning the raw encoded data returned from making all the calls in on_chain_input_data
#[cfg(not(target_arch = "wasm32"))]
pub async fn read_on_chain_inputs<M: 'static + Middleware>(
    client: Arc<M>,
    address: H160,
    data: &Vec<CallsToAccount>,
) -> Result<(Vec<Bytes>, Vec<u8>), Box<dyn Error>> {
    // Iterate over all on-chain inputs
    let mut fetched_inputs = vec![];
    let mut decimals = vec![];
    for on_chain_data in data {
        // Construct the address
        let contract_address_bytes = hex::decode(on_chain_data.address.clone())?;
        let contract_address = H160::from_slice(&contract_address_bytes);
        for (call_data, decimal) in &on_chain_data.call_data {
            let call_data_bytes = hex::decode(call_data.clone())?;
            let tx: TypedTransaction = TransactionRequest::default()
                .to(contract_address)
                .from(address)
                .data(call_data_bytes)
                .into();
            debug!("transaction {:#?}", tx);

            let result = client.call(&tx, None).await?;
            debug!("return data {:#?}", result);
            fetched_inputs.push(result);
            decimals.push(*decimal);
        }
    }
    Ok((fetched_inputs, decimals))
}

///
#[cfg(not(target_arch = "wasm32"))]
pub async fn evm_quantize<M: 'static + Middleware>(
    client: Arc<M>,
    scales: Vec<f64>,
    data: &(Vec<ethers::types::Bytes>, Vec<u8>),
) -> Result<Vec<Fr>, Box<dyn Error>> {
    // save the sol to a tmp file
    let mut sol_path = std::env::temp_dir();
    sol_path.push("quantizedata.sol");
    std::fs::write(&sol_path, QUANTIZE_DATA_SOL)?;

    let (abi, bytecode, runtime_bytecode) = get_contract_artifacts(sol_path, "QuantizeData", None)?;
    let factory =
        get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone()).unwrap();

    let contract = factory.deploy(())?.send().await?;

    let contract = QuantizeData::new(contract.address(), client.clone());

    let fetched_inputs = data.0.clone();
    let decimals = data.1.clone();

    let fetched_inputs = fetched_inputs
        .iter()
        .map(|x| Result::<_, std::convert::Infallible>::Ok(ethers::types::Bytes::from(x.to_vec())))
        .collect::<Result<Vec<Bytes>, _>>()?;

    let decimals = decimals
        .iter()
        .map(|x| U256::from_dec_str(&x.to_string()))
        .collect::<Result<Vec<U256>, _>>()?;

    let scales = scales
        .iter()
        .map(|x| U256::from_dec_str(&x.to_string()))
        .collect::<Result<Vec<U256>, _>>()?;

    info!("scales: {:#?}", scales);
    info!("decimals: {:#?}", decimals);
    info!("fetched_inputs: {:#?}", fetched_inputs);

    let results = contract
        .quantize_data(fetched_inputs, decimals, scales)
        .call()
        .await;

    let results = results
        .unwrap()
        .iter()
        .map(|x| crate::fieldutils::i128_to_felt(*x))
        .collect::<Vec<Fr>>();
    info!("evm quantization results: {:#?}", results,);
    Ok(results.to_vec())
}

/// Generates the contract factory for a solidity verifier, optionally compiling the code with optimizer runs set on the Solc compiler.
fn get_sol_contract_factory<M: 'static + Middleware>(
    abi: Contract,
    bytecode: Bytes,
    runtime_bytecode: Bytes,
    client: Arc<M>,
) -> Result<ContractFactory<M>, Box<dyn Error>> {
    const MAX_RUNTIME_BYTECODE_SIZE: usize = 24577;
    let size = runtime_bytecode.len();
    debug!("runtime bytecode size: {:#?}", size);
    if size > MAX_RUNTIME_BYTECODE_SIZE {
        // `_runtime_bytecode` exceeds the limit
        panic!(
            "Solidity runtime bytecode size is: {:#?},
            which exceeds 24577 bytes limit.",
            size
        );
    }
    Ok(ContractFactory::new(abi, bytecode, client))
}

/// Compiles a solidity verifier contract and returns the abi, bytecode, and runtime bytecode
#[cfg(not(target_arch = "wasm32"))]
pub fn get_contract_artifacts(
    sol_code_path: PathBuf,
    contract_name: &str,
    runs: Option<usize>,
) -> Result<(Contract, Bytes, Bytes), Box<dyn Error>> {
    assert!(sol_code_path.exists());
    // Create the compiler input, enabling the optimizer and setting the optimzer runs.
    let input: CompilerInput = if let Some(r) = runs {
        let mut i = CompilerInput::new(sol_code_path)?[0].clone().optimizer(r);
        i.settings.optimizer.enable();
        i
    } else {
        CompilerInput::new(sol_code_path)?[0].clone()
    };
    let compiled = Solc::default().compile(&input).unwrap();
    let (abi, bytecode, runtime_bytecode) = compiled
        .find(contract_name)
        .expect("could not find contract")
        .into_parts_or_default();
    Ok((abi, bytecode, runtime_bytecode))
}

use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Reads in raw bytes code and generates equivalent .sol file
/// Can optionally attest to on-chain inputs
pub fn fix_verifier_sol(
    input_file: PathBuf,
    num_instances: u32,
    input_data: Option<(u32, Vec<CallsToAccount>)>,
    output_data: Option<Vec<CallsToAccount>>,
) -> Result<String, Box<dyn Error>> {
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

    let mut max_pubinputs_addr: u32 = 0;
    if num_instances > 0 {
        max_pubinputs_addr = num_instances * 32 - 32;
    }

    let file = File::open(input_file.clone())
        .map_err(|_| format!("failed to load verfier at {}", input_file.display()))?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let mut line = line?;
        let m = bool_pattern.captures(&line);
        if m.is_some() {
            line = line.replace(":bool", "");
        }

        let m = calldata_pattern.captures(&line);
        if let Some(m) = m {
            let calldata_and_addr = m.get(1).unwrap().as_str();
            let addr = m.get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr.strip_prefix("0x").unwrap(), 16)?;
            if addr_as_num <= max_pubinputs_addr {
                let pub_addr = format!("{:#x}", addr_as_num);
                line = line.replace(
                    calldata_and_addr,
                    &format!("calldataload(add(pubInputs, {}))", pub_addr),
                );
            } else {
                let proof_addr = format!("{:#x}", 32 + addr_as_num - max_pubinputs_addr);
                line = line.replace(
                    calldata_and_addr,
                    &format!("calldataload(add(proof, {}))", proof_addr),
                );
            }
        }

        let m = mstore8_pattern.captures(&line);
        if let Some(m) = m {
            let mstore = m.get(1).unwrap().as_str();
            let addr = m.get(2).unwrap().as_str();
            let addr_as_num = addr.parse::<u32>()?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mstore,
                &format!("mstore8(add(transcript, {})", transcript_addr),
            );
        }

        let m = mstoren_pattern.captures(&line);
        if let Some(m) = m {
            let mstore = m.get(1).unwrap().as_str();
            let addr = m.get(2).unwrap().as_str();
            let addr_as_num = addr.parse::<u32>()?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mstore,
                &format!("mstore(add(transcript, {})", transcript_addr),
            );
        }

        let m = modexp_pattern.captures(&line);
        if let Some(m) = m {
            let modexp = m.get(1).unwrap().as_str();
            let start_addr = m.get(2).unwrap().as_str();
            let result_addr = m.get(3).unwrap().as_str();
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
        if let Some(m) = m {
            let ecmul = m.get(1).unwrap().as_str();
            let start_addr = m.get(2).unwrap().as_str();
            let result_addr = m.get(3).unwrap().as_str();
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
        if let Some(m) = m {
            let ecadd = m.get(1).unwrap().as_str();
            let start_addr = m.get(2).unwrap().as_str();
            let result_addr = m.get(3).unwrap().as_str();
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
        if let Some(m) = m {
            let ecpairing = m.get(1).unwrap().as_str();
            let start_addr = m.get(2).unwrap().as_str();
            let result_addr = m.get(3).unwrap().as_str();
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
        if let Some(m) = m {
            let mstore = m.get(1).unwrap().as_str();
            let addr = m.get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr, 16)?;
            let transcript_addr = format!("{:#x}", addr_as_num);
            transcript_addrs.push(addr_as_num);
            line = line.replace(
                mstore,
                &format!("mstore(add(transcript, {})", transcript_addr),
            );
        }

        let m = keccak_pattern.captures(&line);
        if let Some(m) = m {
            let keccak = m.get(1).unwrap().as_str();
            let addr = m.get(2).unwrap().as_str();
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

    let contract = if input_data.is_some() || output_data.is_some() {
        let mut accounts_len = 0;
        let mut contract = ATTESTDATA_SOL.to_string();
        // fill in the quantization params and total calls
        // as constants to the contract to save on gas
        if let Some(input_data) = input_data {
            let input_calls: usize = input_data.1.iter().map(|v| v.call_data.len()).sum();
            let input_scale = input_data.0;
            accounts_len = input_data.1.len();
            contract = contract.replace(
                "uint public constant INPUT_SCALE = 1 << 0;",
                &format!("uint public constant INPUT_SCALE = 1 << {};", input_scale),
            );

            contract = contract.replace(
                "uint256 constant INPUT_CALLS = 0;",
                &format!("uint256 constant INPUT_CALLS = {};", input_calls),
            );
        }
        if let Some(output_data) = output_data {
            let output_calls: usize = output_data.iter().map(|v| v.call_data.len()).sum();
            accounts_len += output_data.len();
            contract = contract.replace(
                "uint256 constant OUTPUT_CALLS = 0;",
                &format!("uint256 constant OUTPUT_CALLS = {};", output_calls),
            );
        }
        contract.replace("AccountCall[]", &format!("AccountCall[{}]", accounts_len))
    } else {
        VERIFIERBASE_SOL.to_string()
    };

    // Insert the max_transcript_addr into the contract string at the correct position.
    let mut contract = contract.replace(
        "bytes32[] memory transcript",
        &format!("bytes32[{}] memory transcript", max_transcript_addr),
    );

    // Hardcode the fixed array length of pubInputs param
    contract = contract.replace(
        "uint256[] calldata",
        &format!("uint256[{}] calldata", num_instances),
    );

    // Find the index of "assembly {"
    let end_index =
        match contract.find("assembly { /* This is where the proof verification happens*/ }") {
            Some(index) => index + 10,
            None => {
                panic!("assembly {{ not found in the contract");
            }
        };

    // Take a slice from the start of the contract string up to the "assembly {" position
    let contract_slice = &contract[..end_index];

    let mut contract_slice_string = contract_slice.to_string();

    // using a boxed Write trait object here to show it works for any Struct impl'ing Write
    // you may also use a std::fs::File here
    let write: Box<&mut dyn Write> = Box::new(&mut contract_slice_string);

    for line in modified_lines[16..modified_lines.len() - 7].iter() {
        write!(write, "{}", line).unwrap();
    }
    writeln!(write, "}} return success; }} }}")?;

    // free memory pointer initialization
    let mut offset = 4;

    // replace all mload(add(pubInputs, 0x...))) with mload(0x...
    contract_slice_string = replace_vars_with_offset(
        &contract_slice_string,
        r"add\(pubInputs, (0x[0-9a-fA-F]+)\)",
        offset,
    );

    offset += 32 * num_instances;

    // replace all mload(add(proof, 0x...))) with mload(0x...
    contract_slice_string = replace_vars_with_offset(
        &contract_slice_string,
        r"add\(proof, (0x[0-9a-fA-F]+)\)",
        offset,
    );

    offset = 128;

    // replace all (add(transcript, 0x...))) with (0x...)
    contract_slice_string = replace_vars_with_offset(
        &contract_slice_string,
        r"add\(transcript, (0x[0-9a-fA-F]+)\)",
        offset,
    );

    Ok(contract_slice_string)
}

fn replace_vars_with_offset(contract: &str, regex_pattern: &str, offset: u32) -> String {
    let re = Regex::new(regex_pattern).unwrap();
    let replaced = re.replace_all(contract, |caps: &regex::Captures| {
        let addr_as_num = u32::from_str_radix(caps[1].strip_prefix("0x").unwrap(), 16).unwrap();
        let new_addr = addr_as_num + offset;
        format!("{:#x}", new_addr)
    });
    replaced.into_owned()
}
