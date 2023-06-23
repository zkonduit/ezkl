use crate::graph::input::{CallsToAccount, DataSource, GraphWitness};
use crate::pfsys::evm::{DeploymentCode, EvmVerificationError};
use crate::pfsys::Snark;
use ethers::abi::Abi;
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

/// Verify a proof using a Solidity verifier contract
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    sol_code_path: Option<PathBuf>,
    sol_bytecode_path: Option<PathBuf>,
) -> Result<bool, Box<dyn Error>> {
    let (anvil, client) = setup_eth_backend(None).await?;

    // sol code supercedes deployment code
    let factory = match sol_code_path {
        Some(path) => get_sol_contract_factory(path, "Verifier", client.clone()).unwrap(),
        None => match sol_bytecode_path {
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

    let contract = factory.deploy(())?.send().await?;
    let addr = contract.address();

    abigen!(Verifier, "./abis/Verifier.json");
    let contract = Verifier::new(addr, client.clone());

    let mut public_inputs = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.as_slice());
        public_inputs.push(u);
    }

    let tx = contract
        .verify(
            public_inputs.clone(),
            ethers::types::Bytes::from(proof.proof.to_vec()),
        )
        .tx;

    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx, None).await?
    );

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
    data: &Vec<Vec<f32>>,
) -> Result<(ContractInstance<Arc<M>, M>, Vec<u8>), Box<dyn Error>> {
    let factory = get_sol_contract_factory(
        PathBuf::from("./contracts/TestReads.sol"),
        "TestReads",
        client.clone(),
    )
    .unwrap();

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
    sol_code_path: PathBuf,
    file_data: PathBuf,
    on_chain_data: PathBuf,
    model_path: PathBuf,
    settings_path: PathBuf,
) -> Result<bool, Box<dyn Error>> {
    use crate::graph::{GraphSettings, GraphCircuit};

    let (anvil, client) = setup_eth_backend(None).await?;

    let on_chain_witness = GraphWitness::from_path(on_chain_data)?;

    let file_witness = GraphWitness::from_path(file_data)?;

    let settings = GraphSettings::load(&settings_path)?;

    let circuit = GraphCircuit::from_settings(&settings, &model_path, crate::circuit::CheckMode::UNSAFE)?;


    let mut scales = vec![];

    // The data that will be stored in the test contracts that will eventually be read from.
    let mut calls_to_accounts = vec![];

    match on_chain_witness.input_data {
        DataSource::OnChain(source) => {
            for call in source.calls {
                calls_to_accounts.push(call);
            }
            if let DataSource::File(floating_points) = file_witness.input_data {
                let (contract, _) = setup_test_contract(client.clone(), &floating_points).await?;
                info!("contract address: {:#?}", contract.address());
            } else {
                panic!("Data source for file_data must be File");
            }
        }
        _ => (),
    };
    match on_chain_witness.output_data {
        DataSource::OnChain(source) => {
            let output_scales = circuit.model.graph.get_output_scales();
            for call in source.calls {
                calls_to_accounts.push(call);
            }
            if let DataSource::File(floating_points) = file_witness.output_data {
                for (i,arr) in floating_points.iter().enumerate() {
                    let scale = output_scales[i];
                    scales.extend(vec![scale; arr.len()])
                }
                let (contract, _) = setup_test_contract(client.clone(), &floating_points).await?;
                info!("contract address: {:#?}", contract.address());
            } else {
                panic!("Data source for file_data must be File");
            }
        }
        _ => (),
    };
    print!("scales: {:#?}", scales);

    let (contract_addresses, call_data, decimals) = if calls_to_accounts.len() > 0 {
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

    let factory =
        get_sol_contract_factory(sol_code_path, "DataAttestationVerifier", client.clone()).unwrap();

    info!("call_data length: {:#?}", call_data);
    info!("contract_addresses length: {:#?}", contract_addresses);
    info!("decimals length: {:#?}", decimals);

    let contract = factory
        .deploy((contract_addresses, call_data, decimals, scales))?
        .send()
        .await?;

    abigen!(
        DataAttestationVerifier,
        "./abis/DataAttestationVerifier.json"
    );
    let contract = DataAttestationVerifier::new(contract.address(), client.clone());

    let mut public_inputs = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.as_slice());
        public_inputs.push(u);
    }

    info!("public_inputs: {:#?}", public_inputs);

    let tx = contract
        .verify_with_data_attestation(
            public_inputs.clone(),
            ethers::types::Bytes::from(proof.proof.to_vec()),
        )
        .tx;

    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx, None).await?
    );

    info!("public_inputs: {:#?}", public_inputs);

    let result = contract
        .verify_with_data_attestation(
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
    data: &Vec<Vec<f32>>,
) -> Result<Vec<CallsToAccount>, Box<dyn Error>> {
    let (contract, decimals) = setup_test_contract(client.clone(), data).await?;

    abigen!(TestReads, "./abis/TestReads.json");

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
) -> Result<Vec<i128>, Box<dyn Error>> {
    let factory = get_sol_contract_factory(
        PathBuf::from("./contracts/QuantizeData.sol"),
        "QuantizeData",
        client.clone(),
    )
    .unwrap();

    let contract = factory.deploy(())?.send().await?;

    abigen!(QuantizeData, "./abis/QuantizeData.json");

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

    let results = results.unwrap();
    info!("evm quantization results: {:#?}", results,);
    Ok(results.to_vec())
}

/// Generates the contract factory for a solidity verifier, optionally compiling the code with optimizer runs set on the Solc compiler.
fn get_sol_contract_factory<M: 'static + Middleware>(
    sol_code_path: PathBuf,
    contract_name: &str,
    client: Arc<M>,
) -> Result<ContractFactory<M>, Box<dyn Error>> {
    const MAX_RUNTIME_BYTECODE_SIZE: usize = 24577;
    // call get_contract_artificacts to get the abi and bytecode
    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, contract_name, None)?;
    let size = runtime_bytecode.len();
    debug!("runtime bytecode size: {:#?}", size);
    if size > MAX_RUNTIME_BYTECODE_SIZE {
        // `_runtime_bytecode` exceeds the limit
        panic!(
            "Solidity runtime bytecode size is: {:#?}, 
            which exceeds 24577 bytes limit.
            Try setting '--optimzer-runs 1' when generating the verifier
            so SOLC can optimize for the smallest deployment",
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
    input_data: Option<(u32, Vec<CallsToAccount>)>,
    output_data: Option<(u32, Vec<CallsToAccount>)>,
) -> Result<String, Box<dyn Error>> {
    let file = File::open(input_file.clone())?;
    let reader = BufReader::new(file);

    let mut transcript_addrs: Vec<u32> = Vec::new();
    let mut modified_lines: Vec<String> = Vec::new();
    let mut proof_size: u32 = 0;

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
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().starts_with("mstore(0x20") && start.is_none() {
            start = Some(i as u32);
        }

        if line.trim().starts_with("mstore(0x0") {
            end = Some(i as u32);
            break;
        }
    }

    let num_pubinputs = if let Some(s) = start {
        end.unwrap() - s
    } else {
        0
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
        if let Some(m) = m {
            let calldata_and_addr = m.get(1).unwrap().as_str();
            let addr = m.get(2).unwrap().as_str();
            let addr_as_num = u32::from_str_radix(addr.strip_prefix("0x").unwrap(), 16)?;
            if addr_as_num <= max_pubinputs_addr {
                let pub_addr = format!("{:#x}", addr_as_num + 32);
                line = line.replace(
                    calldata_and_addr,
                    &format!("mload(add(pubInputs, {}))", pub_addr),
                );
            } else {
                proof_size += 1;
                let proof_addr = format!("{:#x}", addr_as_num - max_pubinputs_addr);
                line = line.replace(
                    calldata_and_addr,
                    &format!("mload(add(proof, {}))", proof_addr),
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

    let mut contract = if input_data.is_some() || output_data.is_some() {
        let mut accounts_len = 0;
        let mut contract = match std::fs::read_to_string("./contracts/AttestData.sol") {
            Ok(file_content) => file_content,
            Err(err) => {
                panic!("Error reading VerifierBase.sol: {}", err);
            }
        };
        // fill in the quantization params and total calls
        // as constants to the contract to save on gas
        if let Some(input_data) = input_data {
            let input_calls: usize = input_data.1.iter().map(|v| v.call_data.len()).sum();
            let input_scale = input_data.0;
            accounts_len = input_data.1.len();
            contract = contract.replace(
                "uint constant public INPUT_SCALE = 1<<0;",
                &format!("uint constant public INPUT_SCALE = 1<<{};", input_scale),
            );
            contract = contract.replace(
                "uint256 constant INPUT_CALLS = 0;",
                &format!("uint256 constant INPUT_CALLS = {};", input_calls),
            );
        }
        if let Some(output_data) = output_data {
            let output_calls: usize = output_data.1.iter().map(|v| v.call_data.len()).sum();
            let output_scale = output_data.0;
            accounts_len += output_data.1.len();
            contract = contract.replace(
                "uint constant public OUTPUT_SCALE = 1<<0;",
                &format!("uint constant public OUTPUT_SCALE = 1<<{};", output_scale),
            );
            contract = contract.replace(
                "uint256 constant OUTPUT_CALLS = 0;",
                &format!("uint256 constant OUTPUT_CALLS = {};", output_calls),
            );
        }
        contract.replace("AccountCall[]", &format!("AccountCall[{}]", accounts_len))
    } else {
        match std::fs::read_to_string("./contracts/VerifierBase.sol") {
            Ok(file_content) => file_content,
            Err(err) => {
                panic!("Error reading VerifierBase.sol: {}", err);
            }
        }
    };

    // Insert the max_transcript_addr into the contract string at the correct position.
    _ = contract.replace(
        "bytes32[] memory transcript",
        &format!("bytes32[{}] memory transcript", max_transcript_addr),
    );

    // Find the index of "assembly {"
    let end_index = match contract.find("assembly {") {
        Some(index) => index,
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
    let mut offset = 128;

    // replace all mload(add(pubInputs, 0x...))) with mload(0x...
    contract = replace_vars_with_offset(&contract, r"add\(pubInputs, (0x[0-9a-fA-F]+)\)", offset);

    offset += 32 * num_pubinputs + 32;

    // replace all mload(add(proof, 0x...))) with mload(0x...
    contract = replace_vars_with_offset(&contract, r"add\(proof, (0x[0-9a-fA-F]+)\)", offset);

    offset += 32 * proof_size + 32;

    // replace all (add(transcript, 0x...))) with (0x...)
    contract = replace_vars_with_offset(&contract, r"add\(transcript, (0x[0-9a-fA-F]+)\)", offset);

    Ok(contract)
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
