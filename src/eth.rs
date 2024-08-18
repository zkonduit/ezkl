use crate::graph::input::{CallsToAccount, FileSourceInner, GraphData};
use crate::graph::modules::POSEIDON_INSTANCES;
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
use ethers::types::TransactionRequest;
use ethers::types::H160;
use ethers::types::U256;
use ethers::types::{Bytes, I256};
#[cfg(not(target_arch = "wasm32"))]
use ethers::{
    prelude::{LocalWallet, Wallet},
    utils::{Anvil, AnvilInstance},
};
use halo2_solidity_verifier::encode_calldata;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use log::{debug, info, warn};
use std::error::Error;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use std::{convert::TryFrom, sync::Arc};

/// A local ethers-rs based client
pub type EthersClient = Arc<SignerMiddleware<Provider<Http>, Wallet<SigningKey>>>;

// Generate contract bindings OUTSIDE the functions so they are part of library
abigen!(TestReads, "./abis/TestReads.json");
abigen!(DataAttestation, "./abis/DataAttestation.json");
abigen!(QuantizeData, "./abis/QuantizeData.json");

const TESTREADS_SOL: &str = include_str!("../contracts/TestReads.sol");
const QUANTIZE_DATA_SOL: &str = include_str!("../contracts/QuantizeData.sol");
const ATTESTDATA_SOL: &str = include_str!("../contracts/AttestData.sol");
const LOADINSTANCES_SOL: &str = include_str!("../contracts/LoadInstances.sol");

/// Return an instance of Anvil and a client for the given RPC URL. If none is provided, a local client is used.
#[cfg(not(target_arch = "wasm32"))]
pub async fn setup_eth_backend(
    rpc_url: Option<&str>,
    private_key: Option<&str>,
) -> Result<(AnvilInstance, EthersClient), Box<dyn Error>> {
    // Launch anvil
    let anvil = Anvil::new()
        .args(["--code-size-limit=41943040", "--disable-block-gas-limit"])
        .spawn();

    let endpoint: String;
    if let Some(rpc_url) = rpc_url {
        endpoint = rpc_url.to_string();
    } else {
        endpoint = anvil.endpoint();
    };

    // Connect to the network
    let provider = Provider::<Http>::try_from(endpoint)?.interval(Duration::from_millis(10u64));

    let chain_id = provider.get_chainid().await?.as_u64();
    info!("using chain {}", chain_id);

    // Instantiate the wallet
    let wallet: LocalWallet;
    if let Some(private_key) = private_key {
        debug!("using private key {}", private_key);
        // Sanity checks for private_key
        let private_key_format_error =
            "Private key must be in hex format, 64 chars, without 0x prefix";
        if private_key.len() != 64 {
            return Err(private_key_format_error.into());
        }
        let private_key_buffer = hex::decode(private_key)?;
        let signing_key = SigningKey::from_slice(&private_key_buffer)?;
        wallet = LocalWallet::from(signing_key);
    } else {
        wallet = anvil.keys()[0].clone().into();
    }

    // Instantiate the client with the signer
    let client = Arc::new(SignerMiddleware::new(
        provider,
        wallet.with_chain_id(chain_id),
    ));

    Ok((anvil, client))
}

///
pub async fn deploy_contract_via_solidity(
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
    contract_name: &str,
) -> Result<ethers::types::Address, Box<dyn Error>> {
    // anvil instance must be alive at least until the factory completes the deploy
    let (anvil, client) = setup_eth_backend(rpc_url, private_key).await?;

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, contract_name, runs)?;

    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone())?;
    let contract = factory.deploy(())?.send().await?;
    let addr = contract.address();

    drop(anvil);
    Ok(addr)
}

///
pub async fn deploy_da_verifier_via_solidity(
    settings_path: PathBuf,
    input: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
) -> Result<ethers::types::Address, Box<dyn Error>> {
    let (anvil, client) = setup_eth_backend(rpc_url, private_key).await?;

    let input = GraphData::from_path(input)?;

    let settings = GraphSettings::load(&settings_path)?;

    let mut scales: Vec<u32> = vec![];
    // The data that will be stored in the test contracts that will eventually be read from.
    let mut calls_to_accounts = vec![];

    let mut instance_shapes = vec![];
    let mut model_instance_offset = 0;

    if settings.run_args.input_visibility.is_hashed() {
        instance_shapes.push(POSEIDON_INSTANCES)
    } else if settings.run_args.input_visibility.is_public() {
        for idx in 0..settings.model_input_scales.len() {
            let shape = &settings.model_instance_shapes[idx];
            instance_shapes.push(shape.iter().product::<usize>());
            model_instance_offset += 1;
        }
    }

    if settings.run_args.param_visibility.is_hashed() {
        return Err(Box::new(EvmVerificationError::InvalidVisibility));
    }

    if settings.run_args.output_visibility.is_hashed() {
        instance_shapes.push(POSEIDON_INSTANCES)
    } else if settings.run_args.output_visibility.is_public() {
        for idx in model_instance_offset..model_instance_offset + settings.model_output_scales.len()
        {
            let shape = &settings.model_instance_shapes[idx];
            instance_shapes.push(shape.iter().product::<usize>());
        }
    }

    let mut instance_idx = 0;
    let mut contract_instance_offset = 0;

    if let DataSource::OnChain(source) = input.input_data {
        if settings.run_args.input_visibility.is_hashed_public() {
            // set scales 1.0
            scales.extend(vec![0; instance_shapes[instance_idx]]);
            instance_idx += 1;
        } else {
            let input_scales = settings.model_input_scales;
            // give each input a scale
            for scale in input_scales {
                scales.extend(vec![scale as u32; instance_shapes[instance_idx]]);
                instance_idx += 1;
            }
        }
        for call in source.calls {
            calls_to_accounts.push(call);
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
        if settings.run_args.output_visibility.is_hashed_public() {
            // set scales 1.0
            scales.extend(vec![0; instance_shapes[instance_idx]]);
        } else {
            let input_scales = settings.model_output_scales;
            // give each output a scale
            for scale in input_scales {
                scales.extend(vec![scale as u32; instance_shapes[instance_idx]]);
                instance_idx += 1;
            }
        }
        for call in source.calls {
            calls_to_accounts.push(call);
        }
    }

    let (contract_addresses, call_data, decimals) = if !calls_to_accounts.is_empty() {
        parse_calls_to_accounts(calls_to_accounts)?
    } else {
        return Err("Data source for either input_data or output_data must be OnChain".into());
    };

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "DataAttestation", runs)?;
    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone())?;

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
            client.address(),
        ))?
        .send()
        .await?;

    drop(anvil);
    Ok(contract.address())
}

type ParsedCallsToAccount = (Vec<H160>, Vec<Vec<Bytes>>, Vec<Vec<U256>>);

fn parse_calls_to_accounts(
    calls_to_accounts: Vec<CallsToAccount>,
) -> Result<ParsedCallsToAccount, Box<dyn Error>> {
    let mut contract_addresses = vec![];
    let mut call_data = vec![];
    let mut decimals: Vec<Vec<U256>> = vec![];
    for (i, val) in calls_to_accounts.iter().enumerate() {
        let contract_address_bytes = hex::decode(val.address.clone())?;
        let contract_address = H160::from_slice(&contract_address_bytes);
        contract_addresses.push(contract_address);
        call_data.push(vec![]);
        decimals.push(vec![]);
        for (call, decimal) in &val.call_data {
            let call_data_bytes = hex::decode(call)?;
            call_data[i].push(ethers::types::Bytes::from(call_data_bytes));
            decimals[i].push(ethers::types::U256::from_dec_str(&decimal.to_string())?);
        }
    }
    Ok((contract_addresses, call_data, decimals))
}

pub async fn update_account_calls(
    addr: H160,
    input: PathBuf,
    rpc_url: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let input = GraphData::from_path(input)?;

    // The data that will be stored in the test contracts that will eventually be read from.
    let mut calls_to_accounts = vec![];

    if let DataSource::OnChain(source) = input.input_data {
        for call in source.calls {
            calls_to_accounts.push(call);
        }
    }

    if let Some(DataSource::OnChain(source)) = input.output_data {
        for call in source.calls {
            calls_to_accounts.push(call);
        }
    }

    let (contract_addresses, call_data, decimals) = if !calls_to_accounts.is_empty() {
        parse_calls_to_accounts(calls_to_accounts)?
    } else {
        return Err("Data source for either input_data or output_data must be OnChain".into());
    };

    let (anvil, client) = setup_eth_backend(rpc_url, None).await?;

    let contract = DataAttestation::new(addr, client.clone());

    contract
        .update_account_calls(
            contract_addresses.clone(),
            call_data.clone(),
            decimals.clone(),
        )
        .send()
        .await?;

    // Instantiate a different wallet
    let wallet: LocalWallet = anvil.keys()[1].clone().into();

    let client = Arc::new(client.with_signer(wallet.with_chain_id(anvil.chain_id())));

    // update contract signer with non admin account
    let contract = DataAttestation::new(addr, client.clone());

    // call to update_account_calls should fail

    if (contract
        .update_account_calls(contract_addresses, call_data, decimals)
        .send()
        .await)
        .is_err()
    {
        info!("update_account_calls failed as expected");
    } else {
        return Err("update_account_calls should have failed".into());
    }

    Ok(())
}

/// Verify a proof using a Solidity verifier contract
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: ethers::types::Address,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    let flattened_instances = proof.instances.into_iter().flatten();

    let encoded = encode_calldata(
        addr_vk.as_ref().map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    info!("encoded: {:#?}", hex::encode(&encoded));
    let (anvil, client) = setup_eth_backend(rpc_url, None).await?;
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
    let result = result?;
    info!("result: {:#?}", result.to_vec());
    // decode return bytes value into uint8
    let result = result.to_vec().last().ok_or("no contract output")? == &1u8;
    if !result {
        return Err(Box::new(EvmVerificationError::InvalidProof));
    }

    let gas = client.estimate_gas(&tx, None).await?;

    info!("estimated verify gas cost: {:#?}", gas);

    // if gas is greater than 30 million warn the user that the gas cost is above ethereum's 30 million block gas limit
    if gas > 30_000_000.into() {
        warn!(
            "Gas cost of verify transaction is greater than 30 million block gas limit. It will fail on mainnet."
        );
    } else if gas > 15_000_000.into() {
        warn!(
            "Gas cost of verify transaction is greater than 15 million, the target block size for ethereum"
        );
    }

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
    data: &[Vec<FileSourceInner>],
) -> Result<(ContractInstance<Arc<M>, M>, Vec<u8>), Box<dyn Error>> {
    // save the abi to a tmp file
    let mut sol_path = std::env::temp_dir();
    sol_path.push("testreads.sol");
    std::fs::write(&sol_path, TESTREADS_SOL)?;

    // Compile the contract
    let (abi, bytecode, runtime_bytecode) = get_contract_artifacts(sol_path, "TestReads", 0)?;

    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone())?;

    let mut decimals = vec![];
    let mut scaled_by_decimals_data = vec![];
    for input in &data[0] {
        if input.is_float() {
            let input = input.to_float() as f32;
            let decimal_places = count_decimal_places(input) as u8;
            let scaled_by_decimals = input * f32::powf(10., decimal_places.into());
            scaled_by_decimals_data.push(I256::from(scaled_by_decimals as i128));
            decimals.push(decimal_places);
        } else if input.is_field() {
            let input = input.to_field(0);
            let hex_str_fr = format!("{:?}", input);
            scaled_by_decimals_data.push(I256::from_raw(U256::from_str_radix(&hex_str_fr, 16)?));
            decimals.push(0);
        }
    }

    let contract = factory.deploy(scaled_by_decimals_data)?.send().await?;
    Ok((contract, decimals))
}

/// Verify a proof using a Solidity DataAttestation contract.
/// Used for testing purposes.
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_with_data_attestation(
    proof: Snark<Fr, G1Affine>,
    addr_verifier: ethers::types::Address,
    addr_da: ethers::types::Address,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    use ethers::abi::{Function, Param, ParamType, StateMutability, Token};

    let mut public_inputs: Vec<U256> = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances.clone() {
        let bytes = val.to_repr();
        let u = U256::from_little_endian(bytes.inner());
        public_inputs.push(u);
    }

    let encoded_verifier = encode_calldata(
        addr_vk.as_ref().map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    info!("encoded: {:#?}", hex::encode(&encoded_verifier));

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
                name: "verifier".to_owned(),
                kind: ParamType::Address,
                internal_type: None,
            },
            Param {
                name: "encoded".to_owned(),
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
        Token::Address(addr_verifier),
        Token::Bytes(encoded_verifier),
    ])?;

    info!("encoded: {:#?}", hex::encode(&encoded));
    let (anvil, client) = setup_eth_backend(rpc_url, None).await?;
    let tx: TypedTransaction = TransactionRequest::default()
        .to(addr_da)
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
    let result = result?;
    info!("result: {:#?}", result);
    // decode return bytes value into uint8
    let result = result.to_vec().last().ok_or("no contract output")? == &1u8;
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
    data: &[Vec<FileSourceInner>],
) -> Result<Vec<CallsToAccount>, Box<dyn Error>> {
    let (contract, decimals) = setup_test_contract(client.clone(), data).await?;

    let contract = TestReads::new(contract.address(), client.clone());

    // Get the encoded call data for each input
    let mut calldata = vec![];
    for (i, _) in data.iter().flatten().enumerate() {
        let function = contract.method::<_, I256>("arr", i as u32)?;
        let call = function.calldata().ok_or("could not get calldata")?;
        // Push (call, decimals) to the calldata vector.
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
    scales: Vec<crate::Scale>,
    data: &(Vec<ethers::types::Bytes>, Vec<u8>),
) -> Result<Vec<Fr>, Box<dyn Error>> {
    // save the sol to a tmp file
    let mut sol_path = std::env::temp_dir();
    sol_path.push("quantizedata.sol");
    std::fs::write(&sol_path, QUANTIZE_DATA_SOL)?;

    let (abi, bytecode, runtime_bytecode) = get_contract_artifacts(sol_path, "QuantizeData", 0)?;
    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone())?;

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
        .await?;

    let felts = contract.to_field_element(results.clone()).call().await?;
    info!("evm quantization contract results: {:#?}", felts,);

    let results = felts
        .iter()
        .map(|x| PrimeField::from_str_vartime(&x.to_string()).unwrap())
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
        warn!(
            "Solidity runtime bytecode size is: {:#?},
            which exceeds 24577 bytes spurious dragon limit.
            Contract will fail to deploy on any chain with 
            EIP 140 enabled",
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
    runs: usize,
) -> Result<(Contract, Bytes, Bytes), Box<dyn Error>> {
    if !sol_code_path.exists() {
        return Err("sol_code_path does not exist".into());
    }
    // Create the compiler input, enabling the optimizer and setting the optimzer runs.
    let input: CompilerInput = if runs > 0 {
        let mut i = CompilerInput::new(sol_code_path)?[0]
            .clone()
            .optimizer(runs);
        i.settings.optimizer.enable();
        i
    } else {
        CompilerInput::new(sol_code_path)?[0].clone()
    };
    let compiled = Solc::default().compile(&input)?;

    let (abi, bytecode, runtime_bytecode) = match compiled.find(contract_name) {
        Some(c) => c.into_parts_or_default(),
        None => {
            return Err("could not find contract".into());
        }
    };
    Ok((abi, bytecode, runtime_bytecode))
}

/// Sets the constants stored in the da verifier
pub fn fix_da_sol(
    input_data: Option<Vec<CallsToAccount>>,
    output_data: Option<Vec<CallsToAccount>>,
) -> Result<String, Box<dyn Error>> {
    let mut accounts_len = 0;
    let mut contract = ATTESTDATA_SOL.to_string();
    let load_instances = LOADINSTANCES_SOL.to_string();
    // replace the import statement with the load_instances contract, not including the
    // `SPDX-License-Identifier: MIT pragma solidity ^0.8.20;` at the top of the file
    contract = contract.replace(
        "import './LoadInstances.sol';",
        &load_instances[load_instances
            .find("contract")
            .ok_or("could not get load-instances contract")?..],
    );

    // fill in the quantization params and total calls
    // as constants to the contract to save on gas
    if let Some(input_data) = input_data {
        let input_calls: usize = input_data.iter().map(|v| v.call_data.len()).sum();
        accounts_len = input_data.len();
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
    contract = contract.replace("AccountCall[]", &format!("AccountCall[{}]", accounts_len));

    Ok(contract)
}
