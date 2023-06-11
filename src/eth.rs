use crate::graph::{OnChainData, GraphInput};
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use ethers::contract::abigen;
use ethers::contract::ContractFactory;
use ethers::core::k256::ecdsa::SigningKey;
use ethers::middleware::SignerMiddleware;
#[cfg(target_arch = "wasm32")]
use ethers::prelude::Wallet;
use ethers::providers::Middleware;
use ethers::providers::{Http, Provider};
use ethers::signers::Signer;
use ethers::types::H160;
use ethers::types::TransactionRequest;
use ethers::types::U256;
use ethers::types::transaction::eip2718::TypedTransaction;
#[cfg(not(target_arch = "wasm32"))]
use ethers::{
    prelude::{LocalWallet, Wallet},
    utils::{Anvil, AnvilInstance},
};
use ethers_solc::{CompilerInput, Solc};
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
pub async fn setup_eth_backend(rpc_url: Option<&str>) -> Result<(AnvilInstance, EthersClient), Box<dyn Error>> {
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
    let provider =
        Provider::<Http>::try_from(endpoint)?.interval(Duration::from_millis(10u64));

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
    sol_code_path: PathBuf,
    runs: Option<usize>,
) -> Result<bool, Box<dyn Error>> {
    let (anvil, client) = setup_eth_backend(None).await?;

    let factory = get_sol_contract_factory(sol_code_path, client.clone(), runs).unwrap();

    let contract = factory.deploy(())?.send().await?;
    let addr = contract.address();

    abigen!(Verifier, "./Verifier.json");
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

/// get_provider returns a JSON RPC HTTP Provider
pub fn get_provider(rpc_url: &str) -> Result<Provider<Http>, Box<dyn Error>> {
    let provider = Provider::<Http>::try_from(rpc_url)?;
    debug!("{:#?}", provider);
    Ok(provider)
}
/// Reads on-chain inputs, casts them as U256, converts them to f32 according to the decimals field in the data and returns them as a vector
pub async fn read_on_chain_inputs (
    rpc_url: Option<&str>,
    data: &mut GraphInput
) -> Result<GraphInput, Box<dyn Error>> {
    let (anvil, client) = setup_eth_backend(rpc_url).await?;
    // Iterate over all on-chain inputs
    if let Some(on_chain_inputs) = &data.on_chain_input_data {
        for on_chain_data in on_chain_inputs {
            // Construct the address
            let contract_address_bytes = hex::decode(on_chain_data.address.clone())?;
            let contract_address = H160::from_slice(&contract_address_bytes);
            // TODO: Set up anvil cllient here to make reading on-chain data easier.
            for call_data in &on_chain_data.call_data {
                let call_data_bytes = hex::decode(call_data.clone())?;
                let tx: TypedTransaction = TransactionRequest::default()
                    .to(contract_address)
                    .from(client.address())
                    .data(call_data_bytes)
                    .into();

                info!("created tx");
                debug!("transaction {:#?}", tx);

                let result = client.call(&tx, None).await?;
                debug!("return data {:#?}", result);

                // Convert bytes to U256 to f32 according to decimals
                let response = U256::from_big_endian(&result[..]);
                debug!("response {:#?}", response);

                // Convert U256 to f32 according to decimals
                let converted = response.low_u64() as f32 / 10f32.powi(on_chain_data.decimals as i32);
                debug!("rconverted {:#?}", converted);

                // Store the result
                data.input_data[0].push(converted);
            }
        }
    }
    drop(anvil);
    Ok(data.clone())
}

/// Generates the contract factory for a solidity verifier, optionally compiling the code with optimizer runs set on the Solc compiler.
fn get_sol_contract_factory<M: 'static + Middleware>(
    sol_code_path: PathBuf,
    client: Arc<M>,
    runs: Option<usize>,
) -> Result<ContractFactory<M>, Box<dyn Error>> {
    const MAX_RUNTIME_BYTECODE_SIZE: usize = 24_577; // Smart contract size limit
                                                     // Create the compiler input, enabling the optimizer and setting the optimzer runs.
    let input: CompilerInput = if let Some(r) = runs {
        let mut i = CompilerInput::new(sol_code_path)?[0].clone().optimizer(r);
        i.settings.optimizer.enable();
        i
    } else {
        CompilerInput::new(sol_code_path)?[0].clone()
    };
    let compiled = Solc::default().compile(&input).unwrap();
    let (abi, bytecode, _runtime_bytecode) = compiled
        .find("Verifier")
        .expect("could not find contract")
        .into_parts_or_default();
    let size = _runtime_bytecode.len();
    debug!("runtime bytecode size: {:#?}", size);
    if size > MAX_RUNTIME_BYTECODE_SIZE {
        // `_runtime_bytecode` exceeds the limit
        panic!(
            "Solidity runtime bytecode size is: {:#?}, 
            which exceeds 24577 bytes limit.
            Try setting '--optimzer-runs 1' 
            so SOLC can optimize for the smallest deployment",
            size
        );
    }
    Ok(ContractFactory::new(abi, bytecode, client))
}

use regex::Regex;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Reads in raw bytes code and generates equivalent .sol file
/// Can optionally attest to on-chain inputs
pub fn fix_verifier_sol(
    input_file: PathBuf, 
    data: Option<Vec<OnChainData>>
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

    let mut contract = if let Some(data) = data {
        format!(
            r#"// SPDX-License-Identifier: MIT
            pragma solidity ^0.8.17;
            
            contract DataAttestationVerifier {{
            
                /**
                 * @notice Struct used to make view only calls to accounts to fetch the data that EZKL reads from.
                 * @param the address of the account to make calls to
                 * @param the abi encoded function calls to make to the `contractAddress`
                 */
                struct AccountCall {{
                    address contractAddress;
                    mapping(uint256 => bytes) callData;
                    uint callCount;
                }}
                AccountCall[{}] public accountCalls;
            
                uint public totalCalls;
            
                /**
                 * @dev Initialize the contract with account calls the EZKL model will read from.
                 * @param _contractAddresses - The calls to all the contracts EZKL reads storage from.
                 * @param _callData - The abi encoded function calls to make to the `contractAddress` that EZKL reads storage from.
                 */
                constructor(address[] memory _contractAddresses, bytes[][] memory _callData) {{
                    require(_contractAddresses.length == _callData.length && accountCalls.length == _contractAddresses.length, "Invalid input length");
                    // fill in the accountCalls storage array
                    for(uint256 i = 0; i < _contractAddresses.length; i++) {{
                        AccountCall storage accountCall = accountCalls[i];
                        accountCall.contractAddress = _contractAddresses[i];
                        accountCall.callCount = _callData[i].length;
                        for(uint256 j = 0; j < _callData[i].length; j++){{
                            accountCall.callData[j] = _callData[i][j];
                        }}
                        // count the total number of storage reads across all of the accounts
                        totalCalls += _callData[i].length;
                    }}
                }}
            
                function staticCall (address target, bytes memory data) internal view returns (bytes memory) {{
                    (bool success, bytes memory returndata) = target.staticcall(data);
                    if (success) {{
                        if (returndata.length == 0) {{
                            // only check isContract if the call was successful and the return data is empty
                            // otherwise we already know that it was a contract
                            require(target.code.length > 0, "Address: call to non-contract");
                        }}
                    return returndata;
                    }} else {{
                        revert("Address: low-level call failed");
                    }}
                }}
            
                function attestData(uint256[] memory pubInputs) internal view {{
                    require(pubInputs.length >= totalCalls, "Invalid public inputs length");
                    uint256 _accountCount = accountCalls.length;
                    uint counter = 0; 
                    for (uint8 i = 0; i < _accountCount; ++i) {{
                        address account = accountCalls[i].contractAddress;
                        for (uint8 j = 0; j < accountCalls[i].callCount; j++) {{
                            bytes memory returnData = staticCall(account, accountCalls[i].callData[j]);
                            require(abi.decode(returnData, (uint256)) == pubInputs[counter], "Public input does not match");
                            counter++;
                        }}
                    }}
                }}
            
                function verifyWithDataAttestation(
                    uint256[] memory pubInputs,
                    bytes memory proof
                ) public view returns (bool) {{
                    bool success = true;
                    bytes32[{}] memory transcript;
                    attestData(pubInputs);
                    assembly {{ 
                "#,
            data.len(),
            max_transcript_addr
        )
        .trim()
        .to_string()
    } else {
        format!(
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
        .to_string()
    };

    // using a boxed Write trait object here to show it works for any Struct impl'ing Write
    // you may also use a std::fs::File here
    let write: Box<&mut dyn Write> = Box::new(&mut contract);

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
