use crate::graph::input::{CallsToAccount, FileSourceInner, GraphData};
use crate::graph::modules::POSEIDON_INSTANCES;
use crate::graph::DataSource;
#[cfg(not(target_arch = "wasm32"))]
use crate::graph::GraphSettings;
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use alloy::contract::CallBuilder;
use alloy::core::primitives::Address as H160;
use alloy::core::primitives::Bytes;
use alloy::core::primitives::U256;
use alloy::dyn_abi::abi::token::{DynSeqToken, PackedSeqToken, WordToken};
use alloy::dyn_abi::abi::TokenSeq;
#[cfg(target_arch = "wasm32")]
use alloy::prelude::Wallet;
// use alloy::providers::Middleware;
use alloy::json_abi::JsonAbi;
use alloy::node_bindings::Anvil;
use alloy::primitives::{B256, I256};
use alloy::providers::fillers::{
    ChainIdFiller, FillProvider, GasFiller, JoinFill, NonceFiller, SignerFiller,
};
use alloy::providers::network::{Ethereum, EthereumSigner};
use alloy::providers::{Identity, Provider, RootProvider};
use alloy::rpc::types::eth::BlockId;
use alloy::rpc::types::eth::TransactionInput;
use alloy::rpc::types::eth::TransactionRequest;
use alloy::signers::wallet::LocalWallet;
use alloy::sol as abigen;
use alloy::transports::http::Http;
use alloy::{node_bindings::AnvilInstance, providers::ProviderBuilder};
use foundry_compilers::artifacts::Settings as SolcSettings;
use foundry_compilers::Solc;
use halo2_solidity_verifier::encode_calldata;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use itertools::Itertools;
use log::{debug, info, warn};
use reqwest::Client;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

// Generate contract bindings OUTSIDE the functions so they are part of library
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc, bytecode="60806040523461012d576102008038038061001981610131565b9283398101602090818382031261012d5782516001600160401b039384821161012d57019080601f8301121561012d5781519384116100fb5760059184831b908480610066818501610131565b80988152019282010192831161012d5784809101915b83831061011d57505050505f5b835181101561010f578281831b850101515f54680100000000000000008110156100fb5760018101805f558110156100e7575f8052845f2001555f1981146100d357600101610089565b634e487b7160e01b5f52601160045260245ffd5b634e487b7160e01b5f52603260045260245ffd5b634e487b7160e01b5f52604160045260245ffd5b60405160a990816101578239f35b825181529181019185910161007c565b5f80fd5b6040519190601f01601f191682016001600160401b038111838210176100fb5760405256fe60808060405260043610156011575f80fd5b5f90813560e01c6371e5ee5f146025575f80fd5b34606f576020366003190112606f576004358254811015606b5782602093527f290decd9548b62a8d60345a988386fc84ba6bc95484008f6362f93160ef3e56301548152f35b8280fd5b5080fdfea2646970667358221220dc28d7ff0d25a49f74c6b97a87c7c6039ee98d715c0f61be72cc4d180d40a41e64736f6c63430008140033")]
    contract TestReads {
        int[] public arr;

        constructor(int256[] memory _numbers) {
            for (uint256 i = 0; i < _numbers.length; i++) {
                arr.push(_numbers[i]);
            }
        }
    }
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    DataAttestation,
    "./abis/DataAttestation.json"
);
abigen!(
    #[allow(missing_docs)]
    #[sol(rpc, bytecode="608080604052346100165761065e908161001b8239f35b5f80fdfe6080604052600480361015610012575f80fd5b5f803560e01c80630a7e4b96146101a95763d3dc6d1f14610031575f80fd5b346101a657602090816003193601126101a65782359067ffffffffffffffff82116101a657366023830112156101a6578184013561007661007182610489565b61044f565b928484838152016024809360051b830101913683116101a2578301905b828210610185575050508251926100b86100af61007186610489565b94808652610489565b84860190601f1901368237835b8251811015610143576100d8818461051d565b5160070b7f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000019081810190878383129112908015821691151617610131579061012c929106610126828961051d565b526104fb565b6100c5565b634e487b7160e01b875260118a528587fd5b8587838760405193838594850191818652518092526040850193925b82811061016e57505050500390f35b83518552869550938101939281019260010161015f565b81358060070b810361019e578152908601908601610093565b8580fd5b8480fd5b80fd5b50346101a65760603660031901126101a65781359067ffffffffffffffff80831161044b573660238401121561044b5782840135916101ea61007184610489565b92839481855260208095016024809360051b830101913683116101a257838101915b8383106103d2575050505080358381116103ce5761022d90369088016104a1565b926044359081116103ce5761024590369088016104a1565b9585519461026161025861007188610489565b96808852610489565b8682019790601f1901368937845b815181101561038c57610282818361051d565b5183818051810103126103885783015186811280610378575b6102a5838a61051d565b51604d81116103665790839291600a0a8d6102c3600196879261051d565b511b6102d0828286610555565b938215610354579082910980861b908082046002149015171561034257101561031b575b6103119350156103165761030790610545565b610126828b61051d565b61026f565b610307565b919281018091116103305761031192916102f4565b634e487b7160e01b8852601186528688fd5b634e487b7160e01b8b5260118952898bfd5b634e487b7160e01b8c5260128a528a8cfd5b634e487b7160e01b8952601187528789fd5b9061038290610545565b9061029b565b8680fd5b87838a8860405193838594850191818652518092526040850193925b8281106103b757505050500390f35b8351855286955093810193928101926001016103a8565b8280fd5b823587811161038857820136604382011215610388578581013588811161043957610405601f8201601f19168b0161044f565b9181835260449036828483010111610435578b838196948296948d9401838601378301015281520192019161020c565b8980fd5b634e487b7160e01b885260418c528688fd5b5080fd5b6040519190601f01601f1916820167ffffffffffffffff81118382101761047557604052565b634e487b7160e01b5f52604160045260245ffd5b67ffffffffffffffff81116104755760051b60200190565b9080601f830112156104f7578135906104bc61007183610489565b9182938184526020808095019260051b8201019283116104f7578301905b8282106104e8575050505090565b813581529083019083016104da565b5f80fd5b5f1981146105095760010190565b634e487b7160e01b5f52601160045260245ffd5b80518210156105315760209160051b010190565b634e487b7160e01b5f52603260045260245ffd5b600160ff1b8114610509575f0390565b915f19828409928281029283808610950394808603951461060657848311156105c95782910960018219018216809204600280826003021880830282030280830282030280830282030280830282030280830282030280920290030293600183805f03040190848311900302920304170290565b60405162461bcd60e51b81526020600482015260156024820152744d6174683a206d756c446976206f766572666c6f7760581b6044820152606490fd5b505080925015610614570490565b634e487b7160e01b5f52601260045260245ffdfea2646970667358221220dcf762b43d6e326747a25e0fba385db1c3a47bd2ee3846a15b7917679cb8fe4d64736f6c63430008140033")]
    contract QuantizeData {
        /**
         * @notice EZKL P value
         * @dev In order to prevent the verifier from accepting two version of the same instance, n and the quantity (n + P),  where n + P <= 2^256, we require that all instances are stricly less than P. a
         * @dev The reason for this is that the assmebly code of the verifier performs all arithmetic operations modulo P and as a consequence can't distinguish between n and n + P.
         */
        uint256 constant ORDER =
            uint256(
                0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
            );

        // /**
        //  * @notice Calculates floor(x * y / denominator) with full precision. Throws if result overflows a uint256 or denominator == 0
        //  * @dev Original credit to Remco Bloemen under MIT license (https://xn--2-umb.com/21/muldiv)
        //  * with further edits by Uniswap Labs also under MIT license.
        //  */
        function mulDiv(
            uint256 x,
            uint256 y,
            uint256 denominator
        ) internal pure returns (uint256 result) {
            unchecked {
                // 512-bit multiply [prod1 prod0] = x * y. Compute the product mod 2^256 and mod 2^256 - 1, then use
                // use the Chinese Remainder Theorem to reconstruct the 512 bit result. The result is stored in two 256
                // variables such that product = prod1 * 2^256 + prod0.
                uint256 prod0; // Least significant 256 bits of the product
                uint256 prod1; // Most significant 256 bits of the product
                assembly {
                    let mm := mulmod(x, y, not(0))
                    prod0 := mul(x, y)
                    prod1 := sub(sub(mm, prod0), lt(mm, prod0))
                }

                // Handle non-overflow cases, 256 by 256 division.
                if (prod1 == 0) {
                    // Solidity will revert if denominator == 0, unlike the div opcode on its own.
                    // The surrounding unchecked block does not change this fact.
                    // See https://docs.soliditylang.org/en/latest/control-structures.html#checked-or-unchecked-arithmetic.
                    return prod0 / denominator;
                }

                // Make sure the result is less than 2^256. Also prevents denominator == 0.
                require(denominator > prod1, "Math: mulDiv overflow");

                ///////////////////////////////////////////////
                // 512 by 256 division.
                ///////////////////////////////////////////////

                // Make division exact by subtracting the remainder from [prod1 prod0].
                uint256 remainder;
                assembly {
                    // Compute remainder using mulmod.
                    remainder := mulmod(x, y, denominator)

                    // Subtract 256 bit number from 512 bit number.
                    prod1 := sub(prod1, gt(remainder, prod0))
                    prod0 := sub(prod0, remainder)
                }

                // Factor powers of two out of denominator and compute largest power of two divisor of denominator. Always >= 1.
                // See https://cs.stackexchange.com/q/138556/92363.

                // Does not overflow because the denominator cannot be zero at this stage in the function.
                uint256 twos = denominator & (~denominator + 1);
                assembly {
                    // Divide denominator by twos.
                    denominator := div(denominator, twos)

                    // Divide [prod1 prod0] by twos.
                    prod0 := div(prod0, twos)

                    // Flip twos such that it is 2^256 / twos. If twos is zero, then it becomes one.
                    twos := add(div(sub(0, twos), twos), 1)
                }

                // Shift in bits from prod1 into prod0.
                prod0 |= prod1 * twos;

                // Invert denominator mod 2^256. Now that denominator is an odd number, it has an inverse modulo 2^256 such
                // that denominator * inv = 1 mod 2^256. Compute the inverse by starting with a seed that is correct for
                // four bits. That is, denominator * inv = 1 mod 2^4.
                uint256 inverse = (3 * denominator) ^ 2;

                // Use the Newton-Raphson iteration to improve the precision. Thanks to Hensel's lifting lemma, this also works
                // in modular arithmetic, doubling the correct bits in each step.
                inverse *= 2 - denominator * inverse; // inverse mod 2^8
                inverse *= 2 - denominator * inverse; // inverse mod 2^16
                inverse *= 2 - denominator * inverse; // inverse mod 2^32
                inverse *= 2 - denominator * inverse; // inverse mod 2^64
                inverse *= 2 - denominator * inverse; // inverse mod 2^128
                inverse *= 2 - denominator * inverse; // inverse mod 2^256

                // Because the division is now exact we can divide by multiplying with the modular inverse of denominator.
                // This will give us the correct result modulo 2^256. Since the preconditions guarantee that the outcome is
                // less than 2^256, this is the final result. We don't need to compute the high bits of the result and prod1
                // is no longer required.
                result = prod0 * inverse;
                return result;
            }
        }

        function quantize_data(
            bytes[] memory data,
            uint256[] memory decimals,
            uint256[] memory scales
        ) external pure returns (int256[] memory quantized_data) {
            quantized_data = new int256[](data.length);
            for (uint i; i < data.length; i++) {
                int x = abi.decode(data[i], (int256));
                bool neg = x < 0;
                if (neg) x = -x;
                uint denom = 10 ** decimals[i];
                uint scale = 1 << scales[i];
                uint output = mulDiv(uint256(x), scale, denom);
                if (mulmod(uint256(x), scale, denom) * 2 >= denom) {
                    output += 1;
                }

                quantized_data[i] = neg ? -int256(output) : int256(output);
            }
        }

        function to_field_element(
            int256[] memory quantized_data
        ) public pure returns (uint256[] memory output) {
            output = new uint256[](quantized_data.length);
            for (uint i; i < quantized_data.length; i++) {
                output[i] = uint256(quantized_data[i] + int(ORDER)) % ORDER;
            }
        }
    }
);

// we have to generate these two contract differently because they are generated dynamically ! and hence the static compilation from above does not suit
const ATTESTDATA_SOL: &str = include_str!("../contracts/AttestData.sol");
const LOADINSTANCES_SOL: &str = include_str!("../contracts/LoadInstances.sol");

pub type EthersClient = Arc<
    FillProvider<
        JoinFill<
            JoinFill<JoinFill<JoinFill<Identity, GasFiller>, NonceFiller>, ChainIdFiller>,
            SignerFiller<EthereumSigner>,
        >,
        RootProvider<Http<Client>>,
        Http<Client>,
        Ethereum,
    >,
>;

pub type ContractFactory<M> = CallBuilder<Http<Client>, Arc<M>, ()>;

/// Return an instance of Anvil and a client for the given RPC URL. If none is provided, a local client is used.
#[cfg(not(target_arch = "wasm32"))]
pub async fn setup_eth_backend(
    rpc_url: Option<&str>,
    private_key: Option<&str>,
) -> Result<(AnvilInstance, EthersClient, alloy::primitives::Address), Box<dyn Error>> {
    // Launch anvil

    let anvil = Anvil::new()
        .args(["--code-size-limit=41943040", "--disable-block-gas-limit"])
        .spawn();

    let endpoint: String;
    if let Some(rpc_url) = rpc_url {
        endpoint = rpc_url.to_string();
    } else {
        endpoint = anvil.endpoint();
    }

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
        wallet = LocalWallet::from_slice(&private_key_buffer)?;
    } else {
        wallet = anvil.keys()[0].clone().into();
    }

    let wallet_address = wallet.address();

    // Connect to the network
    let client = Arc::new(
        ProviderBuilder::new()
            .with_recommended_fillers()
            .signer(EthereumSigner::from(wallet))
            .on_http(endpoint.parse()?),
    );

    let chain_id = client.get_chain_id().await?;
    info!("using chain {}", chain_id);

    Ok((anvil, client, wallet_address))
}

///
pub async fn deploy_contract_via_solidity(
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
    contract_name: &str,
) -> Result<H160, Box<dyn Error>> {
    // anvil instance must be alive at least until the factory completes the deploy
    let (_anvil, client, _) = setup_eth_backend(rpc_url, private_key).await?;

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, contract_name, runs).await?;

    let factory =
        get_sol_contract_factory(abi, bytecode, runtime_bytecode, client.clone(), None::<()>)?;
    let contract = factory.deploy().await?;

    Ok(contract)
}

///
pub async fn deploy_da_verifier_via_solidity(
    settings_path: PathBuf,
    input: PathBuf,
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
) -> Result<H160, Box<dyn Error>> {
    let (_anvil, client, client_address) = setup_eth_backend(rpc_url, private_key).await?;

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
        get_contract_artifacts(sol_code_path, "DataAttestation", runs).await?;

    let factory = get_sol_contract_factory(
        abi,
        bytecode,
        runtime_bytecode,
        client.clone(),
        Some((
            DynSeqToken(
                contract_addresses
                    .iter()
                    .map(|ca| WordToken(ca.into_word()))
                    .collect_vec(),
            ),
            DynSeqToken(
                call_data
                    .iter()
                    .map(|bytes| {
                        DynSeqToken(
                            bytes
                                .iter()
                                .map(|i| PackedSeqToken(i.as_ref()))
                                .collect_vec(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            DynSeqToken(
                decimals
                    .iter()
                    .map(|ints| {
                        DynSeqToken(ints.iter().map(|i| WordToken(B256::from(*i))).collect_vec())
                    })
                    .collect::<Vec<_>>(),
            ),
            DynSeqToken(
                scales
                    .into_iter()
                    .map(|i| WordToken(U256::from(i).into()))
                    .collect_vec(),
            ),
            WordToken(U256::from(contract_instance_offset as u32).into()),
            WordToken(client_address.into_word()),
        )),
    )?;

    debug!("call_data: {:#?}", call_data);
    info!("contract_addresses: {:#?}", contract_addresses);
    info!("decimals: {:#?}", decimals);

    let contract = factory.deploy().await?;

    Ok(contract)
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
            call_data[i].push(Bytes::from(call_data_bytes));
            decimals[i].push(I256::from_dec_str(&decimal.to_string())?.unsigned_abs());
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

    let (_anvil, client, _) = setup_eth_backend(rpc_url, None).await?;

    let contract = DataAttestation::new(addr, client.clone());

    let _ = contract
        .updateAccountCalls(
            contract_addresses.clone(),
            call_data.clone(),
            decimals.clone(),
        )
        .send()
        .await?;

    // update contract signer with non admin account
    let contract = DataAttestation::new(addr, client.clone());

    // call to update_account_calls should fail

    if (contract
        .updateAccountCalls(contract_addresses, call_data, decimals)
        .send()
        .await)
        .is_err()
    {
        info!("updateAccountCalls failed as expected");
    } else {
        return Err("updateAccountCalls should have failed".into());
    }

    Ok(())
}

/// Verify a proof using a Solidity verifier contract
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: H160,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    let flattened_instances = proof.instances.into_iter().flatten();

    let encoded = encode_calldata(
        addr_vk.as_ref().map(|x| x.0).map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    debug!("encoded: {:#?}", hex::encode(&encoded));

    let input: TransactionInput = encoded.into();

    let (_anvil, client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr).input(input);
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx).await;

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

    let gas = client.estimate_gas(&tx, BlockId::default()).await?;

    info!("estimated verify gas cost: {:#?}", gas);

    // if gas is greater than 30 million warn the user that the gas cost is above ethereum's 30 million block gas limit
    if gas > 30_000_000_u128 {
        warn!(
            "Gas cost of verify transaction is greater than 30 million block gas limit. It will fail on mainnet."
        );
    } else if gas > 15_000_000_u128 {
        warn!(
            "Gas cost of verify transaction is greater than 15 million, the target block size for ethereum"
        );
    }

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
pub async fn setup_test_contract<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    data: &[Vec<FileSourceInner>],
) -> Result<(TestReads::TestReadsInstance<Http<Client>, Arc<M>>, Vec<u8>), Box<dyn Error>> {
    let mut decimals = vec![];
    let mut scaled_by_decimals_data = vec![];
    for input in &data[0] {
        if input.is_float() {
            let input = input.to_float() as f32;
            let decimal_places = count_decimal_places(input) as u8;
            let scaled_by_decimals = input * f32::powf(10., decimal_places.into());
            scaled_by_decimals_data.push(I256::from_dec_str(
                &(scaled_by_decimals as i32).to_string(),
            )?);
            decimals.push(decimal_places);
        } else if input.is_field() {
            let input = input.to_field(0);
            let hex_str_fr = format!("{:?}", input);
            scaled_by_decimals_data.push(I256::from_raw(U256::from_str_radix(&hex_str_fr, 16)?));
            decimals.push(0);
        }
    }

    // Compile the contract
    let contract = TestReads::deploy(client, scaled_by_decimals_data).await?;

    Ok((contract, decimals))
}

/// Verify a proof using a Solidity DataAttestation contract.
/// Used for testing purposes.
#[cfg(not(target_arch = "wasm32"))]
pub async fn verify_proof_with_data_attestation(
    proof: Snark<Fr, G1Affine>,
    addr_verifier: H160,
    addr_da: H160,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, Box<dyn Error>> {
    use ethabi::{Function, Param, ParamType, StateMutability, Token};

    let mut public_inputs: Vec<U256> = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances.clone() {
        let bytes = val.to_repr();
        let u = U256::from_le_slice(bytes.as_slice());
        public_inputs.push(u);
    }

    let encoded_verifier = encode_calldata(
        addr_vk.as_ref().map(|x| x.0).map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    info!("encoded: {:#?}", hex::encode(&encoded_verifier));

    info!("public_inputs: {:#?}", public_inputs);
    info!("proof: {:#?}", Bytes::from(proof.proof.to_vec()));

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
        Token::Address(addr_verifier.0 .0.into()),
        Token::Bytes(encoded_verifier),
    ])?;

    info!("encoded: {:#?}", hex::encode(&encoded));

    let encoded: TransactionInput = encoded.into();

    let (_anvil, client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr_da).input(encoded);
    debug!("transaction {:#?}", tx);
    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx, BlockId::default()).await?
    );

    let result = client.call(&tx).await;
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

    Ok(true)
}

/// Tests on-chain data storage by deploying a contract that stores the network input and or output
/// data in its storage. It does this by converting the floating point values to integers and storing the
/// the number of decimals of the floating point value on chain.
pub async fn test_on_chain_data<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    data: &[Vec<FileSourceInner>],
) -> Result<Vec<CallsToAccount>, Box<dyn Error>> {
    let (contract, decimals) = setup_test_contract(client.clone(), data).await?;

    // Get the encoded call data for each input
    let mut calldata = vec![];
    for (i, _) in data.iter().flatten().enumerate() {
        let builder = contract.arr(U256::from(i));
        let call = builder.calldata();
        // Push (call, decimals) to the calldata vector.
        calldata.push((hex::encode(call), decimals[i]));
    }
    // Instantiate a new CallsToAccount struct
    let calls_to_account = CallsToAccount {
        call_data: calldata,
        address: hex::encode(contract.address().0 .0),
    };
    info!("calls_to_account: {:#?}", calls_to_account);
    Ok(vec![calls_to_account])
}

/// Reads on-chain inputs, returning the raw encoded data returned from making all the calls in on_chain_input_data
#[cfg(not(target_arch = "wasm32"))]
pub async fn read_on_chain_inputs<M: 'static + Provider<Http<Client>, Ethereum>>(
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
            let input: TransactionInput = call_data_bytes.into();

            let tx = TransactionRequest::default()
                .to(contract_address)
                .from(address)
                .input(input);
            debug!("transaction {:#?}", tx);

            let result = client.call(&tx).await?;
            debug!("return data {:#?}", result);
            fetched_inputs.push(result);
            decimals.push(*decimal);
        }
    }
    Ok((fetched_inputs, decimals))
}

///
#[cfg(not(target_arch = "wasm32"))]
pub async fn evm_quantize<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    scales: Vec<crate::Scale>,
    data: &(Vec<Bytes>, Vec<u8>),
) -> Result<Vec<Fr>, Box<dyn Error>> {
    use alloy::primitives::ParseSignedError;

    let contract = QuantizeData::deploy(&client).await?;

    let fetched_inputs = data.0.clone();
    let decimals = data.1.clone();

    let fetched_inputs = fetched_inputs
        .iter()
        .map(|x| Result::<_, std::convert::Infallible>::Ok(Bytes::from(x.to_vec())))
        .collect::<Result<Vec<Bytes>, _>>()?;

    let decimals = decimals
        .iter()
        .map(|x| Ok(I256::from_dec_str(&x.to_string())?.unsigned_abs()))
        .collect::<Result<Vec<U256>, ParseSignedError>>()?;

    let scales = scales
        .iter()
        .map(|x| Ok(I256::from_dec_str(&x.to_string())?.unsigned_abs()))
        .collect::<Result<Vec<U256>, ParseSignedError>>()?;

    info!("scales: {:#?}", scales);
    info!("decimals: {:#?}", decimals);
    info!("fetched_inputs: {:#?}", fetched_inputs);

    let results = contract
        .quantize_data(fetched_inputs, decimals, scales)
        .call()
        .await?
        .quantized_data;

    let felts = contract
        .to_field_element(results.clone())
        .call()
        .await?
        .output;
    info!("evm quantization contract results: {:#?}", felts,);

    let results = felts
        .iter()
        .map(|x| PrimeField::from_str_vartime(&x.to_string()).unwrap())
        .collect::<Vec<Fr>>();
    info!("evm quantization results: {:#?}", results,);
    Ok(results.to_vec())
}

/// Generates the contract factory for a solidity verifier, optionally compiling the code with optimizer runs set on the Solc compiler.
fn get_sol_contract_factory<'a, M: 'static + Provider<Http<Client>, Ethereum>, T: TokenSeq<'a>>(
    abi: JsonAbi,
    bytecode: Bytes,
    runtime_bytecode: Bytes,
    client: Arc<M>,
    params: Option<T>,
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

    // Encode the constructor args & concatenate with the bytecode if necessary
    let data: Bytes = match (abi.constructor(), params.is_none()) {
        (None, false) => {
            return Err("Constructor arguments provided but no constructor found".into())
        }
        (None, true) => bytecode.clone(),
        (Some(_), _) => {
            let mut data = bytecode.to_vec();

            if let Some(params) = params {
                let params = alloy::sol_types::abi::encode_sequence(&params);
                data.extend(params);
            }
            data.into()
        }
    };

    Ok(CallBuilder::new_raw_deploy(client.clone(), data))
}

/// Compiles a solidity verifier contract and returns the abi, bytecode, and runtime bytecode
#[cfg(not(target_arch = "wasm32"))]
pub async fn get_contract_artifacts(
    sol_code_path: PathBuf,
    contract_name: &str,
    runs: usize,
) -> Result<(JsonAbi, Bytes, Bytes), Box<dyn Error>> {
    use foundry_compilers::{
        artifacts::{output_selection::OutputSelection, Optimizer},
        compilers::CompilerInput,
        SolcInput, SHANGHAI_SOLC,
    };

    if !sol_code_path.exists() {
        return Err("sol_code_path does not exist".into());
    }

    let mut settings = SolcSettings::default();
    settings.optimizer = Optimizer {
        enabled: Some(true),
        runs: Some(runs),
        details: None,
    };
    settings.output_selection = OutputSelection::default_output_selection();

    let input = SolcInput::build(
        std::collections::BTreeMap::from([(
            sol_code_path.clone(),
            foundry_compilers::artifacts::Source::read(sol_code_path)?,
        )]),
        settings,
        &SHANGHAI_SOLC,
    );

    let solc_opt = Solc::find_svm_installed_version(&SHANGHAI_SOLC.to_string())?;
    let solc = match solc_opt {
        Some(solc) => solc,
        None => {
            info!("required solc version is missing ... installing");
            Solc::install(&SHANGHAI_SOLC).await?
        }
    };

    let compiled: foundry_compilers::CompilerOutput = solc.compile(&input[0])?;

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
