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
use std::str::FromStr;
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
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b506108b18061001d5f395ff3fe608060405234801561000f575f80fd5b5060043610610034575f3560e01c80630a7e4b9614610038578063d3dc6d1f14610061575b5f80fd5b61004b61004636600461047c565b610074565b60405161005891906105bc565b60405180910390f35b61004b61006f3660046105ff565b6101e5565b606083516001600160401b0381111561008f5761008f6103ae565b6040519080825280602002602001820160405280156100b8578160200160208202803683370190505b5090505f5b84518110156101dd575f8582815181106100d9576100d9610699565b60200260200101518060200190518101906100f491906106ad565b90505f8112801561010b57610108826106d8565b91505b5f86848151811061011e5761011e610699565b6020026020010151600a61013291906107d4565b90505f86858151811061014757610147610699565b60200260200101516001901b90505f6101618583856102bf565b9050828380610172576101726107df565b8387096101809060026107f3565b106101935761019060018261080a565b90505b8361019e57806101a7565b6101a7816106d8565b8787815181106101b9576101b9610699565b602002602001018181525050505050505080806101d59061081d565b9150506100bd565b509392505050565b606081516001600160401b03811115610200576102006103ae565b604051908082528060200260200182016040528015610229578160200160208202803683370190505b5090505f5b82518110156102b9577f30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f00000018084838151811061026b5761026b610699565b602002602001015160070b6102809190610835565b61028a919061085c565b82828151811061029c5761029c610699565b6020908102919091010152806102b18161081d565b91505061022e565b50919050565b5f80805f19858709858702925082811083820303915050805f036102f6578382816102ec576102ec6107df565b04925050506103a7565b8084116103415760405162461bcd60e51b81526020600482015260156024820152744d6174683a206d756c446976206f766572666c6f7760581b604482015260640160405180910390fd5b5f848688098519600190810187169687900496828603819004959092119093035f82900391909104909201919091029190911760038402600290811880860282030280860282030280860282030280860282030280860282030280860290910302029150505b9392505050565b634e487b7160e01b5f52604160045260245ffd5b604051601f8201601f191681016001600160401b03811182821017156103ea576103ea6103ae565b604052919050565b5f6001600160401b0382111561040a5761040a6103ae565b5060051b60200190565b5f82601f830112610423575f80fd5b81356020610438610433836103f2565b6103c2565b82815260059290921b84018101918181019086841115610456575f80fd5b8286015b84811015610471578035835291830191830161045a565b509695505050505050565b5f805f6060848603121561048e575f80fd5b83356001600160401b03808211156104a4575f80fd5b818601915086601f8301126104b7575f80fd5b813560206104c7610433836103f2565b82815260059290921b8401810191818101908a8411156104e5575f80fd5b8286015b8481101561056e57803586811115610500575f8081fd5b8701603f81018d13610511575f8081fd5b84810135604088821115610527576105276103ae565b610539601f8301601f191688016103c2565b8281528f8284860101111561054d575f8081fd5b82828501898301375f928101880192909252508452509183019183016104e9565b5097505087013592505080821115610584575f80fd5b61059087838801610414565b935060408601359150808211156105a5575f80fd5b506105b286828701610414565b9150509250925092565b602080825282518282018190525f9190848201906040850190845b818110156105f3578351835292840192918401916001016105d7565b50909695505050505050565b5f6020808385031215610610575f80fd5b82356001600160401b03811115610625575f80fd5b8301601f81018513610635575f80fd5b8035610643610433826103f2565b81815260059190911b82018301908381019087831115610661575f80fd5b928401925b8284101561068e5783358060070b811461067f575f8081fd5b82529284019290840190610666565b979650505050505050565b634e487b7160e01b5f52603260045260245ffd5b5f602082840312156106bd575f80fd5b5051919050565b634e487b7160e01b5f52601160045260245ffd5b5f600160ff1b82016106ec576106ec6106c4565b505f0390565b600181815b8085111561072c57815f1904821115610712576107126106c4565b8085161561071f57918102915b93841c93908002906106f7565b509250929050565b5f82610742575060016107ce565b8161074e57505f6107ce565b8160018114610764576002811461076e5761078a565b60019150506107ce565b60ff84111561077f5761077f6106c4565b50506001821b6107ce565b5060208310610133831016604e8410600b84101617156107ad575081810a6107ce565b6107b783836106f2565b805f19048211156107ca576107ca6106c4565b0290505b92915050565b5f6103a78383610734565b634e487b7160e01b5f52601260045260245ffd5b80820281158282048414176107ce576107ce6106c4565b808201808211156107ce576107ce6106c4565b5f6001820161082e5761082e6106c4565b5060010190565b8082018281125f831280158216821582161715610854576108546106c4565b505092915050565b5f8261087657634e487b7160e01b5f52601260045260245ffd5b50069056fea26469706673582212200b8a0a357f7d2a8895754f5b26e714ec153b59420bdab2d0ad696eb17a0f235164736f6c63430008140033")]
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
            int64[] memory quantized_data
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
    println!("client_address: {:?}", client_address);

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
            // address[] memory _contractAddresses,
            DynSeqToken(
                contract_addresses
                    .iter()
                    .map(|ca| WordToken(ca.into_word()))
                    .collect_vec(),
            ),
            // bytes[][] memory _callData,
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
            // uint256[][] memory _decimals,
            DynSeqToken(
                decimals
                    .iter()
                    .map(|ints| {
                        DynSeqToken(ints.iter().map(|i| WordToken(B256::from(*i))).collect_vec())
                    })
                    .collect::<Vec<_>>(),
            ),
            // uint[] memory _scales,
            DynSeqToken(
                scales
                    .into_iter()
                    .map(|i| WordToken(U256::from(i).into()))
                    .collect_vec(),
            ),
            //  uint8 _instanceOffset,
            WordToken(U256::from(contract_instance_offset as u32).into()),
            //address _admin
            WordToken(client_address.into_word()),
        )),
    )?;

    debug!("call_data: {:#?}", call_data);
    debug!("contract_addresses: {:#?}", contract_addresses);
    debug!("decimals: {:#?}", decimals);

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

    let (_anvil, client, client_address) = setup_eth_backend(rpc_url, None).await?;

    println!("client_address: {:?}", client_address);

    let contract = DataAttestation::new(addr, client.clone());

    info!("contract_addresses: {:#?}", contract_addresses);

    let _ = contract
        .updateAccountCalls(
            contract_addresses.clone(),
            call_data.clone(),
            decimals.clone(),
        )
        .from(client_address)
        .send()
        .await?;

    // update contract signer with non admin account
    let contract = DataAttestation::new(addr, client.clone());

    info!("contract_addresses: {:#?}", contract_addresses);

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
    debug!("result: {:#?}", result.to_vec());
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

    debug!("encoded: {:#?}", hex::encode(&encoded_verifier));

    debug!("public_inputs: {:#?}", public_inputs);
    debug!("proof: {:#?}", Bytes::from(proof.proof.to_vec()));

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

    debug!("encoded: {:#?}", hex::encode(&encoded));

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
    debug!("result: {:#?}", result);
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

    debug!("scales: {:#?}", scales);
    debug!("decimals: {:#?}", decimals);
    debug!("fetched_inputs: {:#?}", fetched_inputs);

    let results = contract
        .quantize_data(fetched_inputs, decimals, scales)
        .call()
        .await?
        .quantized_data;

    debug!("evm quantization results: {:#?}", results);

    let results_i64 = results
        .iter()
        .map(|x| i64::from_str(&x.to_string()).unwrap())
        .collect::<Vec<i64>>();

    let felts = contract.to_field_element(results_i64).call().await?.output;
    debug!("evm quantization contract results: {:#?}", felts,);

    let results = felts
        .iter()
        .map(|x| PrimeField::from_str_vartime(&x.to_string()).unwrap())
        .collect::<Vec<Fr>>();
    debug!("evm quantized felts: {:#?}", results,);
    Ok(results.to_vec())
}

/// Generates the contract factory for a solidity verifier. The factory is used to deploy the contract
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
        return Err(format!("file not found: {:#?}", sol_code_path).into());
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
