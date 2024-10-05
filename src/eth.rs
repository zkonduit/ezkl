use crate::graph::input::{CallsToAccount, FileSourceInner, GraphData};
use crate::graph::modules::POSEIDON_INSTANCES;
use crate::graph::DataSource;
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
use crate::graph::GraphSettings;
use crate::pfsys::evm::EvmVerificationError;
use crate::pfsys::Snark;
use alloy::contract::CallBuilder;
use alloy::core::primitives::Address as H160;
use alloy::core::primitives::Bytes;
use alloy::core::primitives::U256;
use alloy::dyn_abi::abi::token::{DynSeqToken, PackedSeqToken, WordToken};
use alloy::dyn_abi::abi::TokenSeq;
#[cfg(any(target_os = "ios", target_arch = "wasm32"))]
use alloy::prelude::Wallet;
// use alloy::providers::Middleware;
use alloy::json_abi::JsonAbi;
use alloy::node_bindings::Anvil;
use alloy::primitives::ruint::ParseError;
use alloy::primitives::{ParseSignedError, B256, I256};
use alloy::providers::fillers::{
    ChainIdFiller, FillProvider, GasFiller, JoinFill, NonceFiller, SignerFiller,
};
use alloy::providers::network::{Ethereum, EthereumSigner};
use alloy::providers::ProviderBuilder;
use alloy::providers::{Identity, Provider, RootProvider};
use alloy::rpc::types::eth::TransactionInput;
use alloy::rpc::types::eth::TransactionRequest;
use alloy::signers::k256::ecdsa;
use alloy::signers::wallet::{LocalWallet, WalletError};
use alloy::sol as abigen;
use alloy::transports::http::Http;
use alloy::transports::{RpcError, TransportErrorKind};
use foundry_compilers::artifacts::Settings as SolcSettings;
use foundry_compilers::error::{SolcError, SolcIoError};
use foundry_compilers::Solc;
use halo2_solidity_verifier::encode_calldata;
use halo2curves::bn256::{Fr, G1Affine};
use halo2curves::group::ff::PrimeField;
use itertools::Itertools;
use log::{debug, info, warn};
use reqwest::Client;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

const ANVIL_DEFAULT_PRIVATE_KEY: &str =
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

pub const DEFAULT_ANVIL_ENDPOINT: &str = "http://localhost:8545";

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
    #[sol(rpc, bytecode="608060405234801561000f575f80fd5b50610a8b8061001d5f395ff3fe608060405234801561000f575f80fd5b506004361061003f575f3560e01c80630a7e4b9614610043578063b404abab1461006c578063d3dc6d1f1461007f575b5f80fd5b6100566100513660046105b6565b610092565b60405161006391906106f6565b60405180910390f35b61005661007a366004610739565b610203565b61005661008d3660046107c4565b61033c565b606083516001600160401b038111156100ad576100ad6104e8565b6040519080825280602002602001820160405280156100d6578160200160208202803683370190505b5090505f5b84518110156101fb575f8582815181106100f7576100f7610853565b60200260200101518060200190518101906101129190610867565b90505f811280156101295761012682610892565b91505b5f86848151811061013c5761013c610853565b6020026020010151600a610150919061098e565b90505f86858151811061016557610165610853565b60200260200101516001901b90505f61017f8583856103fd565b905082838061019057610190610999565b83870961019e9060026109ad565b106101b1576101ae6001826109c4565b90505b836101bc57806101c5565b6101c581610892565b8787815181106101d7576101d7610853565b602002602001018181525050505050505080806101f3906109d7565b9150506100db565b509392505050565b606081516001600160401b0381111561021e5761021e6104e8565b604051908082528060200260200182016040528015610247578160200160208202803683370190505b5090505f5b8251811015610336575f83828151811061026857610268610853565b6020026020010151121580156102a457505f80516020610a3683398151915283828151811061029957610299610853565b602002602001015111155b6102ed5760405162461bcd60e51b8152602060048201526015602482015274125b9d985b1a5908199a595b1908195b195b595b9d605a1b60448201526064015b60405180910390fd5b8281815181106102ff576102ff610853565b602002602001015182828151811061031957610319610853565b60209081029190910101528061032e816109d7565b91505061024c565b50919050565b606081516001600160401b03811115610357576103576104e8565b604051908082528060200260200182016040528015610380578160200160208202803683370190505b5090505f5b8251811015610336575f80516020610a36833981519152808483815181106103af576103af610853565b602002602001015160070b6103c491906109ef565b6103ce9190610a16565b8282815181106103e0576103e0610853565b6020908102919091010152806103f5816109d7565b915050610385565b5f80805f19858709858702925082811083820303915050805f036104345783828161042a5761042a610999565b04925050506104e1565b80841161047b5760405162461bcd60e51b81526020600482015260156024820152744d6174683a206d756c446976206f766572666c6f7760581b60448201526064016102e4565b5f848688098519600190810187169687900496828603819004959092119093035f82900391909104909201919091029190911760038402600290811880860282030280860282030280860282030280860282030280860282030280860290910302029150505b9392505050565b634e487b7160e01b5f52604160045260245ffd5b604051601f8201601f191681016001600160401b0381118282101715610524576105246104e8565b604052919050565b5f6001600160401b03821115610544576105446104e8565b5060051b60200190565b5f82601f83011261055d575f80fd5b8135602061057261056d8361052c565b6104fc565b82815260059290921b84018101918181019086841115610590575f80fd5b8286015b848110156105ab5780358352918301918301610594565b509695505050505050565b5f805f606084860312156105c8575f80fd5b83356001600160401b03808211156105de575f80fd5b818601915086601f8301126105f1575f80fd5b8135602061060161056d8361052c565b82815260059290921b8401810191818101908a84111561061f575f80fd5b8286015b848110156106a85780358681111561063a575f8081fd5b8701603f81018d1361064b575f8081fd5b84810135604088821115610661576106616104e8565b610673601f8301601f191688016104fc565b8281528f82848601011115610687575f8081fd5b82828501898301375f92810188019290925250845250918301918301610623565b50975050870135925050808211156106be575f80fd5b6106ca8783880161054e565b935060408601359150808211156106df575f80fd5b506106ec8682870161054e565b9150509250925092565b602080825282518282018190525f9190848201906040850190845b8181101561072d57835183529284019291840191600101610711565b50909695505050505050565b5f602080838503121561074a575f80fd5b82356001600160401b0381111561075f575f80fd5b8301601f8101851361076f575f80fd5b803561077d61056d8261052c565b81815260059190911b8201830190838101908783111561079b575f80fd5b928401925b828410156107b9578335825292840192908401906107a0565b979650505050505050565b5f60208083850312156107d5575f80fd5b82356001600160401b038111156107ea575f80fd5b8301601f810185136107fa575f80fd5b803561080861056d8261052c565b81815260059190911b82018301908381019087831115610826575f80fd5b928401925b828410156107b95783358060070b8114610844575f8081fd5b8252928401929084019061082b565b634e487b7160e01b5f52603260045260245ffd5b5f60208284031215610877575f80fd5b5051919050565b634e487b7160e01b5f52601160045260245ffd5b5f600160ff1b82016108a6576108a661087e565b505f0390565b600181815b808511156108e657815f19048211156108cc576108cc61087e565b808516156108d957918102915b93841c93908002906108b1565b509250929050565b5f826108fc57506001610988565b8161090857505f610988565b816001811461091e576002811461092857610944565b6001915050610988565b60ff8411156109395761093961087e565b50506001821b610988565b5060208310610133831016604e8410600b8410161715610967575081810a610988565b61097183836108ac565b805f19048211156109845761098461087e565b0290505b92915050565b5f6104e183836108ee565b634e487b7160e01b5f52601260045260245ffd5b80820281158282048414176109885761098861087e565b808201808211156109885761098861087e565b5f600182016109e8576109e861087e565b5060010190565b8082018281125f831280158216821582161715610a0e57610a0e61087e565b505092915050565b5f82610a3057634e487b7160e01b5f52601260045260245ffd5b50069056fe30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001a26469706673582212200034995b2b5991300d54d46b8b569fdaad34c590716304d33ba67eac46c8a61764736f6c63430008140033")]
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

        function check_is_valid_field_element(
            int256[] memory quantized_data
        ) public pure returns (uint256[] memory output) {
            output = new uint256[](quantized_data.length);
            for (uint i; i < quantized_data.length; i++) {
                // assert it is a positive number and less than ORDER
                require(quantized_data[i] >= 0 && uint256(quantized_data[i]) <= ORDER, "Invalid field element");
                output[i] = uint256(quantized_data[i]);
            }
        }
    }
);

#[derive(Debug, thiserror::Error)]
pub enum EthError {
    #[error("a transport error occurred: {0}")]
    Transport(#[from] RpcError<TransportErrorKind>),
    #[error("a contract error occurred: {0}")]
    Contract(#[from] alloy::contract::Error),
    #[error("a wallet error occurred: {0}")]
    Wallet(#[from] WalletError),
    #[error("failed to parse url {0}")]
    UrlParse(String),
    #[error("evm verification error: {0}")]
    EvmVerification(#[from] EvmVerificationError),
    #[error("Private key must be in hex format, 64 chars, without 0x prefix")]
    PrivateKeyFormat,
    #[error("failed to parse hex: {0}")]
    HexParse(#[from] hex::FromHexError),
    #[error("ecdsa error: {0}")]
    Ecdsa(#[from] ecdsa::Error),
    #[error("failed to load graph data")]
    GraphData,
    #[error("failed to load graph settings")]
    GraphSettings,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Data source for either input_data or output_data must be OnChain")]
    OnChainDataSource,
    #[error("failed to parse signed integer: {0}")]
    SignedIntegerParse(#[from] ParseSignedError),
    #[error("failed to parse unsigned integer: {0}")]
    UnSignedIntegerParse(#[from] ParseError),
    #[error("updateAccountCalls should have failed")]
    UpdateAccountCalls,
    #[error("ethabi error: {0}")]
    EthAbi(#[from] ethabi::Error),
    #[error("conversion error: {0}")]
    Conversion(#[from] std::convert::Infallible),
    // Constructor arguments provided but no constructor found
    #[error("constructor arguments provided but no constructor found")]
    NoConstructor,
    #[error("contract not found at path: {0}")]
    ContractNotFound(String),
    #[error("solc error: {0}")]
    Solc(#[from] SolcError),
    #[error("solc io error: {0}")]
    SolcIo(#[from] SolcIoError),
    #[error("svm error: {0}")]
    Svm(String),
    #[error("no contract output found")]
    NoContractOutput,
}

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
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub async fn setup_eth_backend(
    rpc_url: Option<&str>,
    private_key: Option<&str>,
) -> Result<(EthersClient, alloy::primitives::Address), EthError> {
    // Launch anvil

    let endpoint: String;
    if let Some(rpc_url) = rpc_url {
        endpoint = rpc_url.to_string();
    } else {
        let anvil = Anvil::new()
            .args([
                "--code-size-limit=41943040",
                "--disable-block-gas-limit",
                "-p",
                "8545",
            ])
            .spawn();
        endpoint = anvil.endpoint();
    }

    // Instantiate the wallet
    let wallet: LocalWallet;
    if let Some(private_key) = private_key {
        debug!("using private key {}", private_key);
        if private_key.len() != 64 {
            return Err(EthError::PrivateKeyFormat);
        }
        let private_key_buffer = hex::decode(private_key)?;
        wallet = LocalWallet::from_slice(&private_key_buffer)?;
    } else {
        wallet = LocalWallet::from_str(ANVIL_DEFAULT_PRIVATE_KEY)?;
    }

    let wallet_address = wallet.address();

    // Connect to the network
    let client = Arc::new(
        ProviderBuilder::new()
            .with_recommended_fillers()
            .signer(EthereumSigner::from(wallet))
            .on_http(endpoint.parse().map_err(|_| EthError::UrlParse(endpoint))?),
    );

    let chain_id = client.get_chain_id().await?;
    info!("using chain {}", chain_id);

    Ok((client, wallet_address))
}

///
pub async fn deploy_contract_via_solidity(
    sol_code_path: PathBuf,
    rpc_url: Option<&str>,
    runs: usize,
    private_key: Option<&str>,
    contract_name: &str,
) -> Result<H160, EthError> {
    // anvil instance must be alive at least until the factory completes the deploy
    let (client, _) = setup_eth_backend(rpc_url, private_key).await?;

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, contract_name, runs).await?;

    let factory = get_sol_contract_factory(abi, bytecode, runtime_bytecode, client, None::<()>)?;
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
) -> Result<H160, EthError> {
    let (client, client_address) = setup_eth_backend(rpc_url, private_key).await?;

    let input = GraphData::from_path(input).map_err(|_| EthError::GraphData)?;

    let settings = GraphSettings::load(&settings_path).map_err(|_| EthError::GraphSettings)?;

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
        return Err(EvmVerificationError::InvalidVisibility.into());
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

    let (abi, bytecode, runtime_bytecode) =
        get_contract_artifacts(sol_code_path, "DataAttestation", runs).await?;

    let (contract_addresses, call_data, decimals) = if !calls_to_accounts.is_empty() {
        parse_calls_to_accounts(calls_to_accounts)?
    } else {
        // if calls to accounts is empty then we know need to check that atleast there kzg visibility in the settings file
        let kzg_visibility = settings.run_args.input_visibility.is_polycommit()
            || settings.run_args.output_visibility.is_polycommit()
            || settings.run_args.param_visibility.is_polycommit();
        if !kzg_visibility {
            return Err(EthError::OnChainDataSource);
        }
        let factory =
            get_sol_contract_factory::<_, ()>(abi, bytecode, runtime_bytecode, client, None)?;
        let contract = factory.deploy().await?;
        return Ok(contract);
    };

    let factory = get_sol_contract_factory(
        abi,
        bytecode,
        runtime_bytecode,
        client,
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
            // address _admin
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
) -> Result<ParsedCallsToAccount, EthError> {
    let mut contract_addresses = vec![];
    let mut call_data = vec![];
    let mut decimals: Vec<Vec<U256>> = vec![];
    for (i, val) in calls_to_accounts.iter().enumerate() {
        let contract_address_bytes = hex::decode(&val.address)?;
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
) -> Result<(), EthError> {
    let input = GraphData::from_path(input).map_err(|_| EthError::GraphData)?;

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
        return Err(EthError::OnChainDataSource);
    };

    let (client, client_address) = setup_eth_backend(rpc_url, None).await?;

    let contract = DataAttestation::new(addr, &client);

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
        return Err(EthError::UpdateAccountCalls);
    }

    Ok(())
}

/// Verify a proof using a Solidity verifier contract
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: H160,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, EthError> {
    let flattened_instances = proof.instances.into_iter().flatten();

    let encoded = encode_calldata(
        addr_vk.as_ref().map(|x| x.0).map(|x| x.0),
        &proof.proof,
        &flattened_instances.collect::<Vec<_>>(),
    );

    debug!("encoded: {:#?}", hex::encode(&encoded));

    let input: TransactionInput = encoded.into();

    let (client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr).input(input);
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx).await;

    if let Err(e) = result {
        return Err(EvmVerificationError::SolidityExecution(e.to_string()).into());
    }
    let result = result?;
    debug!("result: {:#?}", result.to_vec());
    // decode return bytes value into uint8
    let result = result.to_vec().last().ok_or(EthError::NoContractOutput)? == &1u8;
    if !result {
        return Err(EvmVerificationError::InvalidProof.into());
    }

    let gas = client.estimate_gas(&tx).await?;

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
) -> Result<(TestReads::TestReadsInstance<Http<Client>, Arc<M>>, Vec<u8>), EthError> {
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
            // remove the 0x prefix
            let hex_str_fr = &hex_str_fr[2..];
            scaled_by_decimals_data.push(I256::from_raw(U256::from_str_radix(hex_str_fr, 16)?));
            decimals.push(0);
        }
    }

    // Compile the contract
    let contract = TestReads::deploy(client, scaled_by_decimals_data).await?;

    Ok((contract, decimals))
}

/// Verify a proof using a Solidity DataAttestation contract.
/// Used for testing purposes.
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub async fn verify_proof_with_data_attestation(
    proof: Snark<Fr, G1Affine>,
    addr_verifier: H160,
    addr_da: H160,
    addr_vk: Option<H160>,
    rpc_url: Option<&str>,
) -> Result<bool, EthError> {
    use ethabi::{Function, Param, ParamType, StateMutability, Token};

    let mut public_inputs: Vec<U256> = vec![];
    let flattened_instances = proof.instances.into_iter().flatten();

    for val in flattened_instances.clone() {
        let bytes = val.to_repr();
        let u = U256::from_le_slice(bytes.inner().as_slice());
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

    let (client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr_da).input(encoded);
    debug!("transaction {:#?}", tx);
    info!(
        "estimated verify gas cost: {:#?}",
        client.estimate_gas(&tx).await?
    );

    let result = client.call(&tx).await;
    if let Err(e) = result {
        return Err(EvmVerificationError::SolidityExecution(e.to_string()).into());
    }
    let result = result?;
    debug!("result: {:#?}", result);
    // decode return bytes value into uint8
    let result = result.to_vec().last().ok_or(EthError::NoContractOutput)? == &1u8;
    if !result {
        return Err(EvmVerificationError::InvalidProof.into());
    }

    Ok(true)
}

/// Tests on-chain data storage by deploying a contract that stores the network input and or output
/// data in its storage. It does this by converting the floating point values to integers and storing the
/// the number of decimals of the floating point value on chain.
pub async fn test_on_chain_data<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    data: &[Vec<FileSourceInner>],
) -> Result<Vec<CallsToAccount>, EthError> {
    let (contract, decimals) = setup_test_contract(client, data).await?;

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
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub async fn read_on_chain_inputs<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    address: H160,
    data: &Vec<CallsToAccount>,
) -> Result<(Vec<Bytes>, Vec<u8>), EthError> {
    // Iterate over all on-chain inputs

    let mut fetched_inputs = vec![];
    let mut decimals = vec![];
    for on_chain_data in data {
        // Construct the address
        let contract_address_bytes = hex::decode(&on_chain_data.address)?;
        let contract_address = H160::from_slice(&contract_address_bytes);
        for (call_data, decimal) in &on_chain_data.call_data {
            let call_data_bytes = hex::decode(call_data)?;
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
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub async fn evm_quantize<M: 'static + Provider<Http<Client>, Ethereum>>(
    client: Arc<M>,
    scales: Vec<crate::Scale>,
    data: &(Vec<Bytes>, Vec<u8>),
) -> Result<Vec<Fr>, EthError> {
    let contract = QuantizeData::deploy(&client).await?;

    let fetched_inputs = &data.0;
    let decimals = &data.1;

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

    let mut felts = vec![];

    for x in results {
        let felt = match i64::from_str(&x.to_string()) {
            Ok(x) => contract.to_field_element(vec![x]).call().await?.output[0],
            Err(_) => {
                contract
                    .check_is_valid_field_element(vec![x])
                    .call()
                    .await?
                    .output[0]
            }
        };
        felts.push(PrimeField::from_str_vartime(&felt.to_string()).unwrap());
    }

    debug!("evm quantized felts: {:#?}", felts,);
    Ok(felts)
}

/// Generates the contract factory for a solidity verifier. The factory is used to deploy the contract
fn get_sol_contract_factory<'a, M: 'static + Provider<Http<Client>, Ethereum>, T: TokenSeq<'a>>(
    abi: JsonAbi,
    bytecode: Bytes,
    runtime_bytecode: Bytes,
    client: Arc<M>,
    params: Option<T>,
) -> Result<ContractFactory<M>, EthError> {
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
            return Err(EthError::NoConstructor);
        }
        (None, true) => bytecode,
        (Some(_), _) => {
            let mut data = bytecode.to_vec();

            if let Some(params) = params {
                let params = alloy::sol_types::abi::encode_sequence(&params);
                data.extend(params);
            }
            data.into()
        }
    };

    Ok(CallBuilder::new_raw_deploy(client, data))
}

/// Compiles a solidity verifier contract and returns the abi, bytecode, and runtime bytecode
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
pub async fn get_contract_artifacts(
    sol_code_path: PathBuf,
    contract_name: &str,
    runs: usize,
) -> Result<(JsonAbi, Bytes, Bytes), EthError> {
    use foundry_compilers::{
        artifacts::{output_selection::OutputSelection, Optimizer},
        compilers::CompilerInput,
        SolcInput, SHANGHAI_SOLC,
    };

    if !sol_code_path.exists() {
        return Err(EthError::ContractNotFound(
            sol_code_path.to_string_lossy().to_string(),
        ));
    }

    let settings = SolcSettings {
        optimizer: Optimizer {
            enabled: Some(true),
            runs: Some(runs),
            details: None,
        },
        output_selection: OutputSelection::default_output_selection(),
        ..Default::default()
    };

    let input = SolcInput::build(
        std::collections::BTreeMap::from([(
            sol_code_path.clone(),
            foundry_compilers::artifacts::Source::read(sol_code_path)?,
        )]),
        settings,
        &SHANGHAI_SOLC,
    );

    let solc_opt = Solc::find_svm_installed_version(SHANGHAI_SOLC.to_string())?;
    let solc = match solc_opt {
        Some(solc) => solc,
        None => {
            info!("required solc version is missing ... installing");
            Solc::install(&SHANGHAI_SOLC)
                .await
                .map_err(|e| EthError::Svm(e.to_string()))?
        }
    };

    let compiled: foundry_compilers::CompilerOutput = solc.compile(&input[0])?;

    let (abi, bytecode, runtime_bytecode) = match compiled.find(contract_name) {
        Some(c) => c.into_parts_or_default(),
        None => {
            return Err(EthError::ContractNotFound(contract_name.to_string()));
        }
    };

    Ok((abi, bytecode, runtime_bytecode))
}

/// Sets the constants stored in the da verifier
pub fn fix_da_sol(
    input_data: Option<Vec<CallsToAccount>>,
    output_data: Option<Vec<CallsToAccount>>,
    commitment_bytes: Option<Vec<u8>>,
) -> Result<String, EthError> {
    let mut accounts_len = 0;
    let mut contract = ATTESTDATA_SOL.to_string();

    // fill in the quantization params and total calls
    // as constants to the contract to save on gas
    if let Some(input_data) = &input_data {
        let input_calls: usize = input_data.iter().map(|v| v.call_data.len()).sum();
        accounts_len = input_data.len();
        contract = contract.replace(
            "uint256 constant INPUT_CALLS = 0;",
            &format!("uint256 constant INPUT_CALLS = {};", input_calls),
        );
    }
    if let Some(output_data) = &output_data {
        let output_calls: usize = output_data.iter().map(|v| v.call_data.len()).sum();
        accounts_len += output_data.len();
        contract = contract.replace(
            "uint256 constant OUTPUT_CALLS = 0;",
            &format!("uint256 constant OUTPUT_CALLS = {};", output_calls),
        );
    }
    contract = contract.replace("AccountCall[]", &format!("AccountCall[{}]", accounts_len));

    // The case where a combination of on-chain data source + kzg commit is provided.
    if commitment_bytes.is_some() && !commitment_bytes.as_ref().unwrap().is_empty() {
        let commitment_bytes = commitment_bytes.as_ref().unwrap();
        let hex_string = hex::encode(commitment_bytes);
        contract = contract.replace(
            "bytes constant COMMITMENT_KZG = hex\"\";",
            &format!("bytes constant COMMITMENT_KZG = hex\"{}\";", hex_string),
        );
    } else {
        // Remove the SwapProofCommitments inheritance and the checkKzgCommits function call if no commitment is provided
        contract = contract.replace(", SwapProofCommitments", "");
        contract = contract.replace(
            "require(checkKzgCommits(encoded), \"Invalid KZG commitments\");",
            "",
        );
    }

    // if both input and output data is none then we will only deploy the DataAttest contract, adding in the verifyWithDataAttestation function
    if input_data.is_none()
        && output_data.is_none()
        && commitment_bytes.as_ref().is_some()
        && !commitment_bytes.as_ref().unwrap().is_empty()
    {
        contract = contract.replace(
            "contract SwapProofCommitments {",
            "contract DataAttestation {",
        );

        // Remove everything past the end of the checkKzgCommits function
        if let Some(pos) = contract.find("    } /// end checkKzgCommits") {
            contract.truncate(pos);
            contract.push('}');
        }

        // Add the Solidity function below checkKzgCommits
        contract.push_str(
            r#"
    function verifyWithDataAttestation(
        address verifier,
        bytes calldata encoded
    ) public view returns (bool) {
        require(verifier.code.length > 0, "Address: call to non-contract");
        require(checkKzgCommits(encoded), "Invalid KZG commitments");
        // static call the verifier contract to verify the proof
        (bool success, bytes memory returndata) = verifier.staticcall(encoded);

        if (success) {
            return abi.decode(returndata, (bool));
        } else {
            revert("low-level call to verifier failed");
        }
    }
}"#,
        );
    }

    Ok(contract)
}
