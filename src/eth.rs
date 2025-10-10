use crate::pfsys::{encode_calldata, Snark};
use alloy::contract::CallBuilder;
use alloy::core::primitives::Address as H160;
use alloy::core::primitives::Bytes;
use alloy::core::primitives::I256;
use alloy::dyn_abi::abi::TokenSeq;
// use alloy::providers::Middleware;
use alloy::json_abi::JsonAbi;
use alloy::primitives::ruint::ParseError;
use alloy::primitives::ParseSignedError;
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
use alloy::transports::http::Http;
use alloy::transports::{RpcError, TransportErrorKind};
use foundry_compilers::artifacts::Settings as SolcSettings;
use foundry_compilers::error::{SolcError, SolcIoError};
use foundry_compilers::Solc;
use halo2_solidity_verifier::encode_register_vk_calldata;
use halo2curves::bn256::{Fr, G1Affine};
use log::{debug, info, warn};
use reqwest::Client;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

const ANVIL_DEFAULT_PRIVATE_KEY: &str =
    "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
///
pub const DEFAULT_ANVIL_ENDPOINT: &str = "http://localhost:8545";

#[derive(Debug, thiserror::Error)]
pub enum RescaleCheckError {
    #[error("rescaled instance #{idx} mismatch: expected {expected}, got {got}")]
    Mismatch {
        idx: usize,
        expected: String,
        got: String,
    },
}

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
    #[error("ethabi error: {0}")]
    EthAbi(#[from] ethabi::Error),
    #[error("conversion error: {0}")]
    Conversion(#[from] std::convert::Infallible),
    // Constructor arguments provided but no constructor found
    #[error("constructor arguments provided but no constructor found")]
    NoConstructor,
    #[error("contract not found at path: {0}")]
    ContractNotFound(String),
    #[error("encoded calldata not found at path: {0}")]
    EncodedCalldataNotFound(String),
    #[error("solc error: {0}")]
    Solc(#[from] SolcError),
    #[error("solc io error: {0}")]
    SolcIo(#[from] SolcIoError),
    #[error("svm error: {0}")]
    Svm(String),
    #[error("no contract output found")]
    NoContractOutput,
    #[error("failed to load vka data: {0}")]
    VkaData(String),
    #[error("rescaled‑instance mismatch: {0}")]
    RescaleCheckError(#[from] RescaleCheckError),
    #[error("evm verification error: {0}")]
    EvmVerificationError(String),
}

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
pub async fn setup_eth_backend(
    rpc_url: &str,
    private_key: Option<&str>,
) -> Result<(EthersClient, alloy::primitives::Address), EthError> {
    // Launch anvil

    let endpoint = rpc_url.to_string();

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
    rpc_url: &str,
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
pub async fn register_vka_via_rv(
    rpc_url: &str,
    private_key: Option<&str>,
    rv_address: H160,
    vka_words: &[[u8; 32]],
) -> Result<Vec<u8>, EthError> {
    let (client, _) = setup_eth_backend(rpc_url, private_key).await?;

    let encoded = encode_register_vk_calldata(vka_words);

    debug!(
        "encoded register vka calldata: {:#?}",
        hex::encode(&encoded)
    );

    let input: TransactionInput = encoded.into();

    let tx = TransactionRequest::default().to(rv_address).input(input);
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx).await;

    if let Err(e) = result {
        return Err(EthError::EvmVerificationError(e.to_string()).into());
    }
    let result = result?;
    debug!("result: {:#?}", result.to_vec());
    // decode return bytes value into uint8
    let output = result.to_vec();

    let gas = client.estimate_gas(&tx).await?;

    info!("estimated vka registration cost: {:#?}", gas);

    // broadcast the transaction

    let result = client.send_transaction(tx).await?;

    result.watch().await?;

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

    Ok(output)
}

/// Verify a proof using a Solidity verifier contract
/// TODO: add param to pass vka_digest and use that to fetch the VKA by indexing the RegisteredVKA events on the RV
pub async fn verify_proof_via_solidity(
    proof: Snark<Fr, G1Affine>,
    addr: H160,
    vka_path: Option<PathBuf>,
    rpc_url: &str,
    encoded_calldata: Option<PathBuf>,
) -> Result<bool, EthError> {
    // Load the vka, which is bincode serialized, from the vka_path
    let vka_buf: Option<Vec<[u8; 32]>> = match vka_path {
        Some(path) => {
            let bytes = std::fs::read(path)?;
            Some(bincode::deserialize(&bytes).map_err(|e| EthError::VkaData(e.to_string()))?)
        }
        None => None,
    };

    let vka: Option<&[[u8; 32]]> = vka_buf.as_deref();

    let encoded = if encoded_calldata.is_none() {
        let flattened_instances = proof.instances.into_iter().flatten();

        encode_calldata(vka, &proof.proof, &flattened_instances.collect::<Vec<_>>())
    } else {
        // Load the bincode serialized calldata from the file path
        let path = encoded_calldata.unwrap();
        std::fs::read(&path).map_err(|e| EthError::EncodedCalldataNotFound(e.to_string()))?
    };

    debug!("encoded: {:#?}", hex::encode(&encoded));

    let input: TransactionInput = encoded.into();

    let (client, _) = setup_eth_backend(rpc_url, None).await?;
    let tx = TransactionRequest::default().to(addr).input(input);
    debug!("transaction {:#?}", tx);

    let result = client.call(&tx).await;

    if let Err(e) = result {
        return Err(EthError::EvmVerificationError(e.to_string()).into());
    }
    let result = result?;
    debug!("result: {:#?}", result.to_vec());
    // if result is larger than 32
    if result.to_vec().len() > 32 {
        // From result[96..], iterate through 32 byte chunks converting them to U256
        let rescaled_instances = result.to_vec()[96..]
            .chunks_exact(32)
            .map(|chunk| I256::try_from_be_slice(chunk).unwrap().to_string())
            .collect::<Vec<_>>();
        if let Some(pretty) = &proof.pretty_public_inputs {
            // 1️⃣ collect reference decimals --------------------------------------
            let mut refs = pretty.rescaled_inputs.clone();
            refs.extend(pretty.rescaled_outputs.clone()); // extend inputs with outputs
            let reference: Vec<String> = refs.into_iter().flatten().collect();

            // 2️⃣ compare element‑wise -------------------------------------------
            for (idx, (inst, exp)) in rescaled_instances.iter().zip(reference.iter()).enumerate() {
                if !scaled_matches(inst, exp) {
                    return Err(EthError::RescaleCheckError(RescaleCheckError::Mismatch {
                        idx,
                        expected: exp.clone(),
                        got: to_decimal_18(inst),
                    }));
                }
            }
            debug!("✅ all rescaled instances match their expected values");
        }
    }
    // decode return bytes value into uint8
    let result = result.to_vec()[..32]
        .last()
        .ok_or(EthError::NoContractOutput)?
        == &1u8;
    if !result {
        return Err(EthError::EvmVerificationError("Invalid proof".into()));
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

/// Convert a 1e‑18 fixed‑point **integer string** into a decimal string.
///
/// `"1541748046875000000"` → `"1.541748046875000000"`
/// `"273690402507781982"`  → `"0.273690402507781982"`
/// `"-892333984375000000"` → `"-0.892333984375000000"`
fn to_decimal_18(s: &str) -> String {
    let is_negative = s.starts_with('-');
    let s = if is_negative { &s[1..] } else { s };
    let s = s.trim_start_matches('0');

    if s.is_empty() {
        return "0".into();
    }

    if s.len() <= 18 {
        // pad on the left so we always have exactly 18 fraction digits
        let result = format!("0.{:0>18}", s);
        return if is_negative {
            format!("-{}", result)
        } else {
            result
        };
    }

    let split = s.len() - 18;
    let result = format!("{}.{}", &s[..split], &s[split..]);
    if is_negative {
        format!("-{}", result)
    } else {
        result
    }
}
/// "Banker's‐round" comparison:  compare the **decimal** produced
/// by `instance` to the reference string `expected`.
///
/// *  Only the first 18 digits of the expected fraction part are compared.
/// *  All digits present in the truncated `expected` (integer part **and** first 18 fraction digits)
///    must match exactly.
/// *  Excess digits in `instance` are ignored **unless** the very first
///    excess digit ≥ 5; in that case we round the last compared digit
///    and check again.
fn scaled_matches(instance: &str, expected: &str) -> bool {
    let inst_dec = to_decimal_18(instance);
    let (inst_int, inst_frac) = inst_dec.split_once('.').unwrap_or((&inst_dec, ""));
    let (exp_int, exp_frac) = expected.split_once('.').unwrap_or((expected, ""));

    // Normalize both integer parts to handle "-" vs "-0"
    let normalized_inst_int = if inst_int == "-" { "-0" } else { inst_int };
    let normalized_exp_int = if exp_int == "-" { "-0" } else { exp_int };

    // integer part must be identical
    if normalized_inst_int != normalized_exp_int {
        return false;
    }

    // If expected has more than 18 decimal places, round it to 18 places
    let exp_frac_truncated = if exp_frac.len() > 18 {
        let truncated = &exp_frac[..18];
        let next_digit = exp_frac.chars().nth(18).unwrap_or('0');

        if next_digit >= '6' {
            // Need to round up the 18th digit
            let mut rounded = truncated.chars().collect::<Vec<_>>();
            let mut carry = true;
            for d in rounded.iter_mut().rev() {
                if !carry {
                    break;
                }
                let v = d.to_digit(10).unwrap() + 1;
                *d = char::from_digit(v % 10, 10).unwrap();
                carry = v == 10;
            }
            if carry {
                // All 18 digits were 9s - this would carry to integer part
                // For now, return the original truncated (this edge case may need special handling)
                truncated.to_string()
            } else {
                rounded.into_iter().collect::<String>()
            }
        } else {
            truncated.to_string()
        }
    } else {
        exp_frac.to_string()
    };
    let exp_frac_truncated = exp_frac_truncated.as_str();

    // fraction‑part comparison with optional rounding
    let cmp_len = exp_frac_truncated.len();
    let inst_cmp = &inst_frac[..cmp_len.min(inst_frac.len())];
    let trailing = inst_frac.chars().nth(cmp_len).unwrap_or('0');

    if inst_cmp == exp_frac_truncated {
        true // exact match
    } else if trailing >= '5' {
        // need to round
        // round the inst_cmp (string) up by one ulp
        let mut rounded = inst_cmp.chars().collect::<Vec<_>>();
        let mut carry = true;
        for d in rounded.iter_mut().rev() {
            if !carry {
                break;
            }
            let v = d.to_digit(10).unwrap() + 1;
            *d = char::from_digit(v % 10, 10).unwrap();
            carry = v == 10;
        }
        if carry {
            // 0.999… → 1.000…
            // Handle negative numbers in the carry case
            let is_negative = normalized_inst_int.starts_with('-');
            let abs_int = normalized_inst_int.trim_start_matches('-');
            let incremented =
                (num::BigUint::parse_bytes(abs_int.as_bytes(), 10).unwrap() + 1u32).to_string();
            let expected_after_carry = if is_negative {
                format!("-{}", incremented)
            } else {
                incremented
            };
            return normalized_exp_int == expected_after_carry
                && exp_frac_truncated.chars().all(|c| c == '0');
        }
        rounded.into_iter().collect::<String>() == exp_frac_truncated
    } else {
        false
    }
}
