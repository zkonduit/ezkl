use crate::circuit::region::RegionSettings;
use crate::circuit::CheckMode;
use crate::commands::CalibrationTarget;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::deploy_contract_via_solidity;
#[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
use crate::eth::register_vka_via_rv;
#[allow(unused_imports)]
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::{get_contract_artifacts, verify_proof_via_solidity};
use crate::graph::input::GraphData;
use crate::graph::{GraphCircuit, GraphSettings, GraphWitness, Model};
use crate::pfsys::evm::aggregation_kzg::{AggregationCircuit, PoseidonTranscript};
use crate::pfsys::{
    create_keys, load_pk, load_vk, save_params, save_pk, Snark, StrategyType, TranscriptType,
};
use crate::pfsys::{
    create_proof_circuit, swap_proof_commitments_polycommit, verify_proof_circuit, ProofSplitCommit,
};
use crate::pfsys::{encode_calldata, save_vk, srs::*};
use crate::tensor::TensorError;
use crate::EZKL_BUF_CAPACITY;
use crate::{commands::*, EZKLError};
use crate::{Commitments, RunArgs};
use colored::Colorize;
#[cfg(unix)]
use gag::Gag;
use halo2_proofs::dev::VerifyFailure;
use halo2_proofs::plonk::{self, Circuit};
use halo2_proofs::poly::commitment::{CommitmentScheme, Params};
use halo2_proofs::poly::commitment::{ParamsProver, Verifier};
use halo2_proofs::poly::ipa::commitment::{IPACommitmentScheme, ParamsIPA};
use halo2_proofs::poly::ipa::multiopen::{ProverIPA, VerifierIPA};
use halo2_proofs::poly::ipa::strategy::AccumulatorStrategy as IPAAccumulatorStrategy;
use halo2_proofs::poly::ipa::strategy::SingleStrategy as IPASingleStrategy;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::AccumulatorStrategy as KZGAccumulatorStrategy;
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer};
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use halo2_solidity_verifier;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
use indicatif::{ProgressBar, ProgressStyle};
use instant::Instant;
use itertools::Itertools;
use lazy_static::lazy_static;
use log::debug;
use log::{info, trace, warn};
use serde::de::DeserializeOwned;
use serde::Serialize;
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::compile;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use snark_verifier::system::halo2::Config;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use std::fs::File;
use std::io::BufWriter;
use std::io::Cursor;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use tabled::Tabled;
use thiserror::Error;
use tract_onnx::prelude::IntoTensor;
use tract_onnx::prelude::Tensor as TractTensor;

lazy_static! {
    #[derive(Debug)]
    /// The path to the ezkl related data.
    pub static ref EZKL_REPO_PATH: String =
        std::env::var("EZKL_REPO_PATH").unwrap_or_else(|_|
            // $HOME/.ezkl/
            format!("{}/.ezkl", std::env::var("HOME").unwrap())
        );

    /// The path to the ezkl related data (SRS)
    pub static ref EZKL_SRS_REPO_PATH: String = format!("{}/srs", *EZKL_REPO_PATH);

}

/// A wrapper for execution errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// verification failed
    #[error("verification failed:\n{}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n"))]
    VerifyError(Vec<VerifyFailure>),
    /// Prover error
    #[error("[mock] {0}")]
    MockProverError(String),
}

lazy_static::lazy_static! {
    // read from env EZKL_WORKING_DIR var or default to current dir
    static ref WORKING_DIR: PathBuf = {
        let wd = std::env::var("EZKL_WORKING_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(wd)
    };
}

/// Run an ezkl command with given args
pub async fn run(command: Commands) -> Result<String, EZKLError> {
    // set working dir
    std::env::set_current_dir(WORKING_DIR.as_path())?;

    match command {
        #[cfg(feature = "empty-cmd")]
        Commands::Empty => Ok(String::new()),
        Commands::GenSrs {
            srs_path,
            logrows,
            commitment,
        } => gen_srs_cmd(
            srs_path,
            logrows as u32,
            commitment.unwrap_or_else(|| Commitments::from_str(DEFAULT_COMMITMENT).unwrap()),
        ),
        Commands::GetSrs {
            srs_path,
            settings_path,
            logrows,
            commitment,
        } => get_srs_cmd(srs_path, settings_path, logrows, commitment).await,
        Commands::Table { model, args } => table(model.unwrap_or(DEFAULT_MODEL.into()), args),
        Commands::GenSettings {
            model,
            settings_path,
            args,
        } => gen_circuit_settings(
            model.unwrap_or(DEFAULT_MODEL.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            args,
        ),
        Commands::GenRandomData {
            model,
            data,
            variables,
            seed,
            min,
            max,
        } => gen_random_data(
            model.unwrap_or(DEFAULT_MODEL.into()),
            data.unwrap_or(DEFAULT_DATA.into()),
            variables,
            seed,
            min,
            max,
        ),
        Commands::CalibrateSettings {
            model,
            settings_path,
            data,
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            max_logrows,
        } => calibrate(
            model.unwrap_or(DEFAULT_MODEL.into()),
            data.unwrap_or(DEFAULT_DATA.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            max_logrows,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::GenWitness {
            data,
            compiled_circuit,
            output,
            vk_path,
            srs_path,
        } => gen_witness(
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            data.unwrap_or(DataField(DEFAULT_DATA.into())).to_string(),
            Some(output.unwrap_or(DEFAULT_WITNESS.into())),
            vk_path,
            srs_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Mock { model, witness } => mock(
            model.unwrap_or(DEFAULT_MODEL.into()),
            witness.unwrap_or(DEFAULT_WITNESS.into()),
        ),
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::CreateEvmVerifier {
            vk_path,
            srs_path,
            settings_path,
            sol_code_path,
            abi_path,
            reusable,
        } => {
            create_evm_verifier(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE.into()),
                abi_path.unwrap_or(DEFAULT_VERIFIER_ABI.into()),
                reusable.unwrap_or(DEFAULT_RENDER_REUSABLE.parse().unwrap()),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::EncodeEvmCalldata {
            proof_path,
            calldata_path,
            vka_path,
        } => encode_evm_calldata(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            calldata_path.unwrap_or(DEFAULT_CALLDATA.into()),
            vka_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(all(
            feature = "eth",
            feature = "reusable-verifier",
            not(target_arch = "wasm32")
        ))]
        Commands::CreateEvmVka {
            vk_path,
            srs_path,
            settings_path,
            vka_path,
            decimals,
        } => {
            create_evm_vka(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
                vka_path.unwrap_or(DEFAULT_VKA.into()),
                decimals.unwrap_or(DEFAULT_DECIMALS.parse().unwrap()),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::CreateEvmVerifierAggr {
            vk_path,
            srs_path,
            sol_code_path,
            abi_path,
            aggregation_settings,
            logrows,
            reusable,
        } => {
            create_evm_aggregate_verifier(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE_AGGREGATED.into()),
                abi_path.unwrap_or(DEFAULT_VERIFIER_AGGREGATED_ABI.into()),
                aggregation_settings,
                logrows.unwrap_or(DEFAULT_AGGREGATED_LOGROWS.parse().unwrap()),
                reusable.unwrap_or(DEFAULT_RENDER_REUSABLE.parse().unwrap()),
            )
            .await
        }
        Commands::CompileCircuit {
            model,
            compiled_circuit,
            settings_path,
        } => compile_circuit(
            model.unwrap_or(DEFAULT_MODEL.into()),
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
        ),
        Commands::Setup {
            compiled_circuit,
            srs_path,
            vk_path,
            pk_path,
            witness,
            disable_selector_compression,
        } => setup(
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            srs_path,
            vk_path.unwrap_or(DEFAULT_VK.into()),
            pk_path.unwrap_or(DEFAULT_PK.into()),
            witness,
            disable_selector_compression
                .unwrap_or(DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap()),
        ),
        Commands::SwapProofCommitments {
            proof_path,
            witness_path,
        } => swap_proof_commitments_cmd(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            witness_path.unwrap_or(DEFAULT_WITNESS.into()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),

        Commands::Prove {
            witness,
            compiled_circuit,
            pk_path,
            proof_path,
            srs_path,
            proof_type,
            check_mode,
        } => prove(
            witness.unwrap_or(DEFAULT_WITNESS.into()),
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            pk_path.unwrap_or(DEFAULT_PK.into()),
            Some(proof_path.unwrap_or(DEFAULT_PROOF.into())),
            srs_path,
            proof_type,
            check_mode.unwrap_or(DEFAULT_CHECKMODE.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::MockAggregate {
            aggregation_snarks,
            logrows,
            split_proofs,
        } => mock_aggregate(
            aggregation_snarks,
            logrows.unwrap_or(DEFAULT_AGGREGATED_LOGROWS.parse().unwrap()),
            split_proofs.unwrap_or(DEFAULT_SPLIT.parse().unwrap()),
        ),
        Commands::SetupAggregate {
            sample_snarks,
            vk_path,
            pk_path,
            srs_path,
            logrows,
            split_proofs,
            disable_selector_compression,
            commitment,
        } => setup_aggregate(
            sample_snarks,
            vk_path.unwrap_or(DEFAULT_VK_AGGREGATED.into()),
            pk_path.unwrap_or(DEFAULT_PK_AGGREGATED.into()),
            srs_path,
            logrows.unwrap_or(DEFAULT_AGGREGATED_LOGROWS.parse().unwrap()),
            split_proofs.unwrap_or(DEFAULT_SPLIT.parse().unwrap()),
            disable_selector_compression
                .unwrap_or(DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap()),
            commitment.into(),
        ),
        Commands::Aggregate {
            proof_path,
            aggregation_snarks,
            pk_path,
            srs_path,
            transcript,
            logrows,
            check_mode,
            split_proofs,
            commitment,
        } => aggregate(
            proof_path.unwrap_or(DEFAULT_PROOF_AGGREGATED.into()),
            aggregation_snarks,
            pk_path.unwrap_or(DEFAULT_PK_AGGREGATED.into()),
            srs_path,
            transcript,
            logrows.unwrap_or(DEFAULT_AGGREGATED_LOGROWS.parse().unwrap()),
            check_mode.unwrap_or(DEFAULT_CHECKMODE.parse().unwrap()),
            split_proofs.unwrap_or(DEFAULT_SPLIT.parse().unwrap()),
            commitment.into(),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Verify {
            proof_path,
            settings_path,
            vk_path,
            srs_path,
            reduced_srs,
        } => verify(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            vk_path.unwrap_or(DEFAULT_VK.into()),
            srs_path,
            reduced_srs.unwrap_or(DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::VerifyAggr {
            proof_path,
            vk_path,
            srs_path,
            reduced_srs,
            logrows,
            commitment,
        } => verify_aggr(
            proof_path.unwrap_or(DEFAULT_PROOF_AGGREGATED.into()),
            vk_path.unwrap_or(DEFAULT_VK_AGGREGATED.into()),
            srs_path,
            logrows.unwrap_or(DEFAULT_AGGREGATED_LOGROWS.parse().unwrap()),
            reduced_srs.unwrap_or(DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse().unwrap()),
            commitment.into(),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::DeployEvm {
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
            contract,
        } => {
            deploy_evm(
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE.into()),
                rpc_url,
                addr_path.unwrap_or(DEFAULT_CONTRACT_ADDRESS.into()),
                optimizer_runs,
                private_key,
                #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
                contract,
                #[cfg(not(all(feature = "reusable-verifier", not(target_arch = "wasm32"))))]
                ContractType::default(),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::VerifyEvm {
            proof_path,
            addr_verifier,
            rpc_url,
            vka_path,
            encoded_calldata,
        } => {
            verify_evm(
                proof_path.unwrap_or(DEFAULT_PROOF.into()),
                addr_verifier,
                rpc_url,
                vka_path,
                encoded_calldata,
            )
            .await
        }
        #[cfg(feature = "reusable-verifier")]
        Commands::RegisterVka {
            addr_verifier,
            vka_path,
            rpc_url,
            vka_digest_path,
            private_key,
        } => {
            register_vka(
                rpc_url,
                addr_verifier,
                vka_path.unwrap_or(DEFAULT_VKA.into()),
                vka_digest_path.unwrap_or(DEFAULT_VKA_DIGEST.into()),
                private_key,
            )
            .await
        }
        #[cfg(not(feature = "no-update"))]
        Commands::Update { version } => update_ezkl_binary(&version).map(|e| e.to_string()),
    }
}

#[cfg(not(feature = "no-update"))]
/// Assert that the version is valid
fn assert_version_is_valid(version: &str) -> Result<(), EZKLError> {
    let err_string = "Invalid version string. Must be in the format v0.0.0";
    if version.is_empty() {
        return Err(err_string.into());
    }
    // safe to unwrap since we know the length is not 0
    if !version.starts_with('v') {
        return Err(err_string.into());
    }

    semver::Version::parse(&version[1..])
        .map_err(|_| "Invalid version string. Must be in the format v0.0.0")?;

    Ok(())
}

#[cfg(not(feature = "no-update"))]
const INSTALL_BYTES: &[u8] = include_bytes!("../install_ezkl_cli.sh");

#[cfg(not(feature = "no-update"))]
fn update_ezkl_binary(version: &Option<String>) -> Result<String, EZKLError> {
    // run the install script with the version
    let install_script = std::str::from_utf8(INSTALL_BYTES)?;
    //  now run as sh script with the version as an argument

    // check if bash is installed
    let command = if std::process::Command::new("bash")
        .arg("--version")
        .status()
        .is_err()
    {
        log::warn!(
            "bash is not installed on this system, trying to run the install script with sh (may fail)"
        );
        "sh"
    } else {
        "bash"
    };

    let mut command = std::process::Command::new(command);
    let mut command = command.arg("-c").arg(install_script);

    if let Some(version) = version {
        assert_version_is_valid(version)?;
        command = command.arg(version)
    };
    let output = command.output()?;

    if output.status.success() {
        info!("updated binary");
        Ok("".to_string())
    } else {
        Err(format!(
            "failed to update binary: {}, {}",
            std::str::from_utf8(&output.stdout)?,
            std::str::from_utf8(&output.stderr)?
        )
        .into())
    }
}

/// Get the srs path
pub fn get_srs_path(logrows: u32, srs_path: Option<PathBuf>, commitment: Commitments) -> PathBuf {
    if let Some(srs_path) = srs_path {
        srs_path
    } else {
        if !Path::new(&*EZKL_SRS_REPO_PATH).exists() {
            std::fs::create_dir_all(&*EZKL_SRS_REPO_PATH).unwrap();
        }
        match commitment {
            Commitments::KZG => Path::new(&*EZKL_SRS_REPO_PATH).join(format!("kzg{}.srs", logrows)),
            Commitments::IPA => Path::new(&*EZKL_SRS_REPO_PATH).join(format!("ipa{}.srs", logrows)),
        }
    }
}

fn srs_exists_check(logrows: u32, srs_path: Option<PathBuf>, commitment: Commitments) -> bool {
    Path::new(&get_srs_path(logrows, srs_path, commitment)).exists()
}

pub(crate) fn gen_srs_cmd(
    srs_path: PathBuf,
    logrows: u32,
    commitment: Commitments,
) -> Result<String, EZKLError> {
    match commitment {
        Commitments::KZG => {
            let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
            save_params::<KZGCommitmentScheme<Bn256>>(&srs_path, &params)?;
        }
        Commitments::IPA => {
            let params = gen_srs::<IPACommitmentScheme<G1Affine>>(logrows);
            save_params::<IPACommitmentScheme<G1Affine>>(&srs_path, &params)?;
        }
    }
    Ok(String::new())
}

async fn fetch_srs(uri: &str) -> Result<Vec<u8>, EZKLError> {
    let pb = {
        let pb = init_spinner();
        pb.set_message("Downloading SRS (this may take a while) ...");
        pb
    };
    let client = reqwest::Client::new();
    // wasm doesn't require it to be mutable
    #[allow(unused_mut)]
    let mut resp = client.get(uri).body(vec![]).send().await?;
    let mut buf = vec![];
    while let Some(chunk) = resp.chunk().await? {
        buf.extend(chunk.to_vec());
    }

    pb.finish_with_message("SRS downloaded.");
    Ok(std::mem::take(&mut buf))
}

pub(crate) fn get_file_hash(path: &PathBuf) -> Result<String, EZKLError> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = vec![];
    let bytes_read = reader.read_to_end(&mut buffer)?;
    info!(
        "read {} bytes from file (vector of len = {})",
        bytes_read,
        buffer.len()
    );

    let hash = sha256::digest(buffer);
    info!("file hash: {}", hash);

    Ok(hash)
}

fn check_srs_hash(
    logrows: u32,
    srs_path: Option<PathBuf>,
    commitment: Commitments,
) -> Result<String, EZKLError> {
    let path = get_srs_path(logrows, srs_path, commitment);
    let hash = get_file_hash(&path)?;

    let predefined_hash = match crate::srs_sha::PUBLIC_SRS_SHA256_HASHES.get(&logrows) {
        Some(h) => h,
        None => return Err(format!("SRS (k={}) hash not found in public set", logrows).into()),
    };

    if hash != *predefined_hash {
        // delete file
        warn!("removing SRS file at {}", path.display());
        std::fs::remove_file(path)?;
        return Err(
            "SRS hash does not match the expected hash. Remote SRS may have been tampered with."
                .into(),
        );
    }
    Ok(hash)
}

pub(crate) async fn get_srs_cmd(
    srs_path: Option<PathBuf>,
    settings_path: Option<PathBuf>,
    logrows: Option<u32>,
    commitment: Option<Commitments>,
) -> Result<String, EZKLError> {
    // logrows overrides settings

    let err_string = "You will need to provide a valid settings file to use the settings option. You should run gen-settings to generate a settings file (and calibrate-settings to pick optimal logrows).";

    let k = if let Some(k) = logrows {
        k
    } else if let Some(settings_p) = &settings_path {
        if settings_p.exists() {
            let settings = GraphSettings::load(settings_p)?;
            settings.run_args.logrows
        } else {
            return Err(err_string.into());
        }
    } else {
        return Err(err_string.into());
    };

    let commitment = if let Some(c) = commitment {
        c
    } else if let Some(settings_p) = settings_path {
        if settings_p.exists() {
            let settings = GraphSettings::load(&settings_p)?;
            settings.run_args.commitment.into()
        } else {
            return Err(err_string.into());
        }
    } else {
        return Err(err_string.into());
    };

    if !srs_exists_check(k, srs_path.clone(), commitment) {
        if matches!(commitment, Commitments::KZG) {
            info!("SRS does not exist, downloading...");
            let srs_uri = format!("{}{}", PUBLIC_SRS_URL, k);
            let mut reader = Cursor::new(fetch_srs(&srs_uri).await?);
            // check the SRS
            let pb = init_spinner();
            pb.set_message("Validating SRS (this may take a while) ...");
            let params = ParamsKZG::<Bn256>::read(&mut reader)?;
            pb.finish_with_message("SRS validated.");

            info!("Saving SRS to disk...");
            let computed_srs_path = get_srs_path(k, srs_path.clone(), commitment);
            let mut file = std::fs::File::create(&computed_srs_path)?;
            let mut buffer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, &mut file);
            params.write(&mut buffer)?;

            info!(
                "Saved SRS to {}.",
                computed_srs_path.as_os_str().to_str().unwrap_or("disk")
            );

            info!("SRS downloaded");
        } else {
            let path = get_srs_path(k, srs_path.clone(), commitment);
            gen_srs_cmd(path, k, commitment)?;
        }
    } else {
        info!("SRS already exists at that path");
    };
    // check the hash
    if matches!(commitment, Commitments::KZG) {
        check_srs_hash(k, srs_path.clone(), commitment)?;
    }

    Ok(String::new())
}

pub(crate) fn table(model: PathBuf, run_args: RunArgs) -> Result<String, EZKLError> {
    let model = Model::from_run_args(&run_args, &model)?;
    info!("\n {}", model.table_nodes());
    Ok(String::new())
}

pub(crate) fn gen_witness(
    compiled_circuit_path: PathBuf,
    data: String,
    output: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
) -> Result<GraphWitness, EZKLError> {
    // these aren't real values so the sanity checks are mostly meaningless

    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;
    let data = GraphData::from_str(&data)?;
    let settings = circuit.settings().clone();

    let vk = if let Some(vk) = vk_path {
        Some(load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
            vk,
            settings.clone(),
        )?)
    } else {
        None
    };

    let mut input = circuit.load_graph_input(&data)?;
    #[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
    let mut input = circuit.load_graph_input(&data)?;

    // if any of the settings have kzg visibility then we need to load the srs

    let commitment: Commitments = settings.run_args.commitment.into();

    let region_settings =
        RegionSettings::all_true(settings.run_args.decomp_base, settings.run_args.decomp_legs);

    let start_time = Instant::now();
    let witness = if settings.module_requires_polycommit() {
        if get_srs_path(settings.run_args.logrows, srs_path.clone(), commitment).exists() {
            match Commitments::from(settings.run_args.commitment) {
                Commitments::KZG => {
                    let srs: ParamsKZG<Bn256> = load_params_prover::<KZGCommitmentScheme<Bn256>>(
                        srs_path.clone(),
                        settings.run_args.logrows,
                        commitment,
                    )?;
                    circuit.forward::<KZGCommitmentScheme<_>>(
                        &mut input,
                        vk.as_ref(),
                        Some(&srs),
                        region_settings,
                    )?
                }
                Commitments::IPA => {
                    let srs: ParamsIPA<G1Affine> =
                        load_params_prover::<IPACommitmentScheme<G1Affine>>(
                            srs_path.clone(),
                            settings.run_args.logrows,
                            commitment,
                        )?;
                    circuit.forward::<IPACommitmentScheme<_>>(
                        &mut input,
                        vk.as_ref(),
                        Some(&srs),
                        region_settings,
                    )?
                }
            }
        } else {
            warn!("SRS for poly commit does not exist (will be ignored)");
            circuit.forward::<KZGCommitmentScheme<Bn256>>(
                &mut input,
                vk.as_ref(),
                None,
                region_settings,
            )?
        }
    } else {
        circuit.forward::<KZGCommitmentScheme<Bn256>>(
            &mut input,
            vk.as_ref(),
            None,
            region_settings,
        )?
    };

    // print each variable tuple (symbol, value) as symbol=value
    trace!(
        "witness generation {:?} took {:?}",
        circuit
            .settings()
            .run_args
            .variables
            .iter()
            .map(|v| { format!("{}={}", v.0, v.1) })
            .collect::<Vec<_>>(),
        start_time.elapsed()
    );

    if let Some(output_path) = output {
        witness.save(output_path)?;
    }

    // print the witness in debug
    debug!("witness: \n {}", witness.as_json()?.to_colored_json_auto()?);

    Ok(witness)
}

/// Generate a circuit settings file
pub(crate) fn gen_circuit_settings(
    model_path: PathBuf,
    params_output: PathBuf,
    run_args: RunArgs,
) -> Result<String, EZKLError> {
    let circuit = GraphCircuit::from_run_args(&run_args, &model_path)?;
    let params = circuit.settings();
    params.save(&params_output)?;
    Ok(String::new())
}

/// Generate a circuit settings file
pub(crate) fn gen_random_data(
    model_path: PathBuf,
    data_path: PathBuf,
    variables: Vec<(String, usize)>,
    seed: u64,
    min: Option<f32>,
    max: Option<f32>,
) -> Result<String, EZKLError> {
    let mut file = std::fs::File::open(&model_path).map_err(|e| {
        crate::graph::errors::GraphError::ReadWriteFileError(
            model_path.display().to_string(),
            e.to_string(),
        )
    })?;

    let (tract_model, _symbol_values) = Model::load_onnx_using_tract(&mut file, &variables)?;

    let input_facts = tract_model
        .input_outlets()
        .map_err(|e| EZKLError::from(e.to_string()))?
        .iter()
        .map(|&i| tract_model.outlet_fact(i))
        .collect::<tract_onnx::prelude::TractResult<Vec<_>>>()
        .map_err(|e| EZKLError::from(e.to_string()))?;

    let min = min.unwrap_or(0.0);
    let max = max.unwrap_or(1.0);

    /// Generates a random tensor of a given size and type.
    fn random(
        sizes: &[usize],
        datum_type: tract_onnx::prelude::DatumType,
        seed: u64,
        min: f32,
        max: f32,
    ) -> TractTensor {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut tensor = TractTensor::zero::<f32>(sizes).unwrap();
        let slice = tensor.as_slice_mut::<f32>().unwrap();
        slice.iter_mut().for_each(|x| *x = rng.gen_range(min..max));
        tensor.cast_to_dt(datum_type).unwrap().into_owned()
    }

    fn tensor_for_fact(
        fact: &tract_onnx::prelude::TypedFact,
        seed: u64,
        min: f32,
        max: f32,
    ) -> TractTensor {
        if let Some(value) = &fact.konst {
            return value.clone().into_tensor();
        }

        random(
            fact.shape
                .as_concrete()
                .expect("Expected concrete shape, found: {fact:?}"),
            fact.datum_type,
            seed,
            min,
            max,
        )
    }

    let generated = input_facts
        .iter()
        .map(|v| tensor_for_fact(v, seed, min, max))
        .collect_vec();

    let data = GraphData::from_tract_data(&generated)?;

    data.save(data_path)?;

    Ok(String::new())
}

// not for wasm targets
pub(crate) fn init_spinner() -> ProgressBar {
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(200));
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {spinner:.blue} {msg}")
            .unwrap()
            .tick_strings(&[
                "------ - ✨ ",
                "------ - ⏳ ",
                "------ - 🌎 ",
                "------ - 🔎 ",
                "------ - 🥹 ",
                "------ - 🫠 ",
                "------ - 👾 ",
            ]),
    );
    pb
}

// not for wasm targets
pub(crate) fn init_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(200));
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("##-");
    pb.set_style(sty);
    pb
}

use colored_json::ToColoredJson;

#[derive(Debug, Clone, Tabled)]
/// Accuracy tearsheet
pub struct AccuracyResults {
    mean_error: f32,
    median_error: f32,
    max_error: f32,
    min_error: f32,
    mean_abs_error: f32,
    median_abs_error: f32,
    max_abs_error: f32,
    min_abs_error: f32,
    mean_squared_error: f32,
    mean_percent_error: f32,
    mean_abs_percent_error: f32,
}

impl AccuracyResults {
    /// Create a new accuracy results struct
    pub fn new(
        mut original_preds: Vec<crate::tensor::Tensor<f32>>,
        mut calibrated_preds: Vec<crate::tensor::Tensor<f32>>,
    ) -> Result<Self, EZKLError> {
        let mut errors = vec![];
        let mut abs_errors = vec![];
        let mut squared_errors = vec![];
        let mut percentage_errors = vec![];
        let mut abs_percentage_errors = vec![];

        for (original, calibrated) in original_preds.iter_mut().zip(calibrated_preds.iter_mut()) {
            original.flatten();
            calibrated.flatten();
            let error = (original.clone() - calibrated.clone())?;
            let abs_error = error.map(|x| x.abs());
            let squared_error = error.map(|x| x.powi(2));
            let percentage_error = error.enum_map(|i, x| {
                // if everything is 0 then we can't divide by 0 so we just return 0
                let res = if original[i] == 0.0 && x == 0.0 {
                    0.0
                } else {
                    x / original[i]
                };
                Ok::<f32, TensorError>(res)
            })?;
            let abs_percentage_error = percentage_error.map(|x| x.abs());

            errors.extend(error);
            abs_errors.extend(abs_error);
            squared_errors.extend(squared_error);
            percentage_errors.extend(percentage_error);
            abs_percentage_errors.extend(abs_percentage_error);
        }

        let mean_percent_error =
            percentage_errors.iter().sum::<f32>() / percentage_errors.len() as f32;
        let mean_abs_percent_error =
            abs_percentage_errors.iter().sum::<f32>() / abs_percentage_errors.len() as f32;
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let median_error = errors[errors.len() / 2];
        let max_error = *errors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let min_error = *errors
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let mean_abs_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
        let median_abs_error = abs_errors[abs_errors.len() / 2];
        let max_abs_error = *abs_errors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let min_abs_error = *abs_errors
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let mean_squared_error = squared_errors.iter().sum::<f32>() / squared_errors.len() as f32;

        Ok(Self {
            mean_error,
            median_error,
            max_error,
            min_error,
            mean_abs_error,
            median_abs_error,
            max_abs_error,
            min_abs_error,
            mean_squared_error,
            mean_percent_error,
            mean_abs_percent_error,
        })
    }
}

/// Calibrate the circuit parameters to a given a dataset
#[allow(trivial_casts)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate(
    model_path: PathBuf,
    data: String,
    settings_path: PathBuf,
    target: CalibrationTarget,
    lookup_safety_margin: f64,
    scales: Option<Vec<crate::Scale>>,
    scale_rebase_multiplier: Vec<u32>,
    max_logrows: Option<u32>,
) -> Result<GraphSettings, EZKLError> {
    use log::error;
    use std::collections::HashMap;
    use tabled::Table;

    use crate::fieldutils::IntegerRep;

    let data = GraphData::from_str(&data)?;
    // load the pre-generated settings
    let settings = GraphSettings::load(&settings_path)?;
    // now retrieve the run args
    // we load the model to get the input and output shapes

    let model = Model::from_run_args(&settings.run_args, &model_path)?;

    let input_shapes = model.graph.input_shapes()?;

    let chunks = data.split_into_batches(input_shapes)?;
    info!("num calibration batches: {}", chunks.len());

    debug!("running onnx predictions...");
    let original_predictions = Model::run_onnx_predictions(
        &settings.run_args,
        &model_path,
        &chunks,
        model.graph.input_shapes()?,
    )?;

    let range = if let Some(scales) = scales {
        scales
    } else {
        (11..14).collect::<Vec<crate::Scale>>()
    };

    let mut found_params: Vec<GraphSettings> = vec![];

    // 2 x 2 grid
    let range_grid = range
        .iter()
        .cartesian_product(range.iter())
        .map(|(a, b)| (*a, *b))
        .collect::<Vec<(crate::Scale, crate::Scale)>>();

    // remove all entries where input_scale > param_scale
    let mut range_grid = range_grid
        .into_iter()
        .filter(|(a, b)| a <= b)
        .collect::<Vec<(crate::Scale, crate::Scale)>>();

    // if all integers
    let all_scale_0 = model
        .graph
        .get_input_types()?
        .iter()
        .all(|t| t.is_integer());
    if all_scale_0 {
        // set all a values to 0 then dedup
        range_grid = range_grid
            .iter()
            .map(|(_, b)| (0, *b))
            .sorted()
            .dedup()
            .collect::<Vec<(crate::Scale, crate::Scale)>>();
    }

    let range_grid = range_grid
        .iter()
        .cartesian_product(scale_rebase_multiplier.iter())
        .map(|(a, b)| (*a, *b))
        .collect::<Vec<((crate::Scale, crate::Scale), u32)>>();

    let mut forward_pass_res = HashMap::new();

    let pb = init_bar(range_grid.len() as u64);
    pb.set_message("calibrating...");

    let mut num_failed = 0;
    let mut num_passed = 0;
    let mut failure_reasons = vec![];

    for ((input_scale, param_scale), scale_rebase_multiplier) in range_grid {
        pb.set_message(format!(
            "i-scale: {}, p-scale: {}, rebase-(x): {}, fail: {}, pass: {}",
            input_scale.to_string().blue(),
            param_scale.to_string().blue(),
            scale_rebase_multiplier.to_string().yellow(),
            num_failed.to_string().red(),
            num_passed.to_string().green()
        ));

        let key = (input_scale, param_scale, scale_rebase_multiplier);
        forward_pass_res.insert(key, vec![]);

        let local_run_args = RunArgs {
            input_scale,
            param_scale,
            scale_rebase_multiplier,
            lookup_range: (IntegerRep::MIN, IntegerRep::MAX),
            ..settings.run_args.clone()
        };

        // if unix get a gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _r = Gag::stdout().ok();
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _g = Gag::stderr().ok();

        let mut circuit = match GraphCircuit::from_run_args(&local_run_args, &model_path) {
            Ok(c) => c,
            Err(e) => {
                error!("circuit creation from run args failed: {:?}", e);
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                    input_scale.to_string().blue(),
                    param_scale.to_string().blue(),
                    scale_rebase_multiplier.to_string().yellow(),
                    e
                ));
                pb.inc(1);
                num_failed += 1;
                continue;
            }
        };

        let forward_res = chunks
            .iter()
            .map(|chunk| {
                let chunk = chunk.clone();

                let data = circuit
                    .load_graph_input(&chunk)
                    .map_err(|e| format!("failed to load circuit inputs: {}", e))?;

                let forward_res = circuit
                    .forward::<KZGCommitmentScheme<Bn256>>(
                        &mut data.clone(),
                        None,
                        None,
                        RegionSettings::all_true(
                            settings.run_args.decomp_base,
                            settings.run_args.decomp_legs,
                        ),
                    )
                    .map_err(|e| format!("failed to forward: {}", e))?;

                // push result to the hashmap
                forward_pass_res
                    .get_mut(&key)
                    .ok_or("key not found")?
                    .push(forward_res);

                Ok(()) as Result<(), String>
            })
            .collect::<Result<Vec<()>, String>>();

        match forward_res {
            Ok(_) => (),
            // typically errors will be due to the circuit overflowing the i64 limit
            Err(e) => {
                error!("forward pass failed: {:?}", e);
                pb.inc(1);
                num_failed += 1;
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                    input_scale.to_string().blue(),
                    param_scale.to_string().blue(),
                    scale_rebase_multiplier.to_string().yellow(),
                    e
                ));
                continue;
            }
        }

        // drop the gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_r);
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_g);

        let result = forward_pass_res.get(&key).ok_or("key not found")?;

        let min_lookup_range = result
            .iter()
            .map(|x| x.min_lookup_inputs)
            .min()
            .unwrap_or(0);

        let max_lookup_range = result
            .iter()
            .map(|x| x.max_lookup_inputs)
            .max()
            .unwrap_or(0);

        let max_range_size = result.iter().map(|x| x.max_range_size).max().unwrap_or(0);

        let res = circuit.calc_min_logrows(
            (min_lookup_range, max_lookup_range),
            max_range_size,
            max_logrows,
            lookup_safety_margin,
        );

        if res.is_ok() {
            let new_settings = circuit.settings().clone();

            let found_run_args = RunArgs {
                input_scale: new_settings.run_args.input_scale,
                param_scale: new_settings.run_args.param_scale,
                lookup_range: new_settings.run_args.lookup_range,
                logrows: new_settings.run_args.logrows,
                scale_rebase_multiplier: new_settings.run_args.scale_rebase_multiplier,
                ..settings.run_args.clone()
            };

            let found_settings = GraphSettings {
                run_args: found_run_args,
                required_lookups: new_settings.required_lookups,
                required_range_checks: new_settings.required_range_checks,
                model_output_scales: new_settings.model_output_scales,
                model_input_scales: new_settings.model_input_scales,
                num_rows: new_settings.num_rows,
                total_assignments: new_settings.total_assignments,
                total_const_size: new_settings.total_const_size,
                dynamic_lookup_params: new_settings.dynamic_lookup_params,
                shuffle_params: new_settings.shuffle_params,
                ..settings.clone()
            };

            found_params.push(found_settings.clone());

            debug!(
                "found settings: \n {}",
                found_settings.as_json()?.to_colored_json_auto()?
            );
            num_passed += 1;
        } else if let Err(res) = res {
            failure_reasons.push(format!(
                "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                input_scale.to_string().blue(),
                param_scale.to_string().blue(),
                scale_rebase_multiplier.to_string().yellow(),
                res.to_string().red()
            ));
            num_failed += 1;
        }

        pb.inc(1);
    }

    pb.finish_with_message("Calibration Done.");

    if found_params.is_empty() {
        if !failure_reasons.is_empty() {
            error!("Calibration failed for the following reasons:");
            for reason in failure_reasons {
                error!("{}", reason);
            }
        }

        return Err("calibration failed, could not find any suitable parameters given the calibration dataset".into());
    }

    debug!("Found {} sets of parameters", found_params.len());

    // now find the best params according to the target
    let mut best_params = match target {
        CalibrationTarget::Resources { .. } => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| p.run_args.logrows);

            let min_logrows = param_iterator
                .next()
                .ok_or("no params found")?
                .run_args
                .logrows;

            // pick the ones that have the minimum logrows but also the largest scale:
            // this is the best tradeoff between resource usage and accuracy
            found_params
                .iter()
                .filter(|p| p.run_args.logrows == min_logrows)
                .max_by_key(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        // we want the largest rebase multiplier as it means we can use less constraints
                        p.run_args.scale_rebase_multiplier,
                    )
                })
                .ok_or("no params found")?
                .clone()
        }
        CalibrationTarget::Accuracy => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| {
                (
                    p.run_args.input_scale,
                    p.run_args.param_scale,
                    // we want the largest rebase multiplier as it means we can use less constraints
                    p.run_args.scale_rebase_multiplier,
                )
            });

            let last = param_iterator.next_back().ok_or("no params found")?;
            let max_scale = (
                last.run_args.input_scale,
                last.run_args.param_scale,
                last.run_args.scale_rebase_multiplier,
            );

            // pick the ones that have the max scale but also the smallest logrows:
            // this is the best tradeoff between resource usage and accuracy
            found_params
                .iter()
                .filter(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        p.run_args.scale_rebase_multiplier,
                    ) == max_scale
                })
                .min_by_key(|p| p.run_args.logrows)
                .ok_or("no params found")?
                .clone()
        }
    };

    let outputs = forward_pass_res
        .get(&(
            best_params.run_args.input_scale,
            best_params.run_args.param_scale,
            best_params.run_args.scale_rebase_multiplier,
        ))
        .ok_or("no params found")?
        .iter()
        .map(|x| x.get_float_outputs(&best_params.model_output_scales))
        .collect::<Vec<_>>();

    let accuracy_res = AccuracyResults::new(
        original_predictions.into_iter().flatten().collect(),
        outputs.into_iter().flatten().collect(),
    )?;

    let tear_sheet_table = Table::new(vec![accuracy_res]);

    warn!(
        "\n\n <------------- Numerical Fidelity Report (input_scale: {}, param_scale: {}, scale_input_multiplier: {}) ------------->\n\n{}\n\n",
        best_params.run_args.input_scale,
        best_params.run_args.param_scale,
        best_params.run_args.scale_rebase_multiplier,
        tear_sheet_table.to_string().as_str()
    );

    if matches!(target, CalibrationTarget::Resources { col_overflow: true }) {
        let lookup_log_rows = best_params.lookup_log_rows_with_blinding();
        let module_log_row = best_params.module_constraint_logrows_with_blinding();
        let instance_logrows = best_params.log2_total_instances_with_blinding();
        let dynamic_lookup_logrows =
            best_params.min_dynamic_lookup_and_shuffle_logrows_with_blinding();

        let range_check_logrows = best_params.range_check_log_rows_with_blinding();

        let mut reduction = std::cmp::max(lookup_log_rows, module_log_row);
        reduction = std::cmp::max(reduction, range_check_logrows);
        reduction = std::cmp::max(reduction, instance_logrows);
        reduction = std::cmp::max(reduction, dynamic_lookup_logrows);
        reduction = std::cmp::max(reduction, crate::graph::MIN_LOGROWS);

        info!(
            "logrows > bits, shrinking logrows: {} -> {}",
            best_params.run_args.logrows, reduction
        );

        best_params.run_args.logrows = reduction;
    }

    best_params.save(&settings_path)?;

    debug!("Saved parameters.");

    Ok(best_params)
}

pub(crate) fn mock(
    compiled_circuit_path: PathBuf,
    data_path: PathBuf,
) -> Result<String, EZKLError> {
    // mock should catch any issues by default so we set it to safe
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    let data = GraphWitness::from_path(data_path)?;

    circuit.load_graph_witness(&data)?;

    let public_inputs = circuit.prepare_public_inputs(&data)?;

    info!("Mock proof");

    let prover = halo2_proofs::dev::MockProver::run(
        circuit.settings().run_args.logrows,
        &circuit,
        vec![public_inputs],
    )
    .map_err(|e| ExecutionError::MockProverError(e.to_string()))?;

    prover.verify().map_err(ExecutionError::VerifyError)?;
    Ok(String::new())
}

#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn create_evm_verifier(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    reusable: bool,
) -> Result<String, EZKLError> {
    let settings = GraphSettings::load(&settings_path)?;
    let commitment: Commitments = settings.run_args.commitment.into();
    let params = load_params_verifier::<KZGCommitmentScheme<Bn256>>(
        srs_path,
        settings.run_args.logrows,
        commitment,
    )?;

    let num_instance = settings.total_instances();
    // create a scales array that is the same length as the number of instances, all populated with 0
    let scales = vec![0; num_instance.len()];
    // let poseidon_instance = settings.module_sizes.num_instances().iter().sum::<usize>();

    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)?;
    trace!("params computed");

    let generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_instance,
        &scales,
        0,
        0,
    );
    let (verifier_solidity, name) = if reusable {
        (generator.render_separately()?.0, "Halo2VerifierReusable") // ignore the rendered vk artifact for now and generate it in create_evm_vka
    } else {
        (generator.render()?, "Halo2Verifier")
    };

    File::create(sol_code_path.clone())?.write_all(verifier_solidity.as_bytes())?;

    // fetch abi of the contract
    let (abi, _, _) = get_contract_artifacts(sol_code_path, name, 0).await?;
    // save abi to file
    serde_json::to_writer(std::fs::File::create(abi_path)?, &abi)?;

    Ok(String::new())
}

#[cfg(feature = "reusable-verifier")]
pub(crate) async fn create_evm_vka(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    settings_path: PathBuf,
    vka_path: PathBuf,
    decimals: usize,
) -> Result<String, EZKLError> {
    log::warn!("Reusable verifier support is experimental and may change in the future. Use at your own risk.");
    let settings = GraphSettings::load(&settings_path)?;
    let commitment: Commitments = settings.run_args.commitment.into();
    let params = load_params_verifier::<KZGCommitmentScheme<Bn256>>(
        srs_path,
        settings.run_args.logrows,
        commitment,
    )?;

    let num_poseidon_instance = settings.module_sizes.num_instances().iter().sum::<usize>();
    let num_fixed_point_instance = settings
        .model_instance_shapes
        .iter()
        .map(|x| x.iter().product::<usize>())
        .collect_vec();

    let scales = settings.get_model_instance_scales();
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)?;
    trace!("params computed");
    // assert that the decimals must be less than or equal to 38 to prevent overflow
    if decimals > 38 {
        return Err("decimals must be less than or equal to 38".into());
    }

    let generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_fixed_point_instance,
        &scales,
        decimals,
        num_poseidon_instance,
    );

    let vka_words: Vec<[u8; 32]> = generator.render_separately_vka_words()?.1;
    let serialized_vka_words = bincode::serialize(&vka_words).or_else(|e| {
        Err(EZKLError::from(format!(
            "Failed to serialize vka words: {}",
            e
        )))
    })?;

    File::create(vka_path.clone())?.write_all(&serialized_vka_words)?;

    // Load in the vka words and deserialize them and check that they match the original
    let bytes = std::fs::read(vka_path)?;
    let vka_buf: Vec<[u8; 32]> = bincode::deserialize(&bytes)
        .map_err(|e| EZKLError::from(format!("Failed to deserialize vka words: {e}")))?;
    if vka_buf != vka_words {
        return Err("vka words do not match".into());
    };

    Ok(String::new())
}
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn deploy_evm(
    sol_code_path: PathBuf,
    rpc_url: String,
    addr_path: PathBuf,
    runs: usize,
    private_key: Option<String>,
    contract: ContractType,
) -> Result<String, EZKLError> {
    let contract_name = match contract {
        ContractType::Verifier { reusable: false } => "Halo2Verifier",
        ContractType::Verifier { reusable: true } => "Halo2VerifierReusable",
    };
    let contract_address = deploy_contract_via_solidity(
        sol_code_path,
        &rpc_url,
        runs,
        private_key.as_deref(),
        contract_name,
    )
    .await?;

    info!("Contract deployed at: {:#?}", contract_address);

    let mut f = File::create(addr_path)?;
    write!(f, "{:#?}", contract_address)?;
    Ok(String::new())
}

#[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
pub(crate) async fn register_vka(
    rpc_url: String,
    rv_addr: H160Flag,
    vka_path: PathBuf,
    vka_digest_path: PathBuf,
    private_key: Option<String>,
) -> Result<String, EZKLError> {
    log::warn!("Reusable verifier support is experimental and may change in the future. Use at your own risk.");
    // Load the vka, which is bincode serialized, from the vka_path
    let bytes = std::fs::read(vka_path)?;
    let vka_buf: Vec<[u8; 32]> = bincode::deserialize(&bytes)
        .map_err(|e| EZKLError::from(format!("Failed to deserialize vka words: {e}")))?;
    let vka_digest = register_vka_via_rv(
        rpc_url.as_ref(),
        private_key.as_deref(),
        rv_addr.into(),
        &vka_buf,
    )
    .await?;

    info!("VKA digest: {:#?}", vka_digest);

    let mut f = File::create(vka_digest_path)?;
    write!(f, "{:#?}", vka_digest)?;
    Ok(String::new())
}

/// Encodes the calldata for the EVM verifier (both aggregated and single proof)
/// TODO: Add a "RV address param" which will query the "RegisteredVKA" events to fetch the
/// VKA from the vka_digest.
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) fn encode_evm_calldata(
    proof_path: PathBuf,
    calldata_path: PathBuf,
    vka_path: Option<PathBuf>,
) -> Result<Vec<u8>, EZKLError> {
    let snark = Snark::load::<IPACommitmentScheme<G1Affine>>(&proof_path)?;

    let flattened_instances = snark.instances.into_iter().flatten();

    // Load the vka, which is bincode serialized, from the vka_path
    let vka_buf: Option<Vec<[u8; 32]>> =
        match vka_path {
            Some(path) => {
                let bytes = std::fs::read(path)?;
                Some(bincode::deserialize(&bytes).map_err(|e| {
                    EZKLError::from(format!("Failed to deserialize vka words: {e}"))
                })?)
            }
            None => None,
        };

    let vka: Option<&[[u8; 32]]> = vka_buf.as_deref();
    let encoded = encode_calldata(vka, &snark.proof, &flattened_instances.collect::<Vec<_>>());

    log::debug!("Encoded calldata: {:?}", encoded);

    File::create(calldata_path)?.write_all(encoded.as_slice())?;

    Ok(encoded)
}

/// TODO: Add an optional vka_digest param that will allow us to fetch the associated VKA
/// from the RegisteredVKA events on the RV.
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn verify_evm(
    proof_path: PathBuf,
    addr_verifier: H160Flag,
    rpc_url: String,
    vka_path: Option<PathBuf>,
    encoded_calldata: Option<PathBuf>,
) -> Result<String, EZKLError> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;

    let result = verify_proof_via_solidity(
        proof.clone(),
        addr_verifier.into(),
        vka_path.map(|s| s.into()),
        rpc_url.as_ref(),
        encoded_calldata.map(|s| s.into()),
    )
    .await?;

    info!("Solidity verification result: {}", result);

    if !result {
        return Err("Solidity verification failed".into());
    }

    Ok(String::new())
}

#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn create_evm_aggregate_verifier(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    circuit_settings: Vec<PathBuf>,
    logrows: u32,
    reusable: bool,
) -> Result<String, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path, Commitments::KZG);
    let params: ParamsKZG<Bn256> = load_srs_verifier::<KZGCommitmentScheme<Bn256>>(srs_path)?;

    let mut settings: Vec<GraphSettings> = vec![];

    for path in circuit_settings.iter() {
        let s = GraphSettings::load(path)?;
        settings.push(s);
    }

    let num_instance: usize = settings
        .iter()
        .map(|s| s.total_instances().iter().sum::<usize>())
        .sum();

    let num_instance = AggregationCircuit::num_instance(num_instance);
    let scales = vec![0; num_instance.len()];
    assert_eq!(num_instance.len(), 1);

    let agg_vk = load_vk::<KZGCommitmentScheme<Bn256>, AggregationCircuit>(vk_path, ())?;

    let mut generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &agg_vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_instance,
        &scales,
        0,
        0,
    );

    let acc_encoding = halo2_solidity_verifier::AccumulatorEncoding::new(
        0,
        AggregationCircuit::num_limbs(),
        AggregationCircuit::num_bits(),
    );

    generator = generator.set_acc_encoding(Some(acc_encoding));

    let verifier_solidity = if reusable {
        generator.render_separately()?.0 // ignore the rendered vk artifact for now and generate it in create_evm_vka
    } else {
        generator.render()?
    };

    File::create(sol_code_path.clone())?.write_all(verifier_solidity.as_bytes())?;

    // fetch abi of the contract
    let (abi, _, _) = get_contract_artifacts(sol_code_path, "Halo2Verifier", 0).await?;
    // save abi to file
    serde_json::to_writer(std::fs::File::create(abi_path)?, &abi)?;

    Ok(String::new())
}

pub(crate) fn compile_circuit(
    model_path: PathBuf,
    compiled_circuit: PathBuf,
    settings_path: PathBuf,
) -> Result<String, EZKLError> {
    let settings = GraphSettings::load(&settings_path)?;
    let circuit = GraphCircuit::from_settings(&settings, &model_path, CheckMode::UNSAFE)?;
    circuit.save(compiled_circuit)?;
    Ok(String::new())
}

pub(crate) fn setup(
    compiled_circuit: PathBuf,
    srs_path: Option<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    witness: Option<PathBuf>,
    disable_selector_compression: bool,
) -> Result<String, EZKLError> {
    // these aren't real values so the sanity checks are mostly meaningless

    let mut circuit = GraphCircuit::load(compiled_circuit)?;

    if let Some(witness) = witness {
        let data = GraphWitness::from_path(witness)?;
        circuit.load_graph_witness(&data)?;
    }

    let logrows = circuit.settings().run_args.logrows;
    let commitment: Commitments = circuit.settings().run_args.commitment.into();

    let pk = match commitment {
        Commitments::KZG => {
            let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(
                srs_path,
                logrows,
                Commitments::KZG,
            )?;
            create_keys::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
                &circuit,
                &params,
                disable_selector_compression,
            )?
        }
        Commitments::IPA => {
            let params = load_params_prover::<IPACommitmentScheme<G1Affine>>(
                srs_path,
                logrows,
                Commitments::IPA,
            )?;
            create_keys::<IPACommitmentScheme<G1Affine>, GraphCircuit>(
                &circuit,
                &params,
                disable_selector_compression,
            )?
        }
    };
    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    Ok(String::new())
}

use crate::pfsys::ProofType;

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove(
    data_path: PathBuf,
    compiled_circuit_path: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
    proof_type: ProofType,
    check_mode: CheckMode,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let data = GraphWitness::from_path(data_path)?;
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    circuit.load_graph_witness(&data)?;

    let pretty_public_inputs = circuit.pretty_public_inputs(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;

    let circuit_settings = circuit.settings().clone();

    let strategy: StrategyType = proof_type.into();
    let transcript: TranscriptType = proof_type.into();
    let proof_split_commits: Option<ProofSplitCommit> = data.into();

    let commitment = circuit_settings.run_args.commitment.into();
    let logrows = circuit_settings.run_args.logrows;
    // creates and verifies the proof
    let mut snark = match commitment {
        Commitments::KZG => {
            let pk =
                load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(pk_path, circuit.params())?;

            let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(
                srs_path,
                logrows,
                Commitments::KZG,
            )?;
            match strategy {
                StrategyType::Single => create_proof_circuit::<
                    KZGCommitmentScheme<Bn256>,
                    _,
                    ProverSHPLONK<_>,
                    VerifierSHPLONK<_>,
                    KZGSingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit,
                    vec![public_inputs],
                    &params,
                    &pk,
                    check_mode,
                    commitment,
                    transcript,
                    proof_split_commits,
                    None,
                ),
                StrategyType::Accum => {
                    let protocol = Some(compile(
                        &params,
                        pk.get_vk(),
                        Config::kzg().with_num_instance(vec![public_inputs.len()]),
                    ));

                    create_proof_circuit::<
                        KZGCommitmentScheme<Bn256>,
                        _,
                        ProverSHPLONK<_>,
                        VerifierSHPLONK<_>,
                        KZGAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        &params,
                        &pk,
                        check_mode,
                        commitment,
                        transcript,
                        proof_split_commits,
                        protocol,
                    )
                }
            }
        }
        Commitments::IPA => {
            let pk =
                load_pk::<IPACommitmentScheme<G1Affine>, GraphCircuit>(pk_path, circuit.params())?;

            let params = load_params_prover::<IPACommitmentScheme<G1Affine>>(
                srs_path,
                circuit_settings.run_args.logrows,
                Commitments::IPA,
            )?;
            match strategy {
                StrategyType::Single => create_proof_circuit::<
                    IPACommitmentScheme<G1Affine>,
                    _,
                    ProverIPA<_>,
                    VerifierIPA<_>,
                    IPASingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit,
                    vec![public_inputs],
                    &params,
                    &pk,
                    check_mode,
                    commitment,
                    transcript,
                    proof_split_commits,
                    None,
                ),
                StrategyType::Accum => {
                    let protocol = Some(compile(
                        &params,
                        pk.get_vk(),
                        Config::ipa().with_num_instance(vec![public_inputs.len()]),
                    ));
                    create_proof_circuit::<
                        IPACommitmentScheme<G1Affine>,
                        _,
                        ProverIPA<_>,
                        VerifierIPA<_>,
                        IPAAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        &params,
                        &pk,
                        check_mode,
                        commitment,
                        transcript,
                        proof_split_commits,
                        protocol,
                    )
                }
            }
        }
    }?;

    snark.pretty_public_inputs = pretty_public_inputs;

    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
    }

    Ok(snark)
}

pub(crate) fn swap_proof_commitments_cmd(
    proof_path: PathBuf,
    witness: PathBuf,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let snark = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
    let witness = GraphWitness::from_path(witness)?;
    let commitments = witness.get_polycommitments();

    let snark_new = swap_proof_commitments_polycommit(&snark, &commitments)?;

    if snark_new.proof != *snark.proof {
        log::warn!("swap proof has created a different proof");
    }

    snark_new.save(&proof_path)?;
    Ok(snark_new)
}

pub(crate) fn mock_aggregate(
    aggregation_snarks: Vec<PathBuf>,
    logrows: u32,
    split_proofs: bool,
) -> Result<String, EZKLError> {
    let mut snarks = vec![];
    for proof_path in aggregation_snarks.iter() {
        match Snark::load::<KZGCommitmentScheme<Bn256>>(proof_path) {
            Ok(snark) => {
                snarks.push(snark);
            }
            Err(_) => {
                return Err(
                    "invalid sample commitment type for aggregation, must be KZG"
                        .to_string()
                        .into(),
                );
            }
        }
    }
    // proof aggregation
    let pb = {
        let pb = init_spinner();
        pb.set_message("Aggregating (may take a while)...");
        pb
    };

    let circuit = AggregationCircuit::new(&G1Affine::generator().into(), snarks, split_proofs)?;

    let prover = halo2_proofs::dev::MockProver::run(logrows, &circuit, vec![circuit.instances()])
        .map_err(|e| ExecutionError::MockProverError(e.to_string()))?;
    prover.verify().map_err(ExecutionError::VerifyError)?;
    pb.finish_with_message("Done.");
    Ok(String::new())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn setup_aggregate(
    sample_snarks: Vec<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    srs_path: Option<PathBuf>,
    logrows: u32,
    split_proofs: bool,
    disable_selector_compression: bool,
    commitment: Commitments,
) -> Result<String, EZKLError> {
    let mut snarks = vec![];
    for proof_path in sample_snarks.iter() {
        match Snark::load::<KZGCommitmentScheme<Bn256>>(proof_path) {
            Ok(snark) => {
                snarks.push(snark);
            }
            Err(_) => {
                return Err(
                    "invalid sample commitment type for aggregation, must be KZG"
                        .to_string()
                        .into(),
                );
            }
        }
    }

    let circuit = AggregationCircuit::new(&G1Affine::generator().into(), snarks, split_proofs)?;

    let pk = match commitment {
        Commitments::KZG => {
            let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(
                srs_path,
                logrows,
                Commitments::KZG,
            )?;

            create_keys::<KZGCommitmentScheme<Bn256>, AggregationCircuit>(
                &circuit,
                &params,
                disable_selector_compression,
            )?
        }
        Commitments::IPA => {
            let params = load_params_prover::<IPACommitmentScheme<G1Affine>>(
                srs_path,
                logrows,
                Commitments::IPA,
            )?;
            create_keys::<IPACommitmentScheme<G1Affine>, AggregationCircuit>(
                &circuit,
                &params,
                disable_selector_compression,
            )?
        }
    };
    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;

    Ok(String::new())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate(
    proof_path: PathBuf,
    aggregation_snarks: Vec<PathBuf>,
    pk_path: PathBuf,
    srs_path: Option<PathBuf>,
    transcript: TranscriptType,
    logrows: u32,
    check_mode: CheckMode,
    split_proofs: bool,
    commitment: Commitments,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let mut snarks = vec![];
    for proof_path in aggregation_snarks.iter() {
        match Snark::load::<KZGCommitmentScheme<Bn256>>(proof_path) {
            Ok(snark) => {
                snarks.push(snark);
            }
            Err(_) => {
                return Err(
                    "invalid sample commitment type for aggregation, must be KZG"
                        .to_string()
                        .into(),
                );
            }
        }
    }

    // proof aggregation
    let pb = {
        let pb = init_spinner();
        pb.set_message("Aggregating (may take a while)...");
        pb
    };

    let now = Instant::now();

    let snark = match commitment {
        Commitments::KZG => {
            let pk = load_pk::<KZGCommitmentScheme<Bn256>, AggregationCircuit>(pk_path, ())?;
            let params: ParamsKZG<Bn256> = load_params_prover::<KZGCommitmentScheme<_>>(
                srs_path.clone(),
                logrows,
                Commitments::KZG,
            )?;
            let circuit = AggregationCircuit::new(
                &ParamsProver::<G1Affine>::get_g(&params)[0].into(),
                snarks,
                split_proofs,
            )?;
            let public_inputs = circuit.instances();
            match transcript {
                TranscriptType::EVM => create_proof_circuit::<
                    KZGCommitmentScheme<Bn256>,
                    _,
                    ProverSHPLONK<_>,
                    VerifierSHPLONK<_>,
                    KZGSingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit,
                    vec![public_inputs],
                    &params,
                    &pk,
                    check_mode,
                    commitment,
                    transcript,
                    None,
                    None,
                ),
                TranscriptType::Poseidon => {
                    let protocol = Some(compile(
                        &params,
                        pk.get_vk(),
                        Config::kzg().with_num_instance(vec![public_inputs.len()]),
                    ));

                    create_proof_circuit::<
                        KZGCommitmentScheme<Bn256>,
                        _,
                        ProverSHPLONK<_>,
                        VerifierSHPLONK<_>,
                        KZGAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        &params,
                        &pk,
                        check_mode,
                        commitment,
                        transcript,
                        None,
                        protocol,
                    )
                }
            }
        }
        Commitments::IPA => {
            let pk = load_pk::<IPACommitmentScheme<_>, AggregationCircuit>(pk_path, ())?;
            let params: ParamsIPA<_> = load_params_prover::<IPACommitmentScheme<_>>(
                srs_path.clone(),
                logrows,
                Commitments::IPA,
            )?;
            let circuit = AggregationCircuit::new(
                &ParamsProver::<G1Affine>::get_g(&params)[0].into(),
                snarks,
                split_proofs,
            )?;
            let public_inputs = circuit.instances();

            match transcript {
                TranscriptType::EVM => create_proof_circuit::<
                    IPACommitmentScheme<G1Affine>,
                    _,
                    ProverIPA<_>,
                    VerifierIPA<_>,
                    IPASingleStrategy<_>,
                    _,
                    EvmTranscript<_, _, _, _>,
                    EvmTranscript<_, _, _, _>,
                >(
                    circuit,
                    vec![public_inputs],
                    &params,
                    &pk,
                    check_mode,
                    commitment,
                    transcript,
                    None,
                    None,
                ),
                TranscriptType::Poseidon => {
                    let protocol = Some(compile(
                        &params,
                        pk.get_vk(),
                        Config::ipa().with_num_instance(vec![public_inputs.len()]),
                    ));

                    create_proof_circuit::<
                        IPACommitmentScheme<G1Affine>,
                        _,
                        ProverIPA<_>,
                        VerifierIPA<_>,
                        IPAAccumulatorStrategy<_>,
                        _,
                        PoseidonTranscript<NativeLoader, _>,
                        PoseidonTranscript<NativeLoader, _>,
                    >(
                        circuit,
                        vec![public_inputs],
                        &params,
                        &pk,
                        check_mode,
                        commitment,
                        transcript,
                        None,
                        protocol,
                    )
                }
            }
        }
    }?;
    // the K used for the aggregation circuit

    let elapsed = now.elapsed();
    info!(
        "Aggregation proof took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    snark.save(&proof_path)?;

    pb.finish_with_message("Done.");

    Ok(snark)
}

pub(crate) fn verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    reduced_srs: bool,
) -> Result<bool, EZKLError> {
    let circuit_settings = GraphSettings::load(&settings_path)?;

    let logrows = circuit_settings.run_args.logrows;
    let commitment = circuit_settings.run_args.commitment.into();

    match commitment {
        Commitments::KZG => {
            let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
            let params: ParamsKZG<Bn256> = if reduced_srs {
                // only need G_0 for the verification with shplonk
                load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, 1, Commitments::KZG)?
            } else {
                load_params_verifier::<KZGCommitmentScheme<Bn256>>(
                    srs_path,
                    logrows,
                    Commitments::KZG,
                )?
            };
            match proof.transcript_type {
                TranscriptType::EVM => {
                    verify_commitment::<
                        KZGCommitmentScheme<Bn256>,
                        VerifierSHPLONK<'_, Bn256>,
                        _,
                        KZGSingleStrategy<_>,
                        EvmTranscript<G1Affine, _, _, _>,
                        GraphCircuit,
                        _,
                    >(proof_path, circuit_settings, vk_path, &params, logrows)
                }
                TranscriptType::Poseidon => {
                    verify_commitment::<
                        KZGCommitmentScheme<Bn256>,
                        VerifierSHPLONK<'_, Bn256>,
                        _,
                        KZGSingleStrategy<_>,
                        PoseidonTranscript<NativeLoader, _>,
                        GraphCircuit,
                        _,
                    >(proof_path, circuit_settings, vk_path, &params, logrows)
                }
            }
        }
        Commitments::IPA => {
            let proof = Snark::load::<IPACommitmentScheme<G1Affine>>(&proof_path)?;
            let params: ParamsIPA<_> = load_params_verifier::<IPACommitmentScheme<G1Affine>>(
                srs_path,
                logrows,
                Commitments::IPA,
            )?;
            match proof.transcript_type {
                TranscriptType::EVM => {
                    verify_commitment::<
                        IPACommitmentScheme<G1Affine>,
                        VerifierIPA<_>,
                        _,
                        IPASingleStrategy<_>,
                        EvmTranscript<G1Affine, _, _, _>,
                        GraphCircuit,
                        _,
                    >(proof_path, circuit_settings, vk_path, &params, logrows)
                }
                TranscriptType::Poseidon => {
                    verify_commitment::<
                        IPACommitmentScheme<G1Affine>,
                        VerifierIPA<_>,
                        _,
                        IPASingleStrategy<_>,
                        PoseidonTranscript<NativeLoader, _>,
                        GraphCircuit,
                        _,
                    >(proof_path, circuit_settings, vk_path, &params, logrows)
                }
            }
        }
    }
}

fn verify_commitment<
    'a,
    Scheme: CommitmentScheme,
    V: Verifier<'a, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    Strategy: VerificationStrategy<'a, Scheme, V>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
    C: Circuit<<Scheme as CommitmentScheme>::Scalar, Params = Params>,
    Params,
>(
    proof_path: PathBuf,
    settings: Params,
    vk_path: PathBuf,
    params: &'a Scheme::ParamsVerifier,
    logrows: u32,
) -> Result<bool, EZKLError>
where
    Scheme::Scalar: FromUniformBytes<64>
        + SerdeObject
        + Serialize
        + DeserializeOwned
        + WithSmallOrderMulGroup<3>,
    Scheme::Curve: SerdeObject + Serialize + DeserializeOwned,
    Scheme::ParamsVerifier: 'a,
{
    let proof = Snark::load::<Scheme>(&proof_path)?;

    let strategy = Strategy::new(params);
    let vk = load_vk::<Scheme, C>(vk_path, settings)?;
    let now = Instant::now();

    let result =
        verify_proof_circuit::<V, _, _, _, TR>(&proof, params, &vk, strategy, 1 << logrows);

    let elapsed = now.elapsed();
    info!(
        "verify took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    info!("verified: {}", result.is_ok());
    result.map_err(|e: plonk::Error| e.into()).map(|_| true)
}

pub(crate) fn verify_aggr(
    proof_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    logrows: u32,
    reduced_srs: bool,
    commitment: Commitments,
) -> Result<bool, EZKLError> {
    match commitment {
        Commitments::KZG => {
            let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
            let params: ParamsKZG<Bn256> = if reduced_srs {
                // only need G_0 for the verification with shplonk
                load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, 1, Commitments::KZG)?
            } else {
                load_params_verifier::<KZGCommitmentScheme<Bn256>>(
                    srs_path,
                    logrows,
                    Commitments::KZG,
                )?
            };
            match proof.transcript_type {
                TranscriptType::EVM => verify_commitment::<
                    KZGCommitmentScheme<Bn256>,
                    VerifierSHPLONK<'_, Bn256>,
                    _,
                    KZGSingleStrategy<_>,
                    EvmTranscript<_, _, _, _>,
                    AggregationCircuit,
                    _,
                >(proof_path, (), vk_path, &params, logrows),
                TranscriptType::Poseidon => {
                    verify_commitment::<
                        KZGCommitmentScheme<Bn256>,
                        VerifierSHPLONK<'_, Bn256>,
                        _,
                        KZGAccumulatorStrategy<_>,
                        PoseidonTranscript<NativeLoader, _>,
                        AggregationCircuit,
                        _,
                    >(proof_path, (), vk_path, &params, logrows)
                }
            }
        }
        Commitments::IPA => {
            let proof = Snark::load::<IPACommitmentScheme<G1Affine>>(&proof_path)?;
            let params: ParamsIPA<_> = load_params_verifier::<IPACommitmentScheme<G1Affine>>(
                srs_path,
                logrows,
                Commitments::IPA,
            )?;
            match proof.transcript_type {
                TranscriptType::EVM => verify_commitment::<
                    IPACommitmentScheme<G1Affine>,
                    VerifierIPA<_>,
                    _,
                    IPASingleStrategy<_>,
                    EvmTranscript<_, _, _, _>,
                    AggregationCircuit,
                    _,
                >(proof_path, (), vk_path, &params, logrows),
                TranscriptType::Poseidon => {
                    verify_commitment::<
                        IPACommitmentScheme<G1Affine>,
                        VerifierIPA<_>,
                        _,
                        IPAAccumulatorStrategy<_>,
                        PoseidonTranscript<NativeLoader, _>,
                        AggregationCircuit,
                        _,
                    >(proof_path, (), vk_path, &params, logrows)
                }
            }
        }
    }
}

/// helper function for load_params
pub(crate) fn load_params_verifier<Scheme: CommitmentScheme>(
    srs_path: Option<PathBuf>,
    logrows: u32,
    commitment: Commitments,
) -> Result<Scheme::ParamsVerifier, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path, commitment);
    let mut params = load_srs_verifier::<Scheme>(srs_path)?;
    if logrows < params.k() {
        info!("downsizing params to {} logrows", logrows);
        params.downsize(logrows);
    }
    Ok(params)
}

/// helper function for load_params
pub(crate) fn load_params_prover<Scheme: CommitmentScheme>(
    srs_path: Option<PathBuf>,
    logrows: u32,
    commitment: Commitments,
) -> Result<Scheme::ParamsProver, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path, commitment);
    let mut params = load_srs_prover::<Scheme>(srs_path)?;
    if logrows < params.k() {
        info!("downsizing params to {} logrows", logrows);
        params.downsize(logrows);
    }
    Ok(params)
}
