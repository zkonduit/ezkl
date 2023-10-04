/// EVM related proving and verification
pub mod evm;

/// SRS generation, processing, verification and downloading
pub mod srs;

use crate::circuit::CheckMode;
use crate::pfsys::evm::aggregation::PoseidonTranscript;
use crate::tensor::TensorType;
use clap::ValueEnum;
use halo2_proofs::circuit::Value;
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey,
};
use halo2_proofs::poly::commitment::{CommitmentScheme, Params, ParamsProver, Prover, Verifier};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverGWC, VerifierGWC};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer};
use halo2curves::ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
use halo2curves::CurveAffine;
use instant::Instant;
use log::{debug, info, trace};
#[cfg(not(feature = "det-prove"))]
use rand::rngs::OsRng;
#[cfg(feature = "det-prove")]
use rand::rngs::StdRng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use snark_verifier::system::halo2::{compile, Config};
use snark_verifier::verifier::plonk::PlonkProtocol;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Write};
use std::ops::Deref;
use std::path::PathBuf;
use thiserror::Error as thisError;

use halo2curves::bn256::{Bn256, Fr, G1Affine};

#[allow(missing_docs)]
#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum ProofType {
    Single,
    ForAggr,
}

impl From<ProofType> for TranscriptType {
    fn from(val: ProofType) -> Self {
        match val {
            ProofType::Single => TranscriptType::EVM,
            ProofType::ForAggr => TranscriptType::Poseidon,
        }
    }
}

impl From<ProofType> for StrategyType {
    fn from(val: ProofType) -> Self {
        match val {
            ProofType::Single => StrategyType::Single,
            ProofType::ForAggr => StrategyType::Accum,
        }
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for ProofType {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            ProofType::Single => "Single".to_object(py),
            ProofType::ForAggr => "ForAggr".to_object(py),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains StrategyType from PyObject (Required for StrategyType to be compatible with Python)
impl<'source> pyo3::FromPyObject<'source> for ProofType {
    fn extract(ob: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
        let trystr = <pyo3::types::PyString as pyo3::PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();
        match strval.to_lowercase().as_str() {
            "single" => Ok(ProofType::Single),
            "for-aggr" => Ok(ProofType::ForAggr),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid value for ProofType",
            )),
        }
    }
}

#[allow(missing_docs)]
#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum StrategyType {
    Single,
    Accum,
}
impl std::fmt::Display for StrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_possible_value()
            .expect("no values are skipped")
            .get_name()
            .fmt(f)
    }
}
#[cfg(feature = "python-bindings")]
/// Converts StrategyType into a PyObject (Required for StrategyType to be compatible with Python)
impl pyo3::IntoPy<PyObject> for StrategyType {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            StrategyType::Single => "single".to_object(py),
            StrategyType::Accum => "accum".to_object(py),
        }
    }
}
#[cfg(feature = "python-bindings")]
/// Obtains StrategyType from PyObject (Required for StrategyType to be compatible with Python)
impl<'source> pyo3::FromPyObject<'source> for StrategyType {
    fn extract(ob: &'source pyo3::PyAny) -> pyo3::PyResult<Self> {
        let trystr = <pyo3::types::PyString as pyo3::PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();
        match strval.to_lowercase().as_str() {
            "single" => Ok(StrategyType::Single),
            "accum" => Ok(StrategyType::Accum),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid value for StrategyType",
            )),
        }
    }
}

#[derive(thisError, Debug)]
/// Errors related to pfsys
pub enum PfSysError {
    /// Packing exponent is too large
    #[error("largest packing exponent exceeds max. try reducing the scale")]
    PackingExponent,
}

#[allow(missing_docs)]
#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum TranscriptType {
    Poseidon,
    EVM,
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for TranscriptType {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            TranscriptType::Poseidon => "Poseidon".to_object(py),
            TranscriptType::EVM => "EVM".to_object(py),
        }
    }
}

#[cfg(feature = "python-bindings")]
///
pub fn g1affine_to_pydict(g1affine_dict: &PyDict, g1affine: &G1Affine) {
    let g1affine_x = field_to_vecu64_montgomery(&g1affine.x);
    let g1affine_y = field_to_vecu64_montgomery(&g1affine.y);
    g1affine_dict.set_item("x", g1affine_x).unwrap();
    g1affine_dict.set_item("y", g1affine_y).unwrap();
}

#[cfg(feature = "python-bindings")]
use halo2curves::bn256::G1;
#[cfg(feature = "python-bindings")]
///
pub fn g1_to_pydict(g1_dict: &PyDict, g1: &G1) {
    let g1_x = field_to_vecu64_montgomery(&g1.x);
    let g1_y = field_to_vecu64_montgomery(&g1.y);
    let g1_z = field_to_vecu64_montgomery(&g1.z);
    g1_dict.set_item("x", g1_x).unwrap();
    g1_dict.set_item("y", g1_y).unwrap();
    g1_dict.set_item("z", g1_z).unwrap();
}

/// converts fp into `Vec<u64>` in Montgomery form
pub fn field_to_vecu64_montgomery<F: PrimeField + SerdeObject + Serialize>(fp: &F) -> [u64; 4] {
    let repr = serde_json::to_string(&fp).unwrap();
    let b: [u64; 4] = serde_json::from_str(&repr).unwrap();
    b
}

/// converts `Vec<u64>` in Montgomery form into fp
pub fn vecu64_to_field_montgomery<F: PrimeField + SerdeObject + Serialize + DeserializeOwned>(
    b: &[u64; 4],
) -> F {
    let repr = serde_json::to_string(&b).unwrap();
    let fp: F = serde_json::from_str(&repr).unwrap();
    fp
}

/// An application snark with proof and instance variables ready for aggregation (raw field element)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snark<F: PrimeField + SerdeObject, C: CurveAffine>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    /// the protocol
    pub protocol: Option<PlonkProtocol<C>>,
    /// public instances of the snark
    pub instances: Vec<Vec<F>>,
    /// the proof
    pub proof: Vec<u8>,
    /// transcript type
    pub transcript_type: TranscriptType,
}

#[cfg(feature = "python-bindings")]
use pyo3::{types::PyDict, PyObject, Python, ToPyObject};
#[cfg(feature = "python-bindings")]
impl<F: PrimeField + SerdeObject + Serialize, C: CurveAffine + Serialize> ToPyObject for Snark<F, C>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        let field_elems: Vec<Vec<[u64; 4]>> = self
            .instances
            .iter()
            .map(|x| x.iter().map(|fp| field_to_vecu64_montgomery(fp)).collect())
            .collect::<Vec<_>>();
        dict.set_item("instances", &field_elems).unwrap();
        let hex_proof = hex::encode(&self.proof);
        dict.set_item("proof", &hex_proof).unwrap();
        dict.set_item("transcript_type", &self.transcript_type)
            .unwrap();
        dict.to_object(py)
    }
}

impl<
        F: PrimeField + SerdeObject + Serialize + FromUniformBytes<64> + DeserializeOwned,
        C: CurveAffine + Serialize + DeserializeOwned,
    > Snark<F, C>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    /// Create a new application snark from proof and instance variables ready for aggregation
    pub fn new(
        protocol: PlonkProtocol<C>,
        instances: Vec<Vec<F>>,
        proof: Vec<u8>,
        transcript_type: TranscriptType,
    ) -> Self {
        Self {
            protocol: Some(protocol),
            instances,
            proof,
            transcript_type,
        }
    }

    /// Saves the Proof to a specified `proof_path`.
    pub fn save(&self, proof_path: &PathBuf) -> Result<(), Box<dyn Error>> {
        let file = std::fs::File::create(proof_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &self)?;
        Ok(())
    }

    /// Load a json serialized proof from the provided path.
    pub fn load<Scheme: CommitmentScheme<Curve = C, Scalar = F>>(
        proof_path: &PathBuf,
    ) -> Result<Self, Box<dyn Error>>
    where
        <C as CurveAffine>::ScalarExt: FromUniformBytes<64>,
    {
        trace!("reading proof");
        let data = std::fs::read_to_string(proof_path)?;
        serde_json::from_str(&data).map_err(|e| e.into())
    }
}

/// An application snark with proof and instance variables ready for aggregation (wrapped field element)
#[derive(Clone, Debug)]
pub struct SnarkWitness<F: PrimeField, C: CurveAffine> {
    protocol: Option<PlonkProtocol<C>>,
    instances: Vec<Vec<Value<F>>>,
    proof: Value<Vec<u8>>,
}

impl<F: PrimeField, C: CurveAffine> SnarkWitness<F, C> {
    fn without_witnesses(&self) -> Self {
        SnarkWitness {
            protocol: self.protocol.clone(),
            instances: self
                .instances
                .iter()
                .map(|instances| vec![Value::unknown(); instances.len()])
                .collect(),
            proof: Value::unknown(),
        }
    }

    fn proof(&self) -> Value<&[u8]> {
        self.proof.as_ref().map(Vec::as_slice)
    }
}

impl<F: PrimeField + SerdeObject, C: CurveAffine> From<Snark<F, C>> for SnarkWitness<F, C>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    fn from(snark: Snark<F, C>) -> Self {
        Self {
            protocol: snark.protocol,
            instances: snark
                .instances
                .into_iter()
                .map(|instances| instances.into_iter().map(Value::known).collect())
                .collect(),
            proof: Value::known(snark.proof),
        }
    }
}

/// Creates a [VerifyingKey] and [ProvingKey] for a [crate::graph::GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
pub fn create_keys<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
) -> Result<ProvingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
where
    C: Circuit<Scheme::Scalar>,
    <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
{
    //	Real proof
    let empty_circuit = <C as Circuit<F>>::without_witnesses(circuit);

    // Initialize verifying key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk(params, &empty_circuit)?;
    let elapsed = now.elapsed();
    info!("VK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());

    // Initialize the proving key
    let now = Instant::now();
    let pk = keygen_pk(params, vk, &empty_circuit)?;
    let elapsed = now.elapsed();
    info!("PK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());
    Ok(pk)
}

/// a wrapper around halo2's create_proof
pub fn create_proof_circuit<
    'params,
    Scheme: CommitmentScheme,
    F: PrimeField + TensorType,
    C: Circuit<F>,
    P: Prover<'params, Scheme>,
    V: Verifier<'params, Scheme>,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
>(
    circuit: C,
    instances: Vec<Vec<Scheme::Scalar>>,
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    strategy: Strategy,
    check_mode: CheckMode,
    transcript_type: TranscriptType,
) -> Result<Snark<Scheme::Scalar, Scheme::Curve>, Box<dyn Error>>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::ParamsVerifier: 'params,
    Scheme::Scalar: Serialize
        + DeserializeOwned
        + SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>
        + Ord,
    Scheme::Curve: Serialize + DeserializeOwned,
{
    let mut transcript = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);
    #[cfg(feature = "det-prove")]
    let mut rng = <StdRng as rand::SeedableRng>::from_seed([0u8; 32]);
    #[cfg(not(feature = "det-prove"))]
    let mut rng = OsRng;
    let number_instance = instances.iter().map(|x| x.len()).collect();
    trace!("number_instance {:?}", number_instance);
    let protocol = compile(
        params,
        pk.get_vk(),
        Config::kzg().with_num_instance(number_instance),
    );

    let pi_inner = instances
        .iter()
        .map(|e| e.deref())
        .collect::<Vec<&[Scheme::Scalar]>>();
    let pi_inner: &[&[&[Scheme::Scalar]]] = &[&pi_inner];
    trace!("instances {:?}", instances);
    trace!(
        "pk num instance column: {:?}",
        pk.get_vk().cs().num_instance_columns()
    );

    info!("proof started...");
    // not wasm32 unknown
    let now = Instant::now();

    create_proof::<Scheme, P, _, _, TW, _>(
        params,
        pk,
        &[circuit],
        pi_inner,
        &mut rng,
        &mut transcript,
    )?;
    let proof = transcript.finalize();

    let checkable_pf = Snark::new(protocol, instances, proof, transcript_type);

    // sanity check that the generated proof is valid
    if check_mode == CheckMode::SAFE {
        debug!("verifying generated proof");
        let verifier_params = params.verifier_params();
        verify_proof_circuit::<F, V, Scheme, Strategy, E, TR>(
            &checkable_pf,
            verifier_params,
            pk.get_vk(),
            strategy,
        )?;
    }
    let elapsed = now.elapsed();
    info!(
        "proof took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    Ok(checkable_pf)
}

/// A wrapper around halo2's verify_proof
pub fn verify_proof_circuit<
    'params,
    F: PrimeField,
    V: Verifier<'params, Scheme>,
    Scheme: CommitmentScheme,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    E: EncodedChallenge<Scheme::Curve>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
>(
    snark: &Snark<Scheme::Scalar, Scheme::Curve>,
    params: &'params Scheme::ParamsVerifier,
    vk: &VerifyingKey<Scheme::Curve>,
    strategy: Strategy,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error>
where
    Scheme::Scalar: SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>
        + Ord
        + Serialize
        + DeserializeOwned,
    Scheme::Curve: Serialize + DeserializeOwned,
{
    let pi_inner = snark
        .instances
        .iter()
        .map(|e| e.deref())
        .collect::<Vec<&[Scheme::Scalar]>>();
    let instances: &[&[&[Scheme::Scalar]]] = &[&pi_inner];
    trace!("instances {:?}", instances);

    let mut transcript = TranscriptReadBuffer::init(Cursor::new(snark.proof.clone()));
    verify_proof::<Scheme, V, _, TR, _>(params, vk, strategy, instances, &mut transcript)
}

/// Loads a [VerifyingKey] at `path`.
pub fn load_vk<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
) -> Result<VerifyingKey<Scheme::Curve>, Box<dyn Error>>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    info!("loading verification key from {:?}", path);
    let f =
        File::open(path.clone()).map_err(|_| format!("failed to load vk at {}", path.display()))?;
    let mut reader = BufReader::new(f);
    VerifyingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        params,
    )
    .map_err(Box::<dyn Error>::from)
}

/// Loads a [ProvingKey] at `path`.
pub fn load_pk<Scheme: CommitmentScheme, F: PrimeField + TensorType, C: Circuit<F>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
) -> Result<ProvingKey<Scheme::Curve>, Box<dyn Error>>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    info!("loading proving key from {:?}", path);
    let f =
        File::open(path.clone()).map_err(|_| format!("failed to load pk at {}", path.display()))?;
    let mut reader = BufReader::new(f);
    ProvingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        halo2_proofs::SerdeFormat::RawBytes,
        params,
    )
    .map_err(Box::<dyn Error>::from)
}

/// Saves a [ProvingKey] to `path`.
pub fn save_pk<Scheme: CommitmentScheme>(
    path: &PathBuf,
    vk: &ProvingKey<Scheme::Curve>,
) -> Result<(), io::Error>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    info!("saving proving key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);
    vk.write(&mut writer, halo2_proofs::SerdeFormat::RawBytes)?;
    writer.flush()?;
    Ok(())
}

/// Saves a [VerifyingKey] to `path`.
pub fn save_vk<Scheme: CommitmentScheme>(
    path: &PathBuf,
    vk: &VerifyingKey<Scheme::Curve>,
) -> Result<(), io::Error>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    info!("saving verification key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);
    vk.write(&mut writer, halo2_proofs::SerdeFormat::RawBytes)?;
    writer.flush()?;
    Ok(())
}

/// Saves [CommitmentScheme] parameters to `path`.
pub fn save_params<Scheme: CommitmentScheme>(
    path: &PathBuf,
    params: &'_ Scheme::ParamsVerifier,
) -> Result<(), io::Error> {
    info!("saving parameters 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);
    params.write(&mut writer)?;
    writer.flush()?;
    Ok(())
}

/// helper function
pub fn create_proof_circuit_kzg<
    'params,
    C: Circuit<Fr>,
    Strategy: VerificationStrategy<'params, KZGCommitmentScheme<Bn256>, VerifierGWC<'params, Bn256>>,
>(
    circuit: C,
    params: &'params ParamsKZG<Bn256>,
    public_inputs: Vec<Fr>,
    pk: &ProvingKey<G1Affine>,
    transcript: TranscriptType,
    strategy: Strategy,
    check_mode: CheckMode,
) -> Result<Snark<Fr, G1Affine>, Box<dyn Error>> {
    let public_inputs = if !public_inputs.is_empty() {
        vec![public_inputs]
    } else {
        vec![]
    };

    match transcript {
        TranscriptType::EVM => create_proof_circuit::<
            KZGCommitmentScheme<_>,
            Fr,
            _,
            ProverGWC<_>,
            VerifierGWC<_>,
            _,
            _,
            EvmTranscript<G1Affine, _, _, _>,
            EvmTranscript<G1Affine, _, _, _>,
        >(
            circuit,
            public_inputs,
            params,
            pk,
            strategy,
            check_mode,
            transcript,
        )
        .map_err(Box::<dyn Error>::from),
        TranscriptType::Poseidon => create_proof_circuit::<
            KZGCommitmentScheme<_>,
            Fr,
            _,
            ProverGWC<_>,
            VerifierGWC<_>,
            _,
            _,
            PoseidonTranscript<NativeLoader, _>,
            PoseidonTranscript<NativeLoader, _>,
        >(
            circuit,
            public_inputs,
            params,
            pk,
            strategy,
            check_mode,
            transcript,
        )
        .map_err(Box::<dyn Error>::from),
    }
}

#[allow(unused)]
/// helper function
pub(crate) fn verify_proof_circuit_kzg<
    'params,
    Strategy: VerificationStrategy<'params, KZGCommitmentScheme<Bn256>, VerifierGWC<'params, Bn256>>,
>(
    params: &'params ParamsKZG<Bn256>,
    proof: Snark<Fr, G1Affine>,
    vk: &VerifyingKey<G1Affine>,
    strategy: Strategy,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error> {
    match proof.transcript_type {
        TranscriptType::EVM => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            _,
            EvmTranscript<G1Affine, _, _, _>,
        >(&proof, params, vk, strategy),
        TranscriptType::Poseidon => verify_proof_circuit::<
            Fr,
            VerifierGWC<'_, Bn256>,
            _,
            _,
            _,
            PoseidonTranscript<NativeLoader, _>,
        >(&proof, params, vk, strategy),
    }
}

////////////////////////

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use std::io::copy;

    use super::*;
    use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use tempfile::Builder;

    #[tokio::test]
    async fn test_can_load_pre_generated_srs() {
        let tmp_dir = Builder::new().prefix("example").tempdir().unwrap();
        // lets hope this link never rots
        let target = "https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/hermez-raw-1";
        let response = reqwest::get(target).await.unwrap();

        let fname = response
            .url()
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|name| if name.is_empty() { None } else { Some(name) })
            .unwrap_or("tmp.bin");

        info!("file to download: '{}'", fname);
        let fname = tmp_dir.path().join(fname);
        info!("will be located under: '{:?}'", fname);
        let mut dest = File::create(fname.clone()).unwrap();
        let content = response.bytes().await.unwrap();
        copy(&mut &content[..], &mut dest).unwrap();
        let res = srs::load_srs::<KZGCommitmentScheme<Bn256>>(fname);
        assert!(res.is_ok())
    }

    #[tokio::test]
    async fn test_can_load_saved_srs() {
        let tmp_dir = Builder::new().prefix("example").tempdir().unwrap();
        let fname = tmp_dir.path().join("kzg.params");
        let srs = srs::gen_srs::<KZGCommitmentScheme<Bn256>>(1);
        let res = save_params::<KZGCommitmentScheme<Bn256>>(&fname, &srs);
        assert!(res.is_ok());
        let res = srs::load_srs::<KZGCommitmentScheme<Bn256>>(fname);
        assert!(res.is_ok())
    }

    #[test]
    fn test_snark_serialization_roundtrip() {
        let snark = Snark::<Fr, G1Affine> {
            proof: vec![1, 2, 3, 4, 5, 6, 7, 8],
            instances: vec![vec![Fr::from(1)], vec![Fr::from(2)]],
            transcript_type: TranscriptType::EVM,
            protocol: None,
        };

        snark
            .save(&"test_snark_serialization_roundtrip.json".into())
            .unwrap();
        let snark2 = Snark::<Fr, G1Affine>::load::<KZGCommitmentScheme<Bn256>>(
            &"test_snark_serialization_roundtrip.json".into(),
        )
        .unwrap();
        assert_eq!(snark.instances, snark2.instances);
        assert_eq!(snark.proof, snark2.proof);
        assert_eq!(snark.transcript_type, snark2.transcript_type);
    }
}
