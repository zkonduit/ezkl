/// EVM related proving and verification
pub mod evm;

/// SRS generation, processing, verification and downloading
pub mod srs;

/// errors related to pfsys
pub mod errors;

pub use errors::PfsysError;

use crate::circuit::CheckMode;
use crate::graph::GraphWitness;
use crate::pfsys::evm::aggregation_kzg::PoseidonTranscript;
use crate::{Commitments, EZKL_BUF_CAPACITY, EZKL_KEY_FORMAT};
use clap::ValueEnum;
use halo2_proofs::circuit::Value;
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk_custom, verify_proof, Circuit, ProvingKey, VerifyingKey,
};
use halo2_proofs::poly::commitment::{CommitmentScheme, Params, ParamsProver, Prover, Verifier};
use halo2_proofs::poly::ipa::commitment::IPACommitmentScheme;
use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
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
use snark_verifier::verifier::plonk::PlonkProtocol;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Write};
use std::ops::Deref;
use std::path::PathBuf;
use thiserror::Error as thisError;
use tosubcommand::ToFlags;

use halo2curves::bn256::{Bn256, Fr, G1Affine};

fn serde_format_from_str(s: &str) -> halo2_proofs::SerdeFormat {
    match s {
        "processed" => halo2_proofs::SerdeFormat::Processed,
        "raw-bytes-unchecked" => halo2_proofs::SerdeFormat::RawBytesUnchecked,
        "raw-bytes" => halo2_proofs::SerdeFormat::RawBytes,
        _ => panic!("invalid serde format"),
    }
}

#[allow(missing_docs)]
#[derive(
    ValueEnum, Copy, Clone, Default, Debug, PartialEq, Eq, Deserialize, Serialize, PartialOrd,
)]
pub enum ProofType {
    #[default]
    Single,
    ForAggr,
}

impl std::fmt::Display for ProofType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ProofType::Single => "single",
                ProofType::ForAggr => "for-aggr",
            }
        )
    }
}

impl ToFlags for ProofType {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
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
#[derive(
    ValueEnum, Default, Copy, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, PartialOrd,
)]
pub enum TranscriptType {
    Poseidon,
    #[default]
    EVM,
}

impl std::fmt::Display for TranscriptType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TranscriptType::Poseidon => "poseidon",
                TranscriptType::EVM => "evm",
            }
        )
    }
}

impl ToFlags for TranscriptType {
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
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
    let g1affine_x = field_to_string(&g1affine.x);
    let g1affine_y = field_to_string(&g1affine.y);
    g1affine_dict.set_item("x", g1affine_x).unwrap();
    g1affine_dict.set_item("y", g1affine_y).unwrap();
}

#[cfg(feature = "python-bindings")]
use halo2curves::bn256::G1;
#[cfg(feature = "python-bindings")]
///
pub fn g1_to_pydict(g1_dict: &PyDict, g1: &G1) {
    let g1_x = field_to_string(&g1.x);
    let g1_y = field_to_string(&g1.y);
    let g1_z = field_to_string(&g1.z);
    g1_dict.set_item("x", g1_x).unwrap();
    g1_dict.set_item("y", g1_y).unwrap();
    g1_dict.set_item("z", g1_z).unwrap();
}

/// converts fp into a little endian Hex string
pub fn field_to_string<F: PrimeField + SerdeObject + Serialize>(fp: &F) -> String {
    let repr = serde_json::to_string(&fp).unwrap();
    let b: String = serde_json::from_str(&repr).unwrap();
    b
}

/// converts a little endian Hex string into a field element
pub fn string_to_field<F: PrimeField + SerdeObject + Serialize + DeserializeOwned>(
    b: &String,
) -> F {
    let repr = serde_json::to_string(&b).unwrap();
    let fp: F = serde_json::from_str(&repr).unwrap();
    fp
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
/// Contains the instances of the circuit in human readable form
pub struct PrettyElements {
    /// the inputs as rescaled floats -- represented as a String for maximum compatibility with Python and JS
    pub rescaled_inputs: Vec<Vec<String>>,
    /// the inputs as felts but 0x strings -- represented as a String for maximum compatibility with Python and JS
    pub inputs: Vec<Vec<String>>,
    /// the processed inputs (eg. hash of the inputs) -- stays as a felt represented as a 0x string for maximum compatibility with Python and JS
    pub processed_inputs: Vec<Vec<String>>,
    /// the processed params (eg. hash of the params) -- stays as a felt represented as a 0x string for maximum compatibility with Python and JS
    pub processed_params: Vec<Vec<String>>,
    /// the processed outputs (eg. hash of the outputs) -- stays as a felt represented as a 0x string for maximum compatibility with Python and JS
    pub processed_outputs: Vec<Vec<String>>,
    /// the outputs as rescaled floats (if any) -- represented as a String for maximum compatibility with Python and JS
    pub rescaled_outputs: Vec<Vec<String>>,
    /// the outputs as felts but 0x strings (if any) -- represented as a String for maximum compatibility with Python and JS
    pub outputs: Vec<Vec<String>>,
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
    /// hex encoded proof
    pub hex_proof: Option<String>,
    /// transcript type
    pub transcript_type: TranscriptType,
    /// the split proof
    pub split: Option<ProofSplitCommit>,
    /// the proof instances as rescaled floats
    pub pretty_public_inputs: Option<PrettyElements>,
    /// timestamp
    pub timestamp: Option<u128>,
    /// commitment
    pub commitment: Option<Commitments>,
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
        let field_elems: Vec<Vec<String>> = self
            .instances
            .iter()
            .map(|x| x.iter().map(|fp| field_to_string(fp)).collect())
            .collect::<Vec<_>>();
        dict.set_item("instances", field_elems).unwrap();
        let hex_proof = hex::encode(&self.proof);
        dict.set_item("proof", format!("0x{}", hex_proof)).unwrap();
        dict.set_item("transcript_type", self.transcript_type)
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
        protocol: Option<PlonkProtocol<C>>,
        instances: Vec<Vec<F>>,
        proof: Vec<u8>,
        hex_proof: Option<String>,
        transcript_type: TranscriptType,
        split: Option<ProofSplitCommit>,
        pretty_public_inputs: Option<PrettyElements>,
        commitment: Option<Commitments>,
    ) -> Self {
        Self {
            protocol,
            instances,
            proof,
            hex_proof,
            transcript_type,
            split,
            pretty_public_inputs,
            // unix timestamp
            timestamp: Some(
                instant::SystemTime::now()
                    .duration_since(instant::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            ),
            commitment,
        }
    }

    /// create hex proof from proof
    pub fn create_hex_proof(&mut self) {
        let hex_proof = hex::encode(&self.proof);
        self.hex_proof = Some(format!("0x{}", hex_proof));
    }

    /// Saves the Proof to a specified `proof_path`.
    pub fn save(&self, proof_path: &PathBuf) -> Result<(), PfsysError> {
        let file = std::fs::File::create(proof_path)
            .map_err(|e| PfsysError::SaveProof(format!("{}", e)))?;
        let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, file);
        serde_json::to_writer(&mut writer, &self)
            .map_err(|e| PfsysError::SaveProof(format!("{}", e)))?;
        Ok(())
    }

    /// Load a json serialized proof from the provided path.
    pub fn load<Scheme: CommitmentScheme<Curve = C, Scalar = F>>(
        proof_path: &PathBuf,
    ) -> Result<Self, PfsysError>
    where
        <C as CurveAffine>::ScalarExt: FromUniformBytes<64>,
    {
        trace!("reading proof");
        let file =
            std::fs::File::open(proof_path).map_err(|e| PfsysError::LoadProof(format!("{}", e)))?;
        let reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, file);
        let proof: Self =
            serde_json::from_reader(reader).map_err(|e| PfsysError::LoadProof(format!("{}", e)))?;
        Ok(proof)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A proof split commit
pub struct ProofSplitCommit {
    /// The start index of the output in the witness
    start: usize,
    /// The end index of the output in the witness
    end: usize,
}

impl From<GraphWitness> for Option<ProofSplitCommit> {
    fn from(witness: GraphWitness) -> Self {
        let mut elem_offset = 0;

        if let Some(input) = witness.processed_inputs {
            if let Some(polycommit) = input.polycommit {
                // flatten and count number of elements
                let num_elements = polycommit
                    .iter()
                    .map(|polycommit| polycommit.len())
                    .sum::<usize>();

                elem_offset += num_elements;
            }
        }

        if let Some(params) = witness.processed_params {
            if let Some(polycommit) = params.polycommit {
                // flatten and count number of elements
                let num_elements = polycommit
                    .iter()
                    .map(|polycommit| polycommit.len())
                    .sum::<usize>();

                elem_offset += num_elements;
            }
        }

        if let Some(output) = witness.processed_outputs {
            if let Some(polycommit) = output.polycommit {
                // flatten and count number of elements
                let num_elements = polycommit
                    .iter()
                    .map(|polycommit| polycommit.len())
                    .sum::<usize>();

                Some(ProofSplitCommit {
                    start: elem_offset,
                    end: elem_offset + num_elements,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// An application snark with proof and instance variables ready for aggregation (wrapped field element)
#[derive(Clone, Debug)]
pub struct SnarkWitness<F: PrimeField, C: CurveAffine> {
    protocol: Option<PlonkProtocol<C>>,
    instances: Vec<Vec<Value<F>>>,
    proof: Value<Vec<u8>>,
    split: Option<ProofSplitCommit>,
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
            split: self.split.clone(),
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
            split: snark.split,
        }
    }
}

/// Creates a [VerifyingKey] and [ProvingKey] for a [crate::graph::GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
pub fn create_keys<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
    disable_selector_compression: bool,
) -> Result<ProvingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
where
    C: Circuit<Scheme::Scalar>,
    <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
{
    //	Real proof
    let empty_circuit = <C as Circuit<Scheme::Scalar>>::without_witnesses(circuit);

    // Initialize verifying key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk_custom(params, &empty_circuit, !disable_selector_compression)?;
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
#[allow(clippy::too_many_arguments)]
pub fn create_proof_circuit<
    'params,
    Scheme: CommitmentScheme,
    C: Circuit<Scheme::Scalar>,
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
    check_mode: CheckMode,
    commitment: Commitments,
    transcript_type: TranscriptType,
    split: Option<ProofSplitCommit>,
    protocol: Option<PlonkProtocol<Scheme::Curve>>,
) -> Result<Snark<Scheme::Scalar, Scheme::Curve>, PfsysError>
where
    Scheme::ParamsVerifier: 'params,
    Scheme::Scalar: Serialize
        + DeserializeOwned
        + SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>,
    Scheme::Curve: Serialize + DeserializeOwned,
{
    let strategy = Strategy::new(params.verifier_params());
    let mut transcript = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);
    #[cfg(feature = "det-prove")]
    let mut rng = <StdRng as rand::SeedableRng>::from_seed([0u8; 32]);
    #[cfg(not(feature = "det-prove"))]
    let mut rng = OsRng;

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
    let hex_proof = format!("0x{}", hex::encode(&proof));

    let checkable_pf = Snark::new(
        protocol,
        instances,
        proof,
        Some(hex_proof),
        transcript_type,
        split,
        None,
        Some(commitment),
    );

    // sanity check that the generated proof is valid
    if check_mode == CheckMode::SAFE {
        debug!("verifying generated proof");
        let verifier_params = params.verifier_params();
        verify_proof_circuit::<V, Scheme, Strategy, E, TR>(
            &checkable_pf,
            verifier_params,
            pk.get_vk(),
            strategy,
            verifier_params.n(),
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

/// Swaps the proof commitments to a new set in the proof
pub fn swap_proof_commitments<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
>(
    snark: &Snark<Scheme::Scalar, Scheme::Curve>,
    commitments: &[Scheme::Curve],
) -> Result<Snark<Scheme::Scalar, Scheme::Curve>, PfsysError>
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
    let proof_first_bytes = get_proof_commitments::<Scheme, E, TW>(commitments)?;

    let mut snark_new = snark.clone();
    // swap the proof bytes for the new ones
    snark_new.proof[..proof_first_bytes.len()].copy_from_slice(&proof_first_bytes);
    snark_new.create_hex_proof();

    Ok(snark_new)
}

/// Returns the bytes encoded proof commitments
pub fn get_proof_commitments<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
>(
    commitments: &[Scheme::Curve],
) -> Result<Vec<u8>, PfsysError>
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
    let mut transcript_new: TW = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);

    // polycommit commitments are the first set of points in the proof, this will always be the first set of advice
    for commit in commitments {
        transcript_new
            .write_point(*commit)
            .map_err(|e| PfsysError::WritePoint(format!("{}", e)))?;
    }

    let proof_first_bytes = transcript_new.finalize();

    if commitments.is_empty() {
        log::warn!("no commitments found in witness");
    }

    Ok(proof_first_bytes)
}

/// Swap the proof commitments to a new set in the proof for KZG
pub fn swap_proof_commitments_polycommit(
    snark: &Snark<Fr, G1Affine>,
    commitments: &[G1Affine],
) -> Result<Snark<Fr, G1Affine>, PfsysError> {
    let proof = match snark.commitment {
        Some(Commitments::KZG) => match snark.transcript_type {
            TranscriptType::EVM => swap_proof_commitments::<
                KZGCommitmentScheme<Bn256>,
                _,
                EvmTranscript<G1Affine, _, _, _>,
            >(snark, commitments)?,
            TranscriptType::Poseidon => swap_proof_commitments::<
                KZGCommitmentScheme<Bn256>,
                _,
                PoseidonTranscript<NativeLoader, _>,
            >(snark, commitments)?,
        },
        Some(Commitments::IPA) => match snark.transcript_type {
            TranscriptType::EVM => swap_proof_commitments::<
                IPACommitmentScheme<G1Affine>,
                _,
                EvmTranscript<G1Affine, _, _, _>,
            >(snark, commitments)?,
            TranscriptType::Poseidon => swap_proof_commitments::<
                IPACommitmentScheme<G1Affine>,
                _,
                PoseidonTranscript<NativeLoader, _>,
            >(snark, commitments)?,
        },
        None => {
            return Err(PfsysError::InvalidCommitmentScheme);
        }
    };

    Ok(proof)
}

/// A wrapper around halo2's verify_proof
pub fn verify_proof_circuit<
    'params,
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
    orig_n: u64,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error>
where
    Scheme::Scalar: SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>
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
    verify_proof::<Scheme, V, _, TR, _>(params, vk, strategy, instances, &mut transcript, orig_n)
}

/// Loads a [VerifyingKey] at `path`.
pub fn load_vk<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
) -> Result<VerifyingKey<Scheme::Curve>, PfsysError>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    debug!("loading verification key from {:?}", path);
    let f = File::open(path.clone()).map_err(|e| PfsysError::LoadVk(format!("{}", e)))?;
    let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
    let vk = VerifyingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        serde_format_from_str(&EZKL_KEY_FORMAT),
        params,
    )
    .map_err(|e| PfsysError::LoadVk(format!("{}", e)))?;
    info!("loaded verification key ✅");
    Ok(vk)
}

/// Loads a [ProvingKey] at `path`.
pub fn load_pk<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
) -> Result<ProvingKey<Scheme::Curve>, PfsysError>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    debug!("loading proving key from {:?}", path);
    let f = File::open(path.clone()).map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
    let pk = ProvingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        serde_format_from_str(&EZKL_KEY_FORMAT),
        params,
    )
    .map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    info!("loaded proving key ✅");
    Ok(pk)
}

/// Saves a [ProvingKey] to `path`.
pub fn save_pk<C: SerdeObject + CurveAffine>(
    path: &PathBuf,
    pk: &ProvingKey<C>,
) -> Result<(), io::Error>
where
    C::ScalarExt: FromUniformBytes<64> + SerdeObject,
{
    debug!("saving proving key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
    pk.write(&mut writer, serde_format_from_str(&EZKL_KEY_FORMAT))?;
    writer.flush()?;
    info!("done saving proving key ✅");
    Ok(())
}

/// Saves a [VerifyingKey] to `path`.
pub fn save_vk<C: CurveAffine + SerdeObject>(
    path: &PathBuf,
    vk: &VerifyingKey<C>,
) -> Result<(), io::Error>
where
    C::ScalarExt: FromUniformBytes<64> + SerdeObject,
{
    debug!("saving verification key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
    vk.write(&mut writer, serde_format_from_str(&EZKL_KEY_FORMAT))?;
    writer.flush()?;
    info!("done saving verification key ✅");
    Ok(())
}

/// Saves [CommitmentScheme] parameters to `path`.
pub fn save_params<Scheme: CommitmentScheme>(
    path: &PathBuf,
    params: &'_ Scheme::ParamsVerifier,
) -> Result<(), io::Error> {
    debug!("saving parameters 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
    params.write(&mut writer)?;
    writer.flush()?;
    Ok(())
}

////////////////////////

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {

    use super::*;
    use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use tempfile::Builder;

    #[tokio::test]
    async fn test_can_load_saved_srs() {
        let tmp_dir = Builder::new().prefix("example").tempdir().unwrap();
        let fname = tmp_dir.path().join("polycommit.params");
        let srs = srs::gen_srs::<KZGCommitmentScheme<Bn256>>(1);
        let res = save_params::<KZGCommitmentScheme<Bn256>>(&fname, &srs);
        assert!(res.is_ok());
        let res = srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(fname);
        assert!(res.is_ok())
    }

    #[test]
    fn test_snark_serialization_roundtrip() {
        let snark = Snark::<Fr, G1Affine> {
            proof: vec![1, 2, 3, 4, 5, 6, 7, 8],
            instances: vec![vec![Fr::from(1)], vec![Fr::from(2)]],
            transcript_type: TranscriptType::EVM,
            protocol: None,
            hex_proof: None,
            split: None,
            pretty_public_inputs: None,
            timestamp: None,
            commitment: None,
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
