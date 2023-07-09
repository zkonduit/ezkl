/// EVM related proving and verification
pub mod evm;

/// SRS generation, processing, verification and downloading
pub mod srs;

use crate::circuit::CheckMode;
use crate::tensor::TensorType;
use clap::ValueEnum;
use halo2_proofs::circuit::Value;
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey,
};
use halo2_proofs::poly::commitment::{CommitmentScheme, Params, ParamsProver, Prover, Verifier};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer};
use halo2curves::ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
use halo2curves::CurveAffine;
use instant::Instant;
use log::{debug, info, trace};
use rand::rngs::OsRng;
use serde::de::DeserializeOwned;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use snark_verifier::system::halo2::{compile, Config};
use snark_verifier::verifier::plonk::PlonkProtocol;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::PathBuf;
use thiserror::Error as thisError;

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
    Blake,
    Poseidon,
    EVM,
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for TranscriptType {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            TranscriptType::Blake => "Blake".to_object(py),
            TranscriptType::Poseidon => "Poseidon".to_object(py),
            TranscriptType::EVM => "EVM".to_object(py),
        }
    }
}

/// converts fp into Vec<u64>
pub fn field_to_vecu64<F: PrimeField + SerdeObject + Serialize>(fp: &F) -> [u64; 4] {
    let bytes: <F as PrimeField>::Repr = fp.to_repr();
    let bytes_first_u64 = u64::from_le_bytes(bytes.as_ref()[0..8][..].try_into().unwrap());
    let bytes_second_u64 = u64::from_le_bytes(bytes.as_ref()[8..16][..].try_into().unwrap());
    let bytes_third_u64 = u64::from_le_bytes(bytes.as_ref()[16..24][..].try_into().unwrap());
    let bytes_fourth_u64 = u64::from_le_bytes(bytes.as_ref()[24..32][..].try_into().unwrap());

    [
        bytes_first_u64,
        bytes_second_u64,
        bytes_third_u64,
        bytes_fourth_u64,
    ]
}
// consider further restricting the associated type: ` where <F as halo2curves::ff::PrimeField>::Repr: From<[u8; 32]>`
/// convert [u64; 4] into field element
pub fn vecu64_to_field<F: PrimeField + SerdeObject + FromUniformBytes<64>>(b: &[u64; 4]) -> F {
    let mut bytes = [0u8; 64];
    bytes[0..8].copy_from_slice(&b[0].to_le_bytes());
    bytes[8..16].copy_from_slice(&b[1].to_le_bytes());
    bytes[16..24].copy_from_slice(&b[2].to_le_bytes());
    bytes[24..32].copy_from_slice(&b[3].to_le_bytes());
    F::from_uniform_bytes(&bytes)
}

/// converts fp into Vec<u64> in Montgomery form
pub fn field_to_vecu64_montgomery<F: PrimeField + SerdeObject + Serialize>(fp: &F) -> [u64; 4] {
    let repr = serde_json::to_string(&fp).unwrap();
    let b: [u64; 4] = serde_json::from_str(&repr).unwrap();
    b
}

/// An application snark with proof and instance variables ready for aggregation (raw field element)
#[derive(Debug, Clone)]
pub struct Snark<F: PrimeField + SerdeObject, C: CurveAffine> {
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
impl<F: PrimeField + SerdeObject + Serialize, C: CurveAffine + Serialize> ToPyObject
    for Snark<F, C>
{
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        let field_elems: Vec<Vec<[u64; 4]>> = self
            .instances
            .iter()
            .map(|x| x.iter().map(|fp| field_to_vecu64(fp)).collect())
            .collect::<Vec<_>>();
        dict.set_item("instances", &field_elems).unwrap();
        let hex_proof = hex::encode(&self.proof);
        dict.set_item("proof", &hex_proof).unwrap();
        dict.set_item("transcript_type", &self.transcript_type)
            .unwrap();
        dict.to_object(py)
    }
}

impl<F: PrimeField + SerdeObject + Serialize, C: CurveAffine + Serialize> Serialize for Snark<F, C>
where
    C::Scalar: serde::Serialize,
    C::ScalarExt: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // leave it untagged
        let mut state = serializer.serialize_struct("", 2)?;
        let field_elems: Vec<Vec<[u64; 4]>> = self
            .instances
            .iter()
            .map(|x| x.iter().map(|fp| field_to_vecu64(fp)).collect())
            .collect::<Vec<_>>();
        state.serialize_field("instances", &field_elems)?;

        let hex_proof = hex::encode(&self.proof);
        state.serialize_field("proof", &hex_proof)?;
        state.serialize_field("transcript_type", &self.transcript_type)?;
        if self.protocol.is_some() {
            state.serialize_field("protocol", &self.protocol)?;
        }
        state.end()
    }
}

impl<
        'de,
        F: PrimeField + SerdeObject + Serialize + FromUniformBytes<64>,
        C: CurveAffine + Serialize,
    > Deserialize<'de> for Snark<F, C>
where
    C::Scalar: serde::Deserialize<'de>,
    C: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // leave it untagged
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Instances,
            Proof,
            #[serde(rename = "transcript_type")]
            TranscriptType,
            Protocol,
        }

        struct SnarkVisitor<F: PrimeField + SerdeObject, C: CurveAffine> {
            _marker: std::marker::PhantomData<(F, C)>,
        }

        impl<'de, F: PrimeField + SerdeObject + FromUniformBytes<64>, C: CurveAffine>
            serde::de::Visitor<'de> for SnarkVisitor<F, C>
        where
            C::Scalar: serde::Deserialize<'de>,
            C: serde::Deserialize<'de>,
        {
            type Value = Snark<F, C>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Snark")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Snark<F, C>, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut instances: Option<Vec<Vec<[u64; 4]>>> = None;
                let mut proof: Option<String> = None;
                let mut transcript_type = None;
                let mut protocol = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Instances => {
                            if instances.is_some() {
                                return Err(serde::de::Error::duplicate_field("instances"));
                            }
                            instances = Some(map.next_value()?);
                        }
                        Field::Proof => {
                            if proof.is_some() {
                                return Err(serde::de::Error::duplicate_field("proof"));
                            }
                            proof = Some(map.next_value()?);
                        }
                        Field::TranscriptType => {
                            if transcript_type.is_some() {
                                return Err(serde::de::Error::duplicate_field("transcript_type"));
                            }
                            transcript_type = Some(map.next_value()?);
                        }
                        Field::Protocol => {
                            if protocol.is_some() {
                                return Err(serde::de::Error::duplicate_field("protocol"));
                            }
                            protocol = Some(map.next_value()?);
                        }
                    }
                }
                let instances =
                    instances.ok_or_else(|| serde::de::Error::missing_field("instances"))?;
                let proof = proof.ok_or_else(|| serde::de::Error::missing_field("proof"))?;
                let transcript_type = transcript_type
                    .ok_or_else(|| serde::de::Error::missing_field("transcript_type"))?;
                // protocol can be optional

                let instances: Vec<Vec<F>> = instances
                    .iter()
                    .map(|x| x.iter().map(|fp| vecu64_to_field(fp)).collect())
                    .collect::<Vec<_>>();

                let proof = hex::decode(proof).map_err(serde::de::Error::custom)?;

                Ok(Snark {
                    protocol,
                    instances,
                    proof,
                    transcript_type,
                })
            }
        }
        deserializer.deserialize_struct(
            "Snark",
            &["instances", "proof", "transcript_type", "protocol"],
            SnarkVisitor {
                _marker: PhantomData,
            },
        )
    }
}

impl<
        F: PrimeField + SerdeObject + Serialize + FromUniformBytes<64>,
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

impl<F: PrimeField + SerdeObject, C: CurveAffine> From<Snark<F, C>> for SnarkWitness<F, C> {
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

/// Creates a [VerifyingKey] and [ProvingKey] for a [ModelCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
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
    // quickly mock prove as a sanity check
    if check_mode == CheckMode::SAFE {
        debug!("running mock prover");
        let prover = MockProver::run(params.k(), &circuit, instances.clone())?;
        prover
            .verify()
            .map_err(|e| Box::<dyn Error>::from(crate::execute::ExecutionError::VerifyError(e)))?;
    }

    let mut transcript = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);
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

    info!("proof started...");
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
    Scheme::Scalar:
        SerdeObject + PrimeField + FromUniformBytes<64> + WithSmallOrderMulGroup<3> + Ord,
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
    let f = File::open(path)?;
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
    let f = File::open(path)?;
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
