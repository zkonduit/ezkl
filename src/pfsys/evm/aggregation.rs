use crate::commands::Cli;
use crate::fieldutils::i32_to_felt;
use crate::pfsys::prepare_circuit_and_public_input;
use crate::pfsys::ModelInput;
use ethereum_types::Address;
use foundry_evm::executor::{fork::MultiFork, Backend, ExecutorBuilder};
use halo2_proofs::plonk::VerifyingKey;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{
        self, create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ConstraintSystem,
        ProvingKey,
    },
    poly::{
        commitment::{Params, ParamsProver},
        kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverGWC, VerifierGWC},
            strategy::AccumulatorStrategy,
        },
        VerificationStrategy,
    },
    transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer},
};
use halo2_wrong_ecc::{
    integer::rns::Rns,
    maingate::{
        MainGate, MainGateConfig, MainGateInstructions, RangeChip, RangeConfig, RangeInstructions,
        RegionCtx,
    },
    EccConfig,
};
use halo2curves::bn256::{Bn256, Fq, Fr, G1Affine};
use itertools::Itertools;
use log::trace;
use rand::rngs::OsRng;
use snark_verifier::{
    loader::evm::{self, encode_calldata, EvmLoader},
    system::halo2::transcript::evm::EvmTranscript,
};
use snark_verifier::{
    loader::native::NativeLoader,
    system::halo2::{compile, Config},
};
use snark_verifier::{
    loader::{self},
    pcs::{
        kzg::{Gwc19, KzgAccumulator, KzgAs, KzgSuccinctVerifyingKey, LimbsEncoding},
        AccumulationScheme, AccumulationSchemeProver,
    },
    system,
    util::arithmetic::{fe_to_limbs, FieldExt},
    verifier::{self, plonk::PlonkProtocol, SnarkVerifier},
};
use std::error::Error;
use std::io::Cursor;
use std::{iter, rc::Rc};
use thiserror::Error;

const LIMBS: usize = 4;
const BITS: usize = 68;
type As = KzgAs<Bn256, Gwc19>;
/// Type for aggregator verification
type PlonkSuccinctVerifier = verifier::plonk::PlonkSuccinctVerifier<As, LimbsEncoding<LIMBS, BITS>>;
type PlonkVerifier = verifier::plonk::PlonkVerifier<As, LimbsEncoding<LIMBS, BITS>>;

const T: usize = 5;
const RATE: usize = 4;
const R_F: usize = 8;
const R_P: usize = 60;

type Svk = KzgSuccinctVerifyingKey<G1Affine>;
type BaseFieldEccChip =
    snark_verifier::loader::halo2::halo2_wrong_ecc::BaseFieldEccChip<G1Affine, LIMBS, BITS>;
/// The loader type used in the transcript definition
type Halo2Loader<'a> = loader::halo2::Halo2Loader<'a, G1Affine, BaseFieldEccChip>;
/// Application snark transcript
pub type PoseidonTranscript<L, S> =
    system::halo2::transcript::halo2::PoseidonTranscript<G1Affine, L, S, T, RATE, R_F, R_P>;

#[derive(Error, Debug)]
/// Errors related to proof aggregation
pub enum AggregationError {
    /// A KZG proof could not be verified
    #[error("failed to verify KZG proof")]
    KZGProofVerification,
    /// EVM execution errors
    #[error("EVM execution of raw code failed")]
    EVMRawExecution,
    /// proof read errors
    #[error("Failed to read proof")]
    ProofRead,
    /// proof verification errors
    #[error("Failed to verify proof")]
    ProofVerify,
    /// proof creation errors
    #[error("Failed to create proof")]
    ProofCreate,
}

/// An application snark with proof and instance variables ready for aggregation (raw field element)
#[derive(Debug)]
pub struct Snark {
    protocol: PlonkProtocol<G1Affine>,
    instances: Vec<Vec<Fr>>,
    proof: Vec<u8>,
}

impl Snark {
    /// Create a new application snark from proof and instance variables ready for aggregation
    pub fn new(protocol: PlonkProtocol<G1Affine>, instances: Vec<Vec<Fr>>, proof: Vec<u8>) -> Self {
        Self {
            protocol,
            instances,
            proof,
        }
    }
}

impl From<Snark> for SnarkWitness {
    fn from(snark: Snark) -> Self {
        Self {
            protocol: snark.protocol,
            instances: snark
                .instances
                .into_iter()
                .map(|instances| instances.into_iter().map(Value::known).collect_vec())
                .collect(),
            proof: Value::known(snark.proof),
        }
    }
}

/// An application snark with proof and instance variables ready for aggregation (wrapped field element)
#[derive(Clone, Debug)]
pub struct SnarkWitness {
    protocol: PlonkProtocol<G1Affine>,
    instances: Vec<Vec<Value<Fr>>>,
    proof: Value<Vec<u8>>,
}

impl SnarkWitness {
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

/// Aggregate one or more application snarks of the same shape into a KzgAccumulator
pub fn aggregate<'a>(
    svk: &Svk,
    loader: &Rc<Halo2Loader<'a>>,
    snarks: &[SnarkWitness],
    as_proof: Value<&'_ [u8]>,
) -> Result<KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>, plonk::Error> {
    let assign_instances = |instances: &[Vec<Value<Fr>>]| {
        instances
            .iter()
            .map(|instances| {
                instances
                    .iter()
                    .map(|instance| loader.assign_scalar(*instance))
                    .collect_vec()
            })
            .collect_vec()
    };

    let mut accumulators = vec![];

    for snark in snarks.iter() {
        let protocol = snark.protocol.loaded(loader);
        let instances = assign_instances(&snark.instances);
        let mut transcript = PoseidonTranscript::<Rc<Halo2Loader>, _>::new(loader, snark.proof());
        let proof = PlonkSuccinctVerifier::read_proof(svk, &protocol, &instances, &mut transcript)
            .map_err(|_| plonk::Error::Synthesis)?;
        let mut accum = PlonkSuccinctVerifier::verify(svk, &protocol, &instances, &proof)
            .map_err(|_| plonk::Error::Synthesis)?;
        accumulators.append(&mut accum);
    }
    let accumulator = {
        let mut transcript = PoseidonTranscript::<Rc<Halo2Loader>, _>::new(loader, as_proof);
        let proof = As::read_proof(&Default::default(), &accumulators, &mut transcript).unwrap();
        As::verify(&Default::default(), &accumulators, &proof).map_err(|_| plonk::Error::Synthesis)
    };
    accumulator
}

/// The Halo2 Config for the aggregation circuit
#[derive(Clone, Debug)]
pub struct AggregationConfig {
    main_gate_config: MainGateConfig,
    range_config: RangeConfig,
}

impl AggregationConfig {
    /// Configure the aggregation circuit
    pub fn configure<F: FieldExt>(
        meta: &mut ConstraintSystem<F>,
        composition_bits: Vec<usize>,
        overflow_bits: Vec<usize>,
    ) -> Self {
        let main_gate_config = MainGate::<F>::configure(meta);
        let range_config =
            RangeChip::<F>::configure(meta, &main_gate_config, composition_bits, overflow_bits);
        AggregationConfig {
            main_gate_config,
            range_config,
        }
    }

    /// Create a MainGate from the aggregation approach
    pub fn main_gate(&self) -> MainGate<Fr> {
        MainGate::new(self.main_gate_config.clone())
    }

    /// Create a range chip to decompose and range check inputs
    pub fn range_chip(&self) -> RangeChip<Fr> {
        RangeChip::new(self.range_config.clone())
    }

    /// Create an ecc chip for ec ops
    pub fn ecc_chip(&self) -> BaseFieldEccChip {
        BaseFieldEccChip::new(EccConfig::new(
            self.range_config.clone(),
            self.main_gate_config.clone(),
        ))
    }
}

/// Aggregation Circuit with a SuccinctVerifyingKey, application snark witnesses (each with a proof and instance variables), and the instance variables and the resulting aggregation circuit proof.
#[derive(Clone, Debug)]
pub struct AggregationCircuit {
    svk: Svk,
    snarks: Vec<SnarkWitness>,
    instances: Vec<Fr>,
    as_proof: Value<Vec<u8>>,
}

impl AggregationCircuit {
    /// Create a new Aggregation Circuit with a SuccinctVerifyingKey, application snark witnesses (each with a proof and instance variables), and the instance variables and the resulting aggregation circuit proof.
    pub fn new(
        params: &ParamsKZG<Bn256>,
        snarks: impl IntoIterator<Item = Snark>,
    ) -> Result<Self, AggregationError> {
        let svk = params.get_g()[0].into();
        let snarks = snarks.into_iter().collect_vec();

        let mut accumulators = vec![];

        for snark in snarks.iter() {
            trace!("Aggregating with snark instances {:?}", snark.instances);
            let mut transcript = PoseidonTranscript::<NativeLoader, _>::new(snark.proof.as_slice());
            let proof = PlonkSuccinctVerifier::read_proof(
                &svk,
                &snark.protocol,
                &snark.instances,
                &mut transcript,
            )
            .map_err(|_| AggregationError::ProofRead)?;
            let mut accum =
                PlonkSuccinctVerifier::verify(&svk, &snark.protocol, &snark.instances, &proof)
                    .map_err(|_| AggregationError::ProofVerify)?;
            accumulators.append(&mut accum);
        }

        trace!("Accumulator");
        let (accumulator, as_proof) = {
            let mut transcript = PoseidonTranscript::<NativeLoader, _>::new(Vec::new());
            let accumulator =
                As::create_proof(&Default::default(), &accumulators, &mut transcript, OsRng)
                    .map_err(|_| AggregationError::ProofCreate)?;
            (accumulator, transcript.finalize())
        };

        trace!("KzgAccumulator");
        let KzgAccumulator { lhs, rhs } = accumulator;
        let instances = [lhs.x, lhs.y, rhs.x, rhs.y]
            .map(fe_to_limbs::<_, _, LIMBS, BITS>)
            .concat();

        Ok(Self {
            svk,
            snarks: snarks.into_iter().map_into().collect(),
            instances,
            as_proof: Value::known(as_proof),
        })
    }

    /// Accumulator indices used in generating verifier.
    pub fn accumulator_indices() -> Vec<(usize, usize)> {
        (0..4 * LIMBS).map(|idx| (0, idx)).collect()
    }

    /// Number of instance variables for the aggregation circuit, used in generating verifier.
    pub fn num_instance() -> Vec<usize> {
        vec![4 * LIMBS]
    }

    /// Instance variables for the aggregation circuit, fed to verifier.
    pub fn instances(&self) -> Vec<Vec<Fr>> {
        vec![self.instances.clone()]
    }

    fn as_proof(&self) -> Value<&[u8]> {
        self.as_proof.as_ref().map(Vec::as_slice)
    }
}

impl Circuit<Fr> for AggregationCircuit {
    type Config = AggregationConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            svk: self.svk,
            snarks: self
                .snarks
                .iter()
                .map(SnarkWitness::without_witnesses)
                .collect(),
            instances: Vec::new(),
            as_proof: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        AggregationConfig::configure(
            meta,
            vec![BITS / LIMBS],
            Rns::<Fq, Fr, LIMBS, BITS>::construct().overflow_lengths(),
        )
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), plonk::Error> {
        let main_gate = config.main_gate();
        let range_chip = config.range_chip();

        range_chip.load_table(&mut layouter)?;

        let (lhs, rhs) = layouter.assign_region(
            || "",
            |region| {
                let ctx = RegionCtx::new(region, 0);

                let ecc_chip = config.ecc_chip();
                let loader = Halo2Loader::new(ecc_chip, ctx);
                let KzgAccumulator { lhs, rhs } =
                    aggregate(&self.svk, &loader, &self.snarks, self.as_proof())?;

                let lhs = lhs.assigned().clone();
                let rhs = rhs.assigned().clone();

                Ok((lhs, rhs))
            },
        )?;

        for (limb, row) in iter::empty()
            .chain(lhs.x().limbs())
            .chain(lhs.y().limbs())
            .chain(rhs.x().limbs())
            .chain(rhs.y().limbs())
            .zip(0..)
        {
            main_gate.expose_public(layouter.namespace(|| ""), limb.into(), row)?;
        }

        Ok(())
    }
}

/// Create proof and instance variables for the application snark
pub fn gen_application_snark(
    params: &ParamsKZG<Bn256>,
    data: &ModelInput,
    args: &Cli,
) -> Result<Snark, Box<dyn Error>> {
    let (circuit, public_inputs) = prepare_circuit_and_public_input::<Fr>(data, args)?;

    let pk = gen_pk(params, &circuit)?;
    let number_instance = public_inputs[0].len();
    trace!("number_instance {:?}", number_instance);
    let protocol = compile(
        params,
        pk.get_vk(),
        Config::kzg().with_num_instance(vec![number_instance]),
    );
    let pi_inner: Vec<Vec<Fr>> = public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i32_to_felt::<Fr>(*e)).collect::<Vec<Fr>>())
        .collect::<Vec<Vec<Fr>>>();
    //    let pi_inner = pi_inner.iter().map(|e| e.deref()).collect::<Vec<&[Fr]>>();
    trace!("pi_inner {:?}", pi_inner);
    let proof = gen_kzg_proof::<
        _,
        _,
        PoseidonTranscript<NativeLoader, _>,
        PoseidonTranscript<NativeLoader, _>,
    >(params, &pk, circuit, pi_inner.clone())?;
    Ok(Snark::new(protocol, pi_inner, proof))
}

/// Create aggregation EVM verifier bytecode
pub fn gen_aggregation_evm_verifier(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    num_instance: Vec<usize>,
    accumulator_indices: Vec<(usize, usize)>,
) -> Result<Vec<u8>, AggregationError> {
    let protocol = compile(
        params,
        vk,
        Config::kzg()
            .with_num_instance(num_instance.clone())
            .with_accumulator_indices(Some(accumulator_indices)),
    );
    let vk = (params.get_g()[0], params.g2(), params.s_g2()).into();

    let loader = EvmLoader::new::<Fq, Fr>();
    let protocol = protocol.loaded(&loader);
    let mut transcript = EvmTranscript::<_, Rc<EvmLoader>, _, _>::new(&loader);

    let instances = transcript.load_instances(num_instance);
    let proof = PlonkVerifier::read_proof(&vk, &protocol, &instances, &mut transcript)
        .map_err(|_| AggregationError::ProofRead)?;
    PlonkVerifier::verify(&vk, &protocol, &instances, &proof)
        .map_err(|_| AggregationError::ProofVerify)?;

    Ok(evm::compile_yul(&loader.yul_code()))
}

/// Verify by executing bytecode with instance variables and proof as input
pub fn evm_verify(
    deployment_code: Vec<u8>,
    instances: Vec<Vec<Fr>>,
    proof: Vec<u8>,
) -> Result<bool, Box<dyn Error>> {
    let calldata = encode_calldata(&instances, &proof);
    let mut evm = ExecutorBuilder::default()
        .with_gas_limit(u64::MAX.into())
        .build(Backend::new(MultiFork::new().0, None));

    let caller = Address::from_low_u64_be(0xfe);
    let verifier = evm
        .deploy(caller, deployment_code.into(), 0.into(), None)
        .map_err(Box::new)?
        .address;
    let result = evm
        .call_raw(caller, verifier, calldata.into(), 0.into())
        .map_err(|_| Box::new(AggregationError::EVMRawExecution))?;

    dbg!(result.gas_used);

    Ok(!result.reverted)
}

/// Generate a structured reference string for testing. Not secure, do not use in production.
pub fn gen_srs(k: u32) -> ParamsKZG<Bn256> {
    ParamsKZG::<Bn256>::setup(k, OsRng)
}

/// Generate the proving key
pub fn gen_pk<C: Circuit<Fr>>(
    params: &ParamsKZG<Bn256>,
    circuit: &C,
) -> Result<ProvingKey<G1Affine>, plonk::Error> {
    let vk = keygen_vk(params, circuit)?;
    keygen_pk(params, vk, circuit)
}

/// Generates proof for either application circuit (model) or aggregation circuit.
pub fn gen_kzg_proof<
    C: Circuit<Fr>,
    E: EncodedChallenge<G1Affine>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, G1Affine, E>,
    TW: TranscriptWriterBuffer<Vec<u8>, G1Affine, E>,
>(
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
    circuit: C,
    instances: Vec<Vec<Fr>>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    MockProver::run(params.k(), &circuit, instances.clone())
        .map_err(Box::new)?
        .assert_satisfied();

    let instances = instances
        .iter()
        .map(|instances| instances.as_slice())
        .collect_vec();
    let mut proof = TW::init(Vec::new());
    create_proof::<KZGCommitmentScheme<Bn256>, ProverGWC<_>, _, _, TW, _>(
        params,
        pk,
        &[circuit],
        &[instances.as_slice()],
        OsRng,
        &mut proof,
    )
    .map_err(Box::new)?;
    let proof = proof.finalize();

    let mut transcript = TR::init(Cursor::new(proof.clone()));
    let verify = verify_proof::<_, VerifierGWC<_>, _, TR, _>(
        params.verifier_params(),
        pk.get_vk(),
        AccumulatorStrategy::new(params.verifier_params()),
        &[instances.as_slice()],
        &mut transcript,
    )
    .map_err(Box::new)?;

    let accept = VerificationStrategy::<_, VerifierGWC<_>>::finalize(verify);

    if !accept {
        return Err(Box::new(AggregationError::KZGProofVerification));
    }
    Ok(proof)
}
