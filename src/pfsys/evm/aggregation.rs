use crate::pfsys::{Snark, SnarkWitness};
use halo2_proofs::circuit::AssignedCell;
use halo2_proofs::plonk::{self};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem},
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
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::trace;
use rand::rngs::OsRng;
use snark_verifier::loader::native::NativeLoader;
use snark_verifier::loader::EcPointLoader;
use snark_verifier::{
    loader,
    pcs::{
        kzg::{
            Bdfg21, KzgAccumulator, KzgAs, KzgSuccinctVerifyingKey, LimbsEncoding,
            LimbsEncodingInstructions,
        },
        AccumulationScheme, AccumulationSchemeProver,
    },
    system,
    util::arithmetic::fe_to_limbs,
    verifier::{self, SnarkVerifier},
};
use std::rc::Rc;
use thiserror::Error;

const LIMBS: usize = 4;
const BITS: usize = 68;
type As = KzgAs<Bn256, Bdfg21>;
/// Type for aggregator verification
type PlonkSuccinctVerifier = verifier::plonk::PlonkSuccinctVerifier<As, LimbsEncoding<LIMBS, BITS>>;

const T: usize = 5;
const RATE: usize = 4;
const R_F: usize = 8;
const R_P: usize = 60;

type Svk = KzgSuccinctVerifyingKey<G1Affine>;
type BaseFieldEccChip = halo2_wrong_ecc::BaseFieldEccChip<G1Affine, LIMBS, BITS>;
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

/// Aggregate one or more application snarks of the same shape into a KzgAccumulator
pub fn aggregate<'a>(
    svk: &Svk,
    loader: &Rc<Halo2Loader<'a>>,
    snarks: &[SnarkWitness<Fr, G1Affine>],
    as_proof: Value<&'_ [u8]>,
    split_proofs: bool,
) -> Result<
    (
        KzgAccumulator<G1Affine, Rc<Halo2Loader<'a>>>,
        Vec<Vec<AssignedCell<Fr, Fr>>>,
    ),
    plonk::Error,
> {
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
    let mut snark_instances = vec![];
    let mut proofs: Vec<
        verifier::plonk::PlonkProof<
            G1Affine,
            Rc<
                loader::halo2::Halo2Loader<
                    '_,
                    G1Affine,
                    halo2_wrong_ecc::BaseFieldEccChip<G1Affine, 4, 68>,
                >,
            >,
            KzgAs<Bn256, Bdfg21>,
        >,
    > = vec![];

    for snark in snarks.iter() {
        let protocol = snark.protocol.as_ref().unwrap().loaded(loader);
        let instances = assign_instances(&snark.instances);

        // get assigned cells
        snark_instances.extend(instances.iter().map(|instance| {
            instance
                .iter()
                .map(|v| v.clone().into_assigned())
                .collect_vec()
        }));

        // loader.ctx().constrain_equal(cell_0, cell_1)
        let mut transcript = PoseidonTranscript::<Rc<Halo2Loader>, _>::new(loader, snark.proof());
        let proof = PlonkSuccinctVerifier::read_proof(svk, &protocol, &instances, &mut transcript)
            .map_err(|_| plonk::Error::Synthesis)?;

        if split_proofs {
            let previous_proof = proofs.last();
            if let Some(previous_proof) = previous_proof {
                // output
                let output = &previous_proof.witnesses[1];
                // input
                let input = &proof.witnesses[0];
                loader
                    .ec_point_assert_eq("assert commits match", output, input)
                    .map_err(|e| {
                        log::error!("Failed to match KZG commits for sequential proofs: {:?}", e);
                        plonk::Error::Synthesis
                    })?;
            }
            proofs.push(proof.clone());
        }

        let mut accum = PlonkSuccinctVerifier::verify(svk, &protocol, &instances, &proof)
            .map_err(|_| plonk::Error::Synthesis)?;
        accumulators.append(&mut accum);
    }
    let accumulator = {
        let mut transcript = PoseidonTranscript::<Rc<Halo2Loader>, _>::new(loader, as_proof);
        let proof = As::read_proof(&Default::default(), &accumulators, &mut transcript).unwrap();
        As::verify(&Default::default(), &accumulators, &proof).map_err(|_| plonk::Error::Synthesis)
    }?;
    Ok((accumulator, snark_instances))
}

/// The Halo2 Config for the aggregation circuit
#[derive(Clone, Debug)]
pub struct AggregationConfig {
    main_gate_config: MainGateConfig,
    range_config: RangeConfig,
}

impl AggregationConfig {
    /// Configure the aggregation circuit
    pub fn configure<F: PrimeField>(
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
    snarks: Vec<SnarkWitness<Fr, G1Affine>>,
    instances: Vec<Fr>,
    as_proof: Value<Vec<u8>>,
    split_proofs: bool,
}

impl AggregationCircuit {
    /// Create a new Aggregation Circuit with a SuccinctVerifyingKey, application snark witnesses (each with a proof and instance variables), and the instance variables and the resulting aggregation circuit proof.
    pub fn new(
        svk: &KzgSuccinctVerifyingKey<G1Affine>,
        snarks: impl IntoIterator<Item = Snark<Fr, G1Affine>>,
        split_proofs: bool,
    ) -> Result<Self, AggregationError> {
        let snarks = snarks.into_iter().collect_vec();

        let mut accumulators = vec![];

        for snark in snarks.iter() {
            trace!("Aggregating with snark instances {:?}", snark.instances);
            let mut transcript = PoseidonTranscript::<NativeLoader, _>::new(snark.proof.as_slice());
            let proof = PlonkSuccinctVerifier::read_proof(
                svk,
                snark.protocol.as_ref().unwrap(),
                &snark.instances,
                &mut transcript,
            )
            .map_err(|e| {
                log::error!("{:?}", e);
                AggregationError::ProofRead
            })?;
            let mut accum = PlonkSuccinctVerifier::verify(
                svk,
                snark.protocol.as_ref().unwrap(),
                &snark.instances,
                &proof,
            )
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
            svk: *svk,
            snarks: snarks.into_iter().map_into().collect(),
            instances,
            as_proof: Value::known(as_proof),
            split_proofs,
        })
    }

    ///
    pub fn num_limbs() -> usize {
        LIMBS
    }
    ///
    pub fn num_bits() -> usize {
        BITS
    }

    /// Accumulator indices used in generating verifier.
    pub fn accumulator_indices() -> Vec<(usize, usize)> {
        (0..4 * LIMBS).map(|idx| (0, idx)).collect()
    }

    /// Number of instance variables for the aggregation circuit, used in generating verifier.
    pub fn num_instance(orginal_circuit_instances: usize) -> Vec<usize> {
        let accumulation_instances = 4 * LIMBS;
        vec![accumulation_instances + orginal_circuit_instances]
    }

    /// Instance variables for the aggregation circuit, fed to verifier.
    pub fn instances(&self) -> Vec<Fr> {
        // also get snark instances here
        let mut snark_instances: Vec<Vec<Vec<Value<Fr>>>> = self
            .snarks
            .iter()
            .map(|snark| snark.instances.clone())
            .collect_vec();

        // reduce from Vec<Vec<Vec<Value<Fr>>>> to Vec<Vec<Value<Fr>>>
        let mut instances: Vec<Fr> = self.instances.clone();
        for snark_instance in snark_instances.iter_mut() {
            for instance in snark_instance.iter_mut() {
                let mut felt_evals = vec![];
                for value in instance.iter_mut() {
                    value.map(|v| felt_evals.push(v));
                }
                instances.extend(felt_evals);
            }
        }

        instances
    }

    fn as_proof(&self) -> Value<&[u8]> {
        self.as_proof.as_ref().map(Vec::as_slice)
    }
}

impl Circuit<Fr> for AggregationCircuit {
    type Config = AggregationConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();

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
            split_proofs: self.split_proofs,
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

        let (accumulator_limbs, snark_instances) = layouter.assign_region(
            || "",
            |region| {
                let ctx = RegionCtx::new(region, 0);

                let ecc_chip = config.ecc_chip();
                let loader = Halo2Loader::new(ecc_chip, ctx);
                let (accumulator, snark_instances) = aggregate(
                    &self.svk,
                    &loader,
                    &self.snarks,
                    self.as_proof(),
                    self.split_proofs,
                )?;

                let accumulator_limbs = [accumulator.lhs, accumulator.rhs]
                    .iter()
                    .map(|ec_point| {
                        loader
                            .ecc_chip()
                            .assign_ec_point_to_limbs(&mut loader.ctx_mut(), ec_point.assigned())
                    })
                    .collect::<Result<Vec<_>, plonk::Error>>()?
                    .into_iter()
                    .flatten();

                Ok((accumulator_limbs, snark_instances))
            },
        )?;

        let mut instance_offset = 0;
        for limb in accumulator_limbs {
            main_gate.expose_public(layouter.namespace(|| ""), limb, instance_offset)?;
            instance_offset += 1;
        }

        for instance in snark_instances.into_iter() {
            for elem in instance.into_iter() {
                main_gate.expose_public(layouter.namespace(|| ""), elem, instance_offset)?;
                instance_offset += 1;
            }
        }

        Ok(())
    }
}
