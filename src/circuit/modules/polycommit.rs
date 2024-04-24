/*
An easy-to-use implementation of the Poseidon Hash in the form of a Halo2 Chip. While the Poseidon Hash function
is already implemented in halo2_gadgets, there is no wrapper chip that makes it easy to use in other circuits.
Thanks to https://github.com/summa-dev/summa-solvency/blob/master/src/chips/poseidon/hash.rs for the inspiration (and also helping us understand how to use this).
*/

use std::collections::HashMap;

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::poly::commitment::{Blind, CommitmentScheme, Params};
use halo2_proofs::{circuit::*, plonk::*};
use halo2curves::bn256::G1Affine;
use halo2curves::group::prime::PrimeCurveAffine;
use halo2curves::group::Curve;
use halo2curves::CurveAffine;

use crate::circuit::region::ConstantsMap;
use crate::tensor::{Tensor, ValTensor, ValType, VarTensor};

use super::Module;

/// The number of instance columns used by the PolyCommit hash function
pub const NUM_INSTANCE_COLUMNS: usize = 0;
/// The number of advice columns used by the PolyCommit hash function
pub const NUM_INNER_COLS: usize = 1;

#[derive(Debug, Clone)]
/// Configuration for the PolyCommit chip
pub struct PolyCommitConfig {
    ///
    pub inputs: VarTensor,
}

type InputAssignments = ();

///
#[derive(Debug)]
pub struct PolyCommitChip {
    config: PolyCommitConfig,
}

impl PolyCommitChip {
    /// Commit to the message using the KZG commitment scheme
    pub fn commit<Scheme: CommitmentScheme<Scalar = Fp, Curve = G1Affine>>(
        message: Vec<Scheme::Scalar>,
        num_unusable_rows: u32,
        params: &Scheme::ParamsProver,
    ) -> Vec<G1Affine> {
        let k = params.k();
        let domain = halo2_proofs::poly::EvaluationDomain::new(2, k);
        let n = 2_u64.pow(k) - num_unusable_rows as u64;
        let num_poly = (message.len() / n as usize) + 1;
        let mut poly = vec![domain.empty_lagrange(); num_poly];

        (0..num_unusable_rows).for_each(|i| {
            for p in &mut poly {
                p[(n + i as u64) as usize] = Blind::default().0;
            }
        });

        for (i, m) in message.iter().enumerate() {
            let x = i / (n as usize);
            let y = i % (n as usize);
            poly[x][y] = *m;
        }

        let mut advice_commitments_projective = vec![];
        for a in poly {
            advice_commitments_projective.push(params.commit_lagrange(&a, Blind::default()))
        }

        let mut advice_commitments =
            vec![G1Affine::identity(); advice_commitments_projective.len()];
        <G1Affine as CurveAffine>::CurveExt::batch_normalize(
            &advice_commitments_projective,
            &mut advice_commitments,
        );
        advice_commitments
    }
}

impl Module<Fp> for PolyCommitChip {
    type Config = PolyCommitConfig;
    type InputAssignments = InputAssignments;
    type RunInputs = Vec<Fp>;
    type Params = (usize, usize);

    fn name(&self) -> &'static str {
        "PolyCommit"
    }

    fn instance_increment_input(&self) -> Vec<usize> {
        vec![0]
    }

    /// Constructs a new PoseidonChip
    fn new(config: Self::Config) -> Self {
        Self { config }
    }

    /// Configuration of the PoseidonChip
    fn configure(meta: &mut ConstraintSystem<Fp>, params: Self::Params) -> Self::Config {
        let inputs = VarTensor::new_unblinded_advice(meta, params.0, NUM_INNER_COLS, params.1);
        Self::Config { inputs }
    }

    fn layout_inputs(
        &self,
        _: &mut impl Layouter<Fp>,
        _: &[ValTensor<Fp>],
        _: &mut ConstantsMap<Fp>,
    ) -> Result<Self::InputAssignments, Error> {
        Ok(())
    }

    /// L is the number of inputs to the hash function
    /// Takes the cells containing the input values of the hash function and return the cell containing the hash output
    /// It uses the pow5_chip to compute the hash
    fn layout(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: &[ValTensor<Fp>],
        _: usize,
        constants: &mut ConstantsMap<Fp>,
    ) -> Result<ValTensor<Fp>, Error> {
        assert_eq!(input.len(), 1);

        let local_constants = constants.clone();
        layouter.assign_region(
            || "PolyCommit",
            |mut region| {
                let mut local_inner_constants = local_constants.clone();
                let res = self.config.inputs.assign(
                    &mut region,
                    0,
                    &input[0],
                    &mut local_inner_constants,
                )?;
                *constants = local_inner_constants;
                Ok(res)
            },
        )
    }

    ///
    fn run(message: Vec<Fp>) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        Ok(vec![message])
    }

    fn num_rows(_: usize) -> usize {
        0
    }
}

#[allow(unused)]
mod tests {

    use crate::circuit::modules::ModulePlanner;

    use super::*;

    use std::marker::PhantomData;

    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem},
    };
    use halo2curves::ff::Field;

    const K: usize = 8;
    const R: usize = 2048;

    struct HashCircuit {
        message: ValTensor<Fp>,
    }

    impl Circuit<Fp> for HashCircuit {
        type Config = PolyCommitConfig;
        type FloorPlanner = ModulePlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            let empty_val: Vec<ValType<Fp>> = vec![Value::<Fp>::unknown().into(); R];
            let message: Tensor<ValType<Fp>> = empty_val.into_iter().into();

            Self {
                message: message.into(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
            let params = (K, R);
            PolyCommitChip::configure(meta, params)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let polycommit_chip = PolyCommitChip::new(config);
            polycommit_chip.layout(
                &mut layouter,
                &[self.message.clone()],
                0,
                &mut HashMap::new(),
            );

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn polycommit_chip_for_a_range_of_input_sizes() {
        let rng = rand::rngs::OsRng;

        #[cfg(not(target_arch = "wasm32"))]
        env_logger::init();

        {
            let i = 32;
            // print a bunch of new lines
            println!(
                "i is {} -------------------------------------------------",
                i
            );

            let message: Vec<Fp> = (0..i).map(|_| Fp::random(rng)).collect::<Vec<_>>();

            let mut message: Tensor<ValType<Fp>> =
                message.into_iter().map(|m| Value::known(m).into()).into();

            let circuit = HashCircuit {
                message: message.into(),
            };
            let prover = halo2_proofs::dev::MockProver::run(K as u32, &circuit, vec![]).unwrap();

            assert_eq!(prover.verify(), Ok(()))
        }
    }

    #[test]
    #[ignore]
    fn polycommit_chip_much_longer_input() {
        #[cfg(not(target_arch = "wasm32"))]
        env_logger::init();

        let rng = rand::rngs::OsRng;

        let mut message: Vec<Fp> = (0..2048).map(|_| Fp::random(rng)).collect::<Vec<_>>();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let circuit = HashCircuit {
            message: message.into(),
        };
        let prover = halo2_proofs::dev::MockProver::run(K as u32, &circuit, vec![]).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }
}
