/*
An easy-to-use implementation of the Poseidon Hash in the form of a Halo2 Chip. While the Poseidon Hash function
is already implemented in halo2_gadgets, there is no wrapper chip that makes it easy to use in other circuits.
Thanks to https://github.com/summa-dev/summa-solvency/blob/master/src/chips/poseidon/hash.rs for the inspiration (and also helping us understand how to use this).
*/

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};

use crate::tensor::{Tensor, ValTensor, ValType, VarTensor};

use super::Module;

/// The number of instance columns used by the KZG hash function
pub const NUM_INSTANCE_COLUMNS: usize = 0;

#[derive(Debug, Clone)]
/// WIDTH, RATE and L are const generics for the struct, which represent the width, rate, and number of inputs for the Poseidon hash function, respectively.
/// This means they are values that are known at compile time and can be used to specialize the implementation of the struct.
/// The actual chip provided by halo2_gadgets is added to the parent Chip.
pub struct KZGConfig {
    ///
    pub hash_inputs: VarTensor,
}

type InputAssignments = ();

/// PoseidonChip is a wrapper around the Pow5Chip that adds a set of advice columns to the gadget Chip to store the inputs of the hash
#[derive(Debug, Clone)]
pub struct KZGChip {
    config: KZGConfig,
}

impl Module<Fp> for KZGChip {
    type Config = KZGConfig;
    type InputAssignments = InputAssignments;
    type RunInputs = Vec<Fp>;
    type Params = (usize, usize);

    fn name(&self) -> &'static str {
        "KZG"
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
        let hash_inputs = VarTensor::new_unblinded_advice(meta, params.0, params.1);
        Self::Config { hash_inputs }
    }

    fn layout_inputs(
        &self,
        _: &mut impl Layouter<Fp>,
        _: &[ValTensor<Fp>],
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
    ) -> Result<ValTensor<Fp>, Error> {
        assert_eq!(input.len(), 1);
        layouter.assign_region(
            || "kzg commit",
            |mut region| self.config.hash_inputs.assign(&mut region, 0, &input[0]),
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
        type Config = KZGConfig;
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
            KZGChip::configure(meta, params)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let kzg_chip = KZGChip::new(config);
            kzg_chip.layout(&mut layouter, &[self.message.clone()], 0);

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn kzg_for_a_range_of_input_sizes() {
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

            assert_eq!(prover.verify_par(), Ok(()))
        }
    }

    #[test]
    #[ignore]
    fn kzg_commit_much_longer_input() {
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
        assert_eq!(prover.verify_par(), Ok(()))
    }
}