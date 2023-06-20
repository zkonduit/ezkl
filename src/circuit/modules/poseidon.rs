/*
An easy-to-use implementation of the Poseidon Hash in the form of a Halo2 Chip. While the Poseidon Hash function
is already implemented in halo2_gadgets, there is no wrapper chip that makes it easy to use in other circuits.
Thanks to https://github.com/summa-dev/summa-solvency/blob/master/src/chips/poseidon/hash.rs for the inspiration (and also helping us understand how to use this).
*/

pub mod poseidon_params;
pub mod spec;

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
// compared to `hash_with_instance` this version doesn't use any instance column.
use halo2_gadgets::poseidon::{primitives::*, Hash, Pow5Chip, Pow5Config};
use halo2_proofs::arithmetic::Field;
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};

use std::marker::PhantomData;

use crate::tensor::{Tensor, ValTensor, ValType};

use super::Module;

/// The number of instance columns used by the Poseidon hash function
pub const NUM_INSTANCE_COLUMNS: usize = 1;

#[derive(Debug, Clone)]
/// WIDTH, RATE and L are const generics for the struct, which represent the width, rate, and number of inputs for the Poseidon hash function, respectively.
/// This means they are values that are known at compile time and can be used to specialize the implementation of the struct.
/// The actual chip provided by halo2_gadgets is added to the parent Chip.
pub struct PoseidonConfig<const WIDTH: usize, const RATE: usize> {
    ///
    pub hash_inputs: Vec<Column<Advice>>,
    ///
    pub instance: Column<Instance>,
    ///
    pub pow5_config: Pow5Config<Fp, WIDTH, RATE>,
}

type InputAssignments = (Vec<AssignedCell<Fp, Fp>>, AssignedCell<Fp, Fp>);

/// PoseidonChip is a wrapper around the Pow5Chip that adds a set of advice columns to the gadget Chip to store the inputs of the hash
#[derive(Debug, Clone)]
pub struct PoseidonChip<
    S: Spec<Fp, WIDTH, RATE>,
    const WIDTH: usize,
    const RATE: usize,
    const L: usize,
> {
    config: PoseidonConfig<WIDTH, RATE>,
    _marker: PhantomData<S>,
}

impl<S: Spec<Fp, WIDTH, RATE>, const WIDTH: usize, const RATE: usize, const L: usize> Module<Fp>
    for PoseidonChip<S, WIDTH, RATE, L>
{
    type Config = PoseidonConfig<WIDTH, RATE>;
    type InputAssignments = InputAssignments;
    type RunInputs = Vec<Fp>;

    fn name(&self) -> &'static str {
        "Poseidon"
    }

    fn instance_increment_input(&self, _: Vec<usize>) -> Vec<usize> {
        vec![1]
    }

    fn instance_increment_module(&self) -> Vec<usize> {
        vec![0]
    }

    /// Constructs a new PoseidonChip
    fn new(config: Self::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Configuration of the PoseidonChip
    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        //  instantiate the required columns
        let hash_inputs = (0..WIDTH).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for input in &hash_inputs {
            meta.enable_equality(*input);
        }

        let partial_sbox = meta.advice_column();
        let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

        for input in hash_inputs.iter().take(WIDTH) {
            meta.enable_equality(*input);
        }
        meta.enable_constant(rc_b[0]);

        let pow5_config = Pow5Chip::configure::<S>(
            meta,
            hash_inputs.clone().try_into().unwrap(),
            partial_sbox,
            rc_a.try_into().unwrap(),
            rc_b.try_into().unwrap(),
        );

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        PoseidonConfig {
            pow5_config,
            instance,
            hash_inputs,
        }
    }

    fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<Fp>,
        message: &[ValTensor<Fp>],
    ) -> Result<Self::InputAssignments, Error> {
        assert_eq!(message.len(), 1);
        let message = message[0].clone();

        layouter.assign_region(
            || "load message",
            |mut region| {
                let message_word = |i: usize| {
                    let value = &message.get_inner_tensor().unwrap()[i];
                    let x = i % WIDTH;
                    let y = i / WIDTH;

                    match value {
                        ValType::Value(v) => region.assign_advice(
                            || format!("load message_{}", i),
                            self.config.hash_inputs[x],
                            y,
                            || *v,
                        ),
                        ValType::PrevAssigned(v) => Ok(v.clone()),
                        _ => panic!("wrong input type, must be previously assigned"),
                    }
                };

                let message: Result<Vec<AssignedCell<Fp, Fp>>, Error> =
                    (0..message.len()).map(message_word).collect();
                let message = message?;

                let offset = message.len() / WIDTH + 1;

                let zero_val = region
                    .assign_advice_from_constant(
                        || "",
                        self.config.hash_inputs[0],
                        offset,
                        Fp::ZERO,
                    )
                    .unwrap();

                Ok((message, zero_val))
            },
        )
    }

    /// L is the number of inputs to the hash function
    /// Takes the cells containing the input values of the hash function and return the cell containing the hash output
    /// It uses the pow5_chip to compute the hash
    fn layout(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: &[ValTensor<Fp>],
        row_offset: Vec<usize>,
    ) -> Result<ValTensor<Fp>, Error> {
        let (input, zero_val) = self.layout_inputs(layouter, input)?;

        // iterate over the input cells in blocks of L
        let mut input_cells = input.clone();

        // do the Tree dance baby
        while input_cells.len() > 1 {
            let mut hashes = vec![];
            for block in input_cells.chunks(L) {
                let mut block = block.to_vec();
                let remainder = block.len() % L;

                if remainder != 0 {
                    block.extend(vec![zero_val.clone(); L - remainder].into_iter());
                }

                let pow5_chip = Pow5Chip::construct(self.config.pow5_config.clone());
                // initialize the hasher
                let hasher = Hash::<_, _, S, ConstantLength<L>, WIDTH, RATE>::init(
                    pow5_chip,
                    layouter.namespace(|| "block_hasher"),
                )?;

                // you may need to 0 pad the inputs so they fit
                let hash = hasher.hash(
                    layouter.namespace(|| "hash"),
                    block.to_vec().try_into().map_err(|_| Error::Synthesis)?,
                );

                hashes.push(hash?);
            }
            input_cells = hashes;
        }

        let result = Tensor::from(input_cells.iter().map(|e| ValType::from(e.clone())));

        let output = match result[0].clone() {
            ValType::PrevAssigned(v) => v,
            _ => panic!(),
        };

        layouter.assign_region(
            || "constrain output",
            |mut region| {
                let expected_var = region.assign_advice_from_instance(
                    || "pub input anchor",
                    self.config.instance,
                    row_offset[0],
                    self.config.hash_inputs[0],
                    0,
                )?;

                region.constrain_equal(output.cell(), expected_var.cell())
            },
        )?;

        let assigned_input: Tensor<ValType<Fp>> =
            input.iter().map(|e| ValType::from(e.clone())).into();

        Ok(assigned_input.into())
    }

    ///
    fn run(message: Vec<Fp>) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        let mut hash_inputs = message;
        // do the Tree dance baby
        while hash_inputs.len() > 1 {
            let mut hashes: Vec<Fp> = vec![];
            for block in hash_inputs.chunks(L) {
                let mut block = block.to_vec();
                let remainder = block.len() % L;
                if remainder != 0 {
                    block.extend(vec![Fp::ZERO; L - remainder].iter());
                }
                let hash = halo2_gadgets::poseidon::primitives::Hash::<
                    _,
                    S,
                    ConstantLength<L>,
                    { WIDTH },
                    { RATE },
                >::init()
                .hash(block.clone().try_into().unwrap());
                hashes.push(hash);
            }
            hash_inputs = hashes;
        }

        Ok(vec![hash_inputs])
    }
}

#[allow(unused)]
mod tests {

    use super::{
        spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
        *,
    };

    use std::marker::PhantomData;

    use halo2_gadgets::poseidon::primitives::Spec;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem},
    };
    use halo2curves::ff::Field;

    const WIDTH: usize = POSEIDON_WIDTH;
    const RATE: usize = POSEIDON_RATE;
    const R: usize = 240;

    struct HashCircuit<S: Spec<Fp, WIDTH, RATE>, const L: usize> {
        message: ValTensor<Fp>,
        _spec: PhantomData<S>,
    }

    impl<S: Spec<Fp, WIDTH, RATE>, const L: usize> Circuit<Fp> for HashCircuit<S, L> {
        type Config = PoseidonConfig<WIDTH, RATE>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            let empty_val: Vec<ValType<Fp>> = vec![Value::<Fp>::unknown().into()];
            let message: Tensor<ValType<Fp>> = empty_val.into_iter().into();

            Self {
                message: message.into(),
                _spec: PhantomData,
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> PoseidonConfig<WIDTH, RATE> {
            PoseidonChip::<PoseidonSpec, WIDTH, RATE, L>::configure(meta)
        }

        fn synthesize(
            &self,
            config: PoseidonConfig<WIDTH, RATE>,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let chip: PoseidonChip<PoseidonSpec, WIDTH, RATE, L> = PoseidonChip::new(config);
            chip.layout(
                &mut layouter,
                &[self.message.clone()],
                vec![0; NUM_INSTANCE_COLUMNS],
            )?;
            Ok(())
        }
    }

    #[test]
    fn poseidon_hash() {
        let rng = rand::rngs::OsRng;

        let message = [Fp::random(rng), Fp::random(rng)];
        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE, 2>::run(message.to_vec()).unwrap();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let k = 9;
        let circuit = HashCircuit::<PoseidonSpec, 2> {
            message: message.into(),
            _spec: PhantomData,
        };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }

    #[test]
    fn poseidon_hash_longer_input() {
        let rng = rand::rngs::OsRng;

        let message = [Fp::random(rng), Fp::random(rng), Fp::random(rng)];
        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE, 3>::run(message.to_vec()).unwrap();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let k = 9;
        let circuit = HashCircuit::<PoseidonSpec, 3> {
            message: message.into(),
            _spec: PhantomData,
        };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }

    #[test]
    #[ignore]
    fn poseidon_hash_much_longer_input() {
        let rng = rand::rngs::OsRng;

        let mut message: Vec<Fp> = (0..2048).map(|_| Fp::random(rng)).collect::<Vec<_>>();

        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE, 25>::run(message.clone()).unwrap();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let k = 17;
        let circuit = HashCircuit::<PoseidonSpec, 25> {
            message: message.into(),
            _spec: PhantomData,
        };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }
}