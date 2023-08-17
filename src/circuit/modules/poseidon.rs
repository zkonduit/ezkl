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
// use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSlice;

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
    pub instance: Option<Column<Instance>>,
    ///
    pub pow5_config: Pow5Config<Fp, WIDTH, RATE>,
}

type InputAssignments = (Vec<AssignedCell<Fp, Fp>>, AssignedCell<Fp, Fp>);

/// PoseidonChip is a wrapper around the Pow5Chip that adds a set of advice columns to the gadget Chip to store the inputs of the hash
#[derive(Debug, Clone)]
pub struct PoseidonChip<
    S: Spec<Fp, WIDTH, RATE> + Sync,
    const WIDTH: usize,
    const RATE: usize,
    const L: usize,
> {
    config: PoseidonConfig<WIDTH, RATE>,
    _marker: PhantomData<S>,
}

impl<S: Spec<Fp, WIDTH, RATE> + Sync, const WIDTH: usize, const RATE: usize, const L: usize>
    PoseidonChip<S, WIDTH, RATE, L>
{
    /// Creates a new PoseidonChip
    pub fn configure_with_cols(
        meta: &mut ConstraintSystem<Fp>,
        partial_sbox: Column<Advice>,
        rc_a: [Column<Fixed>; WIDTH],
        rc_b: [Column<Fixed>; WIDTH],
        hash_inputs: Vec<Column<Advice>>,
        instance: Option<Column<Instance>>,
    ) -> PoseidonConfig<WIDTH, RATE> {
        let pow5_config = Pow5Chip::configure::<S>(
            meta,
            hash_inputs.clone().try_into().unwrap(),
            partial_sbox,
            rc_a,
            rc_b,
        );

        PoseidonConfig {
            pow5_config,
            instance,
            hash_inputs,
        }
    }
}

impl<S: Spec<Fp, WIDTH, RATE> + Sync, const WIDTH: usize, const RATE: usize, const L: usize>
    Module<Fp> for PoseidonChip<S, WIDTH, RATE, L>
{
    type Config = PoseidonConfig<WIDTH, RATE>;
    type InputAssignments = InputAssignments;
    type RunInputs = Vec<Fp>;

    fn name(&self) -> &'static str {
        "Poseidon"
    }

    fn instance_increment_input(&self) -> Vec<usize> {
        vec![1]
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

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        Self::configure_with_cols(
            meta,
            partial_sbox,
            rc_a.try_into().unwrap(),
            rc_b.try_into().unwrap(),
            hash_inputs,
            Some(instance),
        )
    }

    fn layout_inputs(
        &self,
        layouter: &mut impl Layouter<Fp>,
        message: &[ValTensor<Fp>],
    ) -> Result<Self::InputAssignments, Error> {
        assert_eq!(message.len(), 1);
        let message = message[0].clone();

        let start_time = instant::Instant::now();

        let res = layouter.assign_region(
            || "load message",
            |mut region| {
                let assigned_message: Result<Vec<AssignedCell<Fp, Fp>>, Error> = match &message {
                    ValTensor::Value { inner: v, .. } => v
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let x = i % WIDTH;
                            let y = i / WIDTH;

                            match value {
                                ValType::Value(v) => region.assign_advice(
                                    || format!("load message_{}", i),
                                    self.config.hash_inputs[x],
                                    y,
                                    || *v,
                                ),
                                ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                                    Ok(v.clone())
                                }
                                ValType::Constant(f) => region.assign_advice_from_constant(
                                    || format!("load message_{}", i),
                                    self.config.hash_inputs[x],
                                    y,
                                    *f,
                                ),
                                e => {
                                    panic!("wrong input type {:?}, must be previously assigned", e)
                                }
                            }
                        })
                        .collect(),
                    ValTensor::Instance {
                        inner: col, dims, ..
                    } => {
                        // this should never ever fail
                        let num_elems = dims.iter().product::<usize>();
                        (0..num_elems)
                            .map(|i| {
                                let x = i % WIDTH;
                                let y = i / WIDTH;
                                region.assign_advice_from_instance(
                                    || "pub input anchor",
                                    *col,
                                    i,
                                    self.config.hash_inputs[x],
                                    y,
                                )
                            })
                            .collect()
                    }
                };

                let offset = message.len() / WIDTH + 1;

                let zero_val = region
                    .assign_advice_from_constant(
                        || "",
                        self.config.hash_inputs[0],
                        offset,
                        Fp::ZERO,
                    )
                    .unwrap();

                Ok((assigned_message?, zero_val))
            },
        );
        log::trace!(
            "input (N={:?}) layout took: {:?}",
            message.len(),
            start_time.elapsed()
        );
        res
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
        let (mut input_cells, zero_val) = self.layout_inputs(layouter, input)?;
        // extract the values from the input cells
        let assigned_input: Tensor<ValType<Fp>> =
            input_cells.iter().map(|e| ValType::from(e.clone())).into();
        let len = assigned_input.len();

        let start_time = instant::Instant::now();

        // do the Tree dance baby
        while input_cells.len() > 1 {
            let hashes: Result<Vec<AssignedCell<Fp, Fp>>, Error> = input_cells
                .chunks(L)
                .enumerate()
                .map(|(i, block)| {
                    let _start_time = instant::Instant::now();

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

                    let hash = hasher.hash(
                        layouter.namespace(|| "hash"),
                        block.to_vec().try_into().map_err(|_| Error::Synthesis)?,
                    );

                    if i == 0 {
                        log::trace!("block (L={:?}) took: {:?}", L, _start_time.elapsed());
                    }

                    hash
                })
                .collect();

            log::trace!("hashes (N={:?}) took: {:?}", len, start_time.elapsed());

            input_cells = hashes?;
        }

        let duration = start_time.elapsed();
        log::trace!("layout (N={:?}) took: {:?}", len, duration);

        let result = Tensor::from(input_cells.iter().map(|e| ValType::from(e.clone())));

        let output = match result[0].clone() {
            ValType::PrevAssigned(v) => v,
            _ => panic!(),
        };

        if let Some(instance) = self.config.instance {
            layouter.assign_region(
                || "constrain output",
                |mut region| {
                    let expected_var = region.assign_advice_from_instance(
                        || "pub input anchor",
                        instance,
                        row_offset[0],
                        self.config.hash_inputs[0],
                        0,
                    )?;

                    region.constrain_equal(output.cell(), expected_var.cell())
                },
            )?;

            Ok(assigned_input.into())
        } else {
            Ok(result.into())
        }
    }

    ///
    fn run(message: Vec<Fp>) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        let mut hash_inputs = message;
        let len = hash_inputs.len();

        let start_time = instant::Instant::now();

        // do the Tree dance baby
        while hash_inputs.len() > 1 {
            let hashes: Vec<Fp> = hash_inputs
                .par_chunks(L)
                .map(|block| {
                    let mut block = block.to_vec();
                    let remainder = block.len() % L;

                    if remainder != 0 {
                        block.extend(vec![Fp::ZERO; L - remainder].iter());
                    }
                    halo2_gadgets::poseidon::primitives::Hash::<
                        _,
                        S,
                        ConstantLength<L>,
                        { WIDTH },
                        { RATE },
                    >::init()
                    .hash(block.clone().try_into().unwrap())
                })
                .collect();

            hash_inputs = hashes;
        }

        let duration = start_time.elapsed();
        log::trace!("run (N={:?}) took: {:?}", len, duration);

        Ok(vec![hash_inputs])
    }

    fn num_rows(mut input_len: usize) -> usize {
        // this was determined by running the circuit and looking at the number of constraints
        // in the test called hash_for_a_range_of_input_sizes, then regressing in python to find the slope
        let fixed_cost: usize = 41 * L;

        let mut num_rows = 0;

        loop {
            // the number of times the input_len is divisible by L
            let num_chunks = input_len / L + 1;
            num_rows += num_chunks * fixed_cost;
            if num_chunks == 1 {
                break;
            }
            input_len = num_chunks;
        }

        num_rows
    }
}

#[allow(unused)]
mod tests {

    use crate::circuit::modules::ModulePlanner;

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
        type FloorPlanner = ModulePlanner;
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
        assert_eq!(prover.verify_par(), Ok(()))
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
        assert_eq!(prover.verify_par(), Ok(()))
    }

    #[test]
    #[ignore]
    fn hash_for_a_range_of_input_sizes() {
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
            let output =
                PoseidonChip::<PoseidonSpec, WIDTH, RATE, 32>::run(message.clone()).unwrap();

            let mut message: Tensor<ValType<Fp>> =
                message.into_iter().map(|m| Value::known(m).into()).into();

            let k = 17;
            let circuit = HashCircuit::<PoseidonSpec, 32> {
                message: message.into(),
                _spec: PhantomData,
            };
            let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();

            assert_eq!(prover.verify_par(), Ok(()))
        }
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
        assert_eq!(prover.verify_par(), Ok(()))
    }
}
