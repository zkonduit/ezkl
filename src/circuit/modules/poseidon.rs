/*
An easy-to-use implementation of the Poseidon Hash in the form of a Halo2 Chip. While the Poseidon Hash function
is already implemented in halo2_gadgets, there is no wrapper chip that makes it easy to use in other circuits.
Thanks to https://github.com/summa-dev/summa-solvency/blob/master/zk_prover/src/chips/poseidon/hash.rs for the inspiration (and also helping us understand how to use this).
*/

pub mod poseidon_params;
pub mod spec;

// This chip adds a set of advice columns to the gadget Chip to store the inputs of the hash
use halo2_gadgets::poseidon::{
    primitives::VariableLength, primitives::*, Hash, Pow5Chip, Pow5Config,
};
use halo2_proofs::halo2curves::bn256::Fr as Fp;
use halo2_proofs::{circuit::*, plonk::*};

use std::marker::PhantomData;

use crate::circuit::region::ConstantsMap;
use crate::tensor::{Tensor, ValTensor, ValType};

use super::errors::ModuleError;
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

type InputAssignments = Vec<AssignedCell<Fp, Fp>>;

/// PoseidonChip is a wrapper around the Pow5Chip that adds a set of advice columns to the gadget Chip to store the inputs of the hash
#[derive(Debug, Clone)]
pub struct PoseidonChip<S: Spec<Fp, WIDTH, RATE> + Sync, const WIDTH: usize, const RATE: usize> {
    config: PoseidonConfig<WIDTH, RATE>,
    _marker: PhantomData<S>,
}

impl<S: Spec<Fp, WIDTH, RATE> + Sync, const WIDTH: usize, const RATE: usize>
    PoseidonChip<S, WIDTH, RATE>
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

impl<S: Spec<Fp, WIDTH, RATE> + Sync, const WIDTH: usize, const RATE: usize>
    PoseidonChip<S, WIDTH, RATE>
{
    /// Configuration of the PoseidonChip
    pub fn configure_with_optional_instance(
        meta: &mut ConstraintSystem<Fp>,
        instance: Option<Column<Instance>>,
    ) -> PoseidonConfig<WIDTH, RATE> {
        //  instantiate the required columns
        let hash_inputs = (0..WIDTH).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for input in &hash_inputs {
            meta.enable_equality(*input);
        }

        let partial_sbox = meta.advice_column();
        let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

        meta.enable_constant(rc_b[0]);

        Self::configure_with_cols(
            meta,
            partial_sbox,
            rc_a.try_into().unwrap(),
            rc_b.try_into().unwrap(),
            hash_inputs,
            instance,
        )
    }
}

impl<S: Spec<Fp, WIDTH, RATE> + Sync, const WIDTH: usize, const RATE: usize> Module<Fp>
    for PoseidonChip<S, WIDTH, RATE>
{
    type Config = PoseidonConfig<WIDTH, RATE>;
    type InputAssignments = InputAssignments;
    type RunInputs = Vec<Fp>;
    type Params = ();

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
    fn configure(meta: &mut ConstraintSystem<Fp>, _: Self::Params) -> Self::Config {
        //  instantiate the required columns
        let hash_inputs = (0..WIDTH).map(|_| meta.advice_column()).collect::<Vec<_>>();
        for input in &hash_inputs {
            meta.enable_equality(*input);
        }

        let partial_sbox = meta.advice_column();
        let rc_a = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..WIDTH).map(|_| meta.fixed_column()).collect::<Vec<_>>();

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
        constants: &mut ConstantsMap<Fp>,
    ) -> Result<Self::InputAssignments, ModuleError> {
        if message.len() != 1 {
            return Err(ModuleError::InputWrongLength(message.len()));
        }

        let message = message[0].clone();

        let start_time = instant::Instant::now();

        let local_constants = constants.clone();

        let res = layouter.assign_region(
            || "load message",
            |mut region| {
                let assigned_message: Result<Vec<AssignedCell<Fp, Fp>>, _> = match &message {
                    ValTensor::Value { inner: v, .. } => v
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let x = i % WIDTH;
                            let y = i / WIDTH;

                            match value {
                                ValType::Value(v) => region
                                    .assign_advice(
                                        || format!("load message_{}", i),
                                        self.config.hash_inputs[x],
                                        y,
                                        || *v,
                                    )
                                    .map_err(|e| e.into()),
                                ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                                    Ok(v.clone())
                                }
                                ValType::Constant(f) => {
                                    if local_constants.contains_key(f) {
                                        Ok(constants
                                            .get(f)
                                            .unwrap()
                                            .assigned_cell()
                                            .ok_or(ModuleError::ConstantNotAssigned)?)
                                    } else {
                                        let res = region.assign_advice_from_constant(
                                            || format!("load message_{}", i),
                                            self.config.hash_inputs[x],
                                            y,
                                            *f,
                                        )?;

                                        constants
                                            .insert(*f, ValType::AssignedConstant(res.clone(), *f));

                                        Ok(res)
                                    }
                                }
                                e => Err(ModuleError::WrongInputType(
                                    format!("{:?}", e),
                                    "AssignedValue".to_string(),
                                )),
                            }
                        })
                        .collect(),
                    ValTensor::Instance {
                        dims,
                        inner: col,
                        idx,
                        initial_offset,
                        ..
                    } => {
                        // this should never ever fail
                        let num_elems = dims[*idx].iter().product::<usize>();
                        (0..num_elems)
                            .map(|i| {
                                let x = i % WIDTH;
                                let y = i / WIDTH;
                                region.assign_advice_from_instance(
                                    || "pub input anchor",
                                    *col,
                                    initial_offset + i,
                                    self.config.hash_inputs[x],
                                    y,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| e.into())
                    }
                };

                Ok(assigned_message?)
            },
        );
        log::trace!(
            "input (N={:?}) layout took: {:?}",
            message.len(),
            start_time.elapsed()
        );
        res.map_err(|e| e.into())
    }

    /// L is the number of inputs to the hash function
    /// Takes the cells containing the input values of the hash function and return the cell containing the hash output
    /// It uses the pow5_chip to compute the hash
    fn layout(
        &self,
        layouter: &mut impl Layouter<Fp>,
        input: &[ValTensor<Fp>],
        row_offset: usize,
        constants: &mut ConstantsMap<Fp>,
    ) -> Result<ValTensor<Fp>, ModuleError> {
        let input_cells = self.layout_inputs(layouter, input, constants)?;

        // empty hash case
        if input_cells.is_empty() {
            return Ok(input[0].clone());
        }

        // extract the values from the input cells
        let mut assigned_input: Tensor<ValType<Fp>> =
            input_cells.iter().map(|e| ValType::from(e.clone())).into();
        let len = assigned_input.len();

        let start_time = instant::Instant::now();

        let pow5_chip = Pow5Chip::construct(self.config.pow5_config.clone());
        // initialize the hasher
        let hasher = Hash::<_, _, S, VariableLength, WIDTH, RATE>::init(
            pow5_chip,
            layouter.namespace(|| "block_hasher"),
        )?;

        let hash: AssignedCell<Fp, Fp> = hasher.hash(
            layouter.namespace(|| "hash"),
            input_cells
                .to_vec()
                .try_into()
                .map_err(|_| Error::Synthesis)?,
        )?;

        let duration = start_time.elapsed();
        log::trace!("layout (N={:?}) took: {:?}", len, duration);

        let result = Tensor::from(vec![ValType::from(hash.clone())].into_iter());

        let output = match result[0].clone() {
            ValType::PrevAssigned(v) => v,
            _ => {
                log::error!("wrong input type, must be previously assigned");
                return Err(Error::Synthesis.into());
            }
        };

        if let Some(instance) = self.config.instance {
            layouter.assign_region(
                || "constrain output",
                |mut region| {
                    let expected_var = region.assign_advice_from_instance(
                        || "pub input anchor",
                        instance,
                        row_offset,
                        self.config.hash_inputs[0],
                        0,
                    )?;

                    region.constrain_equal(output.cell(), expected_var.cell())
                },
            )?;

            assigned_input.reshape(input[0].dims()).map_err(|e| {
                log::error!("reshape failed: {:?}", e);
                Error::Synthesis
            })?;

            Ok(assigned_input.into())
        } else {
            Ok(result.into())
        }
    }

    ///
    fn run(message: Vec<Fp>) -> Result<Vec<Vec<Fp>>, ModuleError> {
        let len = message.len();
        if len == 0 {
            return Ok(vec![vec![]]);
        }

        let start_time = instant::Instant::now();

        let hash = halo2_gadgets::poseidon::primitives::Hash::<
            _,
            S,
            VariableLength,
            { WIDTH },
            { RATE },
        >::init()
        .hash(message);

        let duration = start_time.elapsed();
        log::trace!("run (N={:?}) took: {:?}", len, duration);

        Ok(vec![vec![hash]])
    }

    fn num_rows(input_len: usize) -> usize {
        // this was determined by running the circuit and looking at the number of constraints
        // in the test called hash_for_a_range_of_input_sizes, then regressing in python to find the slope
        // import numpy as np
        // from scipy import stats

        // x = np.array([32, 64, 96, 128, 160, 192])
        // y = np.array([1298, 2594, 3890, 5186, 6482, 7778])

        // slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        // print(f"slope: {slope}")
        // print(f"intercept: {intercept}")
        // print(f"R^2: {r_value**2}")

        // # Predict for any x
        // def predict(x):
        // return slope * x + intercept

        // # Test prediction
        // test_x = 256
        // print(f"Predicted value for x={test_x}: {predict(test_x)}")
        // our output:
        // slope: 40.5
        // intercept: 2.0
        // R^2: 1.0
        // Predicted value for x=256: 10370.0
        let fixed_cost: usize = 41 * input_len;

        // the cost of the hash function is linear with the number of inputs
        fixed_cost + 2
    }
}

#[allow(unused)]
mod tests {

    use crate::circuit::modules::ModulePlanner;

    use super::{
        spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
        *,
    };

    use std::{collections::HashMap, marker::PhantomData};

    use halo2_gadgets::poseidon::primitives::Spec;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem},
    };
    use halo2curves::ff::Field;

    const WIDTH: usize = POSEIDON_WIDTH;
    const RATE: usize = POSEIDON_RATE;
    const R: usize = 240;

    struct HashCircuit<S: Spec<Fp, WIDTH, RATE>> {
        message: ValTensor<Fp>,
        _spec: PhantomData<S>,
    }

    impl<S: Spec<Fp, WIDTH, RATE>> Circuit<Fp> for HashCircuit<S> {
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
            PoseidonChip::<PoseidonSpec, WIDTH, RATE>::configure(meta, ())
        }

        fn synthesize(
            &self,
            config: PoseidonConfig<WIDTH, RATE>,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            let chip: PoseidonChip<PoseidonSpec, WIDTH, RATE> = PoseidonChip::new(config);
            chip.layout(
                &mut layouter,
                &[self.message.clone()],
                0,
                &mut HashMap::new(),
            )?;

            Ok(())
        }
    }

    #[test]
    fn poseidon_hash_empty() {
        let message = [];
        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(message.to_vec()).unwrap();
        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();
        let k = 9;
        let circuit = HashCircuit::<PoseidonSpec> {
            message: message.into(),
            _spec: PhantomData,
        };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, vec![vec![]]).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }

    #[test]
    fn poseidon_hash() {
        let rng = rand::rngs::OsRng;

        let message = [Fp::random(rng), Fp::random(rng)];
        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(message.to_vec()).unwrap();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let k = 9;
        let circuit = HashCircuit::<PoseidonSpec> {
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
        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(message.to_vec()).unwrap();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let k = 9;
        let circuit = HashCircuit::<PoseidonSpec> {
            message: message.into(),
            _spec: PhantomData,
        };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }

    #[test]
    #[ignore]
    fn hash_for_a_range_of_input_sizes() {
        let rng = rand::rngs::OsRng;

        #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
        env_logger::init();

        for i in (32..128).step_by(32) {
            // print a bunch of new lines
            log::info!(
                "i is {} -------------------------------------------------",
                i
            );

            let message: Vec<Fp> = (0..i).map(|_| Fp::random(rng)).collect::<Vec<_>>();
            let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(message.clone()).unwrap();

            let mut message: Tensor<ValType<Fp>> =
                message.into_iter().map(|m| Value::known(m).into()).into();

            let k = 17;
            let circuit = HashCircuit::<PoseidonSpec> {
                message: message.into(),
                _spec: PhantomData,
            };
            let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();

            assert_eq!(prover.verify(), Ok(()))
        }
    }

    #[test]
    #[ignore]
    fn poseidon_hash_much_longer_input() {
        let rng = rand::rngs::OsRng;

        let mut message: Vec<Fp> = (0..2048).map(|_| Fp::random(rng)).collect::<Vec<_>>();

        let output = PoseidonChip::<PoseidonSpec, WIDTH, RATE>::run(message.clone()).unwrap();

        let mut message: Tensor<ValType<Fp>> =
            message.into_iter().map(|m| Value::known(m).into()).into();

        let k = 17;
        let circuit = HashCircuit::<PoseidonSpec> {
            message: message.into(),
            _spec: PhantomData,
        };
        let prover = halo2_proofs::dev::MockProver::run(k, &circuit, output).unwrap();
        assert_eq!(prover.verify(), Ok(()))
    }
}
