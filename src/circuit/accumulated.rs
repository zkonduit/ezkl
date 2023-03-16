use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use halo2curves::FieldExt;

use crate::tensor::{ops::accumulated, Tensor, TensorType, ValTensor, VarTensor};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use super::{utils, CircuitError};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BaseOp {
    Dot,
    InitDot,
    Add,
}

/// Matches a [BaseOp] to an operation over inputs
impl BaseOp {
    /// forward func
    pub fn f<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        &self,
        inputs: (T, T, T),
    ) -> T {
        let (a, b, m) = inputs;
        match &self {
            BaseOp::InitDot => a * b,
            BaseOp::Dot => a * b + m,
            BaseOp::Add => b + m,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            BaseOp::InitDot => "INITDOT",
            BaseOp::Dot => "DOT",
            BaseOp::Add => "ADD",
        }
    }
    fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::InitDot => (0, 1),
            BaseOp::Dot => (-1, 2),
            BaseOp::Add => (-1, 2),
        }
    }
    fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::InitDot => 0,
            BaseOp::Dot => 1,
            BaseOp::Add => 1,
        }
    }
}

impl fmt::Display for BaseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BaseOp::InitDot => write!(f, "base accum init dot"),
            BaseOp::Dot => write!(f, "base accum dot"),
            BaseOp::Add => write!(f, "base accum add"),
        }
    }
}

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Dot,
    Matmul,
    Affine,
}

/// Configuration for an accumulated arg.
#[derive(Clone, Debug)]
pub struct BaseConfig<F: FieldExt + TensorType> {
    /// the inputs to the accumulated operations.
    pub inputs: Vec<VarTensor>,
    /// the (currently singular) output of the accumulated operations.
    pub output: VarTensor,
    /// [Selectors] generated when configuring the layer. We use a BTreeMap as we expect to configure many base gates.
    pub selectors: BTreeMap<BaseOp, Selector>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> BaseConfig<F> {
    /// Configures the sequence of operations into a circuit gate.
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations.
    /// * `output` - The variable representing the (currently singular) output of the operations.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor; 2],
        output: &VarTensor,
    ) -> Self {
        // setup a selector per base op
        let mut selectors = BTreeMap::new();
        for input in inputs {
            // we don't support multiple columns rn
            assert!(input.num_cols() == 1);
        }
        selectors.insert(BaseOp::Dot, meta.selector());
        selectors.insert(BaseOp::InitDot, meta.selector());

        let config = Self {
            selectors,
            inputs: inputs.to_vec(),
            output: output.clone(),
            _marker: PhantomData,
        };

        for (base_op, selector) in config.selectors.iter() {
            meta.create_gate(base_op.as_str(), |meta| {
                let selector = meta.query_selector(*selector);

                let qis = config
                    .inputs
                    .iter()
                    .map(|input| {
                        input
                            .query_rng(meta, 0, 1)
                            .expect("accum: input query failed")[0]
                            .clone()
                    })
                    .collect::<Vec<_>>();

                // Get output expressions for each input channel
                let (offset, rng) = base_op.query_offset_rng();

                let expected_output: Tensor<Expression<F>> = config
                    .output
                    .query_rng(meta, offset, rng)
                    .expect("poly: output query failed");

                let res = base_op.f((qis[0].clone(), qis[1].clone(), expected_output[0].clone()));

                let constraints = vec![expected_output[base_op.constraint_idx()].clone() - res];

                Constraints::with_selector(selector, constraints)
            });
        }

        config
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    /// * `offset` - Offset to assign.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>],
        offset: usize,
        op: Op,
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        match op {
            Op::Dot => self.layout_dot(layouter, values.try_into()?, offset),
            Op::Matmul => self.layout_matmul(layouter, values.try_into()?, offset),
            Op::Affine => {
                let (kernel, bias, mut input) =
                    (values[0].clone(), values[1].clone(), values[2].clone());

                input.pad_row_ones()?;
                let params = kernel.append_to_row(bias)?;

                self.layout_matmul(layouter, &[params, input], offset)
            }
        }
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout_dot(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>; 2],
        offset: usize,
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        if values.len() != self.inputs.len() {
            return Err(Box::new(CircuitError::DimMismatch(
                "accum dot layout".to_string(),
            )));
        }

        let t = match layouter.assign_region(
            || "assign inputs",
            |mut region| {
                let mut inputs = vec![];
                for (i, input) in values.iter().enumerate() {
                    let inp = utils::value_muxer(
                        &self.inputs[i],
                        &{
                            let res = self.inputs[i].assign(&mut region, offset, input)?;
                            res.map(|e| e.value_field().evaluate())
                        },
                        input,
                    );
                    inputs.push(inp);
                }

                // Now we can assign the dot product
                let accumulated_dot = accumulated::dot(&inputs)
                    .expect("accum poly: dot op failed")
                    .into();
                let output = self.output.assign(&mut region, offset, &accumulated_dot)?;

                for i in 0..inputs[0].len() {
                    let (_, y) = self.inputs[0].cartesian_coord(i);
                    if y == 0 {
                        self.selectors
                            .get(&BaseOp::InitDot)
                            .unwrap()
                            .enable(&mut region, offset + y)?;
                    } else {
                        self.selectors
                            .get(&BaseOp::Dot)
                            .unwrap()
                            .enable(&mut region, offset + y)?;
                    }
                }

                // last element is the result
                Ok(output
                    .get_slice(&[output.len() - 1..output.len()])
                    .expect("accum poly: failed to fetch last elem"))
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                return Err(Box::new(e));
            }
        };

        Ok(ValTensor::from(t))
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout_matmul(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>; 2],
        offset: usize,
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        if values.len() != 2 {
            return Err(Box::new(CircuitError::DimMismatch(
                "accum matmul layout".to_string(),
            )));
        };

        let mut a = values[0].clone();
        let mut b = values[1].clone();
        b.transpose_2d()?;

        let num_a_repeats = b.dims()[0];
        let num_b_tiles = a.dims()[1];
        let b_row_len = b.dims()[1];

        a.repeat_rows(num_a_repeats)?;
        b.tile(num_b_tiles)?;

        let t = match layouter.assign_region(
            || "assign inputs",
            |mut region| {
                let mut inputs = vec![];

                for (i, elem) in vec![a.clone(), b.clone()].iter().enumerate() {
                    let inp = utils::value_muxer(
                        &self.inputs[i],
                        &{
                            let res = self.inputs[i].assign(&mut region, offset, elem)?;
                            res.map(|e| e.value_field().evaluate())
                        },
                        elem,
                    );
                    inputs.push(inp);
                }

                // remove any repeats from the assignment
                if num_a_repeats > 1 {
                    let dims = inputs[0].dims().to_vec();
                    inputs[0].reshape(&[dims[0], dims[1..].iter().product()]);
                    let mut rm_dup = vec![];
                    for i in 0..dims[0] {
                        rm_dup.push(inputs[0].get_slice(&[i..i + 1, 0..dims[1]]).unwrap());
                    }
                    inputs[0] = Tensor::new(Some(&rm_dup), &[rm_dup.len()])
                        .unwrap()
                        .combine()
                        .unwrap();
                }

                inputs[0].reshape(values[0].dims());

                // transpose it back to its normal shape
                inputs[1] = inputs[1].get_slice(&[0..1]).unwrap();
                inputs[1].reshape(&[values[1].dims()[1], values[1].dims()[0]]);
                inputs[1].transpose_2d().unwrap();

                // now perform matrix multiplication on the processed tensors
                let accumulated_matmul =
                    accumulated::matmul(&vec![inputs[0].clone(), inputs[1].clone()])
                        .expect("accum poly: matmul op failed");

                let output = self
                    .output
                    .assign(&mut region, offset, &accumulated_matmul.into())?;

                // these selectors map from
                for i in 0..a.dims().iter().product::<usize>() {
                    let (_, y) = self.inputs[0].cartesian_coord(i);
                    if (i) % b_row_len > 0 {
                        self.selectors
                            .get(&BaseOp::Dot)
                            .unwrap()
                            .enable(&mut region, offset + y)?;
                    } else {
                        self.selectors
                            .get(&BaseOp::InitDot)
                            .unwrap()
                            .enable(&mut region, offset + y)?;
                    }
                }

                let dims = output.dims();
                let mut last_dims = vec![];

                for d in &dims[0..dims.len() - 1] {
                    last_dims.push(0..*d);
                }
                let script_len = dims.last().unwrap();
                last_dims.push(script_len - 1..*script_len);

                let mut last_elem = output
                    .get_slice(&last_dims)
                    .expect("accum poly: failed to fetch last elem");

                last_elem.reshape(&[values[0].dims()[0], values[1].dims()[1]]);
                // Now we can assign the matmul op
                Ok(last_elem)
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                return Err(Box::new(e));
            }
        };

        Ok(ValTensor::from(t))
    }
}

#[cfg(test)]
mod matmul_test {
    use super::*;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    // use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    // use rand::rngs::OsRng;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct AffineCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for AffineCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true, 512);
            let output = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, 1, LEN], true, 512);
            Self::Config::configure(cs, &[a, b], &output)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config.layout(&mut layouter, &self.inputs.clone(), 0, Op::Matmul);
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]);

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]);

        let circuit = AffineCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dottest {
    use super::*;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    // use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    // use rand::rngs::OsRng;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);

            Self::Config::configure(cs, &[a, b], &output)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config.layout(&mut layouter, &self.inputs.clone(), 0, Op::Dot);
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod affinetest {
    use std::marker::PhantomData;

    use super::*;
    use crate::tensor::Tensor;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    // use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    // use rand::rngs::OsRng;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct AffineCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 3],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for AffineCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true, 512);
            let output =
                VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true, 512);
            Self::Config::configure(cs, &[a, b], &output)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config.layout(&mut layouter, &self.inputs.clone(), 0, Op::Affine);
            Ok(())
        }
    }

    #[test]
    fn affinecircuit() {
        // parameters
        let mut w = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, LEN]);

        let mut b = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        b.reshape(&[LEN, 1]);

        let mut x = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        x.reshape(&[LEN, 1]);

        let circuit = AffineCircuit::<F> {
            inputs: [ValTensor::from(w), ValTensor::from(b), ValTensor::from(x)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod compositiontest {
    use super::*;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    // use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    // use rand::rngs::OsRng;

    const K: usize = 9;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);

            Self::Config::configure(cs, &[a, b], &output)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // lots of stacked dot products
            let _ = config.layout(&mut layouter, &self.inputs.clone(), 0, Op::Dot);
            let _ = config.layout(&mut layouter, &self.inputs.clone(), LEN, Op::Dot);
            let _ = config.layout(&mut layouter, &self.inputs.clone(), 2 * LEN, Op::Dot);
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
