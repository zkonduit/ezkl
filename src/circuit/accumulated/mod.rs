/// Layouts for specific functions (composed of base ops)
pub mod layouts;

use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use halo2curves::FieldExt;

use crate::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BaseOp {
    Dot,
    InitDot,
}

#[allow(missing_docs)]
/// An enum representing activating the sanity checks we can perform on the accumulated arguments
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CheckMode {
    SAFE,
    UNSAFE,
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
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            BaseOp::InitDot => "INITDOT",
            BaseOp::Dot => "DOT",
        }
    }
    fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::InitDot => (0, 1),
            BaseOp::Dot => (-1, 2),
        }
    }
    fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::InitDot => 0,
            BaseOp::Dot => 1,
        }
    }
}

impl fmt::Display for BaseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BaseOp::InitDot => write!(f, "base accum init dot"),
            BaseOp::Dot => write!(f, "base accum dot"),
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
    Conv {
        padding: (usize, usize),
        stride: (usize, usize),
    },
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
    /// Activate sanity checks
    pub check_mode: CheckMode,
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
        check_mode: CheckMode,
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
            check_mode,
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
            Op::Dot => layouts::dot(self, layouter, values.try_into()?, offset),
            Op::Matmul => layouts::matmul(self, layouter, values.try_into()?, offset),
            Op::Affine => layouts::affine(self, layouter, values.try_into()?, offset),
            Op::Conv { padding, stride } => {
                layouts::conv(self, layouter, values.try_into()?, padding, stride, offset)
            }
        }
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
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config
                .layout(&mut layouter, &self.inputs.clone(), 0, Op::Matmul)
                .unwrap();
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

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config
                .layout(&mut layouter, &self.inputs.clone(), 0, Op::Dot)
                .unwrap();
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
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config
                .layout(&mut layouter, &self.inputs.clone(), 0, Op::Affine)
                .unwrap();
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
            inputs: [ValTensor::from(x), ValTensor::from(w), ValTensor::from(b)],
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

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // lots of stacked dot products
            let _ = config
                .layout(&mut layouter, &self.inputs.clone(), 0, Op::Dot)
                .unwrap();
            let _ = config
                .layout(&mut layouter, &self.inputs.clone(), LEN, Op::Dot)
                .unwrap();
            let _ = config
                .layout(&mut layouter, &self.inputs.clone(), 2 * LEN, Op::Dot)
                .unwrap();
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
mod convtest {
    use std::marker::PhantomData;

    use super::*;
    use crate::tensor::Tensor;
    use halo2_proofs::{
        arithmetic::{Field, FieldExt},
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    use rand::rngs::OsRng;

    const K: usize = 22;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: FieldExt + TensorType> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true, 100000);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true, 100000);
            let output =
                VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true, 100000);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config
                .layout(
                    &mut layouter,
                    &self.inputs.clone(),
                    0,
                    Op::Conv {
                        padding: (1, 1),
                        stride: (2, 2),
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn convcircuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 3;
        let image_height = 5;
        let image_width = 7;
        let in_channels = 3;
        let out_channels = 2;

        let mut image = Tensor::from(
            (0..in_channels * image_height * image_width)
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        image.reshape(&[in_channels, image_height, image_width]);
        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        kernels.reshape(&[out_channels, in_channels, kernel_height, kernel_width]);

        let bias =
            Tensor::from((0..{ out_channels }).map(|_| Value::known(pallas::Base::random(OsRng))));

        let circuit = ConvCircuit::<F> {
            inputs: [
                ValTensor::from(image),
                ValTensor::from(kernels),
                ValTensor::from(bias),
            ]
            .to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn convcircuitnobias() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = 4;
        let image_width = 5;
        let in_channels = 3;
        let out_channels = 2;

        let mut image = Tensor::from(
            (0..in_channels * image_height * image_width).map(|i| Value::known(F::from(i as u64))),
        );
        image.reshape(&[in_channels, image_height, image_width]);
        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| Value::known(F::from(i as u64))),
        );
        kernels.reshape(&[out_channels, in_channels, kernel_height, kernel_width]);

        let circuit = ConvCircuit::<F> {
            inputs: [ValTensor::from(image), ValTensor::from(kernels)].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
