/// Layouts for specific functions (composed of base ops)
pub mod layouts;

/// Tests
#[cfg(test)]
mod tests;

use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use halo2curves::FieldExt;

use crate::tensor::{Tensor, TensorError, TensorType, ValTensor, VarTensor};
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
    Identity,
    Add,
    Mult,
    Sub,
    Sum,
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
            BaseOp::Add => a + b,
            BaseOp::Identity => a,
            BaseOp::Sum => a + m,
            BaseOp::Sub => a - b,
            BaseOp::Mult => a * b,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            BaseOp::InitDot => "INITDOT",
            BaseOp::Identity => "IDENTITY",
            BaseOp::Dot => "DOT",
            BaseOp::Add => "ADD",
            BaseOp::Sub => "SUB",
            BaseOp::Mult => "MULT",
            BaseOp::Sum => "SUM",
        }
    }
    fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::InitDot => (0, 1),
            BaseOp::Identity => (0, 1),
            BaseOp::Dot => (-1, 2),
            BaseOp::Add => (0, 1),
            BaseOp::Sub => (0, 1),
            BaseOp::Mult => (0, 1),
            BaseOp::Sum => (-1, 2),
        }
    }
    fn num_inputs(&self) -> usize {
        match self {
            BaseOp::InitDot => 2,
            BaseOp::Identity => 1,
            BaseOp::Dot => 2,
            BaseOp::Add => 2,
            BaseOp::Sub => 2,
            BaseOp::Mult => 2,
            BaseOp::Sum => 1,
        }
    }
    fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::InitDot => 0,
            BaseOp::Identity => 0,
            BaseOp::Dot => 1,
            BaseOp::Add => 0,
            BaseOp::Sub => 0,
            BaseOp::Mult => 0,
            BaseOp::Sum => 1,
        }
    }
}

impl fmt::Display for BaseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
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
    SumPool {
        padding: (usize, usize),
        stride: (usize, usize),
        kernel_shape: (usize, usize),
    },
    Add,
    Sub,
    Mult,
    Identity,
    Reshape(Vec<usize>),
    Flatten(Vec<usize>),
    BatchNorm,
    ScaleAndShift,
    Pad(usize, usize),
    Sum,
    Pow(u32),
    Pack(u32, u32),
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
        selectors.insert(BaseOp::Add, meta.selector());
        selectors.insert(BaseOp::Sub, meta.selector());
        selectors.insert(BaseOp::Dot, meta.selector());
        selectors.insert(BaseOp::Sum, meta.selector());
        selectors.insert(BaseOp::Mult, meta.selector());
        selectors.insert(BaseOp::InitDot, meta.selector());
        selectors.insert(BaseOp::Identity, meta.selector());

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

                let mut qis = vec![Expression::<F>::zero().unwrap(); 2];
                for i in 0..base_op.num_inputs() {
                    qis[i] = config.inputs[i]
                        .query_rng(meta, 0, 1)
                        .expect("accum: input query failed")[0]
                        .clone()
                }

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
            Op::Sum => layouts::sum(self, layouter, values.try_into()?, offset),
            Op::Matmul => layouts::matmul(self, layouter, values.try_into()?, offset),
            Op::Affine => layouts::affine(self, layouter, values.try_into()?, offset),
            Op::Conv { padding, stride } => {
                layouts::conv(self, layouter, values.try_into()?, padding, stride, offset)
            }
            Op::SumPool {
                padding,
                stride,
                kernel_shape,
            } => layouts::sumpool(
                self,
                layouter,
                values.try_into()?,
                padding,
                stride,
                kernel_shape,
                offset,
            ),
            Op::Add => layouts::pairwise(self, layouter, values.try_into()?, offset, BaseOp::Add),
            Op::Sub => layouts::pairwise(self, layouter, values.try_into()?, offset, BaseOp::Sub),
            Op::Mult => layouts::pairwise(self, layouter, values.try_into()?, offset, BaseOp::Mult),
            Op::Identity => layouts::identity(values.try_into()?),
            Op::Reshape(d) | Op::Flatten(d) => layouts::reshape(values.try_into()?, &d),
            Op::BatchNorm => layouts::scale_and_shift(self, layouter, values.try_into()?, offset),
            Op::ScaleAndShift => {
                layouts::scale_and_shift(self, layouter, values.try_into()?, offset)
            }
            Op::Pad(p1, p2) => {
                if values.len() != 1 {
                    return Err(Box::new(TensorError::DimError));
                }
                let mut input = values[0].clone();
                input.pad((p1, p2))?;
                Ok(input)
            }
            Op::Pow(exp) => layouts::pow(self, layouter, values.try_into()?, exp, offset),
            Op::Pack(base, scale) => {
                layouts::pack(self, layouter, values.try_into()?, base, scale, offset)
            }
        }
    }
}
