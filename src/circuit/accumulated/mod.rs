/// Accumulated matmul op
pub mod affine;
/// Accumulated dot product
pub mod dot;
/// Accumulated matmul op
pub mod matmul;

use halo2_proofs::plonk::{ConstraintSystem, Constraints, Expression, Selector};
use halo2curves::FieldExt;

use crate::tensor::{Tensor, TensorType, VarTensor};
use std::{
    collections::BTreeMap,
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
    Add,
}

/// Matches a [Op] to an operation in the `tensor::ops` module.
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

/// Configuration for an accumulated arg.
#[derive(Clone, Debug)]
pub struct BaseConfig<F: FieldExt + TensorType> {
    /// the inputs to the fused operations.
    pub inputs: Vec<VarTensor>,
    /// the (currently singular) output of the fused operations.
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
        assert!(inputs[0].num_cols() == 1);
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
}
