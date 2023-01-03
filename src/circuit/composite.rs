use super::*;
use crate::circuit::lookup::Config as LookupConfig;
use crate::circuit::lookup::Op as LookupOp;
use crate::circuit::polynomial::Config as PolyConfig;
use crate::circuit::polynomial::InputType as PolyInputType;
use crate::circuit::polynomial::Node as PolyNode;
use crate::circuit::polynomial::Op as PolyOp;
use crate::graph::scale_to_multiplier;
use crate::tensor::TensorType;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Selector},
};
use std::fmt;
use std::marker::PhantomData;

#[allow(missing_docs)]
/// An enum representing the operations that require both lookup and polynomials
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    PReLU { scale: eq_float::F32 },
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::PReLU { scale } => {
                write!(f, "prelu w/ scale: {}", scale)
            }
        }
    }
}

/// Representation of a the inputs a [Node] can ingest. The inner type indexes over each of the types.
#[derive(Clone, Debug)]
pub enum InputType {
    /// an explicit input to the operations
    Input(usize),
    /// the intermediate output of a [Node]
    Inter(usize),
}

/// Configuration for a basic sequence of operations all fused together in a single gate.
#[derive(Clone, Debug)]
pub struct Config<F: FieldExt + TensorType> {
    /// the Op represented.
    op: Op,
    ///
    pub poly_configs: Vec<PolyConfig<F>>,
    ///
    pub lookup_configs: Vec<LookupConfig<F>>,
    /// [Selector] generated when configuring the layer.
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Config<F> {
    /// Configures the sequence of operations into a circuit gate, represented as an array of [Node].
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations. [Node]s index over these inputs using their `input_order` attribute. They can also index over the intermediate outputs of other [Node]s.
    /// * `output` - The variable representing the (currently singular) output of the fused operations.
    /// * `nodes` - The sequence of operations (in order of execution) that constitute the fused operation.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor],
        outputs: &[VarTensor],
        network_scale: i32,
        bits: usize,
        op: &Op,
    ) -> Self {
        let mut lookup_configs = vec![];
        let mut poly_configs = vec![];
        println!("{:?}", inputs[0].dims());
        match op {
            Op::PReLU { scale } => {
                println!("{:?}", scale);

                let relu: LookupConfig<F> = LookupConfig::configure(
                    meta,
                    &inputs[0],
                    &outputs[0],
                    bits,
                    &[LookupOp::ReLU {
                        scale: eq_float::F32(scale.0 / scale_to_multiplier(network_scale)),
                    }],
                );

                let neg = PolyConfig::configure(
                    meta,
                    &[inputs[0].clone()],
                    &outputs[0],
                    &[PolyNode {
                        op: PolyOp::Neg,
                        input_order: vec![PolyInputType::Input(0)],
                    }],
                );
                let relu_neg: LookupConfig<F> = LookupConfig::configure(
                    meta,
                    &neg.output,
                    &outputs[0],
                    bits,
                    &[LookupOp::ReLU { scale: *scale }],
                );
                let prelu = PolyConfig::configure(
                    meta,
                    &[inputs[0].clone(), inputs[1].clone(), inputs[2].clone()],
                    &outputs[0],
                    &[
                        // relu(-x) * slope
                        PolyNode {
                            op: PolyOp::Mult,
                            input_order: vec![PolyInputType::Input(0), PolyInputType::Input(1)],
                        },
                        // relu(x) - relu(-x) * slope
                        PolyNode {
                            op: PolyOp::Sub,
                            input_order: vec![PolyInputType::Input(2), PolyInputType::Inter(0)],
                        },
                    ],
                );
                poly_configs.append(&mut vec![neg, prelu]);
                lookup_configs.append(&mut vec![relu.clone(), relu_neg]);
            }
        };

        Self {
            selector: meta.selector(),
            op: op.clone(),
            poly_configs,
            lookup_configs,
            _marker: PhantomData,
        }
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `inputs` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        inputs: &[ValTensor<F>],
    ) -> ValTensor<F> {
        match &self.op {
            Op::PReLU { .. } => {
                let relu = self.lookup_configs[0].layout(layouter, &inputs[0]);
                let neg = self.poly_configs[0].layout(layouter, &[inputs[0].clone()]);
                let relu_neg = self.lookup_configs[1].layout(layouter, &neg);
                let prelu = self.poly_configs[1]
                    .layout(layouter, &[relu_neg, inputs[1].clone(), relu.clone()]);
                prelu
            }
        }
    }
}
