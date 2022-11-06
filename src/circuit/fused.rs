use super::*;
use crate::abort;
use crate::tensor::ops::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use itertools::Itertools;
use log::error;
use std::fmt;
use std::marker::PhantomData;

/// An enum representing the operations that can be merged into a single circuit gate.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FusedOp {
    Identity,
    Reshape(Vec<usize>),
    Add,
    Sub,
    Sum,
    Mult,
    Matmul,
    Dot,
    Affine,
    BatchNorm,
    ScaleAndShift,
    Conv((usize, usize), (usize, usize)), // padding, stride
    SumPool((usize, usize), (usize, usize), (usize, usize)), // padding, stride, kernel_shape
    Pow(usize),
    Rescaled(Box<FusedOp>, Vec<(usize, usize)>),
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FusedOp::Identity => write!(f, "identity"),
            FusedOp::Reshape(new_dims) => write!(f, "reshape to {:?}", new_dims),
            FusedOp::Add => write!(f, "add"),
            FusedOp::Sub => write!(f, "sub"),
            FusedOp::Sum => write!(f, "sum"),
            FusedOp::Mult => write!(f, "mult"),
            FusedOp::Matmul => write!(f, "matmul"),
            FusedOp::Dot => write!(f, "dot"),
            FusedOp::Affine => write!(f, "affine"),
            FusedOp::BatchNorm => write!(f, "batchnorm"),
            FusedOp::ScaleAndShift => write!(f, "scale & shift"),
            FusedOp::Conv(padding, stride) => {
                write!(f, "conv w/ padding: {:?}, stride: {:?}", padding, stride)
            }
            FusedOp::SumPool(padding, stride, kernel_shape) => {
                write!(
                    f,
                    "avg pl w/ padding: {:?}, stride: {:?}, kernel shape: {:?}",
                    padding, stride, kernel_shape,
                )
            }
            FusedOp::Pow(s) => write!(f, "pow {}", s),
            FusedOp::Rescaled(s, m) => {
                write!(
                    f,
                    "{} w/ scalings: {:?}",
                    **s,
                    m.iter().map(|e| e.1).collect_vec()
                )
            }
        }
    }
}

/// Representation of a the inputs a [FusedNode] can ingest. The inner type indexes over each of the types.
#[derive(Clone, Debug)]
pub enum FusedInputType {
    /// an explicit input to the operations
    Input(usize),
    /// the intermediate output of a [FusedNode]
    Inter(usize),
}

/// Representation of a single fuseable operation.
#[derive(Clone, Debug)]
pub struct FusedNode {
    /// the type of operation
    pub op: FusedOp,
    /// execution order over explicit inputs and intermediate outputs.
    pub input_order: Vec<FusedInputType>,
}

/// Configuration for a basic sequence of operations all fused together in a single gate.
#[derive(Clone, Debug)]
pub struct FusedConfig<F: FieldExt + TensorType> {
    /// the inputs to the fused operations.
    pub inputs: Vec<VarTensor>,
    /// the set of [FusedNode] represented in the operation.
    nodes: Vec<FusedNode>,
    /// the (currently singular) output of the fused operations.
    pub output: VarTensor,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> FusedConfig<F> {
    /// Configures the sequence of operations into a circuit gate, represented as an array of [FusedNode].
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations. [FusedNode]s index over these inputs using their `input_order` attribute. They can also index over the intermediate outputs of other [FusedNode]s.
    /// * `output` - The variable representing the (currently singular) output of the fused operations.
    /// * `nodes` - The sequence of operations (in order of execution) that constitute the fused operation.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor],
        output: &VarTensor,
        nodes: &[FusedNode],
    ) -> Self {
        let mut config = Self {
            selector: meta.selector(),
            nodes: nodes.to_vec(),
            inputs: inputs.to_vec(),
            output: output.clone(),
            _marker: PhantomData,
        };

        meta.create_gate("basic_op", |meta| {
            let selector = meta.query_selector(config.selector);
            let qis = config
                .inputs
                .iter()
                .map(|input| match input.query(meta, 0) {
                    Ok(q) => q,
                    Err(e) => {
                        abort!("failed to query input {:?}", e);
                    }
                })
                .collect::<Vec<_>>();

            let mut config_outputs = vec![];
            for node in config.nodes.iter_mut() {
                Self::apply_op(node, &qis, &mut config_outputs);
            }
            let witnessed_output = &config_outputs[config.nodes.len() - 1];

            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = match config.output.query(meta, 0) {
                Ok(res) => res,
                Err(e) => {
                    abort!("failed to query output during fused layer layout {:?}", e);
                }
            };

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations. [FusedNode]s index over these inputs using their `input_order` attribute. They can also index over the intermediate outputs of other [FusedNode]s.
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>],
    ) -> ValTensor<F> {
        assert_eq!(values.len(), self.inputs.len());

        let t = match layouter.assign_region(
            || "assign inputs",
            |mut region| {
                let offset = 0;
                self.selector.enable(&mut region, offset)?;

                let mut inputs = vec![];
                for (i, input) in values.iter().enumerate() {
                    let inp = utils::value_muxer(
                        &self.inputs[i],
                        &{
                            match self.inputs[i].assign(&mut region, offset, input) {
                                Ok(res) => res.map(|e| e.value_field().evaluate()),
                                Err(e) => {
                                    abort!(
                                        "failed to assign inputs during fused layer layout {:?}",
                                        e
                                    );
                                }
                            }
                        },
                        input,
                    );
                    inputs.push(inp);
                }

                let mut layout_outputs = vec![];

                for node in self.nodes.iter_mut() {
                    Self::apply_op(node, &inputs, &mut layout_outputs);
                }
                let output: ValTensor<F> = match layout_outputs.last() {
                    Some(a) => a.clone().into(),
                    None => {
                        panic!("fused layer has empty outputs");
                    }
                };

                match self.output.assign(&mut region, offset, &output) {
                    Ok(a) => Ok(a),
                    Err(e) => {
                        abort!("failed to assign fused layer output {:?}", e);
                    }
                }
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                abort!("failed to assign fused layer region {:?}", e);
            }
        };

        ValTensor::from(t)
    }

    /// Applies an operation represented by a [FusedOp] to the set of inputs (both explicit and intermediate results) it indexes over.
    pub fn apply_op<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        node: &mut FusedNode,
        inputs: &[Tensor<T>],
        outputs: &mut Vec<Tensor<T>>,
    ) {
        let op_inputs = node
            .input_order
            .iter()
            .map(|input| match input {
                FusedInputType::Input(u) => inputs[*u].clone(),
                FusedInputType::Inter(u) => outputs[*u].clone(),
            })
            .collect_vec();
        outputs.push(Self::match_op(node.op.clone(), op_inputs));
    }

    /// Matches a [FusedOp] to an operation in the `tensor::ops` module.
    fn match_op<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        op: FusedOp,
        mut inputs: Vec<Tensor<T>>,
    ) -> Tensor<T> {
        match op {
            FusedOp::Identity => inputs[0].clone(),
            FusedOp::Reshape(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(&new_dims);
                t
            }
            FusedOp::Add => add(&inputs),
            FusedOp::Sub => sub(&inputs),
            FusedOp::Mult => mult(&inputs),
            FusedOp::Affine => affine(&inputs),
            FusedOp::BatchNorm => scale_and_shift(&inputs),
            FusedOp::ScaleAndShift => scale_and_shift(&inputs),
            FusedOp::Matmul => matmul(&inputs),
            FusedOp::Dot => {
                todo!();
            }
            FusedOp::Conv(padding, stride) => convolution(&inputs, padding, stride),
            FusedOp::SumPool(padding, stride, kernel_shape) => {
                sumpool(&inputs[0], padding, stride, kernel_shape)
            }
            FusedOp::Pow(u) => {
                assert_eq!(inputs.len(), 1);
                pow(&inputs[0], u)
            }
            FusedOp::Sum => {
                assert_eq!(inputs.len(), 1);
                sum(&inputs[0])
            }
            FusedOp::Rescaled(op, m) => {
                assert_eq!(m.len(), inputs.len());

                Self::match_op(
                    *op,
                    inputs
                        .iter_mut()
                        .enumerate()
                        .map(|(i, ri)| {
                            assert_eq!(m[i].0, i);
                            rescale(ri, m[i].1)
                        })
                        .collect_vec(),
                )
            }
        }
    }
}
