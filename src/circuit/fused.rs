use super::*;
use crate::tensor::ops::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use itertools::Itertools;
use std::fmt;
use std::marker::PhantomData;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FusedOp {
    Add,
    Sub,
    Sum,
    Mult,
    Matmul,
    Dot,
    Affine,
    Conv((usize, usize), (usize, usize)),
    Pow(usize),
    Rescaled(Box<FusedOp>, Vec<(usize, usize)>),
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FusedOp::Add => write!(f, "add"),
            FusedOp::Sub => write!(f, "sub"),
            FusedOp::Sum => write!(f, "sum"),
            FusedOp::Mult => write!(f, "mult"),
            FusedOp::Matmul => write!(f, "matmul"),
            FusedOp::Dot => write!(f, "dot"),
            FusedOp::Affine => write!(f, "affine"),
            FusedOp::Conv(padding, stride) => {
                write!(f, "conv w/ padding: {:?}, stride: {:?}", padding, stride)
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

#[derive(Clone, Debug)]
pub enum FusedInputType {
    Input(usize),
    Inter(usize),
}

#[derive(Clone, Debug)]
pub struct FusedNode {
    /// the type of operation
    pub op: FusedOp,
    /// execution order.
    pub input_order: Vec<FusedInputType>,
}

/// Configuration for a basic sequence of operations all fused together in a single gate.
#[derive(Clone, Debug)]
pub struct FusedConfig<F: FieldExt + TensorType> {
    pub inputs: Vec<VarTensor>,
    nodes: Vec<FusedNode>,
    pub output: VarTensor,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Configures the sequence of operations into a circuit gate, represented as an array of `FusedOpNode`.
/// `variables` represents the potential inputs to each operation. `FusedOpNode`s index over these inputs using their `input_idx` attribute.
/// They can also ingest the intermediate outputs of other nodes, as represented by the `node_idx` attribute.
impl<F: FieldExt + TensorType> FusedConfig<F> {
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
                .map(|input| input.query(meta, 0))
                .collect::<Vec<_>>();

            let mut config_outputs = vec![];
            for node in config.nodes.iter_mut() {
                Self::apply_op(node, &qis, &mut config_outputs);
            }
            let witnessed_output = &config_outputs[config.nodes.len() - 1];

            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = config.output.query(meta, 0);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>],
    ) -> ValTensor<F> {
        assert_eq!(values.len(), self.inputs.len());

        let t = layouter
            .assign_region(
                || "assign inputs",
                |mut region| {
                    let offset = 0;
                    self.selector.enable(&mut region, offset)?;

                    let mut inputs = vec![];
                    for (i, input) in values.iter().enumerate() {
                        let inp = utils::value_muxer(
                            &self.inputs[i],
                            &self.inputs[i]
                                .assign(&mut region, offset, input)
                                .map(|e| e.value_field().evaluate()),
                            input,
                        );
                        inputs.push(inp);
                    }

                    let mut layout_outputs = vec![];

                    for node in self.nodes.iter_mut() {
                        Self::apply_op(node, &inputs, &mut layout_outputs);
                    }
                    let output: ValTensor<F> = layout_outputs.last().unwrap().clone().into();

                    Ok(self.output.assign(&mut region, offset, &output))
                },
            )
            .unwrap();

        ValTensor::from(t)
    }

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

    fn match_op<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        op: FusedOp,
        mut inputs: Vec<Tensor<T>>,
    ) -> Tensor<T> {
        match op {
            FusedOp::Add => add(&inputs),
            FusedOp::Sub => sub(&inputs),
            FusedOp::Mult => mult(&inputs),
            FusedOp::Affine => affine(&inputs),
            FusedOp::Matmul => matmul(&inputs),
            FusedOp::Dot => {
                todo!();
            }
            FusedOp::Conv(padding, stride) => convolution(&inputs, padding, stride),
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
