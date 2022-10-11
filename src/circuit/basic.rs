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

#[derive(Clone, Debug, PartialEq, Eq, Copy)]
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
                let op_inputs = node
                    .input_order
                    .iter()
                    .map(|input| match input {
                        FusedInputType::Input(u) => &qis[*u],
                        FusedInputType::Inter(u) => &config_outputs[*u],
                    })
                    .collect_vec();
                match node.op {
                    FusedOp::Add => {
                        config_outputs.push(add(&op_inputs));
                    }
                    FusedOp::Sub => {
                        config_outputs.push(sub(&op_inputs));
                    }
                    FusedOp::Mult => {
                        config_outputs.push(mult(&op_inputs));
                    }
                    FusedOp::Affine => {
                        config_outputs.push(affine(&op_inputs));
                    }
                    FusedOp::Matmul => {
                        config_outputs.push(matmul(&op_inputs));
                    }
                    FusedOp::Dot => {
                        todo!();
                    }
                    FusedOp::Conv(padding, stride) => {
                        config_outputs.push(convolution(&op_inputs, padding, stride));
                    }
                    FusedOp::Pow(u) => {
                        assert_eq!(op_inputs.len(), 1);
                        config_outputs.push(pow(op_inputs[0], u));
                    }
                    FusedOp::Sum => {
                        assert_eq!(op_inputs.len(), 1);
                        config_outputs.push(sum(op_inputs[0]));
                    }
                }
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
                        let op_inputs = node
                            .input_order
                            .iter()
                            .map(|input| match input {
                                FusedInputType::Input(u) => &inputs[*u],
                                FusedInputType::Inter(u) => &layout_outputs[*u],
                            })
                            .collect_vec();

                        match node.op {
                            FusedOp::Add => {
                                layout_outputs.push(add(&op_inputs));
                            }
                            FusedOp::Sub => {
                                layout_outputs.push(sub(&op_inputs));
                            }
                            FusedOp::Mult => {
                                layout_outputs.push(mult(&op_inputs));
                            }
                            FusedOp::Affine => {
                                layout_outputs.push(affine(&op_inputs));
                            }
                            FusedOp::Matmul => {
                                layout_outputs.push(matmul(&op_inputs));
                            }
                            FusedOp::Dot => {
                                todo!();
                            }
                            FusedOp::Conv(padding, stride) => {
                                layout_outputs.push(convolution(&op_inputs, padding, stride));
                            }
                            FusedOp::Pow(u) => {
                                assert_eq!(op_inputs.len(), 1);
                                layout_outputs.push(pow(op_inputs[0], u));
                            }
                            FusedOp::Sum => {
                                assert_eq!(op_inputs.len(), 1);
                                layout_outputs.push(sum(op_inputs[0]));
                            }
                        }
                    }
                    let output: ValTensor<F> = layout_outputs.last().unwrap().clone().into();

                    Ok(self.output.assign(&mut region, offset, &output))
                },
            )
            .unwrap();

        ValTensor::from(t)
    }
}
