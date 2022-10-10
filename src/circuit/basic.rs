use super::*;
use crate::tensor::ops::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use std::fmt;
use std::marker::PhantomData;

#[derive(Clone, Debug, Copy)]
pub enum BasicOp {
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

impl fmt::Display for BasicOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BasicOp::Add => write!(f, "add"),
            BasicOp::Sub => write!(f, "sub"),
            BasicOp::Sum => write!(f, "sum"),
            BasicOp::Mult => write!(f, "mult"),
            BasicOp::Matmul => write!(f, "matmul"),
            BasicOp::Dot => write!(f, "dot"),
            BasicOp::Affine => write!(f, "affine"),
            BasicOp::Conv(_, _) => write!(f, "conv"),
            BasicOp::Pow(s) => write!(f, "pow {}", s),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasicOpNode {
    /// the type of operation
    pub op: BasicOp,
    /// indices for the `VarTensor` inputs the operation should ingest.
    pub input_idx: Vec<usize>,
    /// indices for the outputs of other nodes the operation should ingest.
    pub node_idx: Vec<usize>,
}

/// Configuration for a basic sequence of operations all fused together in a single gate.
#[derive(Clone, Debug)]
pub struct BasicConfig<F: FieldExt + TensorType> {
    pub inputs: Vec<VarTensor>,
    nodes: Vec<BasicOpNode>,
    pub output: VarTensor,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

/// Configures the sequence of operations into a circuit gate, represented as an array of `BasicOpNode`.
/// `variables` represents the potential inputs to each operation. `BasicOpNode`s index over these inputs using their `input_idx` attribute.
/// They can also ingest the intermediate outputs of other nodes, as represented by the `node_idx` attribute. 
impl<F: FieldExt + TensorType> BasicConfig<F> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        nodes: &[BasicOpNode],
    ) -> Self {
        let inputs = variables[0..variables.len() - 1].to_vec();
        let output = variables[variables.len() - 1].clone();

        let mut config = Self {
            selector: meta.selector(),
            nodes: nodes.to_vec(),
            inputs,
            output,
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
                let mut op_inputs = node.input_idx.iter().map(|i| &qis[*i]).collect::<Vec<_>>();
                let mut node_inputs = node
                    .node_idx
                    .iter()
                    .map(|i| &config_outputs[*i])
                    .collect::<Vec<_>>();
                op_inputs.append(&mut node_inputs);
                match node.op {
                    BasicOp::Add => {
                        config_outputs.push(add(&op_inputs));
                    }
                    BasicOp::Sub => {
                        config_outputs.push(sub(&op_inputs));
                    }
                    BasicOp::Mult => {
                        config_outputs.push(mult(&op_inputs));
                    }
                    BasicOp::Affine => {
                        config_outputs.push(affine(&op_inputs));
                    }
                    BasicOp::Matmul => {
                        config_outputs.push(matmul(&op_inputs));
                    }
                    BasicOp::Dot => {
                        todo!();
                    }
                    BasicOp::Conv(padding, stride) => {
                        config_outputs.push(convolution(&op_inputs, padding, stride));
                    }
                    BasicOp::Pow(u) => {
                        assert_eq!(op_inputs.len(), 1);
                        config_outputs.push(pow(&op_inputs[0], u));
                    }
                    BasicOp::Sum => {
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
                                .assign(&mut region, offset, &input)
                                .map(|e| e.value_field().evaluate()),
                            &input,
                        );
                        inputs.push(inp);
                    }

                    let mut layout_outputs = vec![];

                    for node in self.nodes.iter_mut() {
                        let mut op_inputs = node
                            .input_idx
                            .iter()
                            .map(|i| &inputs[*i])
                            .collect::<Vec<_>>();
                        let mut node_inputs = node
                            .node_idx
                            .iter()
                            .map(|i| &layout_outputs[*i])
                            .collect::<Vec<_>>();
                        op_inputs.append(&mut node_inputs);
                        match node.op {
                            BasicOp::Add => {
                                layout_outputs.push(add(&op_inputs));
                            }
                            BasicOp::Sub => {
                                layout_outputs.push(sub(&op_inputs));
                            }
                            BasicOp::Mult => {
                                layout_outputs.push(mult(&op_inputs));
                            }
                            BasicOp::Affine => {
                                layout_outputs.push(affine(&op_inputs));
                            }
                            BasicOp::Matmul => {
                                layout_outputs.push(matmul(&op_inputs));
                            }
                            BasicOp::Dot => {
                                todo!();
                            }
                            BasicOp::Conv(padding, stride) => {
                                layout_outputs.push(convolution(&op_inputs, padding, stride));
                            }
                            BasicOp::Pow(u) => {
                                assert_eq!(op_inputs.len(), 1);
                                layout_outputs.push(pow(&op_inputs[0], u));
                            }
                            BasicOp::Sum => {
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
