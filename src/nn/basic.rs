use super::*;
use crate::tensor::ops::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use std::fmt;
use std::marker::PhantomData;

#[derive(Clone, Debug, Copy)]
pub enum BasicOp {
    Add,
    Mult,
    Pow(usize),
}

impl fmt::Display for BasicOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BasicOp::Add => write!(f, "add"),
            BasicOp::Mult => write!(f, "mult"),
            BasicOp::Pow(s) => write!(f, "pow {}", s),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BasicOpNode<F: FieldExt + TensorType> {
    pub op: BasicOp,
    pub input_idx: Vec<usize>,
    pub node_idx: Vec<usize>,
    pub output_config: Option<Tensor<Expression<F>>>,
    pub output_layout: Option<Tensor<Value<F>>>,
}

/// Configuration for an affine layer which (mat)multiplies a weight kernel to an input and adds
/// a bias vector to the result.
#[derive(Clone, Debug)]
pub struct BasicConfig<F: FieldExt + TensorType> {
    pub inputs: Vec<VarTensor>,
    nodes: Vec<BasicOpNode<F>>,
    pub output: VarTensor,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> BasicConfig<F> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        nodes: &[BasicOpNode<F>],
    ) -> Self {
        let inputs = variables[0..variables.len() - 1].to_vec();
        let output = variables[variables.len() - 1].clone();

        for input in inputs.iter() {
            assert_eq!(input.dims(), output.dims());
        }

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

            for node in config.nodes.iter_mut() {
                let mut op_inputs = node
                    .input_idx
                    .iter()
                    .map(|i| qis[*i].clone())
                    .collect::<Vec<_>>();
                let mut node_inputs = node
                    .node_idx
                    .iter()
                    .map(|i| nodes[*i].clone().output_config.unwrap())
                    .collect::<Vec<_>>();
                op_inputs.append(&mut node_inputs);
                match node.op {
                    BasicOp::Add => {
                        node.output_config = Some(add(&op_inputs));
                    }
                    BasicOp::Mult => {
                        node.output_config = Some(mult(&op_inputs));
                    }
                    BasicOp::Pow(u) => {
                        assert_eq!(op_inputs.len(), 1);
                        node.output_config = Some(pow(op_inputs[0].clone(), u));
                    }
                }
            }
            let witnessed_output = nodes[nodes.len() - 1].clone().output_config.unwrap();

            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = config.output.query(meta, 0);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns values to the affine gate variables created when calling `configure`.
    /// Values are supplied as a 3-element array of `[weights, bias, input]` VarTensors.
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

                    let mut nodes = self.nodes.clone();

                    for node in nodes.iter_mut() {
                        let mut op_inputs = node
                            .input_idx
                            .iter()
                            .map(|i| inputs[*i].clone())
                            .collect::<Vec<_>>();
                        let mut node_inputs = node
                            .node_idx
                            .iter()
                            .map(|i| self.nodes[*i].clone().output_layout.unwrap())
                            .collect::<Vec<_>>();
                        op_inputs.append(&mut node_inputs);
                        match node.op {
                            BasicOp::Add => {
                                node.output_layout = Some(add(&op_inputs));
                            }
                            BasicOp::Mult => {
                                node.output_layout = Some(mult(&op_inputs));
                            }
                            BasicOp::Pow(u) => {
                                assert_eq!(op_inputs.len(), 1);
                                node.output_layout = Some(pow(op_inputs[0].clone(), u));
                            }
                        }
                    }
                    let output: ValTensor<F> = self.nodes[self.nodes.len() - 1]
                        .clone()
                        .output_layout
                        .unwrap()
                        .into();

                    Ok(self.output.assign(&mut region, offset, &output))
                },
            )
            .unwrap();

        ValTensor::from(t)
    }
}
