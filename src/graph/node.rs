use super::utilities::node_output_shapes;
use crate::circuit::Op;
use crate::graph::new_op_from_onnx;
use crate::graph::GraphError;
use crate::tensor::TensorType;
use anyhow::Result;
use halo2curves::ff::PrimeField;
use log::trace;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use tabled::Tabled;
use tract_onnx;
use tract_onnx::prelude::Node as OnnxNode;
use tract_onnx::prelude::TypedFact;
use tract_onnx::prelude::TypedOp;

fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if !v.is_empty() {
        format!("{:?}", v)
    } else {
        String::new()
    }
}

fn display_opkind<F: PrimeField + TensorType + PartialOrd>(v: &Box<dyn Op<F>>) -> String {
    v.as_string()
}

/// A single operation in a Model.
/// # Arguments:
/// * `opkind` - [OpKind] enum, i.e what operation this node represents.
/// * `out_scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
/// * `out_dims` - The shape of the activations which enter and leave the self.
/// * `inputs` - The indices of other nodes that feed into this self.
/// * `idx` - The node's unique identifier.
#[derive(Clone, Debug, Tabled)]
pub struct Node<F: PrimeField + TensorType + PartialOrd> {
    /// [OpKind] enum, i.e what operation this node represents.
    #[tabled(display_with = "display_opkind")]
    pub opkind: Box<dyn Op<F>>,
    /// The denominator in the fixed point representation for the node's output. Tensors of differing scales should not be combined.
    pub out_scale: u32,
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    #[tabled(display_with = "display_vector")]
    /// The indices of the node's inputs.
    pub inputs: Vec<usize>,
    #[tabled(display_with = "display_vector")]
    /// Dimensions of output.
    pub out_dims: Vec<usize>,
    /// The node's unique identifier.
    pub idx: usize,
}

impl<F: PrimeField + TensorType + PartialOrd> Node<F> {
    /// Converts a tract [OnnxNode] into an ezkl [Node].
    /// # Arguments:
    /// * `node` - [OnnxNode]
    /// * `other_nodes` - [BTreeMap] of other previously initialized [Node]s in the computational graph.
    /// * `public_params` - flag if parameters of model are public
    /// * `idx` - The node's unique identifier.
    pub fn new(
        mut node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, super::NodeType<F>>,
        scale: u32,
        public_params: bool,
        idx: usize,
    ) -> Result<Self, Box<dyn Error>> {
        trace!("Create {:?}", node);
        trace!("Create op {:?}", node.op);

        // load the node inputs
        let mut inputs = vec![];

        for i in node.inputs.iter_mut() {
            match other_nodes.get(&i.node) {
                Some(n) => inputs.push(n.clone()),
                None => return Err(Box::new(GraphError::MissingNode(i.node))),
            }
        }

        let mut opkind = new_op_from_onnx(idx, scale, public_params, node.clone(), &mut inputs)?; // parses the op name

        // rescale the inputs if necessary to get consistent fixed points
        let in_scales: Vec<u32> = inputs.iter().map(|n| n.out_scales()[0]).collect();
        opkind = opkind.rescale(in_scales.clone(), scale);
        let out_scale = match in_scales.len() {
            0 => scale,
            _ => opkind.out_scale(in_scales, scale),
        };

        // get the output shape
        let out_dims = {
            let output_shapes = match node_output_shapes(&node) {
                Ok(s) => Some(s),
                _ => None,
            };

            if let Some([Some(v)]) = output_shapes.as_deref() {
                v.to_vec()
            } else {
                panic!("Could not get output shape for node {:?}", node);
            }
        };

        Ok(Node {
            idx,
            opkind,
            inputs: inputs.iter().map(|i| i.idx()).collect(),
            out_dims,
            out_scale,
        })
    }
}
