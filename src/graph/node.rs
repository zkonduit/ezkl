use super::utilities::{node_output_shapes, scale_to_multiplier};
use crate::circuit::Op;
use crate::graph::new_op_from_onnx;
use crate::graph::GraphError;
use crate::tensor::TensorType;
use anyhow::Result;
use halo2_proofs::arithmetic::FieldExt;
use itertools::Itertools;
use log::{info, trace};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use tabled::Tabled;
use tract_onnx;
use tract_onnx::prelude::Node as OnnxNode;
use tract_onnx::prelude::TypedFact;
use tract_onnx::prelude::TypedOp;

/// Representation of an execution graph divided into execution 'buckets'.
pub type NodeGraph<F> = BTreeMap<usize, Node<F>>;

fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if !v.is_empty() {
        format!("{:?}", v)
    } else {
        String::new()
    }
}

fn display_opkind<F: FieldExt + TensorType>(v: &Box<dyn Op<F>>) -> String {
    v.as_str().to_string()
}

/// A single operation in a Model.
/// # Arguments:
/// * `opkind` - [OpKind] enum, i.e what operation this node represents.
/// * `output_max` - The inferred maximum value that can appear in the output tensor given previous quantization choices.
/// * `in_scale, out_scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
/// * `in_dims, out_dims` - The shape of the activations which enter and leave the self.
/// * `inputs` - The indices of other nodes that feed into this self.
/// * `const_value` - The constants potentially associated with this self.
/// * `idx` - The node's unique identifier.
/// * `bucket` - The execution bucket this node has been assigned to.
#[derive(Clone, Debug, Tabled)]
pub struct Node<F: FieldExt + TensorType> {
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

impl<F: FieldExt + TensorType> Node<F> {
    /// Converts a tract [OnnxNode] into an ezkl [Node].
    /// # Arguments:
    /// * `node` - [OnnxNode]
    /// * `other_nodes` - [BTreeMap] of other previously initialized [Node]s in the computational graph.
    /// * `scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
    /// * `idx` - The node's unique identifier.
    pub fn new(
        mut node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
        other_nodes: &mut BTreeMap<usize, Node<F>>,
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

        // if the op requires 3d inputs, we need to make sure the input shape is consistent with that
        if opkind.has_3d_input() {
            let input_node = other_nodes.get_mut(&node.inputs[0].node).unwrap();
            Self::format_3d_inputs(input_node)?;
            inputs[0] = input_node.clone();
        };

        // creates a rescaled op if the inputs are not homogenous
        if opkind.requires_homogenous_input_scales() {
            opkind = Self::homogenize_input_scales(opkind, inputs.clone())?;
        }

        // rescale the inputs if necessary to get consistent fixed points
        let in_scales: Vec<u32> = inputs.iter().map(|i| i.out_scale).collect();
        opkind = opkind.rescale(in_scales.clone(), scale);
        let out_scale = match in_scales.len() {
            0 => scale,
            _ => opkind.out_scale(in_scales, scale),
        };

        // get the output shape
        let mut out_dims = {
            let output_shapes = match node_output_shapes(&node) {
                Ok(s) => Some(s),
                _ => None,
            };

            if let Some([Some(v)]) = output_shapes.as_deref() {
                v.to_vec()
            } else {
                // Turn  `outputs: [?,3,32,32,F32 >3/0]` into `vec![3,32,32]`  in two steps
                node.outputs[0]
                    .fact
                    .shape
                    .iter()
                    // .filter_map(|x| x.concretize())
                    .map(|x| x.to_i64().unwrap() as usize)
                    .collect()
            }
        };

        // rm batch
        if !out_dims.is_empty() && out_dims[0] == 1 && out_dims.len() > 1 {
            out_dims = out_dims[1..].to_vec();
        }
        if out_dims.iter().product::<usize>() == 1 {
            out_dims = vec![1];
        };

        Ok(Node {
            idx,
            opkind,
            inputs: inputs.iter().map(|i| i.idx).collect(),
            out_dims,
            out_scale,
        })
    }

    /// Ensures all inputs to a node have the same fixed point denominator.
    fn homogenize_input_scales(
        opkind: Box<dyn Op<F>>,
        inputs: Vec<Self>,
    ) -> Result<Box<dyn Op<F>>, Box<dyn Error>> {
        let mut multipliers = vec![1; inputs.len()];
        let out_scales = inputs.windows(1).map(|w| w[0].out_scale).collect_vec();
        if !out_scales.windows(2).all(|w| w[0] == w[1]) {
            let max_scale = out_scales.iter().max().unwrap();
            let _ = inputs
                .iter()
                .enumerate()
                .map(|(idx, input)| {
                    let scale_diff = max_scale - input.out_scale;
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        multipliers[idx] = mult as usize;
                        info!(
                            "------ scaled op node input {:?}: {:?} -> {:?}",
                            input.idx,
                            input.out_scale,
                            input.out_scale + scale_diff
                        );
                    }
                })
                .collect_vec();
        }

        // only rescale if need to
        if multipliers.iter().sum::<usize>() > multipliers.len() {
            Ok(Box::new(crate::circuit::Rescaled {
                inner: opkind,
                scale: (0..inputs.len()).zip(multipliers).collect_vec(),
            }))
        } else {
            Ok(opkind)
        }
    }

    /// Formats 3d inputs if they have under or overspecified dims (casting 2D -> 3D and nD -> 3D)
    fn format_3d_inputs(mut node: &mut Self) -> Result<(), Box<dyn Error>> {
        // input_nodes come in all shapes and sizes we gotta homogenize, especially for 2D (single channel images)
        if node.out_dims.len() == 2 {
            node = Self::pad_channel_input_node(node)?;
        } else if node.out_dims.len() > 3 {
            node = Self::rm_redundant_3d_channels(node)?;
        };

        if node.out_dims.len() != 3 {
            return Err(Box::new(GraphError::InvalidDims(
                node.idx,
                node.clone().opkind.as_str().to_string(),
            )));
        }
        Ok(())
    }

    /// Adds an extra channel dim to nodes that need it.
    fn pad_channel_input_node(node: &mut Self) -> Result<&mut Self, Box<dyn Error>> {
        let mut dims = vec![1];
        dims.append(&mut node.out_dims);
        node.out_dims = dims;
        Ok(node)
    }

    /// Removes excess channels for an image
    fn rm_redundant_3d_channels(node: &mut Self) -> Result<&mut Self, Box<dyn Error>> {
        let dims = &node.out_dims;
        let last_dims = &dims[dims.len() - 3..];
        let channel_dims = &dims[..dims.len() - 3];
        for dim in channel_dims {
            if *dim != 1 {
                return Err(Box::new(GraphError::InvalidDims(
                    node.idx,
                    node.opkind.as_str().to_string(),
                )));
            }
        }
        node.out_dims = last_dims.to_vec();
        Ok(node)
    }
}
