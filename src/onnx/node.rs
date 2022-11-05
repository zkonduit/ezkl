use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::circuit::eltwise::{DivideBy, EltwiseConfig, ReLu, Sigmoid};
use crate::circuit::fused::*;

use crate::tensor::ops::const_mult;
use crate::tensor::Tensor;
use crate::tensor::TensorType;
use anyhow::Result;

use halo2_proofs::arithmetic::FieldExt;
use itertools::Itertools;
use log::{error, info, warn};
use std::cmp::max;
use std::collections::BTreeMap;
use std::fmt;

use tabled::Tabled;
use tract_onnx;
use tract_onnx::prelude::{InferenceFact, Node, OutletId};
use tract_onnx::tract_hir::{
    infer::Factoid,
    internal::InferenceOp,
    ops::cnn::Conv,
    ops::expandable::Expansion,
    ops::nn::DataFormat,
    tract_core::ops::{
        cnn::{conv::KernelFormat, PaddingSpec},
        konst::Const,
    },
};

// Initially, some of these OpKinds will be folded into others (for example, Const nodes that
// contain parameters will be handled at the consuming self.
// Eventually, though, we probably want to keep them and treat them directly (layouting and configuring
// at each type of node)
/// Enum of the different kinds of operations `ezkl` can support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    ReLU(usize),
    Sigmoid(usize),
    Div(usize),
    Const,
    Input,
    Fused(FusedOp),
    Unknown(String),
}

impl OpKind {
    pub fn is_fused(&self) -> bool {
        matches!(self, OpKind::Fused(_))
    }

    pub fn is_const(&self) -> bool {
        matches!(self, OpKind::Const)
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::ReLU(s) => write!(f, "relu w/ scaling: {}", s),
            OpKind::Div(s) => write!(f, "div  w/ scaling: {}", s),
            OpKind::Sigmoid(s) => write!(f, "sigmoid  w/ scaling: {}", s),
            OpKind::Const => write!(f, "const"),
            OpKind::Input => write!(f, "input"),
            OpKind::Fused(s) => write!(f, "{}", s),
            OpKind::Unknown(c) => write!(f, "? {}", c),
        }
    }
}

/// Enum of the different kinds of node configurations `ezkl` can support.
#[derive(Clone, Default, Debug)]
pub enum NodeConfigTypes<F: FieldExt + TensorType> {
    ReLU(EltwiseConfig<F, ReLu<F>>, Vec<usize>),
    Sigmoid(EltwiseConfig<F, Sigmoid<F>>, Vec<usize>),
    Divide(EltwiseConfig<F, DivideBy<F>>, Vec<usize>),
    Fused(FusedConfig<F>, Vec<usize>),
    Const,
    Input,
    #[default]
    NotConfigured,
}

/// A circuit configuration for a single self.
#[derive(Clone, Default, Debug)]
pub struct NodeConfig<F: FieldExt + TensorType> {
    pub config: NodeConfigTypes<F>,
    pub onnx_idx: Vec<usize>,
}

fn display_option<T: fmt::Debug>(o: &Option<T>) -> String {
    match o {
        Some(s) => format!("{:?}", s),
        None => String::new(),
    }
}

fn display_inputs(o: &Vec<OutletId>) -> String {
    if !o.is_empty() {
        let mut nodes = vec![];
        for id in o.iter() {
            nodes.push(id.node);
        }
        format!("{:?}", nodes)
    } else {
        String::new()
    }
}

fn display_tensor(o: &Option<Tensor<i32>>) -> String {
    match o {
        Some(s) => format!("[{:#?}...]", s[0]),
        None => String::new(),
    }
}

/// A single operation in an [OnnxModel].
/// # Arguments:
/// * `node` - The raw Tract Node data structure.
/// * `opkind` - [OpKind] enum, i.e what operation this node represents.
/// * `output_max` - The inferred maximum value that can appear in the output tensor given previous quantization choices.
/// * `in_scale, out_scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
/// * `in_dims, out_dims` - The shape of the activations which enter and leave the self.
/// * `inputs` - The indices of other nodes that feed into this self.
/// * `const_value` - The constants potentially associated with this self.
/// * `idx` - The node's unique identifier.
/// * `bucket` - The execution bucket this node has been assigned to.
#[derive(Clone, Debug, Tabled)]
pub struct OnnxNode {
    pub node: Node<InferenceFact, Box<dyn InferenceOp>>,
    pub opkind: OpKind,
    pub output_max: f32,
    pub min_cols: usize,
    pub in_scale: i32,
    pub out_scale: i32,
    #[tabled(display_with = "display_tensor")]
    pub const_value: Option<Tensor<i32>>, // float value * 2^qscale if applicable.
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    #[tabled(display_with = "display_inputs")]
    pub inputs: Vec<OutletId>,
    #[tabled(display_with = "display_option")]
    pub in_dims: Option<Vec<usize>>,
    #[tabled(display_with = "display_option")]
    pub out_dims: Option<Vec<usize>>,
    pub idx: usize,
    #[tabled(display_with = "display_option")]
    pub bucket: Option<usize>,
}

impl OnnxNode {
    pub fn new(
        mut node: Node<InferenceFact, Box<dyn InferenceOp>>,
        other_nodes: &mut BTreeMap<usize, OnnxNode>,
        scale: i32,
        idx: usize,
    ) -> Self {
        let mut opkind = match node.op().name().as_ref() {
            "Clip" => OpKind::ReLU(1),
            "Sigmoid" => OpKind::Sigmoid(1),
            "Div" => OpKind::Div(1),
            "Const" => OpKind::Const,
            "Source" => OpKind::Input,
            "Add" => OpKind::Fused(FusedOp::Add),
            "Sub" => OpKind::Fused(FusedOp::Sub),
            "Mul" => OpKind::Fused(FusedOp::Mult),
            "Gemm" => OpKind::Fused(FusedOp::Affine),
            "MatMulInference" => OpKind::Fused(FusedOp::Matmul),
            "Dot" => OpKind::Fused(FusedOp::Dot),
            "Reduce<Sum>" => OpKind::Fused(FusedOp::Sum),
            "Pow" => OpKind::Fused(FusedOp::Pow(1)),
            "Conv" => OpKind::Fused(FusedOp::Conv((1, 1), (1, 1))),
            "ConvHir" => OpKind::Fused(FusedOp::Conv((1, 1), (1, 1))),

            c => {
                warn!("{:?} is not currently supported", c);
                OpKind::Unknown(c.to_string())
            }
        };
        let output_shapes = match node_output_shapes(&node) {
            Ok(s) => Some(s),
            _ => None,
        };

        let inputs: Vec<OnnxNode> = node
            .inputs
            .iter_mut()
            // this shouldn't fail
            .map(|i| other_nodes.get(&i.node).unwrap().clone())
            .collect();

        // Set some default values, then figure out more specific values if possible based on the opkind.
        let mut min_cols = 1;
        let mut const_value = None;
        let mut in_scale = scale;
        let mut out_scale = 0i32;
        let mut in_dims = None;
        let mut out_dims = None;
        let mut output_max = f32::INFINITY;
        let bucket = None;

        match opkind {
            OpKind::Sigmoid(_) => {
                let input_node = &inputs[0];
                in_dims = input_node.out_dims.clone();
                out_dims = input_node.out_dims.clone();
                in_scale = input_node.out_scale;
                out_scale = scale;
                let scale_diff = in_scale;
                if scale_diff > 0 {
                    let mult = scale_to_multiplier(scale_diff);
                    opkind = OpKind::Sigmoid(mult as usize);
                }

                output_max = scale_to_multiplier(out_scale);

                min_cols = max(1, in_dims.as_ref().unwrap().iter().product());
            }

            OpKind::ReLU(_) => {
                let input_node = &inputs[0];
                in_dims = input_node.out_dims.clone();
                out_dims = input_node.out_dims.clone();
                output_max = input_node.output_max;
                in_scale = input_node.out_scale;
                out_scale = scale;
                let scale_diff = in_scale - out_scale;
                // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                if scale_diff > 0 {
                    let mult = scale_to_multiplier(scale_diff);
                    opkind = OpKind::ReLU(mult as usize); // now the input will be scaled down to match
                    output_max = input_node.output_max / mult;
                }
                min_cols = max(1, in_dims.as_ref().unwrap().iter().product());
            }
            OpKind::Div(_) => {
                let input_node = &inputs[0];
                in_dims = input_node.out_dims.clone();
                out_dims = input_node.out_dims.clone();

                // rescale the divider
                let mult = scale_to_multiplier(scale);
                node.inputs.pop();
                if inputs[1].out_dims.clone().unwrap() != [1] {
                    error!("ezkl currently only supports division by a constant");
                    unimplemented!()
                }
                let div = inputs[1].output_max / mult;

                in_scale = input_node.out_scale;
                out_scale = scale;
                let scale_diff = in_scale - out_scale;
                // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                if scale_diff > 0 {
                    let mult = scale_to_multiplier(scale_diff);
                    opkind = OpKind::Div((div * mult) as usize); // now the input will be scaled down to match
                    output_max = input_node.output_max / (div * mult);
                } else {
                    opkind = OpKind::Div(div as usize); // now the input will be scaled down to match
                    output_max = input_node.output_max / (div);
                }
                min_cols = max(1, in_dims.as_ref().unwrap().iter().product());
            }
            OpKind::Fused(ref s) => {
                let input_node = &inputs[0];
                in_dims = input_node.out_dims.clone();
                out_dims = input_node.out_dims.clone();
                min_cols = inputs
                    .iter()
                    .map(|input| input.out_dims.clone().unwrap().iter().product::<usize>() as f32)
                    .sum::<f32>() as usize;

                inputs
                    .iter()
                    .tuple_windows()
                    .all(|(a, b)| a.in_scale == b.in_scale);
                match s {
                    FusedOp::Dot => todo!(),
                    FusedOp::Conv(_, _) => {
                        let (input_node, weight_node, bias_node) =
                            (&inputs[0], &inputs[1], &inputs[2]);

                        // Extract the padding and stride layer hyperparams
                        let op = Box::new(node.op());

                        let conv_node: &Conv = match op.downcast_ref::<Box<dyn Expansion>>() {
                            Some(b) => match (*b).as_any().downcast_ref() {
                                Some(b) => b,
                                None => {
                                    error!("not a conv!");
                                    panic!()
                                }
                            },
                            None => {
                                error!("op is not a Tract Expansion!");
                                panic!()
                            }
                        };

                        // only support pytorch type formatting for now
                        assert_eq!(conv_node.data_format, DataFormat::NCHW);
                        assert_eq!(conv_node.kernel_fmt, KernelFormat::OIHW);

                        let stride = conv_node.strides.clone().unwrap();
                        let padding = match &conv_node.padding {
                            PaddingSpec::Explicit(p, _, _) => p,
                            _ => panic!("padding is not explicitly specified"),
                        };

                        in_scale = input_node.out_scale;
                        out_scale = weight_node.out_scale + input_node.out_scale;
                        let scale_diff = out_scale - bias_node.out_scale;
                        let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                        bias_node = Self::scale_up_const_node(bias_node, scale_diff);

                        assert_eq!(
                            input_node.out_scale + weight_node.out_scale,
                            bias_node.out_scale
                        );

                        let oihw = weight_node.out_dims.as_ref().unwrap();
                        let (out_channels, _, kernel_height, kernel_width) =
                            (oihw[0], oihw[1], oihw[2], oihw[3]);

                        let (padding_h, padding_w, stride_h, stride_w) =
                            (padding[0], padding[1], stride[0], stride[1]);

                        in_dims = input_node.out_dims.clone();

                        let input_height = in_dims.as_ref().unwrap()[1];
                        let input_width = in_dims.as_ref().unwrap()[2];

                        let out_height =
                            (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                        let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                        out_dims = Some(vec![out_channels, out_height, out_width]);

                        output_max = input_node.output_max
                            * weight_node.output_max
                            * ((kernel_height * kernel_width) as f32);

                        opkind = OpKind::Fused(FusedOp::Conv(
                            (padding_h, padding_w),
                            (stride_h, stride_w),
                        ));
                    }
                    FusedOp::Matmul => {
                        let (a_node, b_node) = (&inputs[0], &inputs[1]);

                        let in_dim = a_node.out_dims.as_ref().unwrap()[1];
                        in_dims = Some(vec![in_dim]);

                        let a_dims = a_node.out_dims.as_ref().unwrap();
                        let b_dims = b_node.out_dims.as_ref().unwrap();
                        let mut dims = Vec::from(&a_dims[0..a_dims.len() - 2]);
                        dims.push(a_dims[a_dims.len() - 2]);
                        dims.push(b_dims[a_dims.len() - 1]);

                        out_dims = Some(dims.clone());

                        output_max = input_node.output_max * a_node.output_max * (in_dim as f32);

                        in_scale = input_node.out_scale;

                        out_scale = a_node.out_scale + input_node.out_scale;
                    }
                    FusedOp::Affine => {
                        let (input_node, weight_node, bias_node) =
                            (&inputs[0], &inputs[1], &inputs[2]);

                        in_scale = input_node.out_scale;
                        out_scale = weight_node.out_scale + input_node.out_scale;
                        let scale_diff = out_scale - bias_node.out_scale;
                        let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                        bias_node = Self::scale_up_const_node(bias_node, scale_diff);

                        assert_eq!(
                            input_node.out_scale + weight_node.out_scale,
                            bias_node.out_scale
                        );

                        let in_dim = weight_node.out_dims.as_ref().unwrap()[1];
                        let out_dim = weight_node.out_dims.as_ref().unwrap()[0];
                        in_dims = Some(vec![in_dim]);
                        out_dims = Some(vec![out_dim]);

                        output_max =
                            input_node.output_max * weight_node.output_max * (in_dim as f32);
                    }
                    FusedOp::Add => {
                        opkind = Self::homogenize_input_scales(opkind, inputs.clone());
                        if let OpKind::Fused(FusedOp::Rescaled(_, mult)) = &opkind {
                            output_max = (inputs
                                .iter()
                                .enumerate()
                                .map(|(idx, n)| {
                                    ((mult[idx].1 as f32) * (n.output_max.ceil())) as i32
                                })
                                .max()
                                .unwrap() as f32)
                                * (inputs.len() as f32);
                        } else {
                            error!("failed to homogenize input scalings for node {}", idx);
                            panic!()
                        }

                        in_scale = inputs.iter().map(|input| input.out_scale).max().unwrap();
                        out_scale = in_scale;
                    }
                    FusedOp::Sum => {
                        assert!(inputs.len() == 1);
                        output_max = inputs[0].output_max
                            * inputs[0].in_dims.clone().unwrap().iter().product::<usize>() as f32;
                        in_scale = inputs.iter().map(|input| input.out_scale).max().unwrap();
                        out_scale = in_scale;
                        out_dims = Some(vec![1]);
                    }
                    FusedOp::Sub => {
                        opkind = Self::homogenize_input_scales(opkind, inputs.clone());
                        if let OpKind::Fused(FusedOp::Rescaled(_, mult)) = &opkind {
                            output_max = (inputs
                                .iter()
                                .enumerate()
                                .map(|(idx, n)| {
                                    ((mult[idx].1 as f32) * (n.output_max.ceil())) as i32
                                })
                                .max()
                                .unwrap() as f32)
                                * (inputs.len() as f32);
                        } else {
                            error!("failed to homogenize input scalings for node {}", idx);
                            panic!()
                        }
                        in_scale = inputs.iter().map(|input| input.out_scale).max().unwrap();
                        out_scale = in_scale;
                    }
                    FusedOp::Mult => {
                        output_max = f32::powf(
                            inputs
                                .iter()
                                .map(|input| input.output_max.ceil() as i32)
                                .max()
                                .unwrap() as f32,
                            inputs.len() as f32,
                        );
                        in_scale = input_node.out_scale;
                        out_scale = inputs.iter().map(|input| input.out_scale).sum::<i32>();
                    }
                    FusedOp::Pow(_) => {
                        let mult = scale_to_multiplier(scale);
                        node.inputs.pop();
                        if inputs[1].out_dims.clone().unwrap() != [1] {
                            error!(
                                "ezkl currently only supports raising to the power by a constant"
                            );
                            unimplemented!()
                        }
                        let pow = inputs[1].output_max / mult;
                        output_max = f32::powf(
                            inputs
                                .iter()
                                .map(|input| input.output_max.ceil() as i32)
                                .max()
                                .unwrap() as f32,
                            pow as f32,
                        );
                        in_scale = input_node.out_scale;
                        out_scale = in_scale * (pow as i32);

                        opkind = OpKind::Fused(FusedOp::Pow(pow as usize));
                    }
                    FusedOp::Rescaled(_, _) => {
                        error!("operations should not already be rescaled at this stage")
                    }
                }
                // output size
                min_cols += out_dims.clone().unwrap()[0..out_dims.clone().unwrap().len() - 1]
                    .iter()
                    .product::<usize>()
                    + 1;
            }
            OpKind::Const => {
                // Extract the padding and stride layer hyperparams
                let op = Box::new(node.op());
                let const_node: &Const = match op.as_any().downcast_ref() {
                    Some(b) => b,
                    None => {
                        error!("op is not a const!");
                        panic!()
                    }
                };
                let vec = const_node.0.as_slice::<f32>().unwrap().to_vec();
                let mut dims = const_node.0.shape().to_vec();
                if dims.is_empty() {
                    dims.push(1)
                }
                out_scale = in_scale;
                let t = vector_to_quantized(&vec, &dims, 0f32, out_scale).unwrap();
                out_dims = Some(t.dims().to_vec());
                output_max = t.iter().map(|x| x.abs()).max().unwrap() as f32;
                const_value = Some(t);
            }
            OpKind::Input => {
                let dims = if let Some([Some(v)]) = output_shapes.as_deref() {
                    v.to_vec()
                } else {
                    // Turn  `outputs: [?,3,32,32,F32 >3/0]` into `vec![3,32,32]`  in two steps
                    let the_shape: Result<Vec<i64>> = node.outputs[0]
                        .fact
                        .shape
                        .dims()
                        .filter_map(|x| x.concretize())
                        .map(|x| x.to_i64())
                        .collect();

                    the_shape
                        .unwrap()
                        .iter()
                        .map(|x| (*x as i32) as usize)
                        .collect()
                };
                // remove batch dim for now
                if dims[0] == 1 && dims.len() > 1 {
                    out_dims = Some(dims[1..].to_vec())
                } else {
                    out_dims = Some(dims)
                }

                output_max = 256.0;
                out_scale = in_scale;
            }
            _ => {}
        }

        OnnxNode {
            inputs: node.inputs.clone(),
            node: node.clone(),
            opkind,
            output_max,
            min_cols,
            in_scale,
            out_scale,
            const_value,
            in_dims,
            out_dims,
            idx,
            bucket,
        }
    }

    pub fn output_shapes(&self) -> Result<Vec<Option<Vec<usize>>>> {
        let mut shapes = Vec::new();
        let outputs = self.node.outputs.to_vec();
        for output in outputs {
            let mv = output
                .fact
                .shape
                .clone()
                .as_concrete_finite()?
                .map(|x| x.to_vec());
            shapes.push(mv)
        }
        Ok(shapes)
    }

    pub fn name(&self) -> String {
        self.node.name.clone()
    }

    /// Ensures all inputs to a node have the same floating point denominator.
    pub fn homogenize_input_scales(opkind: OpKind, inputs: Vec<OnnxNode>) -> OpKind {
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
        if let OpKind::Fused(c) = &opkind {
            OpKind::Fused(FusedOp::Rescaled(
                Box::new(c.clone()),
                (0..inputs.len()).zip(multipliers).collect_vec(),
            ))
        } else {
            error!("should not homegenize input scales for non fused ops.");
            panic!()
        }
    }

    /// Re-quantizes a constant value node to a new scale.
    pub fn scale_up_const_node(node: &mut OnnxNode, scale_diff: i32) -> &mut OnnxNode {
        assert!(matches!(node.opkind, OpKind::Const));
        if scale_diff > 0 {
            if let Some(val) = &node.const_value {
                let mult = scale_to_multiplier(scale_diff);
                node.const_value = Some(const_mult(val, mult as i32));
                info!(
                    "------ scaled const node {:?}: {:?} -> {:?}",
                    node.idx,
                    node.in_scale,
                    node.out_scale + scale_diff
                );
                node.output_max *= mult;
                node.out_scale += scale_diff;
            }
        }
        node
    }
}
