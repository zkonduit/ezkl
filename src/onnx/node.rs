use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::circuit::eltwise::{DivideBy, EltwiseConfig, ReLu, Sigmoid};
use crate::circuit::fused::*;

use crate::abort;
use crate::tensor::ops::{add, const_mult, div, mult};
use crate::tensor::Tensor;
use crate::tensor::TensorType;
use anyhow::Result;

use halo2_proofs::arithmetic::FieldExt;
use itertools::Itertools;
use log::{error, info, trace, warn};
use std::cmp::max;
use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt;

use tabled::Tabled;
use tract_onnx;
use tract_onnx::prelude::{DatumType, InferenceFact, Node as OnnxNode, OutletId};
use tract_onnx::tract_hir::{
    infer::Factoid,
    internal::InferenceOp,
    ops::cnn::{Conv, PoolSpec, SumPool}, //MaxPool,},
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum OpKind {
    ReLU(usize),
    Sigmoid(usize),
    Div(usize),
    Const,
    Input,
    Fused(FusedOp),
    Unknown(String),
    #[default]
    None,
}

impl OpKind {
    pub fn new(name: &str) -> Self {
        match name {
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
            "SumPool" => OpKind::Fused(FusedOp::SumPool((1, 1), (1, 1), (1, 1))),
            "Reshape" => OpKind::Fused(FusedOp::Reshape(Vec::new())),
            "BatchNorm" => OpKind::Fused(FusedOp::BatchNorm),
            "Pad" => OpKind::Fused(FusedOp::Identity),
            c => {
                warn!("{:?} is not currently supported", c);
                OpKind::Unknown(c.to_string())
            }
        }
    }
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
            OpKind::None => write!(f, "n/a",),
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

/// Representation of an execution graph divided into execution 'buckets'.
#[derive(Clone, Default, Debug)]
pub struct NodeGraph(pub BTreeMap<Option<usize>, BTreeMap<usize, Node>>);

impl NodeGraph {
    pub fn new() -> Self {
        NodeGraph(BTreeMap::new())
    }

    pub fn insert(&mut self, idx: Option<usize>, node_idx: usize, config: Node) {
        match self.0.entry(idx) {
            Entry::Vacant(e) => {
                e.insert(BTreeMap::from([(node_idx, config)]));
            }
            Entry::Occupied(mut e) => {
                e.get_mut().insert(node_idx, config);
            }
        }
    }

    pub fn flatten(&self) -> Vec<Node> {
        let a = self
            .0
            .clone()
            .into_values()
            .map(|d| d.into_values().collect())
            .collect::<Vec<Vec<Node>>>();
        let mut c: Vec<Node> = a
            .iter()
            .flatten()
            .collect::<Vec<&Node>>()
            .iter()
            .map(|e| (*e).clone())
            .collect();

        c.sort_by_key(|v| v.idx);
        c
    }

    pub fn filter(&self, idx: usize) -> Node {
        let a = self.flatten();
        let c = &a
            .iter()
            .filter(|i| i.idx == idx)
            .cloned()
            .collect::<Vec<Node>>()[0];
        c.clone()
    }
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

fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    format!("{:?}", v)
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

fn display_tensorf32(o: &Option<Tensor<f32>>) -> String {
    match o {
        Some(s) => format!("[{:#?}...]", s[0]),
        None => String::new(),
    }
}

/// A single operation in an [OnnxModel].
/// # Arguments:
/// * `opkind` - [OpKind] enum, i.e what operation this node represents.
/// * `output_max` - The inferred maximum value that can appear in the output tensor given previous quantization choices.
/// * `in_scale, out_scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
/// * `in_dims, out_dims` - The shape of the activations which enter and leave the self.
/// * `inputs` - The indices of other nodes that feed into this self.
/// * `const_value` - The constants potentially associated with this self.
/// * `idx` - The node's unique identifier.
/// * `bucket` - The execution bucket this node has been assigned to.
#[derive(Clone, Debug, Default, Tabled)]
pub struct Node {
    pub opkind: OpKind,
    pub output_max: f32,
    pub min_cols: usize,
    pub in_scale: i32,
    pub out_scale: i32,
    #[tabled(display_with = "display_tensor")]
    pub const_value: Option<Tensor<i32>>, // float value * 2^qscale if applicable.
    #[tabled(display_with = "display_tensorf32")]
    pub raw_const_value: Option<Tensor<f32>>,
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    #[tabled(display_with = "display_inputs")]
    pub inputs: Vec<OutletId>,
    #[tabled(display_with = "display_vector")]
    pub in_dims: Vec<usize>,
    #[tabled(display_with = "display_vector")]
    pub out_dims: Vec<usize>,
    pub idx: usize,
    #[tabled(display_with = "display_option")]
    pub bucket: Option<usize>,
}

impl Node {
    pub fn new(
        mut node: OnnxNode<InferenceFact, Box<dyn InferenceOp>>,
        other_nodes: &mut BTreeMap<usize, Node>,
        scale: i32,
        idx: usize,
    ) -> Self {
        trace!("Create {:?}", node);
        let output_shapes = match node_output_shapes(&node) {
            Ok(s) => Some(s),
            _ => None,
        };

        let mut inputs: Vec<Node> = node
            .inputs
            .iter_mut()
            // this shouldn't fail
            .map(|i| {
                match other_nodes.get(&i.node) {
                    Some(n) => n,
                    None => {
                        abort!("input {} has not been initialized", i.node);
                    }
                }
                .clone()
            })
            .collect();

        let mut mn = Node {
            opkind: OpKind::new(node.op().name().as_ref()), // parses the op name
            inputs: node.inputs.clone(),
            in_scale: scale,
            idx,
            ..Default::default()
        };

        match mn.opkind {
            OpKind::Sigmoid(_) => {
                let input_node = &inputs[0];
                mn.in_dims = input_node.out_dims.clone();
                mn.out_dims = input_node.out_dims.clone();
                mn.in_scale = input_node.out_scale;
                mn.out_scale = scale;
                let scale_diff = mn.in_scale;
                if scale_diff > 0 {
                    let mult = scale_to_multiplier(scale_diff);
                    mn.opkind = OpKind::Sigmoid(mult as usize);
                }

                mn.output_max = scale_to_multiplier(mn.out_scale);

                mn.min_cols = max(1, mn.in_dims.iter().product());
            }

            OpKind::ReLU(_) => {
                let input_node = &inputs[0];
                mn.in_dims = input_node.out_dims.clone();
                mn.out_dims = input_node.out_dims.clone();
                mn.output_max = input_node.output_max;
                mn.in_scale = input_node.out_scale;
                mn.out_scale = scale;
                let scale_diff = mn.in_scale - mn.out_scale;
                // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                if scale_diff > 0 {
                    let mult = scale_to_multiplier(scale_diff);
                    mn.opkind = OpKind::ReLU(mult as usize); // now the input will be scaled down to match
                    mn.output_max = input_node.output_max / mult;
                }
                mn.min_cols = max(1, mn.in_dims.iter().product());
            }
            OpKind::Div(_) => {
                let input_node = &inputs[0];
                mn.in_dims = input_node.out_dims.clone();
                mn.out_dims = input_node.out_dims.clone();

                // rescale the divider
                let mult = scale_to_multiplier(scale);
                mn.inputs.pop();
                if inputs[1].out_dims.clone() != [1] {
                    abort!("ezkl currently only supports division by a constant");
                }
                let div = inputs[1].output_max / mult;

                mn.in_scale = input_node.out_scale;
                mn.out_scale = scale;
                let scale_diff = mn.in_scale - mn.out_scale;
                // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                if scale_diff > 0 {
                    let mult = scale_to_multiplier(scale_diff);
                    mn.opkind = OpKind::Div((div * mult) as usize); // now the input will be scaled down to match
                    mn.output_max = input_node.output_max / (div * mult);
                } else {
                    mn.opkind = OpKind::Div(div as usize); // now the input will be scaled down to match
                    mn.output_max = input_node.output_max / (div);
                }
                mn.min_cols = max(1, mn.in_dims.iter().product());
            }
            OpKind::Fused(ref s) => {
                let input_node = &inputs[0];
                mn.in_dims = input_node.out_dims.clone();
                mn.out_dims = input_node.out_dims.clone();
                mn.min_cols = inputs
                    .iter()
                    .map(|input| input.out_dims.clone().iter().product::<usize>() as f32)
                    .sum::<f32>() as usize;

                inputs
                    .iter()
                    .tuple_windows()
                    .all(|(a, b)| a.in_scale == b.in_scale);
                match s {
                    FusedOp::Dot => todo!(),
                    FusedOp::Conv(_, _) => {
                        let (input_node, weight_node) = (&inputs[0], &inputs[1]);

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

                        let stride = match conv_node.strides.clone() {
                            Some(s) => s,
                            None => {
                                abort!("strides for node {} has not been initialized", idx);
                            }
                        };
                        let padding = match &conv_node.padding {
                            PaddingSpec::Explicit(p, _, _) => p,
                            _ => panic!("padding is not explicitly specified"),
                        };

                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = weight_node.out_scale + input_node.out_scale;

                        if inputs.len() == 3 {
                            let bias_node = &inputs[2];
                            let scale_diff = mn.out_scale - bias_node.out_scale;
                            let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                            bias_node = Self::scale_up_const_node(bias_node, scale_diff);
                            assert_eq!(
                                input_node.out_scale + weight_node.out_scale,
                                bias_node.out_scale
                            );
                        }

                        let oihw = weight_node.out_dims.clone();
                        let (out_channels, _, kernel_height, kernel_width) =
                            (oihw[0], oihw[1], oihw[2], oihw[3]);

                        let (padding_h, padding_w, stride_h, stride_w) =
                            (padding[0], padding[1], stride[0], stride[1]);

                        mn.in_dims = input_node.out_dims.clone();
                        trace!("{:?}", mn.in_dims);
                        let input_height = mn.in_dims[1];
                        let input_width = mn.in_dims[2];

                        let out_height =
                            (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                        let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                        mn.out_dims = vec![out_channels, out_height, out_width];

                        mn.output_max = input_node.output_max
                            * weight_node.output_max
                            * ((kernel_height * kernel_width) as f32);

                        mn.opkind = OpKind::Fused(FusedOp::Conv(
                            (padding_h, padding_w),
                            (stride_h, stride_w),
                        ));
                    }

                    FusedOp::SumPool(_, _, _) => {
                        let input_node = &inputs[0];

                        // Extract the padding and stride layer hyperparams
                        let op = Box::new(node.op());
                        let sumpool_node: &SumPool = match op.downcast_ref() {
                            Some(b) => b,
                            None => panic!("op isn't a SumPool!"),
                        };

                        let pool_spec: &PoolSpec = &sumpool_node.pool_spec;

                        // only support pytorch type formatting for now
                        assert_eq!(pool_spec.data_format, DataFormat::NCHW);

                        let stride = pool_spec.strides.clone().unwrap();
                        let padding = match &pool_spec.padding {
                            PaddingSpec::Explicit(p, _, _) => p,
                            _ => panic!("padding is not explicitly specified"),
                        };
                        let kernel_shape = &pool_spec.kernel_shape;

                        let weight_scale = input_node.out_scale;
                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = input_node.out_scale;

                        let (padding_h, padding_w, stride_h, stride_w) =
                            (padding[0], padding[1], stride[0], stride[1]);
                        let (kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1]);

                        mn.in_dims = input_node.out_dims.clone();

                        let input_channels = mn.in_dims[0];
                        let input_height = mn.in_dims[1];
                        let input_width = mn.in_dims[2];

                        let out_height =
                            (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                        let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                        mn.out_dims = vec![input_channels, out_height, out_width];
                        mn.output_max = input_node.output_max * f32::powi(2.0, weight_scale);

                        mn.opkind = OpKind::Fused(FusedOp::SumPool(
                            (padding_h, padding_w),
                            (stride_h, stride_w),
                            (kernel_height, kernel_width),
                        ));
                    }

                    FusedOp::Matmul => {
                        let (a_node, b_node) = (&inputs[0], &inputs[1]);
                        let a_dims = a_node.out_dims.clone();
                        let b_dims = b_node.out_dims.clone();
                        let in_dim = a_dims[1];
                        mn.in_dims = vec![in_dim];

                        let mut dims = Vec::from(&a_dims[0..a_dims.len() - 2]);
                        dims.push(a_dims[a_dims.len() - 2]);
                        dims.push(b_dims[a_dims.len() - 1]);

                        mn.out_dims = dims.clone();

                        mn.output_max = input_node.output_max * a_node.output_max * (in_dim as f32);

                        mn.in_scale = input_node.out_scale;

                        mn.out_scale = a_node.out_scale + input_node.out_scale;
                    }
                    FusedOp::Affine | FusedOp::ScaleAndShift => {
                        let (input_node, weight_node, bias_node) =
                            (&inputs[0], &inputs[1], &inputs[2]);

                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = weight_node.out_scale + input_node.out_scale;
                        let scale_diff = mn.out_scale - bias_node.out_scale;
                        let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                        bias_node = Self::scale_up_const_node(bias_node, scale_diff);

                        assert_eq!(
                            input_node.out_scale + weight_node.out_scale,
                            bias_node.out_scale
                        );

                        let in_dim = weight_node.out_dims.clone()[1];
                        let out_dim = weight_node.out_dims.clone()[0];
                        mn.in_dims = vec![in_dim];
                        mn.out_dims = vec![out_dim];

                        mn.output_max =
                            input_node.output_max * weight_node.output_max * (in_dim as f32);
                    }
                    // BatchNorm take four parameters, does some f32 arithmetic and then quantizes
                    // while ScaleAndShift takes the final two parameters immediately.
                    // We will also reach back and quantize
                    FusedOp::BatchNorm => {
                        //Compute scale and shift from the four inputs,
                        // then replace the first two, and change this node to a ScaleAndShift

                        // let (input_node, mut gamma_node, mut beta_node, mean_node, var_node) = (
                        //     &mut inputs[0],
                        //     &mut inputs[1],
                        //     &mut inputs[2],
                        //     &mut inputs[3],
                        //     &mut inputs[4],
                        // );
                        let gamma = inputs[1].raw_const_value.as_ref().unwrap();
                        let beta = inputs[2].raw_const_value.as_ref().unwrap();
                        let mu = inputs[3].raw_const_value.as_ref().unwrap();
                        let sigma = inputs[4].raw_const_value.as_ref().unwrap();
                        let num_entries = gamma.len();

                        let a = div(gamma.clone(), sigma.clone());
                        let amu: Tensor<f32> = mult(&vec![a.clone(), mu.clone()]);
                        let amupb: Tensor<f32> = add(&vec![amu, beta.clone()]);
                        let b = const_mult(&amupb, -1f32);

                        mn.in_scale = inputs[0].out_scale;
                        mn.out_scale = 2 * inputs[0].out_scale;
                        // gamma node becomes the scale (weigh) in scale and shift
                        inputs[1].raw_const_value = Some(a);
                        inputs[1].quantize_const_to_scale(mn.in_scale);

                        // beta node becomes the shift (bias)
                        inputs[2].raw_const_value = Some(b);
                        inputs[2].quantize_const_to_scale(mn.out_scale);

                        // this node becomes a ScaleAndShift with former gamma and beta as params
                        mn.opkind = OpKind::Fused(FusedOp::ScaleAndShift);

                        mn.in_dims = inputs[0].out_dims.clone();
                        mn.out_dims = inputs[0].out_dims.clone();

                        mn.output_max = //is gamma output max still accurate?
                            inputs[0].output_max * inputs[1].output_max * (num_entries as f32);
                    }

                    FusedOp::Add => {
                        mn.opkind = Self::homogenize_input_scales(mn.opkind, inputs.clone());
                        if let OpKind::Fused(FusedOp::Rescaled(_, mult)) = &mn.opkind {
                            mn.output_max = (inputs
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

                        mn.in_scale = inputs.iter().map(|input| input.out_scale).max().unwrap();
                        mn.out_scale = mn.in_scale;
                    }
                    FusedOp::Sum => {
                        assert!(inputs.len() == 1);
                        mn.output_max = inputs[0].output_max
                            * inputs[0].in_dims.iter().product::<usize>() as f32;
                        mn.in_scale = inputs.iter().map(|input| input.out_scale).max().unwrap();
                        mn.out_scale = mn.in_scale;
                        mn.out_dims = vec![1];
                    }
                    FusedOp::Sub => {
                        mn.opkind = Self::homogenize_input_scales(mn.opkind, inputs.clone());
                        if let OpKind::Fused(FusedOp::Rescaled(_, mult)) = &mn.opkind {
                            mn.output_max = (inputs
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
                        mn.in_scale = inputs.iter().map(|input| input.out_scale).max().unwrap();
                        mn.out_scale = mn.in_scale;
                    }
                    FusedOp::Mult => {
                        mn.output_max = f32::powf(
                            inputs
                                .iter()
                                .map(|input| input.output_max.ceil() as i32)
                                .max()
                                .unwrap() as f32,
                            inputs.len() as f32,
                        );
                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = inputs.iter().map(|input| input.out_scale).sum::<i32>();
                    }
                    FusedOp::Pow(_) => {
                        let mult = scale_to_multiplier(scale);
                        mn.inputs.pop();
                        if inputs[1].out_dims != [1] {
                            error!(
                                "ezkl currently only supports raising to the power by a constant"
                            );
                            unimplemented!()
                        }
                        let pow = inputs[1].output_max / mult;
                        mn.output_max = f32::powf(
                            inputs
                                .iter()
                                .map(|input| input.output_max.ceil() as i32)
                                .max()
                                .unwrap() as f32,
                            pow as f32,
                        );
                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = mn.in_scale * (pow as i32);

                        mn.opkind = OpKind::Fused(FusedOp::Pow(pow as usize));
                    }
                    FusedOp::Rescaled(_, _) => {
                        error!("operations should not already be rescaled at this stage")
                    }
                    FusedOp::Identity => {
                        mn.output_max = input_node.output_max;
                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = input_node.out_scale;
                    }
                    FusedOp::Reshape(_) => {
                        let shape_const_node = &inputs[1];
                        let shape_const = match shape_const_node.const_value.as_ref() {
                            Some(sc) => sc,
                            None => {
                                abort!("missing shape constant");
                            }
                        };
                        let shapes = shape_const[0..].iter();
                        let new_dims: Vec<usize> = shapes
                            .map(|x| {
                                assert!(x > &0);
                                *x as usize
                            })
                            .collect();
                        mn.opkind = OpKind::Fused(FusedOp::Reshape(new_dims.clone()));
                        mn.output_max = input_node.output_max;
                        mn.in_scale = input_node.out_scale;
                        mn.out_scale = input_node.out_scale;
                        mn.out_dims = new_dims;
                    }
                }
                // output size
                mn.min_cols += mn.out_dims[0..mn.out_dims.len() - 1]
                    .iter()
                    .product::<usize>()
                    + 1;
            }
            OpKind::Const => {
                let op = Box::new(node.op());
                let const_node: &Const = match op.as_any().downcast_ref() {
                    Some(b) => b,
                    None => {
                        abort!("op is not a const!");
                    }
                };
                let dt = const_node.0.datum_type();
                let mut dims = const_node.0.shape().to_vec();
                if dims.is_empty() {
                    dims.push(1)
                }

                match dt {
                    DatumType::F32 => {
                        mn.out_scale = mn.in_scale;
                        let vec = const_node.0.as_slice::<f32>().unwrap().to_vec();
                        let raw: Tensor<f32> = Tensor::new(Some(&vec), &dims).unwrap();
                        let t = vector_to_quantized(&vec, &dims, 0f32, mn.out_scale).unwrap();
                        mn.out_dims = t.dims().to_vec();
                        mn.in_dims = mn.out_dims.clone();
                        mn.output_max = t.iter().map(|x| x.abs()).max().unwrap() as f32;
                        mn.const_value = Some(t);
                        mn.raw_const_value = Some(raw);
                    }

                    DatumType::I64 => {
                        // Generally a shape or hyperparam
                        mn.out_scale = 0;
                        let vec = const_node.0.as_slice::<i64>().unwrap().to_vec();
                        let cast: Vec<i32> = vec.iter().map(|x| *x as i32).collect();
                        let t = Tensor::<i32>::new(Some(&cast), &dims).unwrap();
                        mn.out_dims = t.dims().to_vec();
                        mn.in_dims = mn.out_dims.clone();
                        mn.output_max = cast.iter().map(|x| x.abs()).max().unwrap() as f32;
                        mn.const_value = Some(t);
                        mn.raw_const_value = None;
                    }
                    _ => todo!(),
                }
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
                    mn.out_dims = dims[1..].to_vec();
                } else {
                    mn.out_dims = dims;
                }
                mn.in_dims = mn.out_dims.clone();

                mn.output_max = 256.0;
                mn.out_scale = mn.in_scale;
            }
            OpKind::Unknown(_) => {
                warn!("{:?}", mn);
            }
            _ => {}
        }
        mn
    }

    /// Ensures all inputs to a node have the same floating point denominator.
    pub fn homogenize_input_scales(opkind: OpKind, inputs: Vec<Node>) -> OpKind {
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

    pub fn quantize_const_to_scale(self: &mut Self, scale: i32) {
        assert!(matches!(self.opkind, OpKind::Const));
        let raw = self.raw_const_value.as_ref().unwrap();
        self.out_scale = scale;
        println!("vtq {}", scale);
        let t = vector_to_quantized(&*raw, &raw.dims(), 0f32, self.out_scale).unwrap();
        self.output_max = 0f32; //t.iter().map(|x| x.abs()).max().unwrap() as f32;
        self.const_value = Some(t);
    }

    /// Re-quantizes a constant value node to a new scale.
    pub fn scale_up_const_node(node: &mut Node, scale_diff: i32) -> &mut Node {
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
