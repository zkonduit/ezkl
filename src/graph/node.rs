use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::abort;
use crate::circuit::lookup::Config as LookupConfig;
use crate::circuit::lookup::Op as LookupOp;
use crate::circuit::polynomial::Config as PolyConfig;
use crate::circuit::polynomial::Op as PolyOp;
use crate::tensor::ops::{add, const_mult, div, mult};
use crate::tensor::Tensor;
use crate::tensor::TensorType;
use anyhow::Result;
use halo2_proofs::arithmetic::FieldExt;
use itertools::Itertools;
use log::{error, info, trace, warn};
use std::collections::{btree_map::Entry, BTreeMap};
use std::fmt;
use std::ops::Deref;
use tabled::Tabled;
use tract_onnx;
use tract_onnx::prelude::{DatumType, InferenceFact, Node as OnnxNode, OutletId};
use tract_onnx::tract_hir::{
    infer::Factoid,
    internal::InferenceOp,
    ops::activations::LeakyRelu,
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Ord, PartialOrd)]
pub enum OpKind {
    /// A nonlinearity
    Lookup(LookupOp),
    /// A fused op, combining affine layers or other arithmetic
    Poly(PolyOp),
    /// Constant
    Const,
    /// Input node
    Input,
    /// Unable to parse the node type
    Unknown(String),
    #[allow(missing_docs)]
    #[default]
    None,
}

impl OpKind {
    /// Produce an OpKind from a `&str` onnx name  
    pub fn new(name: &str) -> Self {
        match name {
            "Clip" => OpKind::Lookup(LookupOp::ReLU { scale: 1 }),
            "Prelu" => OpKind::Lookup(LookupOp::PReLU {
                scale: 1,
                slopes: vec![],
            }),
            "LeakyRelu" => OpKind::Lookup(LookupOp::LeakyReLU {
                scale: 1,
                slope: eq_float::F32(0.0),
            }),
            "Sigmoid" => OpKind::Lookup(LookupOp::Sigmoid { scales: (1, 1) }),
            "Div" => OpKind::Lookup(LookupOp::Div { scale: 1 }),
            "Const" => OpKind::Const,
            "Source" => OpKind::Input,
            "Add" => OpKind::Poly(PolyOp::Add),
            "Sub" => OpKind::Poly(PolyOp::Sub),
            "Mul" => OpKind::Poly(PolyOp::Mult),
            "Gemm" => OpKind::Poly(PolyOp::Affine),
            "MatMulInference" => OpKind::Poly(PolyOp::Matmul),
            "Dot" => OpKind::Poly(PolyOp::Dot),
            "Reduce<Sum>" => OpKind::Poly(PolyOp::Sum),
            "Pow" => OpKind::Poly(PolyOp::Pow(1)),
            "Conv" => OpKind::Poly(PolyOp::Conv {
                padding: (1, 1),
                stride: (1, 1),
            }),
            "ConvHir" => OpKind::Poly(PolyOp::Conv {
                padding: (1, 1),
                stride: (1, 1),
            }),
            "SumPool" => OpKind::Poly(PolyOp::SumPool {
                padding: (1, 1),
                stride: (1, 1),
                kernel_shape: (1, 1),
            }),
            "GlobalAvgPool" => OpKind::Poly(PolyOp::GlobalSumPool),
            "Reshape" => OpKind::Poly(PolyOp::Reshape(Vec::new())),
            "Flatten" => OpKind::Poly(PolyOp::Flatten(Vec::new())),
            "BatchNorm" => OpKind::Poly(PolyOp::BatchNorm),
            "Pad" => OpKind::Poly(PolyOp::Identity),
            c => {
                warn!("{:?} is not currently supported", c);
                OpKind::Unknown(c.to_string())
            }
        }
    }
    /// Identify fused OpKind
    pub fn is_poly(&self) -> bool {
        matches!(self, OpKind::Poly(_))
    }

    /// Identify fused OpKind
    pub fn is_lookup(&self) -> bool {
        matches!(self, OpKind::Lookup(_))
    }

    /// Identify fused OpKind
    pub fn is_input(&self) -> bool {
        matches!(self, OpKind::Input)
    }

    /// Identify constant OpKind
    pub fn is_const(&self) -> bool {
        matches!(self, OpKind::Const)
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::Const => write!(f, "const"),
            OpKind::Input => write!(f, "input"),
            OpKind::Lookup(s) => write!(f, "{}", s),
            OpKind::Poly(s) => write!(f, "{}", s),
            OpKind::Unknown(c) => write!(f, "? {}", c),
            OpKind::None => write!(f, "n/a",),
        }
    }
}

/// Enum of the different kinds of node configurations `ezkl` can support.
#[allow(missing_docs)]
#[derive(Clone, Default, Debug)]
pub enum NodeConfig<F: FieldExt + TensorType> {
    Lookup(LookupConfig<F>, Vec<usize>),
    Poly(PolyConfig<F>, Vec<usize>),
    Const,
    Input,
    #[default]
    NotConfigured,
}

/// Representation of an execution graph divided into execution 'buckets'.
#[derive(Clone, Default, Debug)]
pub struct NodeGraph(pub BTreeMap<Option<usize>, BTreeMap<usize, Node>>);

impl NodeGraph {
    /// Create an empty NodeGraph
    pub fn new() -> Self {
        NodeGraph(BTreeMap::new())
    }

    /// Insert the node with given tract `node_idx` and config at `idx`  
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

    /// Flattens the inner [BTreeMap] into a [Vec] of [Node]s.
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

    /// Retrieves a node, as specified by idx, from the Graph of bucketed nodes.
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
#[derive(Clone, Debug, Default, Tabled)]
pub struct Node {
    /// [OpKind] enum, i.e what operation this node represents.
    pub opkind: OpKind,
    /// The inferred maximum value that can appear in the output tensor given previous quantization choices.
    pub output_max: f32,
    /// The denominator in the fixed point representation for the node's input. Tensors of differing scales should not be combined.
    pub in_scale: i32,
    /// The denominator in the fixed point representation for the node's output. Tensors of differing scales should not be combined.
    pub out_scale: i32,
    #[tabled(display_with = "display_tensor")]
    /// The quantized constants potentially associated with this self.
    pub const_value: Option<Tensor<i32>>,
    #[tabled(display_with = "display_tensorf32")]
    /// The un-quantized constants potentially associated with this self.
    pub raw_const_value: Option<Tensor<f32>>,
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    #[tabled(display_with = "display_inputs")]
    /// The indices of the node's inputs.
    pub inputs: Vec<OutletId>,
    #[tabled(display_with = "display_vector")]
    /// Dimensions of input.
    pub in_dims: Vec<Vec<usize>>,
    #[tabled(display_with = "display_vector")]
    /// Dimensions of output.
    pub out_dims: Vec<usize>,
    /// The node's unique identifier.
    pub idx: usize,
    #[tabled(display_with = "display_option")]
    /// The execution bucket this node has been assigned to.
    pub bucket: Option<usize>,
}

impl Node {
    /// Converts a tract [OnnxNode] into an ezkl [Node].
    /// # Arguments:
    /// * `node` - [OnnxNode]
    /// * `other_nodes` - [BTreeMap] of other previously initialized [Node]s in the computational graph.
    /// * `scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
    /// * `idx` - The node's unique identifier.
    pub fn new(
        mut node: OnnxNode<InferenceFact, Box<dyn InferenceOp>>,
        other_nodes: &mut BTreeMap<usize, Node>,
        scale: i32,
        idx: usize,
    ) -> Self {
        trace!("Create {:?}", node);
        trace!("Create op {:?}", node.op);
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

        let mut opkind = OpKind::new(node.op().name().as_ref()); // parses the op name

        let mn = match opkind {
            OpKind::Lookup(ref s) => {
                match s {
                    LookupOp::Sigmoid { .. } => {
                        let input_node = &inputs[0];
                        let scale_diff = input_node.out_scale;
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::Sigmoid {
                                scales: (mult as usize, scale_to_multiplier(scale) as usize),
                            });
                        } else {
                            opkind = OpKind::Lookup(LookupOp::Sigmoid {
                                scales: (1, scale_to_multiplier(scale) as usize),
                            });
                        }

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: input_node.out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: scale,
                            output_max: scale_to_multiplier(scale),
                            ..Default::default()
                        }
                    }

                    LookupOp::ReLU { .. } => {
                        let input_node = &inputs[0];
                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        let mut output_max = input_node.output_max;
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::ReLU {
                                scale: mult as usize,
                            }); // now the input will be scaled down to match
                            output_max = input_node.output_max / mult;
                        }
                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: input_node.out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: scale,
                            output_max,
                            ..Default::default()
                        }
                    }
                    LookupOp::LeakyReLU {
                        scale: mut layer_scale,
                        ..
                    } => {
                        let input_node = &inputs[0];

                        // Extract the slope layer hyperparams
                        let op = Box::new(node.op());

                        let leaky_op: &LeakyRelu = match op.downcast_ref::<Box<dyn Expansion>>() {
                            Some(b) => match (*b).as_any().downcast_ref() {
                                Some(b) => b,
                                None => {
                                    panic!("not a leaky relu!");
                                }
                            },
                            None => {
                                panic!("op is not a Tract Expansion!");
                            }
                        };

                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        let mut output_max = input_node.output_max;
                        if scale_diff > 0 {
                            layer_scale = scale_to_multiplier(scale_diff) as usize;
                            output_max = input_node.output_max / (layer_scale as f32);
                        }

                        opkind = OpKind::Lookup(LookupOp::LeakyReLU {
                            scale: layer_scale,
                            slope: eq_float::F32(leaky_op.0),
                        }); // now the input will be scaled down to match

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: input_node.out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: scale,
                            output_max,
                            ..Default::default()
                        }
                    }
                    LookupOp::PReLU {
                        scale: mut layer_scale,
                        ..
                    } => {
                        let input_node = &inputs[0];
                        // Extract the slope layer hyperparams
                        let slopes = inputs[1]
                            .clone()
                            .raw_const_value
                            .unwrap()
                            .deref()
                            .iter()
                            .map(|value| eq_float::F32(*value))
                            .collect_vec();
                        node.inputs.pop();

                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        let mut output_max = input_node.output_max;
                        if scale_diff > 0 {
                            layer_scale = scale_to_multiplier(scale_diff) as usize;
                            output_max = input_node.output_max / (layer_scale as f32);
                        }

                        opkind = OpKind::Lookup(LookupOp::PReLU {
                            scale: layer_scale,
                            slopes: slopes.clone(),
                        }); // now the input will be scaled down to match

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs,
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: input_node.out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: scale,
                            output_max,
                            ..Default::default()
                        }
                    }
                    LookupOp::Div { .. } => {
                        if inputs[1].out_dims.clone() != [1] {
                            abort!("ezkl currently only supports division by a constant");
                        }
                        let mult = scale_to_multiplier(scale);
                        let div = inputs[1].output_max / mult;
                        let input_node = &inputs[0];

                        let mut input_outlets = node.inputs.clone();
                        input_outlets.pop();

                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        let output_max: f32;
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::Div {
                                scale: (div * mult) as usize,
                            }); // now the input will be scaled down to match
                            output_max = input_node.output_max / (div * mult);
                        } else {
                            opkind = OpKind::Lookup(LookupOp::Div {
                                scale: div as usize,
                            }); // now the input will be scaled down to match
                            output_max = input_node.output_max / (div);
                        }

                        Node {
                            idx,
                            opkind,
                            inputs: input_outlets,
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: input_node.out_dims.clone(),
                            // in scale is the same as the input
                            in_scale: input_node.out_scale,
                            // same for the output scale
                            out_scale: scale,
                            output_max,
                            ..Default::default()
                        }
                    }
                }
            }
            OpKind::Poly(ref s) => {
                match s {
                    PolyOp::Dot => todo!(),
                    PolyOp::Conv { .. } => {
                        let (input_node, weight_node) = (&inputs[0], &inputs[1]);

                        // Extract the padding and stride layer hyperparams
                        let op = Box::new(node.op());

                        let conv_node: &Conv = match op.downcast_ref::<Box<dyn Expansion>>() {
                            Some(b) => match (*b).as_any().downcast_ref() {
                                Some(b) => b,
                                None => {
                                    panic!("not a conv!");
                                }
                            },
                            None => {
                                panic!("op is not a Tract Expansion!");
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

                        if inputs.len() == 3 {
                            let bias_node = &inputs[2];
                            let scale_diff =
                                weight_node.out_scale + input_node.out_scale - bias_node.out_scale;
                            let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                            bias_node = Self::scale_up_const_node(bias_node, scale + scale_diff);
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

                        let input_height = input_node.out_dims[1];
                        let input_width = input_node.out_dims[2];

                        let out_height =
                            (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                        let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::Conv {
                                padding: (padding_h, padding_w),
                                stride: (stride_h, stride_w),
                            }),
                            inputs: node.inputs.clone(),
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: vec![out_channels, out_height, out_width],
                            in_scale: input_node.out_scale,
                            out_scale: weight_node.out_scale + input_node.out_scale,
                            output_max: input_node.output_max
                                * weight_node.output_max
                                * ((kernel_height * kernel_width) as f32),
                            ..Default::default()
                        }
                    }

                    PolyOp::SumPool { .. } => {
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

                        let (padding_h, padding_w, stride_h, stride_w) =
                            (padding[0], padding[1], stride[0], stride[1]);
                        let (kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1]);

                        let input_channels = input_node.out_dims[0];
                        let input_height = input_node.out_dims[1];
                        let input_width = input_node.out_dims[2];

                        let out_height =
                            (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                        let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::SumPool {
                                padding: (padding_h, padding_w),
                                stride: (stride_h, stride_w),
                                kernel_shape: (kernel_height, kernel_width),
                            }),
                            inputs: node.inputs.clone(),
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: vec![input_channels, out_height, out_width],
                            in_scale: input_node.out_scale,
                            out_scale: input_node.out_scale,
                            output_max: input_node.output_max
                                * f32::powi(2.0, input_node.out_scale),
                            ..Default::default()
                        }
                    }

                    PolyOp::GlobalSumPool => {
                        let input_node = &inputs[0];
                        let input_channels = input_node.out_dims[0];
                        let input_height = input_node.out_dims[1];
                        let input_width = input_node.out_dims[2];

                        let (padding_h, padding_w, stride_h, stride_w) = (0, 0, 1, 1);
                        let (kernel_height, kernel_width) = (input_height, input_width);

                        // These are 1 if padding is 0,0 and stride is 1,1
                        let out_height =
                            (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                        let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::SumPool {
                                padding: (padding_h, padding_w),
                                stride: (stride_h, stride_w),
                                kernel_shape: (kernel_height, kernel_width),
                            }),
                            inputs: node.inputs.clone(),
                            in_dims: vec![input_node.out_dims.clone()],
                            out_dims: vec![input_channels, out_height, out_width],
                            in_scale: input_node.out_scale,
                            out_scale: input_node.out_scale,
                            output_max: input_node.output_max
                                * (input_height as f32)
                                * (input_width as f32),
                            ..Default::default()
                        }
                    }

                    PolyOp::Matmul => {
                        let (a_node, b_node) = (&inputs[0], &inputs[1]);
                        let a_dims = a_node.out_dims.clone();
                        let b_dims = b_node.out_dims.clone();
                        let in_dim = a_dims[1];

                        let mut dims = Vec::from(&a_dims[0..a_dims.len() - 2]);
                        dims.push(a_dims[a_dims.len() - 2]);
                        dims.push(b_dims[a_dims.len() - 1]);

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: vec![vec![in_dim]],
                            out_dims: dims.clone(),
                            in_scale: a_node.out_scale,
                            out_scale: a_node.out_scale + b_node.out_scale,
                            output_max: a_node.output_max * b_node.output_max * (in_dim as f32),
                            ..Default::default()
                        }
                    }
                    PolyOp::Affine | PolyOp::ScaleAndShift => {
                        let (input_node, weight_node, bias_node) =
                            (&inputs[0], &inputs[1], &inputs[2]);

                        let scale_diff =
                            weight_node.out_scale + input_node.out_scale - bias_node.out_scale;
                        let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                        bias_node = Self::scale_up_const_node(bias_node, scale + scale_diff);

                        assert_eq!(
                            input_node.out_scale + weight_node.out_scale,
                            bias_node.out_scale
                        );

                        let in_dim = weight_node.out_dims.clone()[1];
                        let out_dim = weight_node.out_dims.clone()[0];

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: vec![out_dim],
                            in_scale: input_node.out_scale,
                            out_scale: weight_node.out_scale + input_node.out_scale,
                            output_max: input_node.output_max
                                * weight_node.output_max
                                * (in_dim as f32),
                            ..Default::default()
                        }
                    }
                    // BatchNorm take four parameters, does some f32 arithmetic and then quantizes
                    // while ScaleAndShift takes the final two parameters immediately.
                    // We will also reach back and quantize
                    PolyOp::BatchNorm => {
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

                        let in_scale = inputs[0].out_scale;
                        let out_scale = 2 * inputs[0].out_scale;
                        // gamma node becomes the scale (weigh) in scale and shift
                        inputs[1].raw_const_value = Some(a);
                        inputs[1].quantize_const_to_scale(in_scale);

                        // beta node becomes the shift (bias)
                        inputs[2].raw_const_value = Some(b);
                        inputs[2].quantize_const_to_scale(out_scale);

                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::ScaleAndShift),
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: inputs[0].out_dims.clone(),
                            in_scale,
                            out_scale,
                            output_max: inputs[0].output_max
                                * inputs[1].output_max
                                * (num_entries as f32),
                            ..Default::default()
                        }
                    }

                    PolyOp::Add => {
                        opkind = Self::homogenize_input_scales(opkind, inputs.clone());
                        let output_max =
                            if let OpKind::Poly(PolyOp::Rescaled { scale, .. }) = &opkind {
                                (inputs
                                    .iter()
                                    .enumerate()
                                    .map(|(idx, n)| {
                                        ((scale[idx].1 as f32) * (n.output_max.ceil())) as i32
                                    })
                                    .max()
                                    .unwrap() as f32)
                                    * (inputs.len() as f32)
                            } else {
                                error!("failed to homogenize input scalings for node {}", idx);
                                panic!()
                            };

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: inputs[0].out_dims.clone(),
                            in_scale: inputs.iter().map(|input| input.out_scale).max().unwrap(),
                            out_scale: inputs.iter().map(|input| input.out_scale).max().unwrap(),
                            output_max,
                            ..Default::default()
                        }
                    }
                    PolyOp::Sum => {
                        assert!(inputs.len() == 1);

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: vec![1],
                            in_scale: inputs.iter().map(|input| input.out_scale).max().unwrap(),
                            out_scale: inputs.iter().map(|input| input.out_scale).max().unwrap(),
                            output_max: inputs[0].output_max
                                * inputs[0].out_dims.iter().product::<usize>() as f32,
                            ..Default::default()
                        }
                    }
                    PolyOp::Sub => {
                        opkind = Self::homogenize_input_scales(opkind, inputs.clone());
                        let output_max =
                            if let OpKind::Poly(PolyOp::Rescaled { inner: _, scale }) = &opkind {
                                (inputs
                                    .iter()
                                    .enumerate()
                                    .map(|(idx, n)| {
                                        ((scale[idx].1 as f32) * (n.output_max.ceil())) as i32
                                    })
                                    .max()
                                    .unwrap() as f32)
                                    * (inputs.len() as f32)
                            } else {
                                error!("failed to homogenize input scalings for node {}", idx);
                                panic!()
                            };

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: inputs[0].out_dims.clone(),
                            in_scale: inputs.iter().map(|input| input.out_scale).max().unwrap(),
                            out_scale: inputs.iter().map(|input| input.out_scale).max().unwrap(),
                            output_max,
                            ..Default::default()
                        }
                    }
                    PolyOp::Mult => {
                        let input_node = &inputs[0];

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: inputs[0].out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: inputs.iter().map(|input| input.out_scale).sum::<i32>(),
                            output_max: f32::powf(
                                inputs
                                    .iter()
                                    .map(|input| input.output_max.ceil() as i32)
                                    .max()
                                    .unwrap() as f32,
                                inputs.len() as f32,
                            ),
                            ..Default::default()
                        }
                    }
                    PolyOp::Pow(_) => {
                        let input_node = &inputs[0];
                        let pow = inputs[1].clone().raw_const_value.unwrap()[0];
                        node.inputs.pop();
                        if inputs[1].out_dims != [1] {
                            error!(
                                "ezkl currently only supports raising to the power by a constant"
                            );
                            unimplemented!()
                        }

                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::Pow(pow as usize)),
                            inputs: node.inputs,
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: input_node.out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: input_node.out_scale * (pow as i32),
                            output_max: f32::powf(
                                inputs
                                    .iter()
                                    .map(|input| input.output_max.ceil() as i32)
                                    .max()
                                    .unwrap() as f32,
                                pow,
                            ),
                            ..Default::default()
                        }
                    }
                    PolyOp::Rescaled { .. } => {
                        error!("operations should not already be rescaled at this stage");
                        panic!()
                    }
                    PolyOp::Identity => {
                        let input_node = &inputs[0];
                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: input_node.out_dims.clone(),
                            in_scale: input_node.out_scale,
                            out_scale: input_node.out_scale,
                            output_max: input_node.output_max,
                            ..Default::default()
                        }
                    }
                    PolyOp::Flatten(_) => {
                        let input_node = &inputs[0];
                        let new_dims: Vec<usize> =
                            vec![inputs[0].out_dims.iter().product::<usize>()];
                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::Flatten(new_dims.clone())),
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: new_dims,
                            in_scale: input_node.out_scale,
                            out_scale: input_node.out_scale,
                            output_max: input_node.output_max,
                            ..Default::default()
                        }
                    }
                    PolyOp::Reshape(_) => {
                        let input_node = &inputs[0];
                        let shape_const_node = &inputs[1];
                        let shape_const = match shape_const_node.const_value.as_ref() {
                            Some(sc) => sc,
                            None => {
                                abort!("missing shape constant");
                            }
                        };
                        let shapes = &shape_const[0..];
                        let new_dims: Vec<usize> = if shapes.iter().all(|x| x > &0) {
                            shapes
                                .iter()
                                .map(|x| {
                                    assert!(x > &0);
                                    *x as usize
                                })
                                .collect()
                        } else {
                            let num_entries: usize = input_node.out_dims.iter().product();
                            let explicit_prod: i32 = shapes.iter().filter(|x| *x > &0).product();
                            assert!(explicit_prod > 0);
                            let inferred = num_entries / (explicit_prod as usize);
                            let mut new_dims: Vec<usize> = Vec::new();
                            for i in shapes {
                                match i {
                                    -1 => new_dims.push(inferred),
                                    0 => continue,
                                    x => new_dims.push(*x as usize),
                                }
                            }
                            new_dims
                        };

                        Node {
                            idx,
                            opkind: OpKind::Poly(PolyOp::Reshape(new_dims.clone())),
                            inputs: node.inputs.clone(),
                            in_dims: inputs.iter().map(|inp| inp.out_dims.clone()).collect(),
                            out_dims: new_dims,
                            in_scale: input_node.out_scale,
                            out_scale: input_node.out_scale,
                            output_max: input_node.output_max,
                            ..Default::default()
                        }
                    }
                }
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
                        let vec = const_node.0.as_slice::<f32>().unwrap().to_vec();
                        let raw: Tensor<f32> = Tensor::new(Some(&vec), &dims).unwrap();
                        let t = vector_to_quantized(&vec, &dims, 0f32, scale).unwrap();

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: vec![dims.clone()],
                            out_dims: dims,
                            in_scale: scale,
                            out_scale: scale,
                            output_max: t.iter().map(|x| x.abs()).max().unwrap() as f32,
                            const_value: Some(t),
                            raw_const_value: Some(raw),
                            ..Default::default()
                        }
                    }

                    DatumType::I64 => {
                        // Generally a shape or hyperparam
                        let vec = const_node.0.as_slice::<i64>().unwrap().to_vec();
                        let cast: Vec<i32> = vec.iter().map(|x| *x as i32).collect();
                        let t = Tensor::<i32>::new(Some(&cast), &dims).unwrap();

                        Node {
                            idx,
                            opkind,
                            inputs: node.inputs.clone(),
                            in_dims: vec![dims.clone()],
                            out_dims: dims,
                            in_scale: scale,
                            out_scale: 0,
                            output_max: cast.iter().map(|x| x.abs()).max().unwrap() as f32,
                            const_value: Some(t),
                            raw_const_value: None,
                            ..Default::default()
                        }
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
                let out_dims = if dims[0] == 1 && dims.len() > 1 {
                    dims[1..].to_vec()
                } else {
                    dims
                };

                Node {
                    idx,
                    opkind,
                    inputs: node.inputs.clone(),
                    in_dims: vec![out_dims.clone()],
                    out_dims,
                    in_scale: scale,
                    out_scale: scale,
                    output_max: 256.0,
                    ..Default::default()
                }
            }
            OpKind::Unknown(_) => {
                warn!("{:?} is unknown", opkind);
                Node::default()
            }
            o => {
                error!("unsupported op {:?}", o);
                panic!()
            }
        };
        mn
    }

    /// Ensures all inputs to a node have the same fixed point denominator.
    fn homogenize_input_scales(opkind: OpKind, inputs: Vec<Node>) -> OpKind {
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
        if let OpKind::Poly(c) = &opkind {
            OpKind::Poly(PolyOp::Rescaled {
                inner: Box::new(c.clone()),
                scale: (0..inputs.len()).zip(multipliers).collect_vec(),
            })
        } else {
            error!("should not homegenize input scales for non fused ops.");
            panic!()
        }
    }

    fn quantize_const_to_scale(&mut self, scale: i32) {
        assert!(matches!(self.opkind, OpKind::Const));
        let raw = self.raw_const_value.as_ref().unwrap();
        self.out_scale = scale;
        let t = vector_to_quantized(raw, raw.dims(), 0f32, self.out_scale).unwrap();
        self.output_max = 0f32; //t.iter().map(|x| x.abs()).max().unwrap() as f32;
        self.const_value = Some(t);
    }

    /// Re-quantizes a constant value node to a new scale.
    fn scale_up_const_node(node: &mut Node, scale: i32) -> &mut Node {
        assert!(matches!(node.opkind, OpKind::Const));
        if scale > 0 {
            if let Some(raw) = &node.raw_const_value {
                let mult = scale_to_multiplier(scale);
                let t = vector_to_quantized(&raw, raw.dims(), 0f32, scale).unwrap();
                node.const_value = Some(t);
                info!(
                    "------ scaled const node {:?}: {:?} -> {:?}",
                    node.idx, node.in_scale, scale
                );
                node.output_max *= mult;
                node.out_scale = scale;
            }
        }
        node
    }
}
