use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::circuit::hybrid::HybridOp;
use crate::circuit::ops::lookup::LookupOp;
use crate::circuit::ops::poly::PolyOp;
use crate::circuit::utils;
use crate::circuit::BaseConfig;
use crate::circuit::Op;
use crate::graph::GraphError;
use crate::tensor::Tensor;
use crate::tensor::TensorType;
use anyhow::Result;
use halo2_proofs::arithmetic::FieldExt;
use itertools::Itertools;
use log::{info, trace, warn};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::rc::Rc;
use tabled::Tabled;
use tract_onnx;
use tract_onnx::prelude::{DatumType, InferenceFact, Node as OnnxNode};
use tract_onnx::tract_hir::{
    infer::Factoid,
    internal::InferenceOp,
    ops::activations::LeakyRelu,
    ops::array::{Pad, PadMode},
    ops::cnn::{Conv, MaxPool, PoolSpec, SumPool},
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Ord, PartialOrd, Deserialize, Serialize)]
pub enum OpKind {
    /// A nonlinearity
    Lookup(LookupOp),
    /// A nonlinearity
    Hybrid(HybridOp),
    /// A fused op, combining affine layers or other arithmetic
    Poly(PolyOp),
    /// Constant
    Const {
        /// The quantized constants potentially associated with this self.
        const_value: Tensor<i128>,
        /// The un-quantized constants potentially associated with this self.
        raw_const_value: Option<Tensor<utils::F32>>,
    },
    /// Input node
    Input,
    /// Unable to parse the node type
    #[default]
    Unknown,
}

impl From<PolyOp> for OpKind {
    fn from(op: PolyOp) -> Self {
        OpKind::Poly(op)
    }
}

impl From<LookupOp> for OpKind {
    fn from(op: LookupOp) -> Self {
        OpKind::Lookup(op)
    }
}

impl OpKind {
    /// Produce an OpKind from a `&str` onnx name  
    pub fn new(
        idx: usize,
        scale: u32,
        node: OnnxNode<InferenceFact, Box<dyn InferenceOp>>,
    ) -> Result<Self, Box<dyn Error>> {
        let op = match node.op().name().as_ref() {
            "Reduce<Min>" => OpKind::Hybrid(HybridOp::Min),
            "Reduce<Max>" => OpKind::Hybrid(HybridOp::Max),
            "Clip" => OpKind::Lookup(LookupOp::ReLU { scale: 1 }),
            "Prelu" => OpKind::Hybrid(HybridOp::PReLU {
                scale: 1,
                slopes: vec![],
            }),
            "LeakyRelu" => {
                // Extract the slope layer hyperparams
                let op = Box::new(node.op());

                let leaky_op: &LeakyRelu = match op.downcast_ref::<Box<dyn Expansion>>() {
                    Some(b) => match (*b).as_any().downcast_ref() {
                        Some(b) => b,
                        None => {
                            return Err(Box::new(GraphError::OpMismatch(
                                idx,
                                "leaky relu".to_string(),
                            )));
                        }
                    },
                    None => {
                        return Err(Box::new(GraphError::OpMismatch(
                            idx,
                            "leaky relu".to_string(),
                        )));
                    }
                };
                OpKind::Lookup(LookupOp::LeakyReLU {
                    scale: 1,
                    slope: crate::circuit::utils::F32(leaky_op.0),
                })
            }
            "Sigmoid" => OpKind::Lookup(LookupOp::Sigmoid { scales: (1, 1) }),
            "Sqrt" => OpKind::Lookup(LookupOp::Sqrt { scales: (1, 1) }),
            "Tanh" => OpKind::Lookup(LookupOp::Tanh { scales: (1, 1) }),
            "onnx.Erf" => OpKind::Lookup(LookupOp::Erf { scales: (1, 1) }),
            "Div" => OpKind::Lookup(LookupOp::Div {
                denom: utils::F32(1.0),
            }),

            "Const" => {
                let op = Box::new(node.op());
                let const_node: &Const = match op.as_any().downcast_ref() {
                    Some(b) => b,
                    None => {
                        return Err(Box::new(GraphError::OpMismatch(idx, "const".to_string())));
                    }
                };
                let dt = const_node.0.datum_type();
                let mut dims = const_node.0.shape().to_vec();
                if dims.is_empty() {
                    dims.push(1)
                }

                let const_value: Tensor<i128>;
                let mut raw_const_value = None;
                match dt {
                    DatumType::F32 => {
                        let vec = const_node.0.as_slice::<f32>().unwrap().to_vec();
                        let raw: Tensor<f32> = Tensor::new(Some(&vec), &dims).unwrap();
                        let t = vector_to_quantized(&vec, &dims, 0f32, scale).unwrap();
                        const_value = t;
                        raw_const_value = Some(raw.map(|f| utils::F32(f)));
                    }

                    DatumType::I64 => {
                        // Generally a shape or hyperparam
                        let vec = const_node.0.as_slice::<i64>().unwrap().to_vec();
                        let cast: Vec<i128> = vec.iter().map(|x| *x as i128).collect();
                        let t = Tensor::<i128>::new(Some(&cast), &dims).unwrap();
                        const_value = t;
                    }
                    _ => todo!(),
                }
                OpKind::Const {
                    const_value,
                    raw_const_value,
                }
            }
            "Source" => OpKind::Input,
            "Add" => OpKind::Poly(PolyOp::Add),
            "Sub" => OpKind::Poly(PolyOp::Sub),
            "Mul" => OpKind::Poly(PolyOp::Mult),
            "Gemm" => OpKind::Poly(PolyOp::Affine),
            "MatMulInference" => OpKind::Poly(PolyOp::Matmul),
            "MaxPool" => {
                // Extract the padding and stride layer hyperparams
                let op = Box::new(node.op());
                let sumpool_node: &MaxPool = match op.downcast_ref() {
                    Some(b) => b,
                    None => {
                        return Err(Box::new(GraphError::OpMismatch(idx, "Maxpool".to_string())));
                    }
                };

                let pool_spec: &PoolSpec = &sumpool_node.pool_spec;

                // only support pytorch type formatting for now
                if pool_spec.data_format != DataFormat::NCHW {
                    return Err(Box::new(GraphError::MissingParams(
                        "data in wrong format".to_string(),
                    )));
                }

                let stride = pool_spec.strides.clone().unwrap();
                let padding = match &pool_spec.padding {
                    PaddingSpec::Explicit(p, _, _) => p,
                    _ => {
                        return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                    }
                };
                let kernel_shape = &pool_spec.kernel_shape;

                let (padding_h, padding_w, stride_h, stride_w) =
                    (padding[0], padding[1], stride[0], stride[1]);
                let (kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1]);

                OpKind::Hybrid(HybridOp::MaxPool2d {
                    padding: (padding_h, padding_w),
                    stride: (stride_h, stride_w),
                    pool_dims: (kernel_height, kernel_width),
                })
            }
            "Dot" => OpKind::Poly(PolyOp::Dot),
            "Reduce<Sum>" => OpKind::Poly(PolyOp::Sum),
            "Reduce<Mean>" => OpKind::Hybrid(HybridOp::Mean { scale: 1 }),
            "Pow" => OpKind::Poly(PolyOp::Pow(1)),
            "Conv" | "ConvHir" => {
                // Extract the padding and stride layer hyperparams
                let op = Box::new(node.op());

                let conv_node: &Conv = match op.downcast_ref::<Box<dyn Expansion>>() {
                    Some(b) => match (*b).as_any().downcast_ref() {
                        Some(b) => b,
                        None => {
                            return Err(Box::new(GraphError::OpMismatch(idx, "conv".to_string())));
                        }
                    },
                    None => {
                        return Err(Box::new(GraphError::OpMismatch(idx, "conv".to_string())));
                    }
                };

                if (conv_node.data_format != DataFormat::NCHW)
                    || (conv_node.kernel_fmt != KernelFormat::OIHW)
                {
                    return Err(Box::new(GraphError::MisformedParams(
                        "data or kernel in wrong format".to_string(),
                    )));
                }

                let stride = match conv_node.strides.clone() {
                    Some(s) => s,
                    None => {
                        return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                    }
                };
                let padding = match &conv_node.padding {
                    PaddingSpec::Explicit(p, _, _) => p,
                    _ => {
                        return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                    }
                };

                let (padding_h, padding_w, stride_h, stride_w) =
                    (padding[0], padding[1], stride[0], stride[1]);
                OpKind::Poly(PolyOp::Conv {
                    padding: (padding_h, padding_w),
                    stride: (stride_h, stride_w),
                })
            }

            "SumPool" => {
                // Extract the padding and stride layer hyperparams
                let op = Box::new(node.op());
                let sumpool_node: &SumPool = match op.downcast_ref() {
                    Some(b) => b,
                    None => {
                        return Err(Box::new(GraphError::OpMismatch(idx, "sumpool".to_string())));
                    }
                };

                let pool_spec: &PoolSpec = &sumpool_node.pool_spec;

                // only support pytorch type formatting for now
                if pool_spec.data_format != DataFormat::NCHW {
                    return Err(Box::new(GraphError::MissingParams(
                        "data in wrong format".to_string(),
                    )));
                }

                let stride = pool_spec.strides.clone().unwrap();
                let padding = match &pool_spec.padding {
                    PaddingSpec::Explicit(p, _, _) => p,
                    _ => {
                        return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                    }
                };
                let kernel_shape = &pool_spec.kernel_shape;

                let (padding_h, padding_w, stride_h, stride_w) =
                    (padding[0], padding[1], stride[0], stride[1]);
                let (kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1]);

                OpKind::Poly(PolyOp::SumPool {
                    padding: (padding_h, padding_w),
                    stride: (stride_h, stride_w),
                    kernel_shape: (kernel_height, kernel_width),
                })
            }
            "InstanceNorm" => OpKind::Hybrid(HybridOp::InstanceNorm2d {
                epsilon: utils::F32(1e-5),
            }),
            "GlobalAvgPool" => OpKind::Poly(PolyOp::GlobalSumPool),
            "Pad" => {
                let pad_node: &Pad = match node.op().downcast_ref::<Pad>() {
                    Some(b) => b,
                    None => {
                        return Err(Box::new(GraphError::OpMismatch(idx, "pad".to_string())));
                    }
                };
                // we only support constant 0 padding
                if pad_node.mode
                    != PadMode::Constant(tract_onnx::prelude::Arc::new(
                        tract_onnx::prelude::Tensor::zero::<f32>(&[])?,
                    ))
                {
                    return Err(Box::new(GraphError::MisformedParams(
                        "pad mode or pad type".to_string(),
                    )));
                }

                let padding_len = pad_node.pads.len();

                // we only support symmetrical padding that affects the last 2 dims (height and width params)
                for (i, pad_params) in pad_node.pads.iter().enumerate() {
                    if (i < padding_len - 2) && ((pad_params.0 != 0) || (pad_params.1 != 0)) {
                        return Err(Box::new(GraphError::MisformedParams(
                            "ezkl currently only supports padding height and width dimensions"
                                .to_string(),
                        )));
                    }
                    if pad_params.0 != pad_params.1 {
                        return Err(Box::new(GraphError::MisformedParams(
                            "ezkl currently only supports symmetric padding".to_string(),
                        )));
                    }
                }

                let (padding_h, padding_w) = (
                    pad_node.pads[padding_len - 2].0,
                    pad_node.pads[padding_len - 1].0,
                );
                OpKind::Poly(PolyOp::Pad(padding_h, padding_w))
            }
            "Reshape" => OpKind::Poly(PolyOp::Reshape(Vec::new())),
            "Flatten" => OpKind::Poly(PolyOp::Flatten(Vec::new())),
            "BatchNorm" => OpKind::Poly(PolyOp::BatchNorm),
            c => {
                warn!("{:?} is not currently supported", c);
                OpKind::Unknown
            }
        };
        Ok(op)
    }

    /// Get the inner op
    pub fn get_inner<F: FieldExt + TensorType>(&self) -> Box<dyn Op<F>> {
        match self {
            OpKind::Poly(op) => Box::new(op.clone()),
            OpKind::Lookup(op) => Box::new(op.clone()),
            OpKind::Hybrid(op) => Box::new(op.clone()),
            OpKind::Const { .. } => Box::new(crate::circuit::ops::Const),
            OpKind::Input => Box::new(crate::circuit::ops::Input),
            OpKind::Unknown => Box::new(crate::circuit::ops::Unknown),
        }
    }

    /// is ploy type constrant
    pub fn is_poly(&self) -> bool {
        matches!(self, OpKind::Poly(_))
    }

    /// is lookup based op
    pub fn is_lookup(&self) -> bool {
        matches!(self, OpKind::Lookup(_))
    }

    /// is hybriud based op
    pub fn is_hybrid(&self) -> bool {
        matches!(self, OpKind::Hybrid(_))
    }

    /// is lookup based op
    pub fn is_parameterized(&self) -> bool {
        match self {
            OpKind::Poly(PolyOp::Affine) | OpKind::Poly(PolyOp::Conv { .. }) => true,
            _ => false,
        }
    }

    /// is rescaled op
    pub fn is_rescaled(&self) -> bool {
        matches!(self, OpKind::Poly(PolyOp::Rescaled { .. }))
    }

    /// is input
    pub fn is_input(&self) -> bool {
        matches!(self, OpKind::Input)
    }

    /// is const
    pub fn is_const(&self) -> bool {
        matches!(self, OpKind::Const { .. })
    }
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::Const { .. } => write!(f, "const"),
            OpKind::Input => write!(f, "input"),
            OpKind::Hybrid(s) => write!(f, "{:#?}", s),
            OpKind::Lookup(s) => write!(f, "{:#?}", s),
            OpKind::Poly(s) => write!(f, "{}", s),
            OpKind::Unknown => write!(f, "?"),
        }
    }
}
/// Enum of the different kinds of node configurations `ezkl` can support.
#[allow(missing_docs)]
#[derive(Clone, Default, Debug)]
pub enum NodeConfig<F: FieldExt + TensorType> {
    Op {
        config: Rc<RefCell<BaseConfig<F>>>,
        inputs: Vec<usize>,
        op: OpKind,
    },
    Const,
    Input,
    #[default]
    NotConfigured,
}

/// Representation of an execution graph divided into execution 'buckets'.
pub type NodeGraph = BTreeMap<usize, Node>;

fn display_vector<T: fmt::Debug>(v: &Vec<T>) -> String {
    if v.len() > 0 {
        format!("{:?}", v)
    } else {
        format!("")
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
#[derive(Clone, Debug, Default, Tabled, Serialize, Deserialize)]
pub struct Node {
    /// [OpKind] enum, i.e what operation this node represents.
    pub opkind: OpKind,
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

impl Node {
    /// Converts a tract [OnnxNode] into an ezkl [Node].
    /// # Arguments:
    /// * `node` - [OnnxNode]
    /// * `other_nodes` - [BTreeMap] of other previously initialized [Node]s in the computational graph.
    /// * `scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
    /// * `idx` - The node's unique identifier.
    pub fn new<F: FieldExt + TensorType>(
        mut node: OnnxNode<InferenceFact, Box<dyn InferenceOp>>,
        other_nodes: &mut BTreeMap<usize, Node>,
        scale: u32,
        idx: usize,
    ) -> Result<Self, Box<dyn Error>> {
        trace!("Create {:?}", node);
        trace!("Create op {:?}", node.op);

        // input_nodes come in all shapes and sizes we gotta homogenize, especially for 2D (single channel images)
        let mut opkind = OpKind::new(idx, scale, node.clone())?; // parses the op name
        let inner_op = opkind.get_inner::<F>();
        if inner_op.has_3d_input() {
            let input_node = other_nodes.get_mut(&node.inputs[0].node).unwrap();
            Self::format_3d_inputs(input_node)?;
        };

        let mut inputs = vec![];
        for i in node.inputs.iter_mut() {
            match other_nodes.get(&i.node) {
                Some(n) => inputs.push(n.clone()),
                None => return Err(Box::new(GraphError::MissingNode(i.node))),
            }
        }

        let in_dims: Vec<Vec<usize>> = inputs.iter().map(|i| i.out_dims.clone()).collect();
        let out_dims = match in_dims.len() {
            0 => {
                // remove batch dim for now
                match opkind {
                    OpKind::Const {
                        ref const_value, ..
                    } => const_value.dims().to_vec(),
                    _ => {
                        let output_shapes = match node_output_shapes(&node) {
                            Ok(s) => Some(s),
                            _ => None,
                        };

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
                                .map(|x| (*x as i128) as usize)
                                .collect()
                        };
                        if dims.len() > 0 && dims[0] == 1 && dims.len() > 1 {
                            dims[1..].to_vec()
                        } else {
                            dims
                        }
                    }
                }
            }
            _ => inner_op.out_dims(in_dims),
        };
        println!("dims: {:?}", out_dims);
        let in_scales: Vec<u32> = inputs.iter().map(|i| i.out_scale).collect();
        let out_scale = match in_scales.len() {
            0 => scale,
            _ => inner_op.out_scale(in_scales, scale),
        };

        let mut ezkl_node = Node {
            idx,
            opkind: opkind.clone(),
            inputs: node.inputs.iter().map(|i| i.node).collect(),
            out_dims,
            out_scale,
        };

        // we now run a forward pass to re-quantize the inputs to the node
        // this is necessary because the inputs to the node may have been quantized differently
        // we also homogenize the inputs to the node to be 3D if required.
        match opkind {
            OpKind::Hybrid(ref s) => match s {
                HybridOp::Mean { .. } => {
                    let input_node = &inputs[0];
                    let scale_diff = input_node.out_scale - scale;
                    // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        opkind = OpKind::Hybrid(HybridOp::Mean {
                            scale: mult as usize,
                        }); // now the input will be scaled down to match
                    }
                }
                HybridOp::PReLU {
                    scale: mut layer_scale,
                    ..
                } => {
                    let input_node = &inputs[0];
                    // Extract the slope layer hyperparams

                    let slopes = match &inputs[1].opkind {
                        OpKind::Const {
                            raw_const_value, ..
                        } => raw_const_value.as_ref().unwrap(),
                        _ => {
                            return Err(Box::new(GraphError::MissingParams("slopes".to_string())));
                        }
                    };

                    let scale_diff = input_node.out_scale - scale;
                    // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                    if scale_diff > 0 {
                        layer_scale = scale_to_multiplier(scale_diff) as usize;
                    }

                    opkind = OpKind::Hybrid(HybridOp::PReLU {
                        scale: layer_scale,
                        slopes: slopes.clone().into_iter().collect(),
                    }); // now the input will be scaled down to match
                }
                _ => {}
            },

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
                    }

                    LookupOp::Sqrt { .. } => {
                        let input_node = &inputs[0];
                        let scale_diff = input_node.out_scale;
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::Sqrt {
                                scales: (mult as usize, scale_to_multiplier(scale) as usize),
                            });
                        } else {
                            opkind = OpKind::Lookup(LookupOp::Sqrt {
                                scales: (1, scale_to_multiplier(scale) as usize),
                            });
                        }
                    }

                    LookupOp::Tanh { .. } => {
                        let input_node = &inputs[0];
                        let scale_diff = input_node.out_scale;
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::Tanh {
                                scales: (mult as usize, scale_to_multiplier(scale) as usize),
                            });
                        } else {
                            opkind = OpKind::Lookup(LookupOp::Tanh {
                                scales: (1, scale_to_multiplier(scale) as usize),
                            });
                        }
                    }

                    LookupOp::Erf { .. } => {
                        let input_node = &inputs[0];
                        let scale_diff = input_node.out_scale;
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::Erf {
                                scales: (mult as usize, scale_to_multiplier(scale) as usize),
                            });
                        } else {
                            opkind = OpKind::Lookup(LookupOp::Erf {
                                scales: (1, scale_to_multiplier(scale) as usize),
                            });
                        }
                    }

                    LookupOp::ReLU { .. } => {
                        let input_node = &inputs[0];
                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::ReLU {
                                scale: mult as usize,
                            }); // now the input will be scaled down to match
                        }
                    }

                    LookupOp::LeakyReLU {
                        scale: mut layer_scale,
                        slope,
                        ..
                    } => {
                        let input_node = &inputs[0];

                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        if scale_diff > 0 {
                            layer_scale = scale_to_multiplier(scale_diff) as usize;
                        }

                        opkind = OpKind::Lookup(LookupOp::LeakyReLU {
                            scale: layer_scale,
                            slope: slope.clone(),
                        }); // now the input will be scaled down to match
                    }

                    LookupOp::Div { .. } => {
                        if (inputs[1].out_dims.clone() != [1])
                            || !matches!(inputs[1].opkind, OpKind::Const { .. })
                        {
                            return Err(Box::new(GraphError::NonConstantDiv));
                        }

                        let input_node = &inputs[0];
                        ezkl_node.inputs.pop();

                        let denom = match &inputs[1].opkind {
                            OpKind::Const {
                                raw_const_value, ..
                            } => raw_const_value.as_ref().unwrap().map(|x| x.0)[0],
                            _ => {
                                return Err(Box::new(GraphError::MissingParams(
                                    "slopes".to_string(),
                                )));
                            }
                        };

                        let scale_diff = input_node.out_scale - scale;
                        // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                        if scale_diff > 0 {
                            let mult = scale_to_multiplier(scale_diff);
                            opkind = OpKind::Lookup(LookupOp::Div {
                                denom: crate::circuit::utils::F32(denom * mult),
                            }); // now the input will be scaled down to match
                        } else {
                            opkind = OpKind::Lookup(LookupOp::Div {
                                denom: crate::circuit::utils::F32(denom),
                            }); // now the input will be scaled down to match
                        }
                    }
                }
            }
            OpKind::Poly(ref s) => {
                match s {
                    PolyOp::Conv { .. } => {
                        let input_node = &inputs[0];

                        let weight_node = &inputs[1];

                        if inputs.len() == 3 {
                            let bias_node = &inputs[2];
                            let scale_diff =
                                weight_node.out_scale + input_node.out_scale - bias_node.out_scale;
                            let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                            bias_node = Self::scale_up_const_node(bias_node, scale + scale_diff)?;
                            if (input_node.out_scale + weight_node.out_scale) != bias_node.out_scale
                            {
                                return Err(Box::new(GraphError::RescalingError(
                                    opkind.to_string(),
                                )));
                            }
                        }
                    }
                    PolyOp::GlobalSumPool => {
                        opkind = OpKind::Poly(PolyOp::SumPool {
                            padding: (0, 0),
                            stride: (1, 1),
                            kernel_shape: (inputs[0].out_dims[1], inputs[0].out_dims[2]),
                        });
                    }
                    PolyOp::Affine | PolyOp::ScaleAndShift => {
                        let (input_node, weight_node, bias_node) =
                            (&inputs[0], &inputs[1], &inputs[2]);

                        let scale_diff =
                            weight_node.out_scale + input_node.out_scale - bias_node.out_scale;
                        let mut bias_node = other_nodes.get_mut(&node.inputs[2].node).unwrap();
                        bias_node = Self::scale_up_const_node(bias_node, scale + scale_diff)?;
                        if (input_node.out_scale + weight_node.out_scale) != bias_node.out_scale {
                            return Err(Box::new(GraphError::RescalingError(opkind.to_string())));
                        }
                    }
                    // BatchNorm take four parameters, does some f32 arithmetic and then quantizes
                    // while ScaleAndShift takes the final two parameters immediately.
                    // We will also reach back and quantize
                    PolyOp::BatchNorm => {
                        //Compute scale and shift from the four inputs,
                        // then replace the first two, and change this node to a ScaleAndShift
                        let gamma = match &inputs[1].opkind {
                            OpKind::Const {
                                raw_const_value, ..
                            } => raw_const_value.as_ref().unwrap().map(|x| x.0),
                            _ => {
                                return Err(Box::new(GraphError::MissingParams(
                                    "bn_gamma".to_string(),
                                )));
                            }
                        };

                        let beta = match &inputs[2].opkind {
                            OpKind::Const {
                                raw_const_value, ..
                            } => raw_const_value.as_ref().unwrap().map(|x| x.0),
                            _ => {
                                return Err(Box::new(GraphError::MissingParams(
                                    "bn_beta".to_string(),
                                )));
                            }
                        };

                        let mu = match &inputs[3].opkind {
                            OpKind::Const {
                                raw_const_value, ..
                            } => raw_const_value.as_ref().unwrap().map(|x| x.0),
                            _ => {
                                return Err(Box::new(GraphError::MissingParams(
                                    "bn_mu".to_string(),
                                )));
                            }
                        };

                        let sigma = match &inputs[4].opkind {
                            OpKind::Const {
                                raw_const_value, ..
                            } => raw_const_value.as_ref().unwrap().map(|x| x.0),
                            _ => {
                                return Err(Box::new(GraphError::MissingParams(
                                    "bn_sigma".to_string(),
                                )));
                            }
                        };

                        let a = (gamma.clone() / sigma.clone())?;
                        let amu: Tensor<f32> = (a.clone() * mu.clone())?;
                        let amupb: Tensor<f32> = (amu + beta.clone())?;
                        let b = (amupb * Tensor::new(Some(&[-1f32]), &[1])?)?;

                        let in_scale = inputs[0].out_scale;
                        let out_scale = 2 * inputs[0].out_scale;
                        // gamma node becomes the scale (weigh) in scale and shift
                        inputs[1].opkind = OpKind::Const {
                            const_value: Tensor::new(None, &[1])?,
                            raw_const_value: Some(a.map(|x| utils::F32(x))),
                        };
                        inputs[1].quantize_const_to_scale(in_scale)?;

                        // beta node becomes the shift (bias)
                        inputs[2].opkind = OpKind::Const {
                            const_value: Tensor::new(None, &[1])?,
                            raw_const_value: Some(b.map(|x| utils::F32(x))),
                        };
                        inputs[2].quantize_const_to_scale(out_scale)?;

                        opkind = OpKind::Poly(PolyOp::ScaleAndShift);
                    }

                    PolyOp::Add => {
                        opkind = Self::homogenize_input_scales(opkind, inputs.clone())?;
                    }
                    PolyOp::Sum => {
                        if inputs.len() != 1 {
                            return Err(Box::new(GraphError::InvalidDims(idx, opkind.to_string())));
                        };
                    }
                    PolyOp::Sub => {
                        opkind = Self::homogenize_input_scales(opkind, inputs.clone())?;
                    }
                    PolyOp::Pow(_) => {
                        ezkl_node.inputs.pop();
                        match &inputs[1].opkind {
                            OpKind::Const {
                                raw_const_value, ..
                            } => {
                                let pow = &raw_const_value.as_ref().unwrap()[0].0;
                                if inputs[1].out_dims != [1] {
                                    {
                                        return Err(Box::new(GraphError::NonConstantPower));
                                    }
                                }
                                opkind = OpKind::Poly(PolyOp::Pow(*pow as u32));
                            }
                            _ => {
                                return Err(Box::new(GraphError::MissingParams("pow".to_string())))
                            }
                        }
                    }
                    PolyOp::Rescaled { .. } => {
                        return Err(Box::new(GraphError::RescalingError(opkind.to_string())));
                    }
                    PolyOp::Flatten(_) => {
                        let new_dims: Vec<usize> =
                            vec![inputs[0].out_dims.iter().product::<usize>()];
                        opkind = OpKind::Poly(PolyOp::Flatten(new_dims.clone()));
                        ezkl_node.out_dims = new_dims;
                    }
                    PolyOp::Reshape(_) => {
                        let input_node = &inputs[0];
                        let shape_const_node = &inputs[1];
                        let shape_const = match &shape_const_node.opkind {
                            OpKind::Const { const_value, .. } => const_value,
                            _ => {
                                return Err(Box::new(GraphError::MissingParams(
                                    "shape constant".to_string(),
                                )));
                            }
                        };

                        let mut shapes = &shape_const[0..];

                        // we remove batch dims as we assume single elem batches
                        if shapes[0] == -1 && shapes.len() > 1 {
                            shapes = &shapes[1..];
                        }

                        let new_dims: Result<Vec<usize>, Box<dyn Error>> =
                            if shapes.iter().all(|x| x > &0) {
                                let mut res = vec![];
                                for x in shapes.iter() {
                                    if x <= &0 {
                                        return Err(Box::new(GraphError::InvalidDims(
                                            idx,
                                            opkind.to_string(),
                                        )));
                                    }
                                    res.push(*x as usize);
                                }
                                Ok(res)
                            } else {
                                let num_entries: usize = input_node.out_dims.iter().product();
                                let explicit_prod: i128 =
                                    shapes.iter().filter(|x| *x > &0).product();
                                if explicit_prod <= 0 {
                                    return Err(Box::new(GraphError::InvalidDims(
                                        idx,
                                        opkind.to_string(),
                                    )));
                                }
                                let inferred = num_entries / (explicit_prod as usize);
                                let mut new_dims: Vec<usize> = Vec::new();
                                for i in shapes {
                                    match i {
                                        -1 => new_dims.push(inferred),
                                        0 => continue,
                                        x => new_dims.push(*x as usize),
                                    }
                                }
                                Ok(new_dims)
                            };

                        let new_dims = new_dims?;
                        ezkl_node.inputs.pop();

                        opkind = OpKind::Poly(PolyOp::Reshape(new_dims.clone()));
                        ezkl_node.out_dims = new_dims;
                    }
                    _ => {}
                }
            }
            _ => {}
        };
        ezkl_node.opkind = opkind;
        Ok(ezkl_node)
    }

    /// Ensures all inputs to a node have the same fixed point denominator.
    fn homogenize_input_scales(
        opkind: OpKind,
        inputs: Vec<Node>,
    ) -> Result<OpKind, Box<dyn Error>> {
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
            // only rescale if need to
            if multipliers.iter().sum::<usize>() > multipliers.len() {
                Ok(OpKind::Poly(PolyOp::Rescaled {
                    inner: Box::new(c.clone()),
                    scale: (0..inputs.len()).zip(multipliers).collect_vec(),
                }))
            } else {
                Ok(opkind)
            }
        } else {
            Err(Box::new(GraphError::RescalingError(opkind.to_string())))
        }
    }

    fn quantize_const_to_scale(&mut self, scale: u32) -> Result<(), Box<dyn Error>> {
        match &self.opkind {
            OpKind::Const {
                raw_const_value, ..
            } => {
                let raw = raw_const_value.as_ref().unwrap();
                self.out_scale = scale;
                let t = vector_to_quantized(&raw.map(|e| e.0), raw.dims(), 0f32, self.out_scale)
                    .unwrap();
                self.opkind = OpKind::Const {
                    const_value: t,
                    raw_const_value: raw_const_value.clone(),
                };
                Ok(())
            }
            _ => {
                return Err(Box::new(GraphError::WrongMethod(
                    self.idx,
                    self.opkind.clone().to_string(),
                )))
            }
        }
    }

    /// Re-quantizes a constant value node to a new scale.
    fn scale_up_const_node(node: &mut Node, scale: u32) -> Result<&mut Node, Box<dyn Error>> {
        if !node.opkind.is_const() {
            return Err(Box::new(GraphError::WrongMethod(
                node.idx,
                node.opkind.clone().to_string(),
            )));
        };
        if scale > 0 {
            match &node.opkind {
                OpKind::Const {
                    const_value,
                    raw_const_value,
                } => {
                    let t = vector_to_quantized(
                        &raw_const_value.as_ref().unwrap().map(|f| f.0),
                        const_value.dims(),
                        0f32,
                        scale,
                    )?;
                    info!(
                        "------ scaled const node {:?}: {:?} -> {:?}",
                        node.idx, node.out_scale, scale
                    );
                    node.out_scale = scale;
                    node.opkind = OpKind::Const {
                        const_value: t,
                        raw_const_value: raw_const_value.clone(),
                    };
                }
                _ => {
                    return Err(Box::new(GraphError::WrongMethod(
                        node.idx,
                        node.opkind.clone().to_string(),
                    )))
                }
            }
        }
        Ok(node)
    }

    /// Formats 3d inputs if they have under or overspecified dims (casting 2D -> 3D and nD -> 3D)
    fn format_3d_inputs(mut node: &mut Node) -> Result<(), Box<dyn Error>> {
        if node.opkind.is_const() {
            return Err(Box::new(GraphError::WrongMethod(
                node.idx,
                node.opkind.clone().to_string(),
            )));
        };
        // input_nodes come in all shapes and sizes we gotta homogenize, especially for 2D (single channel images)
        if node.out_dims.len() == 2 {
            node = Self::pad_channel_input_node(node)?;
        } else if node.out_dims.len() > 3 {
            node = Self::rm_redundant_3d_channels(node)?;
        };

        if node.out_dims.len() != 3 {
            return Err(Box::new(GraphError::InvalidDims(
                node.idx,
                node.clone().opkind.to_string(),
            )));
        }
        Ok(())
    }

    /// Adds an extra channel dim to nodes that need it.
    fn pad_channel_input_node(node: &mut Node) -> Result<&mut Node, Box<dyn Error>> {
        if node.opkind.is_const() {
            return Err(Box::new(GraphError::WrongMethod(
                node.idx,
                node.opkind.clone().to_string(),
            )));
        };
        let mut dims = vec![1];
        dims.append(&mut node.out_dims);
        node.out_dims = dims;
        Ok(node)
    }

    /// Removes excess channels for an image
    fn rm_redundant_3d_channels(node: &mut Node) -> Result<&mut Node, Box<dyn Error>> {
        if node.opkind.is_const() {
            return Err(Box::new(GraphError::WrongMethod(
                node.idx,
                node.opkind.to_string(),
            )));
        };
        let dims = &node.out_dims;
        let last_dims = &dims[dims.len() - 3..];
        let channel_dims = &dims[..dims.len() - 3];
        for dim in channel_dims {
            if *dim != 1 {
                return Err(Box::new(GraphError::InvalidDims(
                    node.idx,
                    node.opkind.to_string(),
                )));
            }
        }
        node.out_dims = last_dims.to_vec();
        Ok(node)
    }
}
