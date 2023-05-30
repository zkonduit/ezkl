use std::sync::Arc;

use super::GraphError;
use crate::circuit::hybrid::HybridOp;
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::fieldutils::i128_to_felt;
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor};
use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;
use log::{debug, warn};
use tract_onnx::prelude::{DatumType, Node as OnnxNode, TypedFact, TypedOp};
use tract_onnx::tract_core::ops::array::Gather;
use tract_onnx::tract_core::ops::array::Slice;
use tract_onnx::tract_core::ops::cnn::DeconvUnary;
use tract_onnx::tract_core::ops::einsum::EinSum;
// use tract_onnx::tract_core::ops::binary::UnaryOp;
// use tract_onnx::tract_core::ops::matmul::MatMulUnary;
use tract_onnx::tract_core::ops::element_wise::ElementWiseOp;
use tract_onnx::tract_core::ops::nn::{LeakyRelu, Reduce, Softmax};
use tract_onnx::tract_hir::internal::{AxisOp, DimLike};
use tract_onnx::tract_hir::ops::cnn::ConvUnary;
use tract_onnx::tract_hir::ops::konst::Const;
use tract_onnx::tract_hir::{
    ops::array::{Pad, PadMode},
    ops::cnn::{MaxPool, PoolSpec, SumPool},
    ops::nn::DataFormat,
    tract_core::ops::cnn::{conv::KernelFormat, PaddingSpec},
};

// Warning: currently ignores stride information
/// Quantizes an iterable of f32s to a [Tensor] of i32s using a fixed point representation.
/// Arguments
///
/// * `vec` - the vector to quantize.
/// * `dims` - the dimensionality of the resulting [Tensor].
/// * `shift` - offset used in the fixed point representation.
/// * `scale` - `2^scale` used in the fixed point representation.
pub fn quantize_float(elem: &f32, shift: f32, scale: u32) -> Result<i128, TensorError> {
    let mult = scale_to_multiplier(scale) as f32;
    let max_value = ((i128::MAX as f32 - shift) / mult).round(); // the maximum value that can be represented w/o sig bit truncation

    if *elem > max_value {
        return Err(TensorError::SigBitTruncatioError);
    }

    // we parallelize the quantization process as it seems to be quite slow at times
    let scaled = (mult * *elem + shift).round() as i128;

    Ok(scaled)
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn scale_to_multiplier(scale: u32) -> f64 {
    f64::powf(2., scale as f64)
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn mult_to_scale(mult: f64) -> u32 {
    mult.log2().round() as u32
}

/// Gets the shape of a onnx node's outlets.
pub fn node_output_shapes(
    node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
) -> Result<Vec<Option<Vec<usize>>>, Box<dyn std::error::Error>> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
    for output in outputs {
        let mv = output.fact.shape.clone().as_concrete().map(|x| x.to_vec());
        shapes.push(mv)
    }
    Ok(shapes)
}

fn extract_tensor_value(
    input: Arc<tract_onnx::prelude::Tensor>,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let dt = input.datum_type();
    let mut dims = input.shape().to_vec();
    if dims.is_empty() {
        dims.push(1)
    } else if dims.iter().product::<usize>() == 1 {
        dims = vec![1];
    };

    let mut const_value: Tensor<f32>;
    match dt {
        DatumType::F32 => {
            let vec = input.as_slice::<f32>()?.to_vec();
            const_value = vec.into_iter().into();
        }

        DatumType::I64 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i64>()?.to_vec();
            let cast: Vec<f32> = vec.iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::Bool => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<bool>()?.to_vec();
            let cast: Vec<f32> = vec.iter().map(|x| *x as usize as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::TDim => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<tract_onnx::prelude::TDim>()?.to_vec();
            let cast: Vec<f32> = vec
                .iter()
                .map(|x| x.to_i64().map_or_else(|_| 1, |e| e) as f32)
                .collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        _ => todo!(),
    }
    const_value.reshape(&dims);

    Ok(const_value)
}

/// Converts a tensor to a [ValTensor] with a given scale.
pub fn tensor_to_valtensor<F: PrimeField + TensorType + PartialOrd>(
    const_value: Tensor<f32>,
    scale: u32,
    public_params: bool,
) -> Result<ValTensor<F>, Box<dyn std::error::Error>> {
    let mut value: ValTensor<F> = if public_params {
        const_value
            .map(|x| {
                crate::tensor::ValType::Constant(i128_to_felt::<F>(
                    quantize_float(&x, 0.0, scale).unwrap(),
                ))
            })
            .into()
    } else {
        const_value
            .map(|x| {
                crate::tensor::ValType::Value(Value::known(i128_to_felt::<F>(
                    quantize_float(&x, 0.0, scale).unwrap(),
                )))
            })
            .into()
    };
    value.set_scale(scale);
    Ok(value)
}

/// Extracts an axis op from an onnx node.
fn load_axis_op(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<AxisOp, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &AxisOp = match op.downcast_ref::<AxisOp>() {
        Some(b) => b,
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
        }
    };

    Ok(op.clone())
}

/// Extracts a Gather op from an onnx node.
fn load_gather_op(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<Gather, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &Gather = match op.downcast_ref::<Gather>() {
        Some(b) => b,
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
        }
    };

    Ok(op.clone())
}

/// Extracts an axis op from an onnx node.
fn load_const(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<Const, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &Const = match op.downcast_ref::<Const>() {
        Some(b) => b,
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
        }
    };

    Ok(op.clone())
}

/// Extracts an axis op from an onnx node.
fn load_reduce_op(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<Reduce, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &Reduce = match op.downcast_ref::<Reduce>() {
        Some(b) => b,
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
        }
    };
    Ok(op.clone())
}

/// Extracts an axis op from an onnx node.
fn load_eltwise_op(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<ElementWiseOp, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams

    let op: &ElementWiseOp = match op.downcast_ref::<ElementWiseOp>() {
        Some(b) => b,
        None => return Err(Box::new(GraphError::OpMismatch(idx, name))),
    };

    Ok(op.clone())
}

fn load_concat_op(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<tract_onnx::tract_core::ops::array::TypedConcat, Box<dyn std::error::Error>> {
    let op: &tract_onnx::tract_core::ops::array::TypedConcat =
        match op.downcast_ref::<tract_onnx::tract_core::ops::array::TypedConcat>() {
            Some(b) => b,
            None => return Err(Box::new(GraphError::OpMismatch(idx, name))),
        };

    Ok(op.clone())
}

/// Extracts a Slice op from an onnx node.
fn load_slice_op(
    op: &dyn tract_onnx::prelude::Op,
    name: String,
) -> Result<Slice, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &Slice = match op.downcast_ref::<Slice>() {
        Some(b) => b,
        None => {
            return Err(Box::new(TensorError::DimMismatch(name)));
        }
    };

    Ok(op.clone())
}

/// Matches an onnx node to a [OpKind] and returns a [Node] with the corresponding [OpKind].  
/// Arguments
/// * `idx` - the index of the node in the graph.
/// * `scale` - the global (circuit) scale.
/// * `public_params` - whether the node's parameters are public.
/// * `node` - the [OnnxNode] to be converted into a [Node].
/// * `inputs` - the node's inputs.
pub fn new_op_from_onnx<F: PrimeField + TensorType + PartialOrd>(
    idx: usize,
    scale: u32,
    public_params: bool,
    node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
    inputs: &mut Vec<super::NodeType<F>>,
) -> Result<Box<dyn crate::circuit::Op<F>>, Box<dyn std::error::Error>> {
    debug!("Loading node: {:?}", node);
    Ok(match node.op().name().as_ref() {
        "Gather" => {
            if inputs.len() != 2 {
                return Err(Box::new(GraphError::InvalidDims(idx, "gather".to_string())));
            };
            let op = load_gather_op(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;

            let boxed_op = inputs[1].clone().opkind();
            let index: Tensor<usize> = match boxed_op
                .as_any()
                .downcast_ref::<crate::circuit::ops::Constant<F>>()
            {
                Some(c) => c.values.map(|e| e as usize),
                None => {
                    warn!("assuming the gather window is over a context variable");
                    // offset by 1
                    let index: Tensor<usize> =
                        (0..node_output_shapes(&node)?[0].as_ref().unwrap().to_vec()[axis + 1])
                            .into();
                    index
                }
            };

            inputs.pop();

            Box::new(crate::circuit::ops::poly::PolyOp::Gather { dim: axis, index })
        }
        "Concat" | "InferenceConcat" => {
            let op = load_concat_op(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;
            Box::new(crate::circuit::ops::poly::PolyOp::Concat { axis })
        }
        "Slice" => {
            let slice = load_slice_op(node.op(), node.op().name().to_string())?;

            let axis = slice.axis;
            let start = slice.start.to_usize()?;
            let end = slice.end.to_usize()?;

            Box::new(PolyOp::Slice { axis, start, end })
        }
        "Const" => {
            let op: Const = load_const(node.op(), idx, node.op().name().to_string())?;
            let dt = op.clone().0.datum_type();
            let value = extract_tensor_value(op.0)?;
            let constant_scale = if dt == DatumType::Bool { 0 } else { scale };
            Box::new(crate::circuit::ops::Constant::new(
                value,
                constant_scale,
                public_params,
            ))
        }
        "Reduce<Min>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.iter().filter(|x| **x != 0).copied().collect();

            Box::new(HybridOp::Min { axes })
        }
        "Reduce<Max>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.iter().filter(|x| **x != 0).copied().collect();

            Box::new(HybridOp::Max { axes })
        }
        "Reduce<Sum>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.iter().filter(|x| **x != 0).copied().collect();

            Box::new(PolyOp::Sum { axes })
        }
        // TODO: this is a hack to get around the fact that onnx replace ReLU with Max(0, x) -- we should probably implement
        "Max" => {
            // Extract the slope layer hyperparams

            let boxed_op = inputs[1].clone().opkind();
            let unit = match boxed_op
                .as_any()
                .downcast_ref::<crate::circuit::ops::Constant<F>>()
            {
                Some(c) => c.values[0],
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "Max".to_string())));
                }
            };

            if inputs.len() == 2 && unit == 0. {
                inputs.pop();
                Box::new(LookupOp::ReLU {
                    scale: inputs[0].out_scales()[0] as usize,
                })
            } else {
                todo!()
            }
        }
        "Recip" => {
            // Extract the slope layer hyperparams
            Box::new(LookupOp::Recip { scale: 1 })
        }

        "LeakyRelu" => {
            // Extract the slope layer hyperparams
            let leaky_op = load_eltwise_op(node.op(), idx, node.op().name().to_string())?;

            let leaky_op: &LeakyRelu = match leaky_op.0.downcast_ref::<LeakyRelu>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(
                        idx,
                        "leaky relu".to_string(),
                    )));
                }
            };

            Box::new(LookupOp::LeakyReLU {
                scale: 1,
                slope: crate::circuit::utils::F32(leaky_op.alpha),
            })
        }
        "Scan" => {
            panic!("should never reach here")
        }
        "Sigmoid" => Box::new(LookupOp::Sigmoid { scales: (1, 1) }),
        "Sqrt" => Box::new(LookupOp::Sqrt { scales: (1, 1) }),
        "Rsqrt" => Box::new(LookupOp::Rsqrt { scales: (1, 1) }),
        "Tanh" => Box::new(LookupOp::Tanh { scales: (1, 1) }),
        "Erf" => Box::new(LookupOp::Erf { scales: (1, 1) }),
        "Source" => Box::new(crate::circuit::ops::Input { scale }),
        "Add" => {
            let mut params = None;

            let max_scale = inputs
                .iter()
                .map(|x| x.out_scales()[0])
                .max()
                .ok_or_else(|| Box::new(GraphError::MissingParams("add".to_string())))?;

            for (idx, inp) in inputs.clone().iter().enumerate() {
                let boxed_op = &inp.opkind();
                if let Some(c) = boxed_op
                    .as_any()
                    .downcast_ref::<crate::circuit::ops::Constant<F>>()
                {
                    inputs.remove(idx);
                    params = Some(tensor_to_valtensor::<F>(
                        c.values.clone(),
                        max_scale,
                        public_params,
                    )?);
                }
            }

            Box::new(PolyOp::Add { a: params })
        }
        "Sub" => Box::new(PolyOp::Sub),
        "Mul" => {
            let mut params = None;

            for (idx, inp) in inputs.clone().iter().enumerate() {
                let boxed_op = &inp.opkind();
                if let Some(c) = boxed_op
                    .as_any()
                    .downcast_ref::<crate::circuit::ops::Constant<F>>()
                {
                    inputs.remove(idx);
                    params = Some(tensor_to_valtensor::<F>(
                        c.values.clone(),
                        scale,
                        public_params,
                    )?);
                }
            }

            Box::new(PolyOp::Mult { a: params })
        }
        "Iff" => Box::new(PolyOp::Iff),
        "Greater" => {
            // Extract the slope layer hyperparams
            let boxed_op = inputs[0].clone().opkind();
            let unit = match boxed_op
                .as_any()
                .downcast_ref::<crate::circuit::ops::Constant<F>>()
            {
                Some(c) => {
                    if c.values.len() == 1 {
                        c.values[0]
                    } else {
                        todo!()
                    }
                }
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "greater".to_string())));
                }
            };

            if inputs.len() == 2 {
                *inputs = vec![inputs[1].clone()];
                Box::new(LookupOp::GreaterThan {
                    a: crate::circuit::utils::F32(unit),
                })
            } else {
                todo!()
            }
        }
        "EinSum" => {
            // Extract the slope layer hyperparams
            let op: &EinSum = match node.op().downcast_ref::<EinSum>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "einsum".to_string())));
                }
            };

            let axes = &op.axes;
            Box::new(PolyOp::Einsum {
                equation: axes.to_string(),
            })
        }
        "Softmax" => {
            // Extract the slope layer hyperparams
            let softmax_op: &Softmax = match node.op().downcast_ref::<Softmax>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "softmax".to_string())));
                }
            };

            // if its not the last dim then we don't support it
            if softmax_op.axes.to_vec() != vec![inputs[0].out_dims()[0].len() - 1] {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "softmax".to_string(),
                )));
            }

            Box::new(HybridOp::Softmax { scales: (1, 1) })
        }
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

            Box::new(HybridOp::MaxPool2d {
                padding: (padding_h, padding_w),
                stride: (stride_h, stride_w),
                pool_dims: (kernel_height, kernel_width),
            })
        }
        "Square" => Box::new(PolyOp::Pow(2)),
        "ConvUnary" => {
            let conv_node: &ConvUnary = match node.op().downcast_ref::<ConvUnary>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "conv".to_string())));
                }
            };

            if let Some(dilations) = &conv_node.pool_spec.dilations {
                if dilations.iter().any(|x| *x != 1) {
                    return Err(Box::new(GraphError::MisformedParams(
                        "non unit dilations not supported".to_string(),
                    )));
                }
            }

            if (conv_node.pool_spec.data_format != DataFormat::NCHW)
                || (conv_node.kernel_fmt != KernelFormat::OIHW)
            {
                return Err(Box::new(GraphError::MisformedParams(
                    "data or kernel in wrong format".to_string(),
                )));
            }

            let stride = match conv_node.pool_spec.strides.clone() {
                Some(s) => s,
                None => {
                    return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                }
            };
            let padding = match &conv_node.pool_spec.padding {
                PaddingSpec::Explicit(p, _, _) => p,
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            let (padding_h, padding_w, stride_h, stride_w) =
                (padding[0], padding[1], stride[0], stride[1]);

            let kernel = extract_tensor_value(conv_node.kernel.clone())?;
            let kernel = tensor_to_valtensor(kernel, scale, public_params)?;

            let bias = match conv_node.bias.clone() {
                Some(b) => {
                    let const_value = extract_tensor_value(b)?;

                    let val = tensor_to_valtensor(
                        const_value,
                        scale + inputs[0].out_scales()[0],
                        public_params,
                    )?;
                    Some(val)
                }
                None => None,
            };

            Box::new(PolyOp::Conv {
                kernel,
                bias,
                padding: (padding_h, padding_w),
                stride: (stride_h, stride_w),
            })
        }
        "DeconvUnary" => {
            let deconv_node: &DeconvUnary = match node.op().downcast_ref::<DeconvUnary>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "deconv".to_string())));
                }
            };

            if let Some(dilations) = &deconv_node.pool_spec.dilations {
                if dilations.iter().any(|x| *x != 1) {
                    return Err(Box::new(GraphError::MisformedParams(
                        "non unit dilations not supported".to_string(),
                    )));
                }
            }

            if (deconv_node.pool_spec.data_format != DataFormat::NCHW)
                || (deconv_node.kernel_format != KernelFormat::OIHW)
            {
                return Err(Box::new(GraphError::MisformedParams(
                    "data or kernel in wrong format".to_string(),
                )));
            }

            let stride = match deconv_node.pool_spec.strides.clone() {
                Some(s) => s,
                None => {
                    return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                }
            };
            let padding = match &deconv_node.pool_spec.padding {
                PaddingSpec::Explicit(p, _, _) => p,
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            let (padding_h, padding_w, stride_h, stride_w) =
                (padding[0], padding[1], stride[0], stride[1]);

            let kernel = extract_tensor_value(deconv_node.kernel.clone())?;
            let kernel = tensor_to_valtensor(kernel, scale, public_params)?;

            let bias = match deconv_node.bias.clone() {
                Some(b) => {
                    let const_value = extract_tensor_value(b)?;

                    let val = tensor_to_valtensor(
                        const_value,
                        scale + inputs[0].out_scales()[0],
                        public_params,
                    )?;
                    Some(val)
                }
                None => None,
            };

            Box::new(PolyOp::DeConv {
                kernel,
                bias,
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

            Box::new(PolyOp::SumPool {
                padding: (padding_h, padding_w),
                stride: (stride_h, stride_w),
                kernel_shape: (kernel_height, kernel_width),
            })
        }
        "GlobalAvgPool" => Box::new(PolyOp::SumPool {
            padding: (0, 0),
            stride: (1, 1),
            kernel_shape: (inputs[0].out_dims()[0][1], inputs[0].out_dims()[0][2]),
        }),
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
            Box::new(PolyOp::Pad(padding_h, padding_w))
        }
        "RmAxis" => {
            // Extract the slope layer hyperparams
            let reshape = load_axis_op(node.op(), idx, node.op().name().to_string())?;

            let new_dims: Vec<usize> = match reshape {
                AxisOp::Rm(_) => inputs[0].out_dims()[0].clone(),
                _ => {
                    return Err(Box::new(GraphError::MisformedParams("reshape".to_string())));
                }
            };

            Box::new(PolyOp::Reshape(new_dims.to_vec()))
        }
        "Reshape" => {
            // Extract the slope layer hyperparams
            let reshape = load_axis_op(node.op(), idx, node.op().name().to_string())?;

            let new_dims: Vec<usize> = match reshape {
                AxisOp::Reshape(_, _shape_from, _shape_to) => {
                    let shapes = node_output_shapes(&node)?;
                    shapes[0].as_ref().unwrap().clone()
                }
                _ => {
                    return Err(Box::new(GraphError::MisformedParams("reshape".to_string())));
                }
            };
            Box::new(PolyOp::Reshape(new_dims.to_vec()))
        }
        "Flatten" => {
            let new_dims: Vec<usize> = vec![inputs[0].out_dims()[0].iter().product::<usize>()];
            Box::new(PolyOp::Flatten(new_dims))
        }
        c => {
            warn!("Unknown op: {}", c);
            Box::new(crate::circuit::ops::Unknown)
        }
    })
}
