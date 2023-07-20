use std::sync::Arc;

use super::{GraphError, Visibility};
use crate::circuit::hybrid::HybridOp;
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor, ValType};
use halo2curves::bn256::Fr as Fp;
use halo2curves::ff::PrimeField;
use log::{debug, warn};
use tract_onnx::prelude::{DatumType, Node as OnnxNode, TypedFact, TypedOp};
use tract_onnx::tract_core::ops::array::Gather;
use tract_onnx::tract_core::ops::array::Slice;
use tract_onnx::tract_core::ops::cnn::DeconvUnary;
use tract_onnx::tract_core::ops::einsum::EinSum;
use tract_onnx::tract_core::ops::Downsample;

use tract_onnx::tract_core::ops::element_wise::ElementWiseOp;

use tract_onnx::tract_core::ops::nn::{LeakyRelu, Reduce, Softmax};
use tract_onnx::tract_hir::internal::DimLike;
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
pub fn quantize_float(elem: &f64, shift: f64, scale: u32) -> Result<i128, TensorError> {
    let mult = scale_to_multiplier(scale);
    let max_value = ((i128::MAX as f64 - shift) / mult).round(); // the maximum value that can be represented w/o sig bit truncation

    if *elem > max_value {
        return Err(TensorError::SigBitTruncationError);
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
pub fn new_op_from_onnx(
    idx: usize,
    scale: u32,
    param_visibility: Visibility,
    node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
    inputs: &mut Vec<super::NodeType>,
) -> Result<Box<dyn crate::circuit::Op<Fp>>, Box<dyn std::error::Error>> {
    debug!("Loading node: {:?}", node);
    Ok(match node.op().name().as_ref() {
        "Gather" => {
            if inputs.len() != 2 {
                return Err(Box::new(GraphError::InvalidDims(idx, "gather".to_string())));
            };
            let op = load_gather_op(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;

            let boxed_op = inputs[1].clone().opkind();

            let index: Tensor<usize> = match extract_const_raw_values(boxed_op) {
                Some(c) => c.map(|e| e as usize),
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
            let dt = op.0.datum_type();
            // Raw values are always f32
            let raw_value = extract_tensor_value(op.0)?;
            // If bool then don't scale
            let constant_scale = if dt == DatumType::Bool { 0 } else { scale };
            // Quantize the raw value
            let quantized_value =
                tensor_to_valtensor(raw_value.clone(), constant_scale, param_visibility)?;
            // Create a constant op
            Box::new(crate::circuit::ops::Constant::new(
                quantized_value,
                raw_value,
            ))
        }
        "Reduce<Min>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "min".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            Box::new(HybridOp::Min { axes })
        }
        "Reduce<Max>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "max".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            Box::new(HybridOp::Max { axes })
        }
        "Reduce<Sum>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            Box::new(PolyOp::Sum { axes })
        }
        "Max" => {
            // Extract the slope layer hyperparams

            let boxed_op = inputs[1].clone().opkind();
            let unit = if let Some(c) = extract_const_raw_values(boxed_op) {
                if c.len() == 1 {
                    c[0]
                } else {
                    todo!()
                }
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Max".to_string())));
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
                let boxed_op = inp.opkind();
                if let Some(c) = extract_const_raw_values(boxed_op) {
                    inputs.remove(idx);
                    params = Some(tensor_to_valtensor(c, max_scale, param_visibility)?);
                }
            }

            Box::new(PolyOp::Add { a: params })
        }
        "Sub" => Box::new(PolyOp::Sub),
        "Mul" => Box::new(PolyOp::Mult { a: None }),
        "Iff" => Box::new(PolyOp::Iff),
        "Greater" => {
            // Extract the slope layer hyperparams
            let boxed_op = inputs[0].clone().opkind();
            let unit = if let Some(c) = extract_const_raw_values(boxed_op) {
                if c.len() == 1 {
                    c[0]
                } else {
                    todo!()
                }
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "greater".to_string())));
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

            if ((conv_node.pool_spec.data_format != DataFormat::NCHW)
                && (conv_node.pool_spec.data_format != DataFormat::CHW))
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
            let kernel = tensor_to_valtensor(kernel, scale, param_visibility)?;

            let bias = match conv_node.bias.clone() {
                Some(b) => {
                    let const_value = extract_tensor_value(b)?;

                    let val = tensor_to_valtensor(
                        const_value,
                        scale + inputs[0].out_scales()[0],
                        param_visibility,
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
            let kernel = tensor_to_valtensor(kernel, scale, param_visibility)?;

            let bias = match deconv_node.bias.clone() {
                Some(b) => {
                    let const_value = extract_tensor_value(b)?;

                    let val = tensor_to_valtensor(
                        const_value,
                        scale + inputs[0].out_scales()[0],
                        param_visibility,
                    )?;
                    Some(val)
                }
                None => None,
            };

            let output_padding = (deconv_node.adjustments[0], deconv_node.adjustments[1]);

            Box::new(PolyOp::DeConv {
                kernel,
                bias,
                padding: (padding_h, padding_w),
                output_padding,
                stride: (stride_h, stride_w),
            })
        }
        "Downsample" => {
            let downsample_node: Downsample = match node.op().downcast_ref::<Downsample>() {
                Some(b) => b.clone(),
                None => {
                    return Err(Box::new(GraphError::OpMismatch(
                        idx,
                        "downsample".to_string(),
                    )));
                }
            };

            Box::new(PolyOp::Downsample {
                axis: downsample_node.axis,
                stride: downsample_node.stride as usize,
                modulo: downsample_node.modulo,
            })
        }

        "Resize" => {
            // this is a bit hacky, but we need to extract the resize node somehow
            // and this is the only way I can think of
            // see https://github.com/sonos/tract/issues/324

            let resize_node = format!("{:?}", node);

            if !resize_node.contains("interpolator: Nearest")
                && !resize_node.contains("nearest: Floor")
            {
                unimplemented!("Only nearest neighbor interpolation is supported")
            }
            let boxed_op = inputs[2].clone().opkind();
            let scale_factor = if let Some(c) = extract_const_raw_values(boxed_op) {
                c.map(|x| x as usize).into_iter().collect::<Vec<usize>>()
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Resize".to_string())));
            };

            // remove the resize node from the inputs
            inputs.pop();
            // remove the scale factor node from the inputs
            inputs.pop();

            Box::new(PolyOp::Resize { scale_factor })
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
        "RmAxis" | "Reshape" => {
            // Extract the slope layer hyperparams
            let shapes = node_output_shapes(&node)?;
            let output_shape = shapes[0].as_ref().unwrap().clone();

            Box::new(PolyOp::Reshape(output_shape))
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

/// Extracts the raw values from a [Constant] op.
pub fn extract_const_raw_values(boxed_op: Box<dyn crate::circuit::Op<Fp>>) -> Option<Tensor<f32>> {
    boxed_op
        .as_any()
        .downcast_ref::<crate::circuit::ops::Constant<Fp>>()
        .map(|c| c.raw_values.clone())
}

/// Extracts the quantized values from a [Constant] op.
pub fn extract_const_quantized_values(
    boxed_op: Box<dyn crate::circuit::Op<Fp>>,
) -> Option<ValTensor<Fp>> {
    boxed_op
        .as_any()
        .downcast_ref::<crate::circuit::ops::Constant<Fp>>()
        .map(|c| c.quantized_values.clone())
}

/// Converts a tensor to a [ValTensor] with a given scale.
pub fn tensor_to_valtensor<F: PrimeField + TensorType + PartialOrd>(
    const_value: Tensor<f32>,
    scale: u32,
    visibility: Visibility,
) -> Result<ValTensor<F>, Box<dyn std::error::Error>> {
    let mut value: ValTensor<F> = match visibility {
        Visibility::Public => const_value
            .map(|x| {
                crate::tensor::ValType::Constant(crate::fieldutils::i128_to_felt::<F>(
                    quantize_float(&x.into(), 0.0, scale).unwrap(),
                ))
            })
            .into(),
        Visibility::Private | Visibility::Hashed | Visibility::Encrypted => const_value
            .map(|x| {
                crate::tensor::ValType::Value(halo2_proofs::circuit::Value::known(
                    crate::fieldutils::i128_to_felt::<F>(
                        quantize_float(&x.into(), 0.0, scale).unwrap(),
                    ),
                ))
            })
            .into(),
    };
    value.set_scale(scale);
    Ok(value)
}

/// Flatten a vector of [ValTensor]s into a single [ValTensor].
pub(crate) fn flatten_valtensors(
    tensors: Vec<ValTensor<Fp>>,
) -> Result<Vec<ValTensor<Fp>>, Box<dyn std::error::Error>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    let mut merged: Vec<ValType<Fp>> = tensors[0]
        .get_inner_tensor()?
        .into_iter()
        .collect::<Vec<_>>();

    for tensor in tensors.iter().skip(1) {
        let vals = tensor.get_inner_tensor()?.into_iter();
        merged.extend(vals);
    }

    let tensor = Tensor::new(Some(&merged), &[merged.len()])?;
    Ok(vec![tensor.into()])
}

/// Split a [ValTensor] into a vector of [ValTensor]s.
pub(crate) fn split_valtensor(
    values: ValTensor<Fp>,
    shapes: Vec<Vec<usize>>,
) -> Result<Vec<ValTensor<Fp>>, Box<dyn std::error::Error>> {
    let mut tensors: Vec<ValTensor<Fp>> = Vec::new();
    let mut start = 0;
    for shape in shapes {
        let end = start + shape.iter().product::<usize>();
        let mut tensor = values.get_slice(&[start..end])?;
        tensor.reshape(&shape)?;
        tensors.push(tensor);
        start = end;
    }
    Ok(tensors)
}

#[cfg(test)]
pub mod tests {

    use super::*;

    #[test]
    fn test_flatten_valtensors() {
        let tensor1: Tensor<Fp> = (0..10).map(|x| x.into()).into();
        let tensor2: Tensor<Fp> = (10..20).map(|x| x.into()).into();
        let tensor3: Tensor<Fp> = (20..30).map(|x| x.into()).into();

        let flattened =
            flatten_valtensors(vec![tensor1.into(), tensor2.into(), tensor3.into()]).unwrap();

        assert_eq!(flattened[0].len(), 30);

        let split =
            split_valtensor(flattened[0].clone(), vec![vec![2, 5], vec![10], vec![5, 2]]).unwrap();

        assert_eq!(split.len(), 3);
        assert_eq!(split[0].len(), 10);
        assert_eq!(split[0].dims(), vec![2, 5]);
        assert_eq!(split[1].len(), 10);
        assert_eq!(split[1].dims(), vec![10]);
        assert_eq!(split[2].dims(), vec![5, 2]);
        assert_eq!(split[2].len(), 10);
    }
}
