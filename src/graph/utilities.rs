use std::sync::Arc;

use super::{node::*, GraphError};
use crate::circuit::hybrid::HybridOp;
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::fieldutils::i128_to_felt;
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor};
use anyhow::Result;
use halo2_proofs::circuit::Value;
use halo2curves::FieldExt;
use log::{trace, warn};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use tract_onnx::prelude::{DatumType, Node as OnnxNode, TypedFact, TypedOp};
use tract_onnx::tract_core::ops::binary::UnaryOp;
use tract_onnx::tract_core::ops::matmul::MatMulUnary;
use tract_onnx::tract_core::ops::nn::{LeakyRelu, Reduce};
use tract_onnx::tract_hir::internal::AxisOp;
use tract_onnx::tract_hir::ops::cnn::ConvUnary;
use tract_onnx::tract_hir::ops::element_wise::ElementWiseOp;
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
pub fn vector_to_quantized(
    vec: &[f32],
    dims: &[usize],
    shift: f32,
    scale: u32,
) -> Result<Tensor<i128>, TensorError> {
    let mult = scale_to_multiplier(scale);
    // we parallelize the quantization process as it seems to be quite slow at times
    let scaled: Vec<i128> = vec
        .par_iter()
        .map(|e| (mult * *e + shift).round() as i128)
        .collect();
    Tensor::new(Some(&scaled), dims)
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn scale_to_multiplier(scale: u32) -> f32 {
    i128::pow(2, scale) as f32
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn mult_to_scale(mult: f32) -> u32 {
    mult.log2().round() as u32
}

/// Gets the shape of a onnx node's outlets.
pub fn node_output_shapes(
    node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
) -> Result<Vec<Option<Vec<usize>>>> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
    for output in outputs {
        let mv = output.fact.shape.clone().as_concrete().map(|x| x.to_vec());
        shapes.push(mv)
    }
    Ok(shapes)
}

fn extract_tensor_value<F: FieldExt + TensorType>(
    input: Arc<tract_onnx::prelude::Tensor>,
    scale: u32,
    public_params: bool,
) -> Result<ValTensor<F>, Box<dyn std::error::Error>> {
    let dt = input.datum_type();
    let mut dims = input.shape().to_vec();
    if dims.is_empty() {
        dims.push(1)
    } else if dims.iter().product::<usize>() == 1 {
        dims = vec![1];
    };

    let const_value: Tensor<i128>;
    match dt {
        DatumType::F32 => {
            let vec = input.as_slice::<f32>()?.to_vec();
            const_value = vector_to_quantized(&vec, &dims, 0f32, scale)?;
        }

        DatumType::I64 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i64>()?.to_vec();
            let cast: Vec<i128> = vec.iter().map(|x| *x as i128).collect();
            const_value = Tensor::<i128>::new(Some(&cast), &dims)?;
        }
        _ => todo!(),
    }

    let mut value: ValTensor<F> = if public_params {
        const_value
            .map(|x| crate::tensor::ValType::Constant(i128_to_felt::<F>(x)))
            .into()
    } else {
        const_value
            .map(|x| crate::tensor::ValType::Value(Value::known(i128_to_felt::<F>(x))))
            .into()
    };
    value.set_scale(scale);
    Ok(value)
}

/// Extracts a unary op from an onnx node.
fn load_unary_op(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<UnaryOp, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &UnaryOp = match op.downcast_ref::<UnaryOp>() {
        Some(b) => b,
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
        }
    };
    Ok(op.clone())
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
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
        }
    };
    Ok(op.clone())
}

/// Matches an onnx node to a [OpKind] and returns a [Node] with the corresponding [OpKind].  
pub fn new_op_from_onnx<F: FieldExt + TensorType>(
    idx: usize,
    scale: u32,
    public_params: bool,
    node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
    inputs: &mut Vec<Node<F>>,
) -> Result<Box<dyn crate::circuit::Op<F>>, Box<dyn std::error::Error>> {
    trace!("Loading node: {:?}", node);
    Ok(match node.op().name().as_ref() {
        "Reduce<Min>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            // subtract 1 from the axes to account for the batch dimension
            let axes = op
                .axes
                .clone()
                .iter()
                .filter(|x| **x != 0)
                .map(|x| x - 1)
                .collect();

            Box::new(HybridOp::Min { axes })
        }
        "Reduce<Max>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            // subtract 1 from the axes to account for the batch dimension
            let axes = op
                .axes
                .clone()
                .iter()
                .filter(|x| **x != 0)
                .map(|x| x - 1)
                .collect();

            Box::new(HybridOp::Max { axes })
        }
        "Reduce<Sum>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            // subtract 1 from the axes to account for the batch dimension
            let axes = op
                .axes
                .clone()
                .iter()
                .filter(|x| **x != 0)
                .map(|x| x - 1)
                .collect();

            Box::new(PolyOp::Sum { axes })
        }
        // TODO: this is a hack to get around the fact that onnx replace ReLU with Max(0, x) -- we should probably implement
        "MaxUnary" => {
            // Extract the slope layer hyperparams
            let max_op = load_unary_op(node.op(), idx, node.op().name().to_string())?;

            if max_op.a.shape().into_iter().product::<usize>() == 1
                && max_op.a.as_slice::<f32>()?.to_vec()[0] == 0.0
            {
                Box::new(LookupOp::ReLU {
                    scale: inputs[0].out_scale as usize,
                })
            } else {
                todo!()
            }
        }
        "Prelu" => {
            unreachable!("Prelu should be converted to a more complex format in Onnx");
        }
        "Recip" => {
            // Extract the slope layer hyperparams
            let _recip = load_eltwise_op(node.op(), idx, node.op().name().to_string())?;
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
        "Sigmoid" => Box::new(LookupOp::Sigmoid { scales: (1, 1) }),
        "Sqrt" => Box::new(LookupOp::Sqrt { scales: (1, 1) }),
        "Rsqrt" => Box::new(LookupOp::Rsqrt { scales: (1, 1) }),
        "Tanh" => Box::new(LookupOp::Tanh { scales: (1, 1) }),
        "onnx.Erf" => Box::new(LookupOp::Erf { scales: (1, 1) }),
        "Source" => Box::new(crate::circuit::ops::Input),
        "Add" => Box::new(PolyOp::Add { a: None }),
        "Sub" => Box::new(PolyOp::Sub),
        "Mul" => Box::new(PolyOp::Mult { a: None }),
        "Greater" => {
            todo!()
        }
        "GreaterUnary" => {
            // Extract the slope layer hyperparams
            let greater = load_unary_op(node.op(), idx, node.op().name().to_string())?;
            if greater.a.len() == 1 {
                Box::new(LookupOp::GreaterThan {
                    a: crate::circuit::utils::F32(greater.a.as_slice::<f32>()?[0]),
                })
            } else {
                todo!()
            }
        }
        "MatMulUnary" => {
            // Extract the slope layer hyperparams
            let mm_op: &MatMulUnary = match node.op().downcast_ref::<MatMulUnary>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "mm".to_string())));
                }
            };

            let matrix = extract_tensor_value(mm_op.a.clone(), scale, public_params)?;

            Box::new(PolyOp::Matmul { a: Some(matrix) })
        }
        "MatMul" => Box::new(PolyOp::Matmul { a: None }),
        "AddUnary" => {
            // Extract the slope layer hyperparams
            let add_op = load_unary_op(node.op(), idx, node.op().name().to_string())?;
            let matrix =
                extract_tensor_value(add_op.a.clone(), inputs[0].out_scale, public_params)?;

            Box::new(PolyOp::Add { a: Some(matrix) })
        }
        "MulUnary" => {
            // Extract the slope layer hyperparams
            let mul_op = load_unary_op(node.op(), idx, node.op().name().to_string())?;

            // this is actually better represented as a DIV op
            if mul_op.a.shape().into_iter().product::<usize>() == 1
                && mul_op.a.as_slice::<f32>()?.to_vec()[0] < 0.1
            {
                let denom = 1.0 / mul_op.a.as_slice::<f32>()?.to_vec()[0];
                Box::new(LookupOp::Div {
                    denom: crate::circuit::utils::F32(denom),
                })
            } else {
                let matrix = extract_tensor_value(mul_op.a.clone(), scale, public_params)?;
                Box::new(PolyOp::Mult { a: Some(matrix) })
            }
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
        "Dot" => Box::new(PolyOp::Dot),

        "Square" => Box::new(PolyOp::Pow(2)),
        "ConvUnary" => {
            let conv_node: &ConvUnary = match node.op().downcast_ref::<ConvUnary>() {
                Some(b) => b,
                None => {
                    return Err(Box::new(GraphError::OpMismatch(idx, "conv".to_string())));
                }
            };

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

            let kernel = extract_tensor_value(conv_node.kernel.clone(), scale, public_params)?;

            let bias = match conv_node.bias.clone() {
                Some(b) => Some(extract_tensor_value(
                    b,
                    scale + inputs[0].out_scale,
                    public_params,
                )?),
                None => None,
            };

            Box::new(PolyOp::Conv {
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
            kernel_shape: (inputs[0].out_dims[1], inputs[0].out_dims[2]),
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
                AxisOp::Rm(_) => inputs[0].out_dims.clone(),
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
                AxisOp::Reshape(_, _shape_from, shape_to) => shape_to
                    .iter()
                    .map(|a| a.to_i64().unwrap() as usize)
                    .collect(),
                _ => {
                    return Err(Box::new(GraphError::MisformedParams("reshape".to_string())));
                }
            };

            Box::new(PolyOp::Reshape(new_dims.to_vec()))
        }
        "Flatten" => {
            let new_dims: Vec<usize> = vec![inputs[0].out_dims.iter().product::<usize>()];
            Box::new(PolyOp::Flatten(new_dims))
        }
        c => {
            warn!("{:?} is not currently supported", c);
            Box::new(crate::circuit::ops::Unknown)
        }
    })
}
