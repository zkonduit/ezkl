#[cfg(not(target_arch = "wasm32"))]
use super::GraphError;
use super::{Rescaled, SupportedOp, Visibility};
#[cfg(not(target_arch = "wasm32"))]
use crate::circuit::hybrid::HybridOp;
#[cfg(not(target_arch = "wasm32"))]
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::circuit::Op;
use crate::tensor::{Tensor, TensorError, TensorType};
use halo2curves::bn256::Fr as Fp;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
#[cfg(not(target_arch = "wasm32"))]
use log::{debug, warn};
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::{DatumType, Node as OnnxNode, TypedFact, TypedOp};
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_core::ops::{
    array::{Gather, Slice},
    change_axes::AxisOp,
    cnn::DeconvUnary,
    einsum::EinSum,
    element_wise::ElementWiseOp,
    nn::{LeakyRelu, Reduce, Softmax},
    Downsample,
};
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_hir::{
    internal::DimLike,
    ops::array::{Pad, PadMode},
    ops::cnn::{ConvUnary, MaxPool, PoolSpec, SumPool},
    ops::konst::Const,
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
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
        DatumType::F64 => {
            let vec = input.as_slice::<f64>()?.to_vec();
            let cast: Vec<f32> = vec.iter().map(|x| *x as f32).collect();
            const_value = cast.into_iter().into();
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
#[cfg(not(target_arch = "wasm32"))]
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

///
#[cfg(not(target_arch = "wasm32"))]
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

/// Extracts a const node from an onnx node.
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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

/// Matches an onnx node to a [crate::circuit::Op].
/// Arguments
/// * `idx` - the index of the node in the graph.
/// * `scale` - the global (circuit) scale.
/// * `param_visibility` - [Visibility] of the node.
/// * `node` - the [OnnxNode] to be matched.
/// * `inputs` - the node's inputs.
#[cfg(not(target_arch = "wasm32"))]
pub fn new_op_from_onnx(
    idx: usize,
    scale: u32,
    param_visibility: Visibility,
    node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
    inputs: &mut [super::NodeType],
) -> Result<(SupportedOp, Vec<usize>), Box<dyn std::error::Error>> {
    debug!("Loading node: {:?}", node);
    let mut deleted_indices = vec![];
    let node = match node.op().name().as_ref() {
        "Gather" => {
            if inputs.len() != 2 {
                return Err(Box::new(GraphError::InvalidDims(idx, "gather".to_string())));
            };
            let op = load_gather_op(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;

            let index: Tensor<usize> = match extract_const_raw_values(inputs[1].opkind()) {
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

            if let Some(node) = inputs.last_mut() {
                node.decrement_const();
                deleted_indices.push(inputs.len() - 1);
            }

            SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::Gather { dim: axis, index })
        }
        "MoveAxis" => {
            let op = load_axis_op(node.op(), idx, node.op().name().to_string())?;
            match op {
                AxisOp::Move(from, to) => {
                    let source = from.to_usize()?;
                    let destination = to.to_usize()?;
                    SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::MoveAxis {
                        source,
                        destination,
                    })
                }
                _ => todo!(),
            }
        }
        "Concat" | "InferenceConcat" => {
            let op = load_concat_op(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;
            SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::Concat { axis })
        }
        "Slice" => {
            let slice = load_slice_op(node.op(), node.op().name().to_string())?;

            let axis = slice.axis;
            let start = slice.start.to_usize()?;
            let end = slice.end.to_usize()?;

            SupportedOp::Linear(PolyOp::Slice { axis, start, end })
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
                quantize_tensor(raw_value.clone(), constant_scale, param_visibility)?;

            let mut c = crate::circuit::ops::Constant::new(quantized_value, raw_value);
            c.num_uses += node.outputs.len();
            // Create a constant op
            SupportedOp::Constant(c)
        }
        "Reduce<Min>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "min".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Hybrid(HybridOp::ReduceMin { axes })
        }
        "Reduce<Max>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "max".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Hybrid(HybridOp::ReduceMax { axes })
        }
        "Reduce<Sum>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_reduce_op(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Linear(PolyOp::Sum { axes })
        }
        "Max" => {
            // Extract the max value
            // first find the input that is a constant
            // and then extract the value
            let const_inputs = inputs
                .iter()
                .enumerate()
                .filter(|(_, n)| n.is_constant())
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            if const_inputs.len() != 1 {
                return Err(Box::new(GraphError::OpMismatch(idx, "Max".to_string())));
            }

            let const_idx = const_inputs[0];
            let boxed_op = inputs[const_idx].opkind();
            let unit = if let Some(c) = extract_const_raw_values(boxed_op) {
                if c.len() == 1 {
                    c[0]
                } else {
                    todo!()
                }
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Max".to_string())));
            };

            if inputs.len() == 2 {
                if let Some(node) = inputs.get_mut(const_idx) {
                    node.decrement_const();
                    deleted_indices.push(const_idx);
                }
                if unit == 0. {
                    SupportedOp::Nonlinear(LookupOp::ReLU { scale: 1 })
                } else {
                    SupportedOp::Nonlinear(LookupOp::Max {
                        scales: (1, 1),
                        a: crate::circuit::utils::F32(unit),
                    })
                }
            } else {
                todo!()
            }
        }
        "Min" => {
            // Extract the min value
            // first find the input that is a constant
            // and then extract the value
            let const_inputs = inputs
                .iter()
                .enumerate()
                .filter(|(_, n)| n.is_constant())
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            if const_inputs.len() != 1 {
                return Err(Box::new(GraphError::OpMismatch(idx, "Min".to_string())));
            }

            let const_idx = const_inputs[0];
            let boxed_op = inputs[const_idx].opkind();
            let unit = if let Some(c) = extract_const_raw_values(boxed_op) {
                if c.len() == 1 {
                    c[0]
                } else {
                    todo!()
                }
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Min".to_string())));
            };

            if inputs.len() == 2 {
                if let Some(node) = inputs.get_mut(const_idx) {
                    node.decrement_const();
                    deleted_indices.push(const_idx);
                }
                SupportedOp::Nonlinear(LookupOp::Min {
                    scales: (1, 1),
                    a: crate::circuit::utils::F32(unit),
                })
            } else {
                todo!()
            }
        }
        "Recip" => {
            // Extract the slope layer hyperparams
            SupportedOp::Nonlinear(LookupOp::Recip { scale: 1 })
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

            SupportedOp::Nonlinear(LookupOp::LeakyReLU {
                scale: 1,
                slope: crate::circuit::utils::F32(leaky_op.alpha),
            })
        }
        "Scan" => {
            panic!("should never reach here")
        }
        "QuantizeLinearU8" | "DequantizeLinearF32" => SupportedOp::Linear(PolyOp::Identity),
        "Abs" => SupportedOp::Hybrid(HybridOp::Abs),
        "Neg" => SupportedOp::Linear(PolyOp::Neg),
        "Sigmoid" => SupportedOp::Nonlinear(LookupOp::Sigmoid { scales: (1, 1) }),
        "Sqrt" => SupportedOp::Nonlinear(LookupOp::Sqrt { scales: (1, 1) }),
        "Rsqrt" => SupportedOp::Nonlinear(LookupOp::Rsqrt { scales: (1, 1) }),
        "Exp" => SupportedOp::Nonlinear(LookupOp::Exp { scales: (1, 1) }),
        "Ln" => SupportedOp::Nonlinear(LookupOp::Ln { scales: (1, 1) }),
        "Sin" => SupportedOp::Nonlinear(LookupOp::Sin { scales: (1, 1) }),
        "Cos" => SupportedOp::Nonlinear(LookupOp::Cos { scales: (1, 1) }),
        "Tan" => SupportedOp::Nonlinear(LookupOp::Tan { scales: (1, 1) }),
        "Asin" => SupportedOp::Nonlinear(LookupOp::ASin { scales: (1, 1) }),
        "Acos" => SupportedOp::Nonlinear(LookupOp::ACos { scales: (1, 1) }),
        "Atan" => SupportedOp::Nonlinear(LookupOp::ATan { scales: (1, 1) }),
        "Sinh" => SupportedOp::Nonlinear(LookupOp::Sinh { scales: (1, 1) }),
        "Cosh" => SupportedOp::Nonlinear(LookupOp::Cosh { scales: (1, 1) }),
        "Tanh" => SupportedOp::Nonlinear(LookupOp::Tanh { scales: (1, 1) }),
        "Asinh" => SupportedOp::Nonlinear(LookupOp::ASinh { scales: (1, 1) }),
        "Acosh" => SupportedOp::Nonlinear(LookupOp::ACosh { scales: (1, 1) }),
        "Atanh" => SupportedOp::Nonlinear(LookupOp::ATanh { scales: (1, 1) }),
        "Erf" => SupportedOp::Nonlinear(LookupOp::Erf { scales: (1, 1) }),
        "Source" => SupportedOp::Input(crate::circuit::ops::Input { scale }),
        "Add" => SupportedOp::Linear(PolyOp::Add),
        "Sub" => SupportedOp::Linear(PolyOp::Sub),
        "Mul" => SupportedOp::Linear(PolyOp::Mult),
        "Iff" => SupportedOp::Linear(PolyOp::Iff),
        "Less" => {
            if inputs.len() == 2 {
                SupportedOp::Hybrid(HybridOp::Less { scales: (1, 1) })
            } else {
                todo!()
            }
        }
        "Greater" => {
            // Extract the slope layer hyperparams
            if inputs.len() == 2 {
                SupportedOp::Hybrid(HybridOp::Greater { scales: (1, 1) })
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
            SupportedOp::Linear(PolyOp::Einsum {
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

            SupportedOp::Hybrid(HybridOp::Softmax { scales: (1, 1) })
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
                PaddingSpec::Explicit(b, a, _) => [(b[0], b[1]), (a[0], a[1])],
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };
            let kernel_shape = &pool_spec.kernel_shape;

            let (stride_h, stride_w) = (stride[0], stride[1]);
            let (kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1]);

            SupportedOp::Hybrid(HybridOp::MaxPool2d {
                padding,
                stride: (stride_h, stride_w),
                pool_dims: (kernel_height, kernel_width),
            })
        }
        "Ceil" | "Floor" | "Round" | "RoundHalfToEven" => {
            warn!("using a round op in the circuit which does not make sense in Field arithmetic");
            SupportedOp::Linear(PolyOp::Identity)
        }
        "Sign" => SupportedOp::Nonlinear(LookupOp::Sign),
        "Cube" => SupportedOp::Linear(PolyOp::Pow(3)),
        "Square" => SupportedOp::Linear(PolyOp::Pow(2)),
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
                Some(s) => (s[0], s[1]),
                None => {
                    return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                }
            };

            let padding = match &conv_node.pool_spec.padding {
                PaddingSpec::Explicit(b, a, _) => [(b[0], b[1]), (a[0], a[1])],
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            let kernel = extract_tensor_value(conv_node.kernel.clone())?;
            let kernel = quantize_tensor(kernel, scale, param_visibility)?;

            let bias = match conv_node.bias.clone() {
                Some(b) => {
                    let const_value = extract_tensor_value(b)?;

                    let val = quantize_tensor(
                        const_value,
                        scale + inputs[0].out_scales()[0],
                        param_visibility,
                    )?;
                    Some(val)
                }
                None => None,
            };

            SupportedOp::Linear(PolyOp::Conv {
                kernel,
                bias,
                padding,
                stride,
            })
        }
        "Not" => {
            change_all_input_scales(inputs, 0);
            SupportedOp::Linear(PolyOp::Not)
        }
        "And" => {
            change_all_input_scales(inputs, 0);
            SupportedOp::Linear(PolyOp::And)
        }
        "Or" => {
            change_all_input_scales(inputs, 0);
            SupportedOp::Linear(PolyOp::Or)
        }
        "Xor" => {
            change_all_input_scales(inputs, 0);
            SupportedOp::Linear(PolyOp::Xor)
        }
        "Equals" => SupportedOp::Hybrid(HybridOp::Equals),
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
                Some(s) => (s[0], s[1]),
                None => {
                    return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                }
            };
            let padding = match &deconv_node.pool_spec.padding {
                PaddingSpec::Explicit(b, a, _) => [(b[0], b[1]), (a[0], a[1])],
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            let kernel = extract_tensor_value(deconv_node.kernel.clone())?;
            let kernel = quantize_tensor(kernel, scale, param_visibility)?;

            let bias = match deconv_node.bias.clone() {
                Some(b) => {
                    let const_value = extract_tensor_value(b)?;

                    let val = quantize_tensor(
                        const_value,
                        scale + inputs[0].out_scales()[0],
                        param_visibility,
                    )?;
                    Some(val)
                }
                None => None,
            };

            let output_padding: (usize, usize) =
                (deconv_node.adjustments[0], deconv_node.adjustments[1]);

            SupportedOp::Linear(PolyOp::DeConv {
                kernel,
                bias,
                padding,
                output_padding,
                stride,
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

            SupportedOp::Linear(PolyOp::Downsample {
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
            let boxed_op = inputs[2].opkind();
            let scale_factor = if let Some(c) = extract_const_raw_values(boxed_op) {
                c.map(|x| x as usize).into_iter().collect::<Vec<usize>>()
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Resize".to_string())));
            };

            // remove the resize node from the inputs
            if let Some(node) = inputs.last_mut() {
                node.decrement_const();
                deleted_indices.push(inputs.len() - 1);
            }
            // remove the scale factor node from the inputs
            if let Some(node) = inputs.get_mut(inputs.len() - 2) {
                node.decrement_const();
                deleted_indices.push(inputs.len() - 2);
            }

            SupportedOp::Linear(PolyOp::Resize { scale_factor })
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
                PaddingSpec::Explicit(b, a, _) => [(b[0], b[1]), (a[0], a[1])],
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };
            let kernel_shape = &pool_spec.kernel_shape;

            let (stride_h, stride_w) = (stride[0], stride[1]);
            let (kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[1]);

            SupportedOp::Linear(PolyOp::SumPool {
                padding,
                stride: (stride_h, stride_w),
                kernel_shape: (kernel_height, kernel_width),
            })
        }
        "GlobalAvgPool" => SupportedOp::Linear(PolyOp::SumPool {
            padding: [(0, 0); 2],
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
            }

            let padding = [
                (
                    pad_node.pads[padding_len - 2].0,
                    pad_node.pads[padding_len - 1].0,
                ),
                (
                    pad_node.pads[padding_len - 2].1,
                    pad_node.pads[padding_len - 1].1,
                ),
            ];
            SupportedOp::Linear(PolyOp::Pad(padding))
        }
        "RmAxis" | "Reshape" | "AddAxis" => {
            // Extract the slope layer hyperparams
            let shapes = node_output_shapes(&node)?;
            let output_shape = shapes[0].as_ref().unwrap().clone();

            SupportedOp::Linear(PolyOp::Reshape(output_shape))
        }
        "Flatten" => {
            let new_dims: Vec<usize> = vec![inputs[0].out_dims()[0].iter().product::<usize>()];
            SupportedOp::Linear(PolyOp::Flatten(new_dims))
        }
        c => {
            warn!("Unknown op: {}", c);
            SupportedOp::Unknown(crate::circuit::ops::Unknown)
        }
    };

    Ok((node, deleted_indices))
}

/// Extracts the raw values from a [crate::circuit::ops::Constant] op.
pub fn extract_const_raw_values(op: SupportedOp) -> Option<Tensor<f32>> {
    match op {
        SupportedOp::Constant(crate::circuit::ops::Constant { raw_values, .. }) => Some(raw_values),
        _ => None,
    }
}

/// Extracts the quantized values from a [crate::circuit::ops::Constant] op.
pub fn extract_const_quantized_values(op: SupportedOp) -> Option<Tensor<Fp>> {
    match op {
        SupportedOp::Constant(crate::circuit::ops::Constant {
            quantized_values, ..
        }) => Some(quantized_values),
        _ => None,
    }
}

/// Extract the quantized values from a conv op
pub fn extract_conv_values(boxed_op: Box<dyn crate::circuit::Op<Fp>>) -> [Option<Tensor<Fp>>; 2] {
    let op = boxed_op
        .as_any()
        .downcast_ref::<crate::circuit::ops::poly::PolyOp<Fp>>();

    if let Some(op) = op {
        match op {
            PolyOp::Conv { kernel, bias, .. } => return [Some(kernel.clone()), bias.clone()],
            _ => {}
        }
    }
    [None, None]
}

/// Converts a tensor to a [ValTensor] with a given scale.
pub fn quantize_tensor<F: PrimeField + TensorType + PartialOrd>(
    const_value: Tensor<f32>,
    scale: u32,
    visibility: Visibility,
) -> Result<Tensor<F>, Box<dyn std::error::Error>> {
    let mut value: Tensor<F> = const_value.map(|x| {
        crate::fieldutils::i128_to_felt::<F>(quantize_float(&x.into(), 0.0, scale).unwrap())
    });
    value.set_scale(scale);
    value.set_visibility(visibility);
    Ok(value)
}

use crate::tensor::ValTensor;
/// Split a [ValTensor] into a vector of [ValTensor]s.
pub(crate) fn split_valtensor(
    values: &ValTensor<Fp>,
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

fn change_all_input_scales(inputs: &mut [super::NodeType], new_scale: u32) {
    for input in inputs {
        if input.is_input() {
            input.bump_scale(new_scale);
        }
    }
}

///
pub fn homogenize_input_scales(
    op: Box<dyn Op<Fp>>,
    input_scales: Vec<u32>,
    inputs_to_scale: Vec<usize>,
) -> Result<Box<dyn Op<Fp>>, Box<dyn Error>> {
    if inputs_to_scale.is_empty() {
        return Ok(op);
    }
    // else if all inputs_scales at inputs_to_scale are the same, we don't need to do anything
    else if input_scales
        .iter()
        .enumerate()
        .filter(|(idx, _)| inputs_to_scale.contains(idx))
        .all(|(_, scale)| scale == &input_scales[0])
    {
        return Ok(op);
    }

    let mut dividers: Vec<u128> = vec![1; input_scales.len()];
    if !input_scales.windows(2).all(|w| w[0] == w[1]) {
        let min_scale = input_scales.iter().min().unwrap();
        let _ = input_scales
            .iter()
            .enumerate()
            .map(|(idx, input_scale)| {
                if !inputs_to_scale.contains(&idx) {
                    return;
                }
                let scale_diff = input_scale - min_scale;
                if scale_diff > 0 {
                    let mult = crate::graph::scale_to_multiplier(scale_diff);
                    dividers[idx] = mult as u128;
                }
            })
            .collect_vec();
    }

    // only rescale if need to
    if dividers.iter().any(|&x| x > 1) {
        Ok(Box::new(Rescaled {
            inner: Box::new(op.into()),
            scale: (0..input_scales.len()).zip(dividers).collect_vec(),
        }))
    } else {
        Ok(op)
    }
}

#[cfg(test)]
pub mod tests {

    use super::*;

    #[test]
    fn test_flatten_valtensors() {
        let tensor1: Tensor<Fp> = (0..10).map(|x| x.into()).into();
        let tensor2: Tensor<Fp> = (10..20).map(|x| x.into()).into();
        let tensor3: Tensor<Fp> = (20..30).map(|x| x.into()).into();

        let mut tensor = Tensor::new(Some(&[tensor1, tensor2, tensor3]), &[3])
            .unwrap()
            .combine()
            .unwrap();

        tensor.set_visibility(Visibility::Public);

        let flattened: ValTensor<Fp> = tensor.into();

        assert_eq!(flattened.len(), 30);

        let split = split_valtensor(&flattened, vec![vec![2, 5], vec![10], vec![5, 2]]).unwrap();

        assert_eq!(split.len(), 3);
        assert_eq!(split[0].len(), 10);
        assert_eq!(split[0].dims(), vec![2, 5]);
        assert_eq!(split[1].len(), 10);
        assert_eq!(split[1].dims(), vec![10]);
        assert_eq!(split[2].dims(), vec![5, 2]);
        assert_eq!(split[2].len(), 10);
    }
}
