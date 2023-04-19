use super::{node::*, GraphError};
use crate::circuit::hybrid::HybridOp;
use crate::circuit::lookup::LookupOp;
use crate::circuit::poly::PolyOp;
use crate::circuit::utils;
use crate::tensor::{Tensor, TensorError, TensorType};
use anyhow::Result;
use halo2curves::FieldExt;
use log::warn;
use tract_onnx::prelude::{DatumType, InferenceFact, Node as OnnxNode};
use tract_onnx::tract_hir::{
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
    let scaled: Vec<i128> = vec
        .iter()
        .map(|e| (mult * e + shift).round() as i128)
        .collect();
    Tensor::new(Some(&scaled), dims)
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn scale_to_multiplier(scale: u32) -> f32 {
    i128::pow(2, scale) as f32
}

/// Gets the shape of a onnx node's outlets.
pub fn node_output_shapes(
    node: &OnnxNode<InferenceFact, Box<dyn InferenceOp>>,
) -> Result<Vec<Option<Vec<usize>>>> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
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

/// Matches an onnx node to a [OpKind] and returns a [Node] with the corresponding [OpKind].  
pub fn new_op_from_onnx<F: FieldExt + TensorType>(
    idx: usize,
    scale: u32,
    node: OnnxNode<InferenceFact, Box<dyn InferenceOp>>,
    inputs: &mut Vec<Node<F>>,
) -> Result<Box<dyn crate::circuit::Op<F>>, Box<dyn std::error::Error>> {
    Ok(match node.op().name().as_ref() {
        "Reduce<Min>" => Box::new(HybridOp::Min),
        "Reduce<Max>" => Box::new(HybridOp::Max),
        "Clip" => Box::new(LookupOp::ReLU { scale: 1 }),
        "Prelu" => {
            let slopes = match inputs[1].opkind.raw_const_value() {
                Some(raw_const_value) => raw_const_value,
                _ => {
                    return Err(Box::new(GraphError::MissingParams("slopes".to_string())));
                }
            };

            Box::new(HybridOp::PReLU {
                scale: 1,
                slopes: slopes.to_vec(),
            })
        }
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
            Box::new(LookupOp::LeakyReLU {
                scale: 1,
                slope: crate::circuit::utils::F32(leaky_op.0),
            })
        }
        "Sigmoid" => Box::new(LookupOp::Sigmoid { scales: (1, 1) }),
        "Sqrt" => Box::new(LookupOp::Sqrt { scales: (1, 1) }),
        "Tanh" => Box::new(LookupOp::Tanh { scales: (1, 1) }),
        "onnx.Erf" => Box::new(LookupOp::Erf { scales: (1, 1) }),
        "Div" => {
            if (inputs[1].out_dims.clone() != [1]) || !inputs[1].opkind.is_const() {
                return Err(Box::new(GraphError::NonConstantDiv));
            }

            let denom = match &inputs[1].opkind.raw_const_value() {
                Some(raw_const_value) => raw_const_value.map(|x| x.0)[0],
                _ => {
                    return Err(Box::new(GraphError::MissingParams("slopes".to_string())));
                }
            };

            inputs.pop();

            Box::new(LookupOp::Div {
                denom: crate::circuit::utils::F32(denom),
            })
        }

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
            Box::new(crate::circuit::ops::Const {
                const_value,
                raw_const_value,
            })
        }
        "Source" => Box::new(crate::circuit::ops::Input),
        "Add" => Box::new(PolyOp::Add),
        "Sub" => Box::new(PolyOp::Sub),
        "Mul" => Box::new(PolyOp::Mult),
        "Gemm" => Box::new(PolyOp::Affine),
        "MatMulInference" => Box::new(PolyOp::Matmul),
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
        "Reduce<Sum>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };

            Box::new(PolyOp::Sum)
        }
        "Reduce<Mean>" => Box::new(HybridOp::Mean {
            scale: 1,
            num_inputs: inputs[0].out_dims.iter().product::<usize>(),
        }),
        "Pow" => match &inputs[1].opkind.raw_const_value() {
            Some(raw_const_value) => {
                let pow = &raw_const_value[0].0;
                if inputs[1].out_dims != [1] {
                    {
                        return Err(Box::new(GraphError::NonConstantPower));
                    }
                }
                Box::new(PolyOp::Pow(*pow as u32))
            }
            _ => return Err(Box::new(GraphError::MissingParams("pow".to_string()))),
        },
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
            Box::new(PolyOp::Conv {
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
        "InstanceNorm" => Box::new(HybridOp::InstanceNorm2d {
            epsilon: utils::F32(1e-5),
        }),
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
        "Reshape" => {
            let input_node = &inputs[0];
            let shape_const_node = &inputs[1];
            let shape_const = match shape_const_node.opkind.const_value() {
                Some(const_value) => const_value,
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

            let new_dims: Result<Vec<usize>, Box<dyn std::error::Error>> =
                if shapes.iter().all(|x| x > &0) {
                    let mut res = vec![];
                    for x in shapes.iter() {
                        if x <= &0 {
                            return Err(Box::new(GraphError::InvalidDims(
                                idx,
                                "reshape".to_string(),
                            )));
                        }
                        res.push(*x as usize);
                    }
                    Ok(res)
                } else {
                    let num_entries: usize = input_node.out_dims.iter().product();
                    let explicit_prod: i128 = shapes.iter().filter(|x| *x > &0).product();
                    if explicit_prod <= 0 {
                        return Err(Box::new(GraphError::InvalidDims(
                            idx,
                            "reshape".to_string(),
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
            inputs.pop();

            Box::new(PolyOp::Reshape(new_dims))
        }
        "Flatten" => {
            let new_dims: Vec<usize> = vec![inputs[0].out_dims.iter().product::<usize>()];
            Box::new(PolyOp::Flatten(new_dims))
        }
        // BatchNorm take four parameters, does some f32 arithmetic and then quantizes
        // while ScaleAndShift takes the final two parameters immediately.
        // We will also reach back and quantize
        "BatchNorm" => {
            //Compute scale and shift from the four inputs,
            // then replace the first two, and change this node to a ScaleAndShift
            let gamma = match &inputs[1].opkind.raw_const_value() {
                Some(raw_const_value, ..) => raw_const_value.map(|x| x.0),
                _ => {
                    return Err(Box::new(GraphError::MissingParams("bn_gamma".to_string())));
                }
            };

            let beta = match &inputs[2].opkind.raw_const_value() {
                Some(raw_const_value, ..) => raw_const_value.map(|x| x.0),
                _ => {
                    return Err(Box::new(GraphError::MissingParams("bn_beta".to_string())));
                }
            };

            let mu = match &inputs[3].opkind.raw_const_value() {
                Some(raw_const_value, ..) => raw_const_value.map(|x| x.0),
                _ => {
                    return Err(Box::new(GraphError::MissingParams("bn_mu".to_string())));
                }
            };

            let sigma = match &inputs[4].opkind.raw_const_value() {
                Some(raw_const_value, ..) => raw_const_value.map(|x| x.0),
                _ => {
                    return Err(Box::new(GraphError::MissingParams("bn_sigma".to_string())));
                }
            };

            let a = (gamma.clone() / sigma.clone())?;
            let amu: Tensor<f32> = (a.clone() * mu.clone())?;
            let amupb: Tensor<f32> = (amu + beta.clone())?;
            let b = (amupb * Tensor::new(Some(&[-1f32]), &[1])?)?;

            let in_scale = inputs[0].out_scale;
            let out_scale = 2 * inputs[0].out_scale;
            // gamma node becomes the scale (weigh) in scale and shift
            inputs[1].opkind = Box::new(crate::circuit::ops::Const {
                const_value: Tensor::new(None, &[1])?,
                raw_const_value: Some(a.map(|x| crate::circuit::utils::F32(x))),
            });
            inputs[1].quantize_const_to_scale(in_scale)?;

            // beta node becomes the shift (bias)
            inputs[2].opkind = Box::new(crate::circuit::ops::Const {
                const_value: Tensor::new(None, &[1])?,
                raw_const_value: Some(b.map(|x| utils::F32(x))),
            });
            inputs[2].quantize_const_to_scale(out_scale)?;

            inputs.pop();
            inputs.pop();

            Box::new(PolyOp::ScaleAndShift)
        }
        c => {
            warn!("{:?} is not currently supported", c);
            Box::new(crate::circuit::ops::Unknown)
        }
    })
}
