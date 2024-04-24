#[cfg(not(target_arch = "wasm32"))]
use super::GraphError;
#[cfg(not(target_arch = "wasm32"))]
use super::VarScales;
use super::{Rescaled, SupportedOp, Visibility};
#[cfg(not(target_arch = "wasm32"))]
use crate::circuit::hybrid::HybridOp;
#[cfg(not(target_arch = "wasm32"))]
use crate::circuit::lookup::LookupOp;
#[cfg(not(target_arch = "wasm32"))]
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
    array::{
        Gather, GatherElements, GatherNd, MultiBroadcastTo, OneHot, ScatterElements, ScatterNd,
        Slice, Topk,
    },
    change_axes::AxisOp,
    cnn::{Conv, Deconv},
    einsum::EinSum,
    element_wise::ElementWiseOp,
    nn::{LeakyRelu, Reduce, Softmax},
    Downsample,
};
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_hir::{
    internal::DimLike,
    ops::array::{Pad, PadMode, TypedConcat},
    ops::cnn::PoolSpec,
    ops::konst::Const,
    ops::nn::DataFormat,
    tract_core::ops::cast::Cast,
    tract_core::ops::cnn::{conv::KernelFormat, MaxPool, PaddingSpec, SumPool},
};

/// Quantizes an iterable of f32s to a [Tensor] of i32s using a fixed point representation.
/// Arguments
///
/// * `vec` - the vector to quantize.
/// * `dims` - the dimensionality of the resulting [Tensor].
/// * `shift` - offset used in the fixed point representation.
/// * `scale` - `2^scale` used in the fixed point representation.
pub fn quantize_float(elem: &f64, shift: f64, scale: crate::Scale) -> Result<i64, TensorError> {
    let mult = scale_to_multiplier(scale);
    let max_value = ((i64::MAX as f64 - shift) / mult).round(); // the maximum value that can be represented w/o sig bit truncation

    if *elem > max_value {
        return Err(TensorError::SigBitTruncationError);
    }

    // we parallelize the quantization process as it seems to be quite slow at times
    let scaled = (mult * *elem + shift).round() as i64;

    Ok(scaled)
}

/// Dequantizes a field element to a f64 using a fixed point representation.
/// Arguments
/// * `felt` - the field element to dequantize.
/// * `scale` - `2^scale` used in the fixed point representation.
/// * `shift` - offset used in the fixed point representation.
pub fn dequantize(felt: Fp, scale: crate::Scale, shift: f64) -> f64 {
    let int_rep = crate::fieldutils::felt_to_i64(felt);
    let multiplier = scale_to_multiplier(scale);
    int_rep as f64 / multiplier - shift
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn scale_to_multiplier(scale: crate::Scale) -> f64 {
    f64::powf(2., scale as f64)
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn multiplier_to_scale(mult: f64) -> crate::Scale {
    mult.log2().round() as crate::Scale
}

/// Gets the shape of a onnx node's outlets.
#[cfg(not(target_arch = "wasm32"))]
pub fn node_output_shapes(
    node: &OnnxNode<TypedFact, Box<dyn TypedOp>>,
    symbol_values: &SymbolValues,
) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
    for output in outputs {
        let shape = output.fact.shape;
        let shape = shape.eval_to_usize(symbol_values)?;
        let mv = shape.to_vec();
        shapes.push(mv)
    }
    Ok(shapes)
}
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::SymbolValues;
#[cfg(not(target_arch = "wasm32"))]
/// Extracts the raw values from a tensor.
pub fn extract_tensor_value(
    input: Arc<tract_onnx::prelude::Tensor>,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    use maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

    let dt = input.datum_type();
    let dims = input.shape().to_vec();

    let mut const_value: Tensor<f32>;
    if dims.is_empty() && input.len() == 0 {
        const_value = Tensor::<f32>::new(None, &dims)?;
        return Ok(const_value);
    }

    match dt {
        DatumType::F16 => {
            let vec = input.as_slice::<tract_onnx::prelude::f16>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| (*x).into()).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::F32 => {
            let vec = input.as_slice::<f32>()?.to_vec();
            const_value = Tensor::<f32>::new(Some(&vec), &dims)?;
        }
        DatumType::F64 => {
            let vec = input.as_slice::<f64>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I64 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i64>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I32 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i32>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I16 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i16>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::I8 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<i8>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U8 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u8>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U16 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u16>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U32 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u32>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::U64 => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<u64>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::Bool => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<bool>()?.to_vec();
            let cast: Vec<f32> = vec.par_iter().map(|x| *x as usize as f32).collect();
            const_value = Tensor::<f32>::new(Some(&cast), &dims)?;
        }
        DatumType::TDim => {
            // Generally a shape or hyperparam
            let vec = input.as_slice::<tract_onnx::prelude::TDim>()?.to_vec();

            let cast: Result<Vec<f32>, &str> = vec
                .par_iter()
                .map(|x| match x.to_i64() {
                    Ok(v) => Ok(v as f32),
                    Err(_) => match x.to_i64() {
                        Ok(v) => Ok(v as f32),
                        Err(_) => Err("could not evaluate tdim"),
                    },
                })
                .collect();

            const_value = Tensor::<f32>::new(Some(&cast?), &dims)?;
        }
        _ => return Err("unsupported data type".into()),
    }
    const_value.reshape(&dims)?;

    Ok(const_value)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_op<C: tract_onnx::prelude::Op + Clone>(
    op: &dyn tract_onnx::prelude::Op,
    idx: usize,
    name: String,
) -> Result<C, Box<dyn std::error::Error>> {
    // Extract the slope layer hyperparams
    let op: &C = match op.downcast_ref::<C>() {
        Some(b) => b,
        None => {
            return Err(Box::new(GraphError::OpMismatch(idx, name)));
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
    scales: &VarScales,
    param_visibility: &Visibility,
    node: OnnxNode<TypedFact, Box<dyn TypedOp>>,
    inputs: &mut [super::NodeType],
    symbol_values: &SymbolValues,
    rebase_frac_zero_constants: bool,
) -> Result<(SupportedOp, Vec<usize>), Box<dyn std::error::Error>> {
    use tract_onnx::tract_core::ops::array::Trilu;

    use crate::circuit::InputType;

    let input_scales = inputs
        .iter()
        .flat_map(|x| x.out_scales())
        .collect::<Vec<_>>();

    let mut replace_const = |scale: crate::Scale,
                             index: usize,
                             default_op: SupportedOp|
     -> Result<SupportedOp, Box<dyn std::error::Error>> {
        let mut constant = inputs[index].opkind();
        let constant = constant.get_mutable_constant();
        if let Some(c) = constant {
            inputs[index].bump_scale(scale);
            c.rebase_scale(scale)?;
            inputs[index].replace_opkind(SupportedOp::Constant(c.clone()));
            Ok(SupportedOp::Linear(PolyOp::Identity {
                out_scale: Some(scale),
            }))
        } else {
            Ok(default_op)
        }
    };

    debug!("Loading node: {:?}", node);
    let mut deleted_indices = vec![];
    let node = match node.op().name().as_ref() {
        "ShiftLeft" => {
            // load shift amount
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                let raw_values = &c.raw_values;
                if raw_values.len() != 1 {
                    return Err(Box::new(GraphError::InvalidDims(
                        idx,
                        "shift left".to_string(),
                    )));
                }
                SupportedOp::Linear(PolyOp::Identity {
                    out_scale: Some(input_scales[0] - raw_values[0] as i32),
                })
            } else {
                return Err(Box::new(GraphError::OpMismatch(
                    idx,
                    "ShiftLeft".to_string(),
                )));
            }
        }
        "ShiftRight" => {
            // load shift amount
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                let raw_values = &c.raw_values;
                if raw_values.len() != 1 {
                    return Err(Box::new(GraphError::InvalidDims(
                        idx,
                        "shift right".to_string(),
                    )));
                }
                SupportedOp::Linear(PolyOp::Identity {
                    out_scale: Some(input_scales[0] + raw_values[0] as i32),
                })
            } else {
                return Err(Box::new(GraphError::OpMismatch(
                    idx,
                    "ShiftRight".to_string(),
                )));
            }
        }
        "MultiBroadcastTo" => {
            let op = load_op::<MultiBroadcastTo>(node.op(), idx, node.op().name().to_string())?;
            let shape = op.shape.clone();
            let shape = shape
                .iter()
                .map(|x| x.to_usize())
                .collect::<Result<Vec<_>, _>>()?;
            SupportedOp::Linear(PolyOp::MultiBroadcastTo { shape })
        }

        "Range" => {
            let mut input_ops = vec![];

            for (i, input) in inputs.iter_mut().enumerate() {
                if !input.opkind().is_constant() {
                    return Err("Range only supports constant inputs in a zk circuit".into());
                } else {
                    input.decrement_use();
                    deleted_indices.push(i);
                    input_ops.push(input.opkind().clone());
                }
            }

            assert_eq!(input_ops.len(), 3, "Range requires 3 inputs");
            let input_ops = input_ops
                .iter()
                .map(|x| x.get_constant().ok_or("Range requires constant inputs"))
                .collect::<Result<Vec<_>, _>>()?;

            let start = input_ops[0].raw_values.map(|x| x as usize)[0];
            let end = input_ops[1].raw_values.map(|x| x as usize)[0];
            let delta = input_ops[2].raw_values.map(|x| x as usize)[0];

            let range = (start..end).step_by(delta).collect::<Vec<_>>();
            let raw_value = range.iter().map(|x| *x as f32).collect::<Tensor<_>>();
            // Quantize the raw value (integers)
            let quantized_value = quantize_tensor(raw_value.clone(), 0, &Visibility::Fixed)?;

            let c = crate::circuit::ops::Constant::new(quantized_value, raw_value);
            // Create a constant op
            SupportedOp::Constant(c)
        }

        "Trilu" => {
            let op = load_op::<Trilu>(node.op(), idx, node.op().name().to_string())?;
            let upper = op.upper;

            // assert second input is a constant
            let diagonal = if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                let raw_values = &c.raw_values;
                if raw_values.len() != 1 {
                    return Err(Box::new(GraphError::InvalidDims(idx, "trilu".to_string())));
                }
                raw_values[0] as i32
            } else {
                return Err("we only support constant inputs for trilu diagonal".into());
            };

            SupportedOp::Linear(PolyOp::Trilu { upper, k: diagonal })
        }

        "Gather" => {
            if inputs.len() != 2 {
                return Err(Box::new(GraphError::InvalidDims(idx, "gather".to_string())));
            };
            let op = load_op::<Gather>(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;

            let mut op = SupportedOp::Hybrid(crate::circuit::ops::hybrid::HybridOp::Gather {
                dim: axis,
                constant_idx: None,
            });

            // if param_visibility.is_public() {
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(inputs.len() - 1);
                op = SupportedOp::Hybrid(crate::circuit::ops::hybrid::HybridOp::Gather {
                    dim: axis,
                    constant_idx: Some(c.raw_values.map(|x| {
                        if x == -1.0 {
                            inputs[0].out_dims()[0][axis] - 1
                        } else {
                            x as usize
                        }
                    })),
                });
            }
            // }

            if inputs[1].opkind().is_input() {
                inputs[1].replace_opkind(SupportedOp::Input(crate::circuit::ops::Input {
                    scale: 0,
                    datum_type: InputType::TDim,
                }));
                inputs[1].bump_scale(0);
            }

            op

            // Extract the max value
        }
        "Topk" => {
            let op = load_op::<Topk>(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;
            // if param_visibility.is_public() {
            let k = if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(inputs.len() - 1);
                c.raw_values.map(|x| x as usize)[0]
            } else {
                op.fallback_k.to_i64()? as usize
            };

            SupportedOp::Hybrid(crate::circuit::ops::hybrid::HybridOp::TopK {
                dim: axis,
                k,
                largest: op.largest,
            })
        }
        "Onehot" => {
            let op = load_op::<OneHot>(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;
            let num_classes = op.dim;

            SupportedOp::Hybrid(crate::circuit::ops::hybrid::HybridOp::OneHot {
                dim: axis,
                num_classes,
            })
        }
        "ScatterElements" => {
            if inputs.len() != 3 {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "scatter elements".to_string(),
                )));
            };
            let op = load_op::<ScatterElements>(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;

            let mut op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::ScatterElements {
                dim: axis,
                constant_idx: None,
            });

            // if param_visibility.is_public() {
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::ScatterElements {
                    dim: axis,
                    constant_idx: Some(c.raw_values.map(|x| x as usize)),
                })
            }
            // }

            if inputs[1].opkind().is_input() {
                inputs[1].replace_opkind(SupportedOp::Input(crate::circuit::ops::Input {
                    scale: 0,
                    datum_type: InputType::TDim,
                }));
                inputs[1].bump_scale(0);
            }

            op

            // Extract the max value
        }
        "ScatterNd" => {
            if inputs.len() != 3 {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "scatter nd".to_string(),
                )));
            };
            // just verify it deserializes correctly
            let _op = load_op::<ScatterNd>(node.op(), idx, node.op().name().to_string())?;

            let mut op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::ScatterND {
                constant_idx: None,
            });

            // if param_visibility.is_public() {
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::ScatterND {
                    constant_idx: Some(c.raw_values.map(|x| x as usize)),
                })
            }
            // }

            if inputs[1].opkind().is_input() {
                inputs[1].replace_opkind(SupportedOp::Input(crate::circuit::ops::Input {
                    scale: 0,
                    datum_type: InputType::TDim,
                }));
                inputs[1].bump_scale(0);
            }

            op
        }

        "GatherNd" => {
            if inputs.len() != 2 {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "gather nd".to_string(),
                )));
            };
            let op = load_op::<GatherNd>(node.op(), idx, node.op().name().to_string())?;
            let batch_dims = op.batch_dims;

            let mut op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::GatherND {
                batch_dims,
                indices: None,
            });

            // if param_visibility.is_public() {
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::GatherND {
                    batch_dims,
                    indices: Some(c.raw_values.map(|x| x as usize)),
                })
            }
            // }

            if inputs[1].opkind().is_input() {
                inputs[1].replace_opkind(SupportedOp::Input(crate::circuit::ops::Input {
                    scale: 0,
                    datum_type: InputType::TDim,
                }));
                inputs[1].bump_scale(0);
            }

            op
        }

        "GatherElements" => {
            if inputs.len() != 2 {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "gather elements".to_string(),
                )));
            };
            let op = load_op::<GatherElements>(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;

            let mut op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::GatherElements {
                dim: axis,
                constant_idx: None,
            });

            // if param_visibility.is_public() {
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                op = SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::GatherElements {
                    dim: axis,
                    constant_idx: Some(c.raw_values.map(|x| x as usize)),
                })
            }
            // }

            if inputs[1].opkind().is_input() {
                inputs[1].replace_opkind(SupportedOp::Input(crate::circuit::ops::Input {
                    scale: 0,
                    datum_type: InputType::TDim,
                }));
                inputs[1].bump_scale(0);
            }

            op

            // Extract the max value
        }
        "MoveAxis" => {
            let op = load_op::<AxisOp>(node.op(), idx, node.op().name().to_string())?;
            match op {
                AxisOp::Move(from, to) => {
                    let source = from.to_usize()?;
                    let destination = to.to_usize()?;
                    SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::MoveAxis {
                        source,
                        destination,
                    })
                }

                _ => {
                    return Err(Box::new(GraphError::OpMismatch(
                        idx,
                        "MoveAxis".to_string(),
                    )))
                }
            }
        }
        "Concat" | "InferenceConcat" => {
            let op = load_op::<TypedConcat>(node.op(), idx, node.op().name().to_string())?;
            let axis = op.axis;
            SupportedOp::Linear(crate::circuit::ops::poly::PolyOp::Concat { axis })
        }
        "Slice" => {
            let slice = load_op::<Slice>(node.op(), idx, node.op().name().to_string())?;

            let axis = slice.axis;
            let start = slice.start.to_usize()?;
            let end = slice.end.to_usize()?;

            SupportedOp::Linear(PolyOp::Slice { axis, start, end })
        }
        "Const" => {
            let op: Const = load_op::<Const>(node.op(), idx, node.op().name().to_string())?;
            let dt = op.0.datum_type();
            // Raw values are always f32
            let raw_value = extract_tensor_value(op.0)?;
            // If bool or a tensor dimension then don't scale
            let mut constant_scale = match dt {
                DatumType::Bool
                | DatumType::TDim
                | DatumType::I64
                | DatumType::I32
                | DatumType::I16
                | DatumType::I8
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64 => 0,
                DatumType::F16 | DatumType::F32 | DatumType::F64 => scales.params,
                _ => return Err(Box::new(GraphError::UnsupportedDataType)),
            };

            // if all raw_values are round then set scale to 0
            let all_round = raw_value.iter().all(|x| (x).fract() == 0.0);
            if all_round && rebase_frac_zero_constants {
                constant_scale = 0;
            }

            // Quantize the raw value
            let quantized_value =
                quantize_tensor(raw_value.clone(), constant_scale, param_visibility)?;
            let c = crate::circuit::ops::Constant::new(quantized_value, raw_value);
            // Create a constant op
            SupportedOp::Constant(c)
        }
        "Reduce<ArgMax(false)>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "argmax".to_string())));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes: Vec<usize> = op.axes.into_iter().collect();
            assert_eq!(axes.len(), 1, "only support argmax over one axis");

            SupportedOp::Hybrid(HybridOp::ReduceArgMax { dim: axes[0] })
        }
        "Reduce<ArgMin(false)>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "argmin".to_string())));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes: Vec<usize> = op.axes.into_iter().collect();
            assert_eq!(axes.len(), 1, "only support argmin over one axis");

            SupportedOp::Hybrid(HybridOp::ReduceArgMin { dim: axes[0] })
        }
        "Reduce<Min>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "min".to_string())));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Hybrid(HybridOp::ReduceMin { axes })
        }
        "Reduce<Max>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "max".to_string())));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Hybrid(HybridOp::ReduceMax { axes })
        }
        "Reduce<Prod>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "prod".to_string())));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes: Vec<usize> = op.axes.into_iter().collect();

            // length of prod along axes
            let len_prod = inputs[0].out_dims()[0]
                .iter()
                .enumerate()
                .filter(|(i, _)| axes.contains(i))
                .map(|(_, v)| v)
                .product::<usize>();

            SupportedOp::Linear(PolyOp::Prod { axes, len_prod })
        }
        "Reduce<Sum>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "sum".to_string())));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Linear(PolyOp::Sum { axes })
        }
        "Reduce<MeanOfSquares>" => {
            if inputs.len() != 1 {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "mean of squares".to_string(),
                )));
            };
            let op = load_op::<Reduce>(node.op(), idx, node.op().name().to_string())?;
            let axes = op.axes.into_iter().collect();

            SupportedOp::Linear(PolyOp::MeanOfSquares { axes })
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
                    return Err(Box::new(GraphError::InvalidDims(idx, "max".to_string())));
                }
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Max".to_string())));
            };

            if inputs.len() == 2 {
                if let Some(node) = inputs.get_mut(const_idx) {
                    node.decrement_use();
                    deleted_indices.push(const_idx);
                }
                if unit == 0. {
                    SupportedOp::Nonlinear(LookupOp::ReLU)
                } else {
                    // get the non-constant index
                    let non_const_idx = if const_idx == 0 { 1 } else { 0 };
                    SupportedOp::Nonlinear(LookupOp::Max {
                        scale: scale_to_multiplier(inputs[non_const_idx].out_scales()[0]).into(),
                        a: crate::circuit::utils::F32(unit),
                    })
                }
            } else {
                return Err(Box::new(GraphError::InvalidDims(idx, "max".to_string())));
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
                    return Err(Box::new(GraphError::InvalidDims(idx, "min".to_string())));
                }
            } else {
                return Err(Box::new(GraphError::OpMismatch(idx, "Min".to_string())));
            };

            if inputs.len() == 2 {
                if let Some(node) = inputs.get_mut(const_idx) {
                    node.decrement_use();
                    deleted_indices.push(const_idx);
                }

                // get the non-constant index
                let non_const_idx = if const_idx == 0 { 1 } else { 0 };

                SupportedOp::Nonlinear(LookupOp::Min {
                    scale: scale_to_multiplier(inputs[non_const_idx].out_scales()[0]).into(),
                    a: crate::circuit::utils::F32(unit),
                })
            } else {
                return Err(Box::new(GraphError::InvalidDims(idx, "min".to_string())));
            }
        }
        "Recip" => {
            let in_scale = inputs[0].out_scales()[0];
            let max_scale = std::cmp::max(scales.get_max(), in_scale);
            // If the input scale is larger than the params scale
            SupportedOp::Hybrid(HybridOp::Recip {
                input_scale: (scale_to_multiplier(in_scale) as f32).into(),
                output_scale: (scale_to_multiplier(max_scale) as f32).into(),
                use_range_check_for_int: true,
            })
        }

        "LeakyRelu" => {
            // Extract the slope layer hyperparams
            let leaky_op = load_op::<ElementWiseOp>(node.op(), idx, node.op().name().to_string())?;

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
                slope: crate::circuit::utils::F32(leaky_op.alpha),
            })
        }
        "Scan" => {
            return Err("scan should never be analyzed explicitly".into());
        }
        "QuantizeLinearU8" | "DequantizeLinearF32" => {
            SupportedOp::Linear(PolyOp::Identity { out_scale: None })
        }
        "Abs" => SupportedOp::Nonlinear(LookupOp::Abs),
        "Neg" => SupportedOp::Linear(PolyOp::Neg),
        "HardSwish" => SupportedOp::Nonlinear(LookupOp::HardSwish {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Sigmoid" => SupportedOp::Nonlinear(LookupOp::Sigmoid {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Sqrt" => SupportedOp::Nonlinear(LookupOp::Sqrt {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Rsqrt" => SupportedOp::Nonlinear(LookupOp::Rsqrt {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Exp" => SupportedOp::Nonlinear(LookupOp::Exp {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Ln" => SupportedOp::Nonlinear(LookupOp::Ln {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Sin" => SupportedOp::Nonlinear(LookupOp::Sin {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Cos" => SupportedOp::Nonlinear(LookupOp::Cos {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Tan" => SupportedOp::Nonlinear(LookupOp::Tan {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Asin" => SupportedOp::Nonlinear(LookupOp::ASin {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Acos" => SupportedOp::Nonlinear(LookupOp::ACos {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Atan" => SupportedOp::Nonlinear(LookupOp::ATan {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Sinh" => SupportedOp::Nonlinear(LookupOp::Sinh {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Cosh" => SupportedOp::Nonlinear(LookupOp::Cosh {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Tanh" => SupportedOp::Nonlinear(LookupOp::Tanh {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Asinh" => SupportedOp::Nonlinear(LookupOp::ASinh {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Acosh" => SupportedOp::Nonlinear(LookupOp::ACosh {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Atanh" => SupportedOp::Nonlinear(LookupOp::ATanh {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Erf" => SupportedOp::Nonlinear(LookupOp::Erf {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Source" => {
            let (scale, datum_type) = match node.outputs[0].fact.datum_type {
                DatumType::Bool => (0, InputType::Bool),
                DatumType::TDim => (0, InputType::TDim),
                DatumType::I64
                | DatumType::I32
                | DatumType::I16
                | DatumType::I8
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64 => (0, InputType::Int),
                DatumType::F16 => (scales.input, InputType::F16),
                DatumType::F32 => (scales.input, InputType::F32),
                DatumType::F64 => (scales.input, InputType::F64),
                _ => return Err(Box::new(GraphError::UnsupportedDataType)),
            };
            SupportedOp::Input(crate::circuit::ops::Input { scale, datum_type })
        }
        "Cast" => {
            let op = load_op::<Cast>(node.op(), idx, node.op().name().to_string())?;
            let dt = op.to;

            assert_eq!(input_scales.len(), 1);

            match dt {
                DatumType::Bool
                | DatumType::TDim
                | DatumType::I64
                | DatumType::I32
                | DatumType::I16
                | DatumType::I8
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64 => {
                    if input_scales[0] != 0 {
                        replace_const(
                            0,
                            0,
                            SupportedOp::Nonlinear(LookupOp::Cast {
                                scale: crate::circuit::utils::F32(scale_to_multiplier(
                                    input_scales[0],
                                )
                                    as f32),
                            }),
                        )?
                    } else {
                        SupportedOp::Linear(PolyOp::Identity { out_scale: None })
                    }
                }
                DatumType::F16 | DatumType::F32 | DatumType::F64 => {
                    SupportedOp::Linear(PolyOp::Identity { out_scale: None })
                }
                _ => return Err(Box::new(GraphError::UnsupportedDataType)),
            }
        }
        "Add" => SupportedOp::Linear(PolyOp::Add),
        "Sub" => SupportedOp::Linear(PolyOp::Sub),
        "Mul" => {
            let mut op = SupportedOp::Linear(PolyOp::Mult);

            let const_idx = inputs
                .iter()
                .enumerate()
                .filter(|(_, n)| n.is_constant())
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            if const_idx.len() > 1 {
                return Err(Box::new(GraphError::InvalidDims(idx, "mul".to_string())));
            }

            if const_idx.len() == 1 {
                let const_idx = const_idx[0];
                if let Some(c) = inputs[const_idx].opkind().get_mutable_constant() {
                    if c.raw_values.len() == 1 && c.raw_values[0] < 1. {
                        // if not divisible by 2 then we need to add a range check
                        let raw_values = 1.0 / c.raw_values[0];
                        if raw_values.log2().fract() == 0.0 {
                            inputs[const_idx].decrement_use();
                            deleted_indices.push(const_idx);
                            op = SupportedOp::Linear(PolyOp::Identity {
                                out_scale: Some(input_scales[0] + raw_values.log2() as i32),
                            });
                        }
                    }
                }
            }
            op
        }
        "Iff" => SupportedOp::Linear(PolyOp::Iff),
        "Less" => {
            if inputs.len() == 2 {
                SupportedOp::Hybrid(HybridOp::Less)
            } else {
                return Err(Box::new(GraphError::InvalidDims(idx, "less".to_string())));
            }
        }
        "LessEqual" => {
            if inputs.len() == 2 {
                SupportedOp::Hybrid(HybridOp::LessEqual)
            } else {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "less equal".to_string(),
                )));
            }
        }
        "Greater" => {
            // Extract the slope layer hyperparams
            if inputs.len() == 2 {
                SupportedOp::Hybrid(HybridOp::Greater)
            } else {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "greater".to_string(),
                )));
            }
        }
        "GreaterEqual" => {
            // Extract the slope layer hyperparams
            if inputs.len() == 2 {
                SupportedOp::Hybrid(HybridOp::GreaterEqual)
            } else {
                return Err(Box::new(GraphError::InvalidDims(
                    idx,
                    "greater equal".to_string(),
                )));
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

            let in_scale = inputs[0].out_scales()[0];
            let max_scale = std::cmp::max(scales.get_max(), in_scale);

            SupportedOp::Hybrid(HybridOp::Softmax {
                input_scale: scale_to_multiplier(in_scale).into(),
                output_scale: scale_to_multiplier(max_scale).into(),
                axes: softmax_op.axes.to_vec(),
            })
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

            let stride = pool_spec
                .strides
                .clone()
                .ok_or(GraphError::MissingParams("stride".to_string()))?;
            let padding = match &pool_spec.padding {
                PaddingSpec::Explicit(b, a) | PaddingSpec::ExplicitOnnxPool(b, a, _) => {
                    b.iter().zip(a.iter()).map(|(b, a)| (*b, *a)).collect()
                }
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };
            let kernel_shape = &pool_spec.kernel_shape;

            SupportedOp::Hybrid(HybridOp::MaxPool {
                padding,
                stride: stride.to_vec(),
                pool_dims: kernel_shape.to_vec(),
            })
        }
        "Ceil" => SupportedOp::Nonlinear(LookupOp::Ceil {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Floor" => SupportedOp::Nonlinear(LookupOp::Floor {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Round" => SupportedOp::Nonlinear(LookupOp::Round {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "RoundHalfToEven" => SupportedOp::Nonlinear(LookupOp::RoundHalfToEven {
            scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
        }),
        "Sign" => SupportedOp::Nonlinear(LookupOp::Sign),
        "Pow" => {
            // Extract the slope layer hyperparams from a const

            // if param_visibility.is_public() {
            if let Some(c) = inputs[1].opkind().get_mutable_constant() {
                inputs[1].decrement_use();
                deleted_indices.push(1);
                if c.raw_values.len() > 1 {
                    unimplemented!("only support scalar pow")
                }
                SupportedOp::Nonlinear(LookupOp::Pow {
                    scale: scale_to_multiplier(inputs[0].out_scales()[0]).into(),
                    a: crate::circuit::utils::F32(c.raw_values[0]),
                })
            } else {
                unimplemented!("only support constant pow for now")
            }
        }
        "Cube" => SupportedOp::Linear(PolyOp::Pow(3)),
        "Square" => SupportedOp::Linear(PolyOp::Pow(2)),
        "Conv" => {
            let conv_node: &Conv = match node.op().downcast_ref::<Conv>() {
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
                Some(s) => s.to_vec(),
                None => {
                    return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                }
            };

            let padding = match &conv_node.pool_spec.padding {
                PaddingSpec::Explicit(b, a) | PaddingSpec::ExplicitOnnxPool(b, a, _) => {
                    b.iter().zip(a.iter()).map(|(b, a)| (*b, *a)).collect()
                }
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            // if bias exists then rescale it to the input + kernel scale
            if input_scales.len() == 3 {
                let bias_scale = input_scales[2];
                let input_scale = input_scales[0];
                let kernel_scale = input_scales[1];

                let output_scale = input_scale + kernel_scale;
                if bias_scale != output_scale {
                    replace_const(
                        output_scale,
                        2,
                        SupportedOp::Unknown(crate::circuit::Unknown),
                    )?;
                }
            }

            SupportedOp::Linear(PolyOp::Conv { padding, stride })
        }
        "Not" => SupportedOp::Linear(PolyOp::Not),
        "And" => SupportedOp::Linear(PolyOp::And),
        "Or" => SupportedOp::Linear(PolyOp::Or),
        "Xor" => SupportedOp::Linear(PolyOp::Xor),
        "Equals" => SupportedOp::Hybrid(HybridOp::Equals),
        "Deconv" => {
            let deconv_node: &Deconv = match node.op().downcast_ref::<Deconv>() {
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
                Some(s) => s.to_vec(),
                None => {
                    return Err(Box::new(GraphError::MissingParams("strides".to_string())));
                }
            };
            let padding = match &deconv_node.pool_spec.padding {
                PaddingSpec::Explicit(b, a) | PaddingSpec::ExplicitOnnxPool(b, a, _) => {
                    b.iter().zip(a.iter()).map(|(b, a)| (*b, *a)).collect()
                }
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            // if bias exists then rescale it to the input + kernel scale
            if input_scales.len() == 3 {
                let bias_scale = input_scales[2];
                let input_scale = input_scales[0];
                let kernel_scale = input_scales[1];

                let output_scale = input_scale + kernel_scale;
                if bias_scale != output_scale {
                    replace_const(
                        output_scale,
                        2,
                        SupportedOp::Unknown(crate::circuit::Unknown),
                    )?;
                }
            }

            SupportedOp::Linear(PolyOp::DeConv {
                padding,
                output_padding: deconv_node.adjustments.to_vec(),
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
            // check if optional scale factor is present
            if inputs.len() != 2 && inputs.len() != 3 {
                return Err(Box::new(GraphError::OpMismatch(idx, "Resize".to_string())));
            }

            let scale_factor_node =  // find optional_scales_input in the string and extract the value inside the Some
            if resize_node.contains("optional_scales_input: None") {
                 None
            } else {
                Some(resize_node
                .split("optional_scales_input: ")
                .collect::<Vec<_>>()[1]
                .split("Some(")
                .collect::<Vec<_>>()[1]
                .split(')')
                .collect::<Vec<_>>()[0]
                .parse::<usize>()?)
            };

            let scale_factor = if let Some(scale_factor_node) = scale_factor_node {
                let boxed_op = inputs[scale_factor_node].opkind();
                if let Some(c) = extract_const_raw_values(boxed_op) {
                    c.map(|x| x as usize).into_iter().collect::<Vec<usize>>()
                } else {
                    return Err(Box::new(GraphError::OpMismatch(idx, "Resize".to_string())));
                }
            } else {
                // default
                vec![1]
            };

            for i in 1..inputs.len() {
                // remove the resize node from the inputs
                if let Some(node) = inputs.get_mut(i) {
                    node.decrement_use();
                    deleted_indices.push(i);
                }
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

            let stride = pool_spec
                .strides
                .clone()
                .ok_or(GraphError::MissingParams("stride".to_string()))?;
            let padding = match &pool_spec.padding {
                PaddingSpec::Explicit(b, a) | PaddingSpec::ExplicitOnnxPool(b, a, _) => {
                    b.iter().zip(a.iter()).map(|(b, a)| (*b, *a)).collect()
                }
                _ => {
                    return Err(Box::new(GraphError::MissingParams("padding".to_string())));
                }
            };

            SupportedOp::Hybrid(HybridOp::SumPool {
                padding,
                stride: stride.to_vec(),
                kernel_shape: pool_spec.kernel_shape.to_vec(),
                normalized: sumpool_node.normalize,
            })
        }
        // "GlobalAvgPool" => SupportedOp::Linear(PolyOp::SumPool {
        //     padding: [(0, 0); 2],
        //     stride: (1, 1),
        //     kernel_shape: (inputs[0].out_dims()[0][1], inputs[0].out_dims()[0][2]),
        // }),
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

            SupportedOp::Linear(PolyOp::Pad(pad_node.pads.to_vec()))
        }
        "RmAxis" | "Reshape" | "AddAxis" => {
            // Extract the slope layer hyperparams
            let shapes = node_output_shapes(&node, symbol_values)?;
            let mut output_shape = shapes[0].clone();
            if output_shape.is_empty() {
                output_shape = vec![1];
            }

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

/// Converts a tensor to a [ValTensor] with a given scale.
pub fn quantize_tensor<F: PrimeField + TensorType + PartialOrd>(
    const_value: Tensor<f32>,
    scale: crate::Scale,
    visibility: &Visibility,
) -> Result<Tensor<F>, Box<dyn std::error::Error>> {
    let mut value: Tensor<F> = const_value.par_enum_map(|_, x| {
        Ok::<_, TensorError>(crate::fieldutils::i64_to_felt::<F>(quantize_float(
            &(x).into(),
            0.0,
            scale,
        )?))
    })?;

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

///
pub fn homogenize_input_scales(
    op: Box<dyn Op<Fp>>,
    input_scales: Vec<crate::Scale>,
    inputs_to_scale: Vec<usize>,
) -> Result<Box<dyn Op<Fp>>, Box<dyn Error>> {
    let relevant_input_scales = input_scales
        .clone()
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| inputs_to_scale.contains(idx))
        .map(|(_, scale)| scale)
        .collect_vec();

    if inputs_to_scale.is_empty() {
        return Ok(op);
    }
    // else if all inputs_scales at inputs_to_scale are the same, we don't need to do anything
    if relevant_input_scales.windows(2).all(|w| w[0] == w[1]) {
        return Ok(op);
    }

    let mut multipliers: Vec<u128> = vec![1; input_scales.len()];

    let max_scale = input_scales.iter().max().ok_or("no max scale")?;
    let _ = input_scales
        .iter()
        .enumerate()
        .map(|(idx, input_scale)| {
            if !inputs_to_scale.contains(&idx) {
                return;
            }
            let scale_diff = max_scale - input_scale;
            if scale_diff > 0 {
                let mult = crate::graph::scale_to_multiplier(scale_diff);
                multipliers[idx] = mult as u128;
            }
        })
        .collect_vec();

    // only rescale if need to
    if multipliers.iter().any(|&x| x > 1) {
        Ok(Box::new(Rescaled {
            inner: Box::new(op.into()),
            scale: (0..input_scales.len()).zip(multipliers).collect_vec(),
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

        tensor.set_visibility(&Visibility::Public);

        let flattened: ValTensor<Fp> = tensor.try_into().unwrap();

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
