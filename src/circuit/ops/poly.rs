use crate::{
    circuit::layouts,
    fieldutils::{felt_to_i128, i128_to_felt},
    tensor::{self, Tensor, TensorError},
};

use super::{base::BaseOp, *};

#[allow(missing_docs)]
/// An enum representing the operations that can be expressed as arithmetic (non lookup) operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PolyOp {
    GatherElements {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
    GatherND {
        batch_dims: usize,
        indices: Option<Tensor<usize>>,
    },
    ScatterElements {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
    ScatterND {
        constant_idx: Option<Tensor<usize>>,
    },
    MultiBroadcastTo {
        shape: Vec<usize>,
    },
    Einsum {
        equation: String,
    },
    Conv {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
    },
    Downsample {
        axis: usize,
        stride: usize,
        modulo: usize,
    },
    DeConv {
        padding: [(usize, usize); 2],
        output_padding: (usize, usize),
        stride: (usize, usize),
    },
    Add,
    Sub,
    Neg,
    Mult,
    Identity {
        out_scale: Option<crate::Scale>,
    },
    Reshape(Vec<usize>),
    MoveAxis {
        source: usize,
        destination: usize,
    },
    Flatten(Vec<usize>),
    Pad([(usize, usize); 2]),
    Sum {
        axes: Vec<usize>,
    },
    MeanOfSquares {
        axes: Vec<usize>,
    },
    Prod {
        axes: Vec<usize>,
        len_prod: usize,
    },
    Pow(u32),
    Concat {
        axis: usize,
    },
    Slice {
        axis: usize,
        start: usize,
        end: usize,
    },
    Iff,
    Resize {
        scale_factor: Vec<usize>,
    },
    Not,
    And,
    Or,
    Xor,
    Trilu {
        upper: bool,
        k: i32,
    },
}

impl<
        F: PrimeField
            + TensorType
            + PartialOrd
            + std::hash::Hash
            + Serialize
            + for<'de> Deserialize<'de>,
    > Op<F> for PolyOp
{
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        match &self {
            PolyOp::GatherElements { dim, .. } => format!("GATHERELEMENTS (dim={})", dim),
            PolyOp::GatherND { batch_dims, .. } => format!("GATHERND (batch_dims={})", batch_dims),
            PolyOp::MeanOfSquares { axes } => format!("MEANOFSQUARES (axes={:?})", axes),
            PolyOp::ScatterElements { dim, .. } => format!("SCATTERELEMENTS (dim={})", dim),
            PolyOp::ScatterND { .. } => "SCATTERND".into(),
            PolyOp::MultiBroadcastTo { shape } => format!("MULTIBROADCASTTO (shape={:?})", shape),
            PolyOp::MoveAxis { .. } => "MOVEAXIS".into(),
            PolyOp::Downsample { .. } => "DOWNSAMPLE".into(),
            PolyOp::Resize { .. } => "RESIZE".into(),
            PolyOp::Iff => "IFF".into(),
            PolyOp::Einsum { equation, .. } => format!("EINSUM {}", equation),
            PolyOp::Identity { out_scale } => {
                format!("IDENTITY (out_scale={:?})", out_scale)
            }
            PolyOp::Reshape(shape) => format!("RESHAPE (shape={:?})", shape),
            PolyOp::Flatten(_) => "FLATTEN".into(),
            PolyOp::Pad(_) => "PAD".into(),
            PolyOp::Add => "ADD".into(),
            PolyOp::Mult => "MULT".into(),
            PolyOp::Sub => "SUB".into(),
            PolyOp::Sum { axes } => format!("SUM (axes={:?})", axes),
            PolyOp::Prod { .. } => "PROD".into(),
            PolyOp::Pow(_) => "POW".into(),
            PolyOp::Conv { .. } => "CONV".into(),
            PolyOp::DeConv { .. } => "DECONV".into(),
            PolyOp::Concat { axis } => format!("CONCAT (axis={})", axis),
            PolyOp::Slice { axis, start, end } => {
                format!("SLICE (axis={}, start={}, end={})", axis, start, end)
            }
            PolyOp::Neg => "NEG".into(),
            PolyOp::Not => "NOT".into(),
            PolyOp::And => "AND".into(),
            PolyOp::Or => "OR".into(),
            PolyOp::Xor => "XOR".into(),
            PolyOp::Trilu { upper, k } => format!("TRILU (upper={}, k={})", upper, k),
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let mut inputs = inputs.to_vec();
        let res = match &self {
            PolyOp::MeanOfSquares { axes } => {
                let x = inputs[0].map(|x| felt_to_i128(x));
                Ok(tensor::ops::nonlinearities::mean_of_squares_axes(&x, axes).map(i128_to_felt))
            }
            PolyOp::MultiBroadcastTo { shape } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch(
                        "multibroadcastto inputs".to_string(),
                    ));
                }
                inputs[0].expand(shape)
            }
            PolyOp::And => tensor::ops::and(&inputs[0], &inputs[1]),
            PolyOp::Or => tensor::ops::or(&inputs[0], &inputs[1]),
            PolyOp::Xor => tensor::ops::xor(&inputs[0], &inputs[1]),
            PolyOp::Not => tensor::ops::not(&inputs[0]),
            PolyOp::Downsample {
                axis,
                stride,
                modulo,
            } => tensor::ops::downsample(&inputs[0], *axis, *stride, *modulo),
            PolyOp::Resize { scale_factor } => tensor::ops::resize(&inputs[0], scale_factor),
            PolyOp::Iff => tensor::ops::iff(&inputs[0], &inputs[1], &inputs[2]),
            PolyOp::Einsum { equation } => tensor::ops::einsum(equation, &inputs),
            PolyOp::Identity { .. } => Ok(inputs[0].clone()),
            PolyOp::Reshape(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims)?;
                Ok(t)
            }
            PolyOp::MoveAxis {
                source,
                destination,
            } => inputs[0].move_axis(*source, *destination),
            PolyOp::Flatten(new_dims) => {
                let mut t = inputs[0].clone();
                t.reshape(new_dims)?;
                Ok(t)
            }
            PolyOp::Pad(p) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pad inputs".to_string()));
                }
                tensor::ops::pad(&inputs[0], *p)
            }
            PolyOp::Add => tensor::ops::add(&inputs),
            PolyOp::Neg => tensor::ops::neg(&inputs[0]),
            PolyOp::Sub => tensor::ops::sub(&inputs),
            PolyOp::Mult => tensor::ops::mult(&inputs),
            PolyOp::Conv { padding, stride } => tensor::ops::conv(&inputs, *padding, *stride),
            PolyOp::DeConv {
                padding,
                output_padding,
                stride,
            } => tensor::ops::deconv(&inputs, *padding, *output_padding, *stride),
            PolyOp::Pow(u) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pow inputs".to_string()));
                }
                inputs[0].pow(*u)
            }
            PolyOp::Sum { axes } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("sum inputs".to_string()));
                }
                tensor::ops::sum_axes(&inputs[0], axes)
            }
            PolyOp::Prod { axes, .. } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("prod inputs".to_string()));
                }
                tensor::ops::prod_axes(&inputs[0], axes)
            }
            PolyOp::Concat { axis } => {
                tensor::ops::concat(&inputs.iter().collect::<Vec<_>>(), *axis)
            }
            PolyOp::Slice { axis, start, end } => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("slice inputs".to_string()));
                }
                tensor::ops::slice(&inputs[0], axis, start, end)
            }
            PolyOp::GatherElements { dim, constant_idx } => {
                let x = inputs[0].clone();
                let y = if let Some(idx) = constant_idx {
                    idx.clone()
                } else {
                    inputs[1].clone().map(|x| felt_to_i128(x) as usize)
                };
                tensor::ops::gather_elements(&x, &y, *dim)
            }
            PolyOp::GatherND {
                indices,
                batch_dims,
            } => {
                let x = inputs[0].clone();
                let y = if let Some(idx) = indices {
                    idx.clone()
                } else {
                    inputs[1].clone().map(|x| felt_to_i128(x) as usize)
                };
                tensor::ops::gather_nd(&x, &y, *batch_dims)
            }
            PolyOp::ScatterElements { dim, constant_idx } => {
                let x = inputs[0].clone();

                let idx = if let Some(idx) = constant_idx {
                    idx.clone()
                } else {
                    inputs[1].clone().map(|x| felt_to_i128(x) as usize)
                };

                let src = if constant_idx.is_some() {
                    inputs[1].clone()
                } else {
                    inputs[2].clone()
                };
                tensor::ops::scatter(&x, &idx, &src, *dim)
            }

            PolyOp::ScatterND { constant_idx } => {
                let x = inputs[0].clone();
                let idx = if let Some(idx) = constant_idx {
                    idx.clone()
                } else {
                    inputs[1].clone().map(|x| felt_to_i128(x) as usize)
                };
                let src = if constant_idx.is_some() {
                    inputs[1].clone()
                } else {
                    inputs[2].clone()
                };
                tensor::ops::scatter_nd(&x, &idx, &src)
            }
            PolyOp::Trilu { upper, k } => tensor::ops::trilu(&inputs[0], *k, *upper),
        }?;

        Ok(ForwardResult { output: res })
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(Some(match self {
            PolyOp::MultiBroadcastTo { shape } => {
                layouts::expand(config, region, values[..].try_into()?, shape)?
            }
            PolyOp::MeanOfSquares { axes } => {
                layouts::mean_of_squares_axes(config, region, values[..].try_into()?, axes)?
            }
            PolyOp::Xor => layouts::xor(config, region, values[..].try_into()?)?,
            PolyOp::Or => layouts::or(config, region, values[..].try_into()?)?,
            PolyOp::And => layouts::and(config, region, values[..].try_into()?)?,
            PolyOp::Not => layouts::not(config, region, values[..].try_into()?)?,
            PolyOp::MoveAxis {
                source,
                destination,
            } => layouts::move_axis(values[..].try_into()?, *source, *destination)?,
            PolyOp::Downsample {
                axis,
                stride,
                modulo,
            } => layouts::downsample(config, region, values[..].try_into()?, axis, stride, modulo)?,
            PolyOp::Resize { scale_factor } => {
                layouts::resize(config, region, values[..].try_into()?, scale_factor)?
            }
            PolyOp::Neg => layouts::neg(config, region, values[..].try_into()?)?,
            PolyOp::Iff => layouts::iff(config, region, values[..].try_into()?)?,
            PolyOp::Einsum { equation } => layouts::einsum(config, region, values, equation)?,
            PolyOp::Sum { axes } => {
                layouts::sum_axes(config, region, values[..].try_into()?, axes)?
            }
            PolyOp::Prod { axes, .. } => {
                layouts::prod_axes(config, region, values[..].try_into()?, axes)?
            }
            PolyOp::Conv { padding, stride } => {
                layouts::conv(config, region, values[..].try_into()?, *padding, *stride)?
            }
            PolyOp::GatherElements { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::gather_elements(values[0].get_inner_tensor()?, idx, *dim)?.into()
                } else {
                    layouts::gather_elements(config, region, values[..].try_into()?, *dim)?.0
                }
            }
            PolyOp::GatherND {
                batch_dims,
                indices,
            } => {
                if let Some(idx) = indices {
                    tensor::ops::gather_nd(values[0].get_inner_tensor()?, idx, *batch_dims)?.into()
                } else {
                    layouts::gather_nd(config, region, values[..].try_into()?, *batch_dims)?.0
                }
            }
            PolyOp::ScatterElements { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::scatter(
                        values[0].get_inner_tensor()?,
                        idx,
                        values[1].get_inner_tensor()?,
                        *dim,
                    )?
                    .into()
                } else {
                    layouts::scatter_elements(config, region, values[..].try_into()?, *dim)?
                }
            }
            PolyOp::ScatterND { constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::scatter_nd(
                        values[0].get_inner_tensor()?,
                        idx,
                        values[1].get_inner_tensor()?,
                    )?
                    .into()
                } else {
                    layouts::scatter_nd(config, region, values[..].try_into()?)?
                }
            }
            PolyOp::DeConv {
                padding,
                output_padding,
                stride,
            } => layouts::deconv(
                config,
                region,
                values[..].try_into()?,
                *padding,
                *output_padding,
                *stride,
            )?,
            PolyOp::Add => layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Add)?,
            PolyOp::Sub => layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Sub)?,
            PolyOp::Mult => {
                layouts::pairwise(config, region, values[..].try_into()?, BaseOp::Mult)?
            }
            PolyOp::Identity { .. } => layouts::identity(config, region, values[..].try_into()?)?,
            PolyOp::Reshape(d) | PolyOp::Flatten(d) => layouts::reshape(values[..].try_into()?, d)?,
            PolyOp::Pad(p) => {
                if values.len() != 1 {
                    return Err(Box::new(TensorError::DimError(
                        "Pad operation requires a single input".to_string(),
                    )));
                }
                let mut input = values[0].clone();
                input.pad(*p)?;
                input
            }
            PolyOp::Pow(exp) => layouts::pow(config, region, values[..].try_into()?, *exp)?,
            PolyOp::Concat { axis } => layouts::concat(values[..].try_into()?, axis)?,
            PolyOp::Slice { axis, start, end } => {
                layouts::slice(config, region, values[..].try_into()?, axis, start, end)?
            }
            PolyOp::Trilu { upper, k } => {
                layouts::trilu(config, region, values[..].try_into()?, k, upper)?
            }
        }))
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let scale = match self {
            PolyOp::MeanOfSquares { .. } => 2 * in_scales[0],
            PolyOp::Xor | PolyOp::Or | PolyOp::And | PolyOp::Not => 0,
            PolyOp::Iff => in_scales[1],
            PolyOp::Einsum { .. } => {
                let mut scale = in_scales[0];
                for s in in_scales.iter().skip(1) {
                    scale += *s;
                }
                scale
            }
            PolyOp::Prod { len_prod, .. } => in_scales[0] * (*len_prod as crate::Scale),
            PolyOp::Sum { .. } => in_scales[0],
            PolyOp::Conv { .. } => {
                let input_scale = in_scales[0];
                let kernel_scale = in_scales[1];
                let output_scale = input_scale + kernel_scale;
                if in_scales.len() == 3 {
                    let bias_scale = in_scales[2];
                    assert_eq!(output_scale, bias_scale);
                }
                output_scale
            }
            PolyOp::DeConv { .. } => {
                let input_scale = in_scales[0];
                let kernel_scale = in_scales[1];
                let output_scale = input_scale + kernel_scale;
                if in_scales.len() == 3 {
                    let bias_scale = in_scales[2];
                    assert_eq!(output_scale, bias_scale);
                }
                output_scale
            }
            PolyOp::Add => {
                let scale_a = in_scales[0];
                let scale_b = in_scales[1];
                assert_eq!(scale_a, scale_b);
                scale_a
            }
            PolyOp::Sub => in_scales[0],
            PolyOp::Mult => {
                let mut scale = in_scales[0];
                scale += in_scales[1];
                scale
            }
            PolyOp::Reshape(_) | PolyOp::Flatten(_) => in_scales[0],
            PolyOp::Pow(pow) => in_scales[0] * (*pow as crate::Scale),
            PolyOp::Identity { out_scale } => out_scale.unwrap_or(in_scales[0]),
            _ => in_scales[0],
        };
        Ok(scale)
    }

    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        if matches!(self, PolyOp::Add { .. } | PolyOp::Sub) {
            vec![0, 1]
        } else if matches!(self, PolyOp::Iff) {
            vec![1, 2]
        } else if matches!(self, PolyOp::Concat { .. }) {
            (0..100).collect()
        } else if matches!(self, PolyOp::ScatterElements { .. })
            | matches!(self, PolyOp::ScatterND { .. })
        {
            vec![0, 2]
        } else {
            vec![]
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
