use crate::{
    circuit::layouts,
    fieldutils::felt_to_i128,
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
    ScatterElements {
        dim: usize,
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
    Prod {
        axes: Vec<usize>,
        len_prod: usize,
    },
    Pow(u32),
    Pack(u32, u32),
    GlobalSumPool,
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
}

impl<F: PrimeField + TensorType + PartialOrd + Serialize + for<'de> Deserialize<'de>> Op<F>
    for PolyOp
{
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        match &self {
            PolyOp::GatherElements { dim, .. } => format!("GATHERELEMENTS (dim={})", dim),
            PolyOp::ScatterElements { dim, .. } => format!("SCATTERELEMENTS (dim={})", dim),
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
            PolyOp::Sum { .. } => "SUM".into(),
            PolyOp::Prod { .. } => "PROD".into(),
            PolyOp::Pow(_) => "POW".into(),
            PolyOp::Pack(_, _) => "PACK".into(),
            PolyOp::GlobalSumPool => "GLOBALSUMPOOL".into(),
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
        }
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let mut inputs = inputs.to_vec();
        let res = match &self {
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
            PolyOp::Pack(base, scale) => {
                if 1 != inputs.len() {
                    return Err(TensorError::DimMismatch("pack inputs".to_string()));
                }

                tensor::ops::pack(&inputs[0], F::from(*base as u64), *scale)
            }
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
            PolyOp::GlobalSumPool => unreachable!(),
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
                    layouts::gather_elements(config, region, values[..].try_into()?, *dim)?
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
            PolyOp::Pack(base, scale) => {
                layouts::pack(config, region, values[..].try_into()?, *base, *scale)?
            }
            PolyOp::GlobalSumPool => unreachable!(),
            PolyOp::Concat { axis } => layouts::concat(values[..].try_into()?, axis)?,
            PolyOp::Slice { axis, start, end } => {
                layouts::slice(config, region, values[..].try_into()?, axis, start, end)?
            }
        }))
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let scale = match self {
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
        } else if matches!(self, PolyOp::ScatterElements { .. }) {
            vec![0, 2]
        } else {
            vec![]
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
