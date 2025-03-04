use super::*;
use crate::{
    circuit::{layouts, utils},
    fieldutils::{integer_rep_to_felt, IntegerRep},
    graph::multiplier_to_scale,
    tensor::{self, DataFormat, Tensor, TensorType, ValTensor},
};
use halo2curves::ff::PrimeField;
use serde::{Deserialize, Serialize};
// import run args from model

#[allow(missing_docs)]
/// An enum representing the operations that consist of both lookups and arithmetic operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HybridOp {
    Ln {
        scale: utils::F32,
    },
    Rsqrt {
        input_scale: utils::F32,
        output_scale: utils::F32,
    },
    Sqrt {
        scale: utils::F32,
    },
    RoundHalfToEven {
        scale: utils::F32,
        legs: usize,
    },
    Ceil {
        scale: utils::F32,
        legs: usize,
    },
    Floor {
        scale: utils::F32,
        legs: usize,
    },
    Round {
        scale: utils::F32,
        legs: usize,
    },
    Recip {
        input_scale: utils::F32,
        output_scale: utils::F32,
    },
    Div {
        denom: utils::F32,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceArgMax {
        dim: usize,
    },
    SumPool {
        padding: Vec<(usize, usize)>,
        stride: Vec<usize>,
        kernel_shape: Vec<usize>,
        normalized: bool,
        data_format: DataFormat,
    },
    MaxPool {
        padding: Vec<(usize, usize)>,
        stride: Vec<usize>,
        pool_dims: Vec<usize>,
        data_format: DataFormat,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    ReduceArgMin {
        dim: usize,
    },
    Max,
    Min,
    Softmax {
        input_scale: utils::F32,
        output_scale: utils::F32,
        axes: Vec<usize>,
    },
    Output {
        decomp: bool,
    },
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equals,
    Gather {
        dim: usize,
        constant_idx: Option<Tensor<usize>>,
    },
    TopK {
        dim: usize,
        k: usize,
        largest: bool,
    },
    OneHot {
        dim: usize,
        num_classes: usize,
    },
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Op<F> for HybridOp {
    ///
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        match self {
            HybridOp::Greater { .. }
            | HybridOp::Less { .. }
            | HybridOp::Equals { .. }
            | HybridOp::GreaterEqual { .. }
            | HybridOp::Max
            | HybridOp::Min
            | HybridOp::LessEqual { .. } => {
                vec![0, 1]
            }
            _ => vec![],
        }
    }

    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        match self {
            HybridOp::Rsqrt {
                input_scale,
                output_scale,
            } => format!(
                "RSQRT (input_scale={}, output_scale={})",
                input_scale, output_scale
            ),
            HybridOp::Sqrt { scale } => format!("SQRT(scale={})", scale),
            HybridOp::Ln { scale } => format!("LN(scale={})", scale),
            HybridOp::RoundHalfToEven { scale, legs } => {
                format!("ROUND_HALF_TO_EVEN(scale={}, legs={})", scale, legs)
            }
            HybridOp::Ceil { scale, legs } => format!("CEIL(scale={}, legs={})", scale, legs),
            HybridOp::Floor { scale, legs } => format!("FLOOR(scale={}, legs={})", scale, legs),
            HybridOp::Round { scale, legs } => format!("ROUND(scale={}, legs={})", scale, legs),

            HybridOp::Max => "MAX".to_string(),
            HybridOp::Min => "MIN".to_string(),
            HybridOp::Recip {
                input_scale,
                output_scale,
            } => format!(
                "RECIP (input_scale={}, output_scale={})",
                input_scale, output_scale
            ),
            HybridOp::Div { denom } => format!("DIV (denom={})", denom),
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized, data_format
            } => format!(
                "SUMPOOL (padding={:?}, stride={:?}, kernel_shape={:?}, normalized={}, data_format={:?})",
                padding, stride, kernel_shape, normalized, data_format
            ),
            HybridOp::ReduceMax { axes } => format!("REDUCEMAX (axes={:?})", axes),
            HybridOp::ReduceArgMax { dim } => format!("REDUCEARGMAX (dim={})", dim),
            HybridOp::MaxPool {
                padding,
                stride,
                pool_dims,
                data_format,
            } => format!(
                "MaxPool (padding={:?}, stride={:?}, pool_dims={:?}, data_format={:?})",
                padding, stride, pool_dims, data_format
            ),
            HybridOp::ReduceMin { axes } => format!("REDUCEMIN (axes={:?})", axes),
            HybridOp::ReduceArgMin { dim } => format!("REDUCEARGMIN (dim={})", dim),
            HybridOp::Softmax {
                input_scale,
                output_scale,
                axes,
            } => {
                format!(
                    "SOFTMAX (input_scale={}, output_scale={}, axes={:?})",
                    input_scale, output_scale, axes
                )
            }
            HybridOp::Output { decomp } => {
                format!("OUTPUT (decomp={})", decomp)
            }
            HybridOp::Greater => "GREATER".to_string(),
            HybridOp::GreaterEqual => "GREATEREQUAL".to_string(),
            HybridOp::Less => "LESS".to_string(),
            HybridOp::LessEqual => "LESSEQUAL".to_string(),
            HybridOp::Equals => "EQUALS".into(),
            HybridOp::Gather { dim, .. } => format!("GATHER (dim={})", dim),
            HybridOp::TopK { k, dim, largest } => {
                format!("TOPK (k={}, dim={}, largest={})", k, dim, largest)
            }
            HybridOp::OneHot { dim, num_classes } => {
                format!("ONEHOT (dim={}, num_classes={})", dim, num_classes)
            }
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        Ok(Some(match self {
            HybridOp::Rsqrt {
                input_scale,
                output_scale,
            } => layouts::rsqrt(
                config,
                region,
                values[..].try_into()?,
                *input_scale,
                *output_scale,
            )?,
            HybridOp::Sqrt { scale } => {
                layouts::sqrt(config, region, values[..].try_into()?, *scale)?
            }
            HybridOp::Ln { scale } => layouts::ln(config, region, values[..].try_into()?, *scale)?,
            HybridOp::RoundHalfToEven { scale, legs } => {
                layouts::round_half_to_even(config, region, values[..].try_into()?, *scale, *legs)?
            }
            HybridOp::Ceil { scale, legs } => {
                layouts::ceil(config, region, values[..].try_into()?, *scale, *legs)?
            }
            HybridOp::Floor { scale, legs } => {
                layouts::floor(config, region, values[..].try_into()?, *scale, *legs)?
            }
            HybridOp::Round { scale, legs } => {
                layouts::round(config, region, values[..].try_into()?, *scale, *legs)?
            }
            HybridOp::Max => layouts::max_comp(config, region, values[..].try_into()?)?,
            HybridOp::Min => layouts::min_comp(config, region, values[..].try_into()?)?,
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized,
                data_format,
            } => layouts::sumpool(
                config,
                region,
                values[..].try_into()?,
                padding,
                stride,
                kernel_shape,
                *normalized,
                *data_format,
            )?,
            HybridOp::Recip {
                input_scale,
                output_scale,
            } => layouts::recip(
                config,
                region,
                values[..].try_into()?,
                integer_rep_to_felt(input_scale.0 as IntegerRep),
                integer_rep_to_felt(output_scale.0 as IntegerRep),
            )?,
            HybridOp::Div { denom, .. } => {
                if denom.0.fract() == 0.0 {
                    layouts::div(
                        config,
                        region,
                        values[..].try_into()?,
                        integer_rep_to_felt(denom.0 as IntegerRep),
                    )?
                } else {
                    layouts::nonlinearity(
                        config,
                        region,
                        values.try_into()?,
                        &LookupOp::Div { denom: *denom },
                    )?
                }
            }
            HybridOp::Gather { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::gather(values[0].get_inner_tensor()?, idx, *dim)?.into()
                } else {
                    layouts::gather(config, region, values[..].try_into()?, *dim)?
                }
            }

            HybridOp::MaxPool {
                padding,
                stride,
                pool_dims,
                data_format,
            } => layouts::max_pool(
                config,
                region,
                values[..].try_into()?,
                padding,
                stride,
                pool_dims,
                *data_format,
            )?,
            HybridOp::ReduceMax { axes } => {
                layouts::max_axes(config, region, values[..].try_into()?, axes)?
            }
            HybridOp::ReduceArgMax { dim } => {
                layouts::argmax_axes(config, region, values[..].try_into()?, *dim)?
            }
            HybridOp::ReduceMin { axes } => {
                layouts::min_axes(config, region, values[..].try_into()?, axes)?
            }
            HybridOp::ReduceArgMin { dim } => {
                layouts::argmin_axes(config, region, values[..].try_into()?, *dim)?
            }
            HybridOp::Softmax {
                input_scale,
                output_scale,
                axes,
            } => layouts::softmax_axes(
                config,
                region,
                values[..].try_into()?,
                *input_scale,
                *output_scale,
                axes,
            )?,
            HybridOp::Output { decomp } => {
                layouts::output(config, region, values[..].try_into()?, *decomp)?
            }
            HybridOp::Greater => layouts::greater(config, region, values[..].try_into()?)?,
            HybridOp::GreaterEqual => {
                layouts::greater_equal(config, region, values[..].try_into()?)?
            }
            HybridOp::Less => layouts::less(config, region, values[..].try_into()?)?,
            HybridOp::LessEqual => layouts::less_equal(config, region, values[..].try_into()?)?,
            HybridOp::Equals => layouts::equals(config, region, values[..].try_into()?)?,
            HybridOp::TopK { dim, k, largest } => {
                layouts::topk_axes(config, region, values[..].try_into()?, *k, *dim, *largest)?
            }
            HybridOp::OneHot { dim, num_classes } => {
                layouts::one_hot_axis(config, region, values[..].try_into()?, *num_classes, *dim)?
            }
        }))
    }

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        let scale = match self {
            HybridOp::Greater { .. }
            | HybridOp::GreaterEqual { .. }
            | HybridOp::Less { .. }
            | HybridOp::LessEqual { .. }
            | HybridOp::ReduceArgMax { .. }
            | HybridOp::OneHot { .. }
            | HybridOp::ReduceArgMin { .. } => 0,

            HybridOp::Recip { output_scale, .. } | HybridOp::Rsqrt { output_scale, .. } => {
                multiplier_to_scale(output_scale.0 as f64)
            }
            HybridOp::Softmax {
                output_scale,
                input_scale,
                ..
            } => multiplier_to_scale((output_scale.0 * input_scale.0) as f64),
            HybridOp::Ln {
                scale: output_scale,
            } => 4 * multiplier_to_scale(output_scale.0 as f64),
            _ => in_scales[0],
        };
        Ok(scale)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
