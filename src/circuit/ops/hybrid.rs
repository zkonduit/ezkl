use super::*;
use crate::{
    circuit::{layouts, utils, Tolerance},
    fieldutils::{felt_to_i128, i128_to_felt},
    graph::multiplier_to_scale,
    tensor::{self, Tensor, TensorError, TensorType, ValTensor},
};
use halo2curves::ff::PrimeField;
use serde::{Deserialize, Serialize};
// import run args from model

#[allow(missing_docs)]
/// An enum representing the operations that consist of both lookups and arithmetic operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HybridOp {
    Recip {
        input_scale: utils::F32,
        output_scale: utils::F32,
        use_range_check_for_int: bool,
    },
    Div {
        denom: utils::F32,
        use_range_check_for_int: bool,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceArgMax {
        dim: usize,
    },
    SumPool {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
        kernel_shape: (usize, usize),
        normalized: bool,
    },
    MaxPool2d {
        padding: [(usize, usize); 2],
        stride: (usize, usize),
        pool_dims: (usize, usize),
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    ReduceArgMin {
        dim: usize,
    },
    Softmax {
        scale: utils::F32,
        axes: Vec<usize>,
    },
    RangeCheck(Tolerance),
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

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for HybridOp {
    ///
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        match self {
            HybridOp::Greater | HybridOp::Less | HybridOp::Equals => vec![0, 1],
            HybridOp::GreaterEqual | HybridOp::LessEqual => vec![0, 1],
            _ => vec![],
        }
    }

    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = inputs[0].clone().map(|x| felt_to_i128(x));

        let res = match &self {
            HybridOp::ReduceMax { axes, .. } => tensor::ops::max_axes(&x, axes)?,
            HybridOp::ReduceMin { axes, .. } => tensor::ops::min_axes(&x, axes)?,
            HybridOp::Div { denom, .. } => {
                crate::tensor::ops::nonlinearities::const_div(&x, denom.0 as f64)
            }
            HybridOp::Recip {
                input_scale,
                output_scale,
                ..
            } => crate::tensor::ops::nonlinearities::recip(
                &x,
                input_scale.0 as f64,
                output_scale.0 as f64,
            ),
            HybridOp::ReduceArgMax { dim } => tensor::ops::argmax_axes(&x, *dim)?,
            HybridOp::ReduceArgMin { dim } => tensor::ops::argmin_axes(&x, *dim)?,
            HybridOp::Gather { dim, constant_idx } => {
                if let Some(idx) = constant_idx {
                    tensor::ops::gather(&x, idx, *dim)?
                } else {
                    let y = inputs[1].clone().map(|x| felt_to_i128(x));
                    tensor::ops::gather(&x, &y.map(|x| x as usize), *dim)?
                }
            }
            HybridOp::OneHot { dim, num_classes } => {
                tensor::ops::one_hot(&x, *num_classes, *dim)?.clone()
            }

            HybridOp::TopK { dim, k, largest } => tensor::ops::topk_axes(&x, *k, *dim, *largest)?,
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
                ..
            } => tensor::ops::max_pool2d(&x, padding, stride, pool_dims)?,
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized,
            } => tensor::ops::sumpool(&x, *padding, *stride, *kernel_shape, *normalized)?,
            HybridOp::Softmax { scale, axes } => {
                tensor::ops::nonlinearities::softmax_axes(&x, scale.into(), axes)
            }
            HybridOp::RangeCheck(tol) => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::nonlinearities::range_check_percent(&[x, y], 128, 128, tol.val)
            }
            HybridOp::Greater => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::greater(&x, &y)?
            }
            HybridOp::GreaterEqual => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::greater_equal(&x, &y)?
            }
            HybridOp::Less => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::less(&x, &y)?
            }
            HybridOp::LessEqual => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::less_equal(&x, &y)?
            }
            HybridOp::Equals => {
                let y = inputs[1].clone().map(|x| felt_to_i128(x));
                tensor::ops::equals(&x, &y)?
            }
        };

        // convert back to felt
        let output = res.map(|x| i128_to_felt(x));

        Ok(ForwardResult { output })
    }

    fn as_string(&self) -> String {
        match self {
            HybridOp::Recip {
                input_scale,
                output_scale,
                use_range_check_for_int,
            } => format!(
                "RECIP (input_scale={}, output_scale={}, use_range_check_for_int={})",
                input_scale, output_scale, use_range_check_for_int
            ),
            HybridOp::Div {
                denom,
                use_range_check_for_int,
            } => format!(
                "DIV (denom={}, use_range_check_for_int={})",
                denom, use_range_check_for_int
            ),
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized,
            } => format!(
                "SUMPOOL (padding={:?}, stride={:?}, kernel_shape={:?}, normalized={})",
                padding, stride, kernel_shape, normalized
            ),
            HybridOp::ReduceMax { axes } => format!("REDUCEMAX (axes={:?})", axes),
            HybridOp::ReduceArgMax { dim } => format!("REDUCEARGMAX (dim={})", dim),
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => format!(
                "MAXPOOL2D (padding={:?}, stride={:?}, pool_dims={:?})",
                padding, stride, pool_dims
            ),
            HybridOp::ReduceMin { axes } => format!("REDUCEMIN (axes={:?})", axes),
            HybridOp::ReduceArgMin { dim } => format!("REDUCEARGMIN (dim={})", dim),
            HybridOp::Softmax { scale, axes } => {
                format!("SOFTMAX (scale={}, axes={:?})", scale, axes)
            }
            HybridOp::RangeCheck(p) => format!("RANGECHECK (tol={:?})", p),
            HybridOp::Greater => "GREATER".into(),
            HybridOp::GreaterEqual => "GREATEREQUAL".into(),
            HybridOp::Less => "LESS".into(),
            HybridOp::LessEqual => "LESSEQUAL".into(),
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
    ) -> Result<Option<ValTensor<F>>, Box<dyn std::error::Error>> {
        Ok(Some(match self {
            HybridOp::SumPool {
                padding,
                stride,
                kernel_shape,
                normalized,
            } => layouts::sumpool(
                config,
                region,
                values[..].try_into()?,
                *padding,
                *stride,
                *kernel_shape,
                *normalized,
            )?,
            HybridOp::Recip {
                input_scale,
                output_scale,
                use_range_check_for_int,
            } => {
                if input_scale.0.fract() == 0.0
                    && output_scale.0.fract() == 0.0
                    && *use_range_check_for_int
                {
                    layouts::recip(
                        config,
                        region,
                        values[..].try_into()?,
                        i128_to_felt(input_scale.0 as i128),
                        i128_to_felt(output_scale.0 as i128),
                    )?
                } else {
                    layouts::nonlinearity(
                        config,
                        region,
                        values.try_into()?,
                        &LookupOp::Recip {
                            input_scale: *input_scale,
                            output_scale: *output_scale,
                        },
                    )?
                }
            }
            HybridOp::Div {
                denom,
                use_range_check_for_int,
                ..
            } => {
                if denom.0.fract() == 0.0 && *use_range_check_for_int {
                    layouts::loop_div(
                        config,
                        region,
                        values[..].try_into()?,
                        i128_to_felt(denom.0 as i128),
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

            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => layouts::max_pool2d(
                config,
                region,
                values[..].try_into()?,
                *padding,
                *stride,
                *pool_dims,
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
            HybridOp::Softmax { scale, axes } => {
                layouts::softmax_axes(config, region, values[..].try_into()?, *scale, axes)?
            }
            HybridOp::RangeCheck(tol) => layouts::range_check_percent(
                config,
                region,
                values[..].try_into()?,
                tol.scale,
                tol.val,
            )?,
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

    fn out_scale(&self, in_scales: Vec<crate::Scale>) -> Result<crate::Scale, Box<dyn Error>> {
        let scale = match self {
            HybridOp::Greater { .. }
            | HybridOp::GreaterEqual { .. }
            | HybridOp::Less { .. }
            | HybridOp::LessEqual { .. }
            | HybridOp::ReduceArgMax { .. }
            | HybridOp::OneHot { .. }
            | HybridOp::ReduceArgMin { .. } => 0,
            HybridOp::Softmax { .. } => 2 * in_scales[0],
            HybridOp::Recip { output_scale, .. } => multiplier_to_scale(output_scale.0 as f64),
            _ => in_scales[0],
        };
        Ok(scale)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
