use std::any::Any;

use crate::{
    circuit::{self, layouts, Tolerance},
    graph::scale_to_multiplier,
    tensor::{self, Tensor, TensorError, TensorType, ValTensor},
};
use halo2_proofs::circuit::Region;
use serde::{Deserialize, Serialize};

use super::{lookup::LookupOp, Op};
use halo2curves::ff::PrimeField;
// import run args from model

#[allow(missing_docs)]
/// An enum representing the operations that consist of both lookups and arithmetic operations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HybridOp {
    Max {
        axes: Vec<usize>,
    },
    MaxPool2d {
        padding: (usize, usize),
        stride: (usize, usize),
        pool_dims: (usize, usize),
    },
    Min {
        axes: Vec<usize>,
    },
    Softmax {
        scales: (usize, usize),
    },
    RangeCheck(Tolerance),
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for HybridOp {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        match &self {
            HybridOp::Max { axes, .. } => Ok(tensor::ops::max_axes(&inputs[0], axes)?),

            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
                ..
            } => tensor::ops::max_pool2d(&inputs[0], padding, stride, pool_dims),
            HybridOp::Min { axes, .. } => Ok(tensor::ops::min_axes(&inputs[0], axes)?),
            HybridOp::Softmax { scales } => Ok(tensor::ops::nonlinearities::multi_dim_softmax(
                &inputs[0], scales.0, scales.1,
            )),
            HybridOp::RangeCheck(..) => Ok(inputs[0].clone()),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            HybridOp::Max { .. } => "MAX",
            HybridOp::MaxPool2d { .. } => "MAXPOOL2D",
            HybridOp::Min { .. } => "MIN",
            HybridOp::Softmax { .. } => "SOFTMAX",
            HybridOp::RangeCheck(..) => "RANGECHECK",
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut Option<&mut Region<F>>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn std::error::Error>> {
        Ok(Some(match self {
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
                offset,
            )?,
            HybridOp::Max { axes } => {
                layouts::max_axes(config, region, values[..].try_into()?, axes, offset)?
            }
            HybridOp::Min { axes } => {
                layouts::min_axes(config, region, values[..].try_into()?, axes, offset)?
            }
            HybridOp::Softmax { scales } => layouts::multi_dim_softmax(
                config,
                region,
                values[..].try_into()?,
                scales.0,
                scales.1,
                offset,
            )?,
            HybridOp::RangeCheck(tol) => match tol {
                Tolerance::Abs { val } => layouts::range_check(
                    config,
                    region,
                    values[..].try_into()?,
                    offset,
                    *val as i32,
                )?,
                Tolerance::Percentage { val, scale } => layouts::range_check_percent(
                    config,
                    region,
                    values[..].try_into()?,
                    *scale,
                    offset,
                    *val,
                )?,
            },
        }))
    }

    fn out_scale(&self, in_scales: Vec<u32>, global_scale: u32) -> u32 {
        match self {
            HybridOp::Softmax { .. } => 2 * global_scale,
            _ => in_scales[0],
        }
    }

    fn rescale(&self, input_scales: Vec<u32>, global_scale: u32) -> Box<dyn Op<F>> {
        match self {
            HybridOp::Softmax { .. } => Box::new(HybridOp::Softmax {
                scales: (
                    scale_to_multiplier(input_scales[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            _ => Box::new(self.clone()),
        }
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        match self {
            HybridOp::Max { .. } | HybridOp::Min { .. } | HybridOp::MaxPool2d { .. } => {
                Op::<F>::required_lookups(&LookupOp::ReLU { scale: 1 })
            }
            HybridOp::Softmax { scales } => {
                vec![
                    LookupOp::Exp { scales: *scales },
                    LookupOp::Recip {
                        scale: scales.1.pow(2),
                    },
                ]
            }
            HybridOp::RangeCheck(tol) => match tol {
                Tolerance::Percentage { val, scale } => {
                    let scale = scale.pow(2);
                    vec![
                        LookupOp::Recip { scale },
                        LookupOp::GreaterThan {
                            a: circuit::utils::F32((val * scale as f32) / 100.0),
                        },
                    ]
                }
                __ => vec![],
            },
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
