use std::any::Any;

use crate::{
    circuit::{self, layouts, Tolerance},
    fieldutils::{felt_to_i128, i128_to_felt},
    graph::scale_to_multiplier,
    tensor::{self, Tensor, TensorError, TensorType, ValTensor},
};
use serde::{Deserialize, Serialize};

use super::{lookup::LookupOp, region::RegionCtx, ForwardResult, Op};
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
    fn f(&self, inputs: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = inputs[0].clone().map(|x| felt_to_i128(x));

        let (res, intermediate_lookups) = match &self {
            HybridOp::Max { axes, .. } => (tensor::ops::max_axes(&x, axes)?, vec![]),
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
                ..
            } => (
                tensor::ops::max_pool2d(&x, padding, stride, pool_dims)?,
                vec![],
            ),
            HybridOp::Min { axes, .. } => (tensor::ops::min_axes(&x, axes)?, vec![]),
            HybridOp::Softmax { scales } => {
                tensor::ops::nonlinearities::multi_dim_softmax(&x, scales.0, scales.1)
            }
            HybridOp::RangeCheck(..) => (x.clone(), vec![]),
        };

        // convert back to felt
        let output = res.map(|x| i128_to_felt(x));

        Ok(ForwardResult {
            output,
            intermediate_lookups,
        })
    }

    fn as_string(&self) -> String {
        let name = match self {
            HybridOp::Max { .. } => "MAX",
            HybridOp::MaxPool2d { .. } => "MAXPOOL2D",
            HybridOp::Min { .. } => "MIN",
            HybridOp::Softmax { .. } => "SOFTMAX",
            HybridOp::RangeCheck(..) => "RANGECHECK",
        };
        name.into()
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
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
            )?,
            HybridOp::Max { axes } => {
                layouts::max_axes(config, region, values[..].try_into()?, axes)?
            }
            HybridOp::Min { axes } => {
                layouts::min_axes(config, region, values[..].try_into()?, axes)?
            }
            HybridOp::Softmax { scales } => layouts::multi_dim_softmax(
                config,
                region,
                values[..].try_into()?,
                scales.0,
                scales.1,
            )?,
            HybridOp::RangeCheck(tol) => layouts::range_check_percent(
                config,
                region,
                values[..].try_into()?,
                tol.scales.0,
                tol.scales.1,
                tol.val,
            )?,
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
            HybridOp::RangeCheck(tol) => {
                let mut lookups = vec![];
                if tol.val > 0.0 {
                    let scale = tol.scales.0 * tol.scales.1;
                    lookups.extend([
                        LookupOp::Recip { scale },
                        LookupOp::GreaterThan {
                            a: circuit::utils::F32((tol.val * scale as f32) / 100.0),
                        },
                    ]);
                }
                lookups
            }
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
