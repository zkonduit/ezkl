use halo2_proofs::circuit::Region;
use halo2curves::FieldExt;

use crate::{
    circuit::{layouts, utils},
    graph::scale_to_multiplier,
    tensor::{self, Tensor, TensorError, TensorType, ValTensor},
};

use super::{lookup::LookupOp, Op};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug)]
pub enum HybridOp {
    Mean {
        scale: usize,
        num_inputs: usize,
    },
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
}

impl<F: FieldExt + TensorType> Op<F> for HybridOp {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        match &self {
            HybridOp::Mean { scale, .. } => {
                Ok(tensor::ops::nonlinearities::mean(&inputs[0], *scale))
            }
            HybridOp::Max { axes, .. } => Ok(tensor::ops::max_axes(&inputs[0], axes)?),

            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
                ..
            } => tensor::ops::max_pool2d(&inputs[0], padding, stride, pool_dims),
            HybridOp::Min { axes, .. } => Ok(tensor::ops::min_axes(&inputs[0], axes)?),
        }
    }

    fn as_str(&self) -> &'static str {
        match &self {
            HybridOp::Mean { .. } => "MEAN",
            HybridOp::Max { .. } => "MAX",
            HybridOp::MaxPool2d { .. } => "MAXPOOL2D",
            HybridOp::Min { .. } => "MIN",
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: Option<&mut Region<F>>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn std::error::Error>> {
        Ok(match self {
            HybridOp::Mean { scale, .. } => Some(layouts::mean(
                config,
                region,
                values[..].try_into()?,
                *scale,
                offset,
            )?),
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => Some(layouts::max_pool2d(
                config,
                region,
                values[..].try_into()?,
                *padding,
                *stride,
                *pool_dims,
                offset,
            )?),
            HybridOp::Max { axes } => Some(layouts::max_axes(
                config,
                region,
                values[..].try_into()?,
                axes,
                offset,
            )?),
            HybridOp::Min { axes } => Some(layouts::min_axes(
                config,
                region,
                values[..].try_into()?,
                axes,
                offset,
            )?),
        })
    }

    fn out_scale(&self, in_scales: Vec<u32>, _: u32) -> u32 {
        in_scales[0]
    }

    fn has_3d_input(&self) -> bool {
        matches!(self, HybridOp::MaxPool2d { .. })
    }

    fn rescale(&self, inputs_scale: Vec<u32>, global_scale: u32) -> Box<dyn Op<F>> {
        let mult = scale_to_multiplier(inputs_scale[0] - global_scale);
        match self {
            HybridOp::Mean {
                scale: _,
                num_inputs,
            } => Box::new(HybridOp::Mean {
                scale: mult as usize,
                num_inputs: *num_inputs,
            }),
            _ => Box::new(self.clone()),
        }
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        match self {
            HybridOp::Max { .. } | HybridOp::Min { .. } | HybridOp::MaxPool2d { .. } => {
                Op::<F>::required_lookups(&LookupOp::ReLU { scale: 1 })
            }
            HybridOp::Mean { scale, num_inputs } => Op::<F>::required_lookups(&LookupOp::Div {
                denom: utils::F32((*scale * *num_inputs) as f32),
            }),
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
