use halo2curves::FieldExt;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    circuit::layouts,
    tensor::{self, Tensor, TensorError, TensorType},
};

use super::Op;

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum HybridOp {
    Mean {
        scale: usize,
    },
    Max,
    MaxPool2d {
        padding: (usize, usize),
        stride: (usize, usize),
        pool_dims: (usize, usize),
    },
    InstanceNorm2d {
        epsilon: crate::circuit::utils::F32,
    },
    Min,
    PReLU {
        scale: usize,
        slopes: Vec<crate::circuit::utils::F32>,
    },
}

impl<F: FieldExt + TensorType> Op<F> for HybridOp {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        match &self {
            HybridOp::Mean { scale } => Ok(tensor::ops::nonlinearities::mean(&inputs[0], *scale)),
            HybridOp::Max => Ok(Tensor::new(
                Some(&[inputs[0].clone().into_iter().max().unwrap()]),
                &[1],
            )?),
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => tensor::ops::max_pool2d(&inputs[0], padding, stride, pool_dims),
            HybridOp::InstanceNorm2d { epsilon } => Ok(tensor::ops::nonlinearities::instance_norm(
                inputs.to_vec().try_into().unwrap(),
                epsilon.0,
            )),
            HybridOp::Min => Ok(Tensor::new(
                Some(&[inputs[0].clone().into_iter().min().unwrap()]),
                &[1],
            )?),
            HybridOp::PReLU { scale, slopes } => Ok(tensor::ops::nonlinearities::prelu(
                &inputs[0],
                *scale,
                &slopes.iter().map(|e| e.0).collect_vec(),
            )),
        }
    }

    fn as_str(&self) -> &'static str {
        match &self {
            HybridOp::Mean { .. } => "MEAN",
            HybridOp::Max => "MAX",
            HybridOp::MaxPool2d { .. } => "MAXPOOL2D",
            HybridOp::InstanceNorm2d { .. } => "INSTANCENORM",
            HybridOp::Min => "MIN",
            HybridOp::PReLU { .. } => "PRELU",
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut halo2_proofs::circuit::Region<F>,
        values: &[tensor::ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<tensor::ValTensor<F>>, Box<dyn std::error::Error>> {
        Ok(match self {
            HybridOp::PReLU { scale, .. } => Some(layouts::prelu(
                config,
                region,
                values[..].try_into()?,
                scale.clone(),
                offset,
            )?),
            HybridOp::Mean { scale, .. } => Some(layouts::mean(
                config,
                region,
                values[..].try_into()?,
                scale.clone(),
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
                padding.clone(),
                stride.clone(),
                pool_dims.clone(),
                offset,
            )?),
            HybridOp::Max => Some(layouts::max(
                config,
                region,
                values[..].try_into()?,
                offset,
            )?),
            HybridOp::Min => Some(layouts::min(
                config,
                region,
                values[..].try_into()?,
                offset,
            )?),
            HybridOp::InstanceNorm2d { epsilon } => Some(layouts::instance_norm(
                config,
                region,
                values[..].try_into()?,
                1,
                epsilon.0 as u64,
                offset,
            )?),
        })
    }

    fn out_scale(&self, in_scales: Vec<u32>, _: u32) -> u32 {
        in_scales[0]
    }

    fn out_dims(&self, in_dims: Vec<Vec<usize>>) -> Vec<usize> {
        match self {
            HybridOp::Mean { .. } => vec![1],
            HybridOp::Max => vec![1],
            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => {
                let (out_channels, kernel_height, kernel_width) =
                    (in_dims[0][0], pool_dims.0, pool_dims.1);

                let (padding_h, padding_w, stride_h, stride_w) =
                    (padding.0, padding.1, stride.0, stride.1);

                let input_height = in_dims[0][1];
                let input_width = in_dims[0][2];

                let out_height = (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                let out_width = (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                vec![out_channels, out_height, out_width]
            }
            HybridOp::InstanceNorm2d { .. } => in_dims[0].clone(),
            HybridOp::Min => vec![1],
            HybridOp::PReLU { .. } => in_dims[0].clone(),
        }
    }

    fn has_3d_input(&self) -> bool {
        match self {
            HybridOp::MaxPool2d { .. } => true,
            HybridOp::InstanceNorm2d { .. } => true,
            _ => false,
        }
    }
}
