use halo2curves::FieldExt;
use itertools::Itertools;

use crate::{
    circuit::{layouts, utils},
    graph::scale_to_multiplier,
    tensor::{self, Tensor, TensorError, TensorType, ValTensor},
};

use super::{lookup::LookupOp, Op};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug)]
pub enum HybridOp<F: FieldExt + TensorType> {
    Mean {
        scale: usize,
        num_inputs: usize,
    },
    Max,
    EltWiseMax {
        a: Option<ValTensor<F>>,
    },
    MaxPool2d {
        padding: (usize, usize),
        stride: (usize, usize),
        pool_dims: (usize, usize),
    },
    Min,
    PReLU {
        scale: usize,
        slopes: Vec<crate::circuit::utils::F32>,
    },
    Greater {
        a: Option<ValTensor<F>>,
    },
}

impl<F: FieldExt + TensorType> Op<F> for HybridOp<F> {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, inputs: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        match &self {
            HybridOp::Mean { scale, .. } => {
                Ok(tensor::ops::nonlinearities::mean(&inputs[0], *scale))
            }
            HybridOp::Greater { .. } => Ok(inputs[0]
                .iter()
                .zip(inputs[1].iter())
                .map(|(a, b)| if a > b { 1 } else { 0 })
                .collect_vec()
                .into_iter()
                .into()),
            HybridOp::Max => Ok(Tensor::new(
                Some(&[inputs[0].clone().into_iter().max().unwrap()]),
                &[1],
            )?),

            HybridOp::EltWiseMax { .. } => Ok(Tensor::new(
                Some(&[inputs[0].clone().into_iter().max().unwrap()]),
                &inputs[0].dims(),
            )?),

            HybridOp::MaxPool2d {
                padding,
                stride,
                pool_dims,
            } => tensor::ops::max_pool2d(&inputs[0], padding, stride, pool_dims),
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
            HybridOp::EltWiseMax { .. } => "ELTWISEMAX",
            HybridOp::Mean { .. } => "MEAN",
            HybridOp::Max => "MAX",
            HybridOp::Greater { .. } => "GREATER",
            HybridOp::MaxPool2d { .. } => "MAXPOOL2D",
            HybridOp::Min => "MIN",
            HybridOp::PReLU { .. } => "PRELU",
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: Option<&mut halo2_proofs::circuit::Region<F>>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn std::error::Error>> {
        let mut values = values.to_vec();
        Ok(match self {
            HybridOp::PReLU { scale, .. } => Some(layouts::prelu(
                config,
                region,
                values[..].try_into()?,
                *scale,
                offset,
            )?),
            HybridOp::EltWiseMax { a } => {
                if let Some(a) = a {
                    values.push(a.clone());
                }
                todo!("EltWiseMax")
            }
            HybridOp::Greater { a } => {
                if let Some(a) = a {
                    values.push(a.clone());
                }
                todo!()
            }
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
            HybridOp::PReLU { scale: _, slopes } => Box::new(HybridOp::PReLU {
                scale: mult as usize,
                slopes: slopes.to_vec(),
            }),
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

    fn required_lookup(&self) -> Option<LookupOp> {
        match self {
            HybridOp::PReLU { scale, .. } => Some(LookupOp::ReLU { scale: *scale }),
            HybridOp::Max
            | HybridOp::Min
            | HybridOp::MaxPool2d { .. }
            | HybridOp::Greater { .. }
            | HybridOp::EltWiseMax { .. } => Some(LookupOp::ReLU { scale: 1 }),
            HybridOp::Mean { scale, num_inputs } => Some(LookupOp::Div {
                denom: utils::F32((*scale * *num_inputs) as f32),
            }),
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
