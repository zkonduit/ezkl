use super::*;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::{
    circuit::{layouts, utils},
    fieldutils::{felt_to_i128, i128_to_felt},
    graph::scale_to_multiplier,
    tensor::{self, Tensor, TensorError, TensorType},
};

use super::Op;
use halo2curves::ff::PrimeField;

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum LookupOp {
    Div { denom: utils::F32 },
    ReLU { scale: usize },
    Sqrt { scales: (usize, usize) },
    Rsqrt { scales: (usize, usize) },
    Recip { scale: usize },
    LeakyReLU { scale: usize, slope: utils::F32 },
    Sigmoid { scales: (usize, usize) },
    Exp { scales: (usize, usize) },
    Tanh { scales: (usize, usize) },
    Erf { scales: (usize, usize) },
    GreaterThan { a: utils::F32 },
}

impl LookupOp {
    /// a value which is always in the table
    pub fn default_pair<F: PrimeField + TensorType + PartialOrd>(&self) -> (F, F) {
        let x = vec![i128_to_felt(0_i128)].into_iter().into();
        (
            <F as TensorType>::zero().unwrap(),
            Op::<F>::f(self, &[x]).unwrap().output[0],
        )
    }
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for LookupOp {
    fn as_any(&self) -> &dyn Any {
        self
    }
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, x: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = x[0].clone().map(|x| felt_to_i128(x));
        let res = match &self {
            LookupOp::GreaterThan { a } => Ok(tensor::ops::nonlinearities::greater_than(
                &x,
                f32::from(*a).into(),
            )),
            LookupOp::Div { denom } => Ok(tensor::ops::nonlinearities::const_div(
                &x,
                f32::from(*denom).into(),
            )),
            LookupOp::Recip { scale } => Ok(tensor::ops::nonlinearities::recip(&x, *scale as u32)),
            LookupOp::ReLU { scale } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x, *scale, 0_f64))
            }

            LookupOp::LeakyReLU { scale, slope } => Ok(tensor::ops::nonlinearities::leakyrelu(
                &x,
                *scale,
                slope.0.into(),
            )),
            LookupOp::Sigmoid { scales } => {
                Ok(tensor::ops::nonlinearities::sigmoid(&x, scales.0, scales.1))
            }
            LookupOp::Sqrt { scales } => {
                Ok(tensor::ops::nonlinearities::sqrt(&x, scales.0, scales.1))
            }
            LookupOp::Rsqrt { scales } => {
                Ok(tensor::ops::nonlinearities::rsqrt(&x, scales.0, scales.1))
            }
            LookupOp::Tanh { scales } => {
                Ok(tensor::ops::nonlinearities::tanh(&x, scales.0, scales.1))
            }
            LookupOp::Erf { scales } => {
                Ok(tensor::ops::nonlinearities::erffunc(&x, scales.0, scales.1))
            }
            LookupOp::Exp { scales } => {
                Ok(tensor::ops::nonlinearities::exp(&x, scales.0, scales.1))
            }
        }?;

        let output = res.map(|x| i128_to_felt(x));

        Ok(ForwardResult {
            output,
            intermediate_lookups: vec![],
        })
    }

    /// Returns the name of the operation
    fn as_string(&self) -> String {
        let name = match self {
            LookupOp::GreaterThan { .. } => "GREATER_THAN",
            LookupOp::Recip { .. } => "RECIP",
            LookupOp::Div { .. } => "DIV",
            LookupOp::ReLU { .. } => "RELU",
            LookupOp::LeakyReLU { .. } => "LEAKY_RELU",
            LookupOp::Sigmoid { .. } => "SIGMOID",
            LookupOp::Sqrt { .. } => "SQRT",
            LookupOp::Tanh { .. } => "TANH",
            LookupOp::Erf { .. } => "ERF",
            LookupOp::Rsqrt { .. } => "RSQRT",
            LookupOp::Exp { .. } => "EXP",
        };
        name.into()
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(Some(layouts::nonlinearity(
            config,
            region,
            values[..].try_into()?,
            self,
        )?))
    }

    fn rescale(&self, inputs_scale: Vec<u32>, global_scale: u32) -> Box<dyn Op<F>> {
        match self {
            LookupOp::Recip { .. } => Box::new(LookupOp::Recip {
                scale: scale_to_multiplier(inputs_scale[0] + global_scale) as usize,
            }),
            LookupOp::Div { denom } => Box::new(LookupOp::Div {
                denom: crate::circuit::utils::F32(
                    ((denom.0 as f64) * scale_to_multiplier(inputs_scale[0] - global_scale)) as f32,
                ),
            }),
            LookupOp::ReLU { .. } => Box::new(LookupOp::ReLU {
                scale: scale_to_multiplier(inputs_scale[0] - global_scale) as usize,
            }),
            LookupOp::LeakyReLU { slope, .. } => Box::new(LookupOp::LeakyReLU {
                scale: scale_to_multiplier(inputs_scale[0] - global_scale) as usize,
                slope: *slope,
            }),
            LookupOp::Sigmoid { .. } => Box::new(LookupOp::Sigmoid {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Sqrt { .. } => Box::new(LookupOp::Sqrt {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Rsqrt { .. } => Box::new(LookupOp::Rsqrt {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Tanh { .. } => Box::new(LookupOp::Tanh {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Erf { .. } => Box::new(LookupOp::Erf {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Exp { .. } => Box::new(LookupOp::Exp {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::GreaterThan { a } => Box::new(LookupOp::GreaterThan {
                a: utils::F32(((a.0 as f64) * scale_to_multiplier(inputs_scale[0])) as f32),
            }),
        }
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        vec![self.clone()]
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
