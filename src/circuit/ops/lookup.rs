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
    Div {
        denom: utils::F32,
    },
    ReLU {
        scale: usize,
    },
    Max {
        scales: (usize, usize),
        a: utils::F32,
    },
    Min {
        scales: (usize, usize),
        a: utils::F32,
    },
    Sqrt {
        scales: (usize, usize),
    },
    Rsqrt {
        scales: (usize, usize),
    },
    Recip {
        scale: usize,
    },
    LeakyReLU {
        scale: usize,
        slope: utils::F32,
    },
    Sigmoid {
        scales: (usize, usize),
    },
    Ln {
        scales: (usize, usize),
    },
    Exp {
        scales: (usize, usize),
    },
    Cos {
        scales: (usize, usize),
    },
    ACos {
        scales: (usize, usize),
    },
    Cosh {
        scales: (usize, usize),
    },
    ACosh {
        scales: (usize, usize),
    },
    Sin {
        scales: (usize, usize),
    },
    ASin {
        scales: (usize, usize),
    },
    Sinh {
        scales: (usize, usize),
    },
    ASinh {
        scales: (usize, usize),
    },
    Tan {
        scales: (usize, usize),
    },
    ATan {
        scales: (usize, usize),
    },
    Tanh {
        scales: (usize, usize),
    },
    ATanh {
        scales: (usize, usize),
    },
    Erf {
        scales: (usize, usize),
    },
    GreaterThan {
        a: utils::F32,
    },
    LessThan {
        a: utils::F32,
    },
    Sign,
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
    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any {
        self
    }
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, x: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let x = x[0].clone().map(|x| felt_to_i128(x));
        let res = match &self {
            LookupOp::Max { scales, a } => Ok(tensor::ops::nonlinearities::max(
                &x,
                scales.0,
                scales.1,
                a.0.into(),
            )),
            LookupOp::Min { scales, a } => Ok(tensor::ops::nonlinearities::min(
                &x,
                scales.0,
                scales.1,
                a.0.into(),
            )),
            LookupOp::Sign => Ok(tensor::ops::nonlinearities::sign(&x)),
            LookupOp::LessThan { a } => Ok(tensor::ops::nonlinearities::less_than(
                &x,
                f32::from(*a).into(),
            )),
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

            LookupOp::LeakyReLU { scale, slope: a } => Ok(tensor::ops::nonlinearities::leakyrelu(
                &x,
                *scale,
                a.0.into(),
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
            LookupOp::Erf { scales } => {
                Ok(tensor::ops::nonlinearities::erffunc(&x, scales.0, scales.1))
            }
            LookupOp::Exp { scales } => {
                Ok(tensor::ops::nonlinearities::exp(&x, scales.0, scales.1))
            }
            LookupOp::Ln { scales } => Ok(tensor::ops::nonlinearities::ln(&x, scales.0, scales.1)),
            LookupOp::Cos { scales } => {
                Ok(tensor::ops::nonlinearities::cos(&x, scales.0, scales.1))
            }
            LookupOp::ACos { scales } => {
                Ok(tensor::ops::nonlinearities::acos(&x, scales.0, scales.1))
            }
            LookupOp::Cosh { scales } => {
                Ok(tensor::ops::nonlinearities::cosh(&x, scales.0, scales.1))
            }
            LookupOp::ACosh { scales } => {
                Ok(tensor::ops::nonlinearities::acosh(&x, scales.0, scales.1))
            }
            LookupOp::Sin { scales } => {
                Ok(tensor::ops::nonlinearities::sin(&x, scales.0, scales.1))
            }
            LookupOp::ASin { scales } => {
                Ok(tensor::ops::nonlinearities::asin(&x, scales.0, scales.1))
            }
            LookupOp::Sinh { scales } => {
                Ok(tensor::ops::nonlinearities::sinh(&x, scales.0, scales.1))
            }
            LookupOp::ASinh { scales } => {
                Ok(tensor::ops::nonlinearities::asinh(&x, scales.0, scales.1))
            }
            LookupOp::Tan { scales } => {
                Ok(tensor::ops::nonlinearities::tan(&x, scales.0, scales.1))
            }
            LookupOp::ATan { scales } => {
                Ok(tensor::ops::nonlinearities::atan(&x, scales.0, scales.1))
            }
            LookupOp::ATanh { scales } => {
                Ok(tensor::ops::nonlinearities::atanh(&x, scales.0, scales.1))
            }
            LookupOp::Tanh { scales } => {
                Ok(tensor::ops::nonlinearities::tanh(&x, scales.0, scales.1))
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
        
        match self {
            LookupOp::Max { scales, a } => format!("MAX w/ {:?} /t {}", scales, a),
            LookupOp::Min { scales, a } => format!("MIN w/ {:?} /t {}", scales, a),
            LookupOp::Sign => "SIGN".into(),
            LookupOp::GreaterThan { .. } => "GREATER_THAN".into(),
            LookupOp::LessThan { .. } => "LESS_THAN".into(),
            LookupOp::Recip { scale, .. } => format!("RECIP w/ {}", scale),
            LookupOp::Div { denom, .. } => format!("DIV w/ {}", denom),
            LookupOp::Ln { scales } => format!("LN w/ {:?}", scales),
            LookupOp::ReLU { scale, .. } => format!("RELU w/ scale {}", scale),
            LookupOp::LeakyReLU { scale, slope: a } => format!("L_RELU w/ {} /s {}", scale, a),
            LookupOp::Sigmoid { scales } => format!("SIGMOID w/ {:?}", scales),
            LookupOp::Sqrt { scales } => format!("SQRT w/ {:?}", scales),
            LookupOp::Erf { scales } => format!("ERF w/ {:?}", scales),
            LookupOp::Rsqrt { scales } => format!("RSQRT w/ {:?}", scales),
            LookupOp::Exp { scales } => format!("EXP w/ {:?}", scales),
            LookupOp::Tan { scales } => format!("TAN w/ {:?}", scales),
            LookupOp::ATan { scales } => format!("ATAN w/ {:?}", scales),
            LookupOp::Tanh { scales } => format!("TANH w/ {:?}", scales),
            LookupOp::ATanh { scales } => format!("ATANH w/ {:?}", scales),
            LookupOp::Cos { scales } => format!("COS w/ {:?}", scales),
            LookupOp::ACos { scales } => format!("ACOS w/ {:?}", scales),
            LookupOp::Cosh { scales } => format!("COSH w/ {:?}", scales),
            LookupOp::ACosh { scales } => format!("ACOSH w/ {:?}", scales),
            LookupOp::Sin { scales } => format!("SIN w/ {:?}", scales),
            LookupOp::ASin { scales } => format!("ASIN w/ {:?}", scales),
            LookupOp::Sinh { scales } => format!("SINH w/ {:?}", scales),
            LookupOp::ASinh { scales } => format!("ASINH w/ {:?}", scales),
        }
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

    /// Returns the scale of the output of the operation.
    fn out_scale(&self, _: Vec<u32>, global_scale: u32) -> u32 {
        match self {
            LookupOp::Sign | LookupOp::GreaterThan { .. } => 0,
            _ => global_scale,
        }
    }

    fn rescale(&self, inputs_scale: Vec<u32>, global_scale: u32) -> Box<dyn Op<F>> {
        match self {
            LookupOp::Sign => Box::new(LookupOp::Sign),
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
            LookupOp::Max { a, .. } => Box::new(LookupOp::Max {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
                a: *a,
            }),
            LookupOp::Min { a, .. } => Box::new(LookupOp::Min {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
                a: *a,
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
            LookupOp::Cos { .. } => Box::new(LookupOp::Cos {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::ACos { .. } => Box::new(LookupOp::ACos {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Cosh { .. } => Box::new(LookupOp::Cosh {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::ACosh { .. } => Box::new(LookupOp::ACosh {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Sin { .. } => Box::new(LookupOp::Sin {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::ASin { .. } => Box::new(LookupOp::ASin {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Sinh { .. } => Box::new(LookupOp::Sinh {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::ASinh { .. } => Box::new(LookupOp::ASinh {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::Tan { .. } => Box::new(LookupOp::Tan {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::ATan { .. } => Box::new(LookupOp::ATan {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::ATanh { .. } => Box::new(LookupOp::ATanh {
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
            LookupOp::Ln { .. } => Box::new(LookupOp::Ln {
                scales: (
                    scale_to_multiplier(inputs_scale[0]) as usize,
                    scale_to_multiplier(global_scale) as usize,
                ),
            }),
            LookupOp::GreaterThan { a } => Box::new(LookupOp::GreaterThan {
                a: utils::F32(((a.0 as f64) * scale_to_multiplier(inputs_scale[0])) as f32),
            }),
            LookupOp::LessThan { a } => Box::new(LookupOp::LessThan {
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
