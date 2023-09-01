use super::*;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::{
    circuit::{layouts, utils},
    fieldutils::{felt_to_i128, i128_to_felt},
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
    ReLU,
    Max {
        scales: (usize, usize),
        a: utils::F32,
    },
    Min {
        scales: (usize, usize),
        a: utils::F32,
    },
    Sqrt {
        scale: utils::F32,
    },
    Rsqrt {
        scale: utils::F32,
    },
    Recip {
        scale: utils::F32,
    },
    LeakyReLU {
        slope: utils::F32,
    },
    Sigmoid {
        scale: utils::F32,
    },
    Ln {
        scale: utils::F32,
    },
    Exp {
        scale: utils::F32,
    },
    Cos {
        scale: utils::F32,
    },
    ACos {
        scale: utils::F32,
    },
    Cosh {
        scale: utils::F32,
    },
    ACosh {
        scale: utils::F32,
    },
    Sin {
        scale: utils::F32,
    },
    ASin {
        scale: utils::F32,
    },
    Sinh {
        scale: utils::F32,
    },
    ASinh {
        scale: utils::F32,
    },
    Tan {
        scale: utils::F32,
    },
    ATan {
        scale: utils::F32,
    },
    Tanh {
        scale: utils::F32,
    },
    ATanh {
        scale: utils::F32,
    },
    Erf {
        scale: utils::F32,
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

    /// Returns the range of values that can be represented by the table
    pub fn bit_range(&self, allocated_bits: usize, num_blinding_factors: usize) -> (i128, i128) {
        let base = 2i128;
        let blinding_offset = (num_blinding_factors / 2) as i128;
        (
            -base.pow(allocated_bits as u32) + blinding_offset,
            base.pow(allocated_bits as u32) - blinding_offset,
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
            LookupOp::Recip { scale } => Ok(tensor::ops::nonlinearities::recip(&x, scale.into())),
            LookupOp::ReLU => Ok(tensor::ops::nonlinearities::leakyrelu(&x, 0_f64)),

            LookupOp::LeakyReLU { slope: a } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x, a.0.into()))
            }
            LookupOp::Sigmoid { scale } => {
                Ok(tensor::ops::nonlinearities::sigmoid(&x, scale.into()))
            }
            LookupOp::Sqrt { scale } => Ok(tensor::ops::nonlinearities::sqrt(&x, scale.into())),
            LookupOp::Rsqrt { scale } => Ok(tensor::ops::nonlinearities::rsqrt(&x, scale.into())),
            LookupOp::Erf { scale } => Ok(tensor::ops::nonlinearities::erffunc(&x, scale.into())),
            LookupOp::Exp { scale } => Ok(tensor::ops::nonlinearities::exp(&x, scale.into())),
            LookupOp::Ln { scale } => Ok(tensor::ops::nonlinearities::ln(&x, scale.into())),
            LookupOp::Cos { scale } => Ok(tensor::ops::nonlinearities::cos(&x, scale.into())),
            LookupOp::ACos { scale } => Ok(tensor::ops::nonlinearities::acos(&x, scale.into())),
            LookupOp::Cosh { scale } => Ok(tensor::ops::nonlinearities::cosh(&x, scale.into())),
            LookupOp::ACosh { scale } => Ok(tensor::ops::nonlinearities::acosh(&x, scale.into())),
            LookupOp::Sin { scale } => Ok(tensor::ops::nonlinearities::sin(&x, scale.into())),
            LookupOp::ASin { scale } => Ok(tensor::ops::nonlinearities::asin(&x, scale.into())),
            LookupOp::Sinh { scale } => Ok(tensor::ops::nonlinearities::sinh(&x, scale.into())),
            LookupOp::ASinh { scale } => Ok(tensor::ops::nonlinearities::asinh(&x, scale.into())),
            LookupOp::Tan { scale } => Ok(tensor::ops::nonlinearities::tan(&x, scale.into())),
            LookupOp::ATan { scale } => Ok(tensor::ops::nonlinearities::atan(&x, scale.into())),
            LookupOp::ATanh { scale } => Ok(tensor::ops::nonlinearities::atanh(&x, scale.into())),
            LookupOp::Tanh { scale } => Ok(tensor::ops::nonlinearities::tanh(&x, scale.into())),
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
            LookupOp::Ln { scale } => format!("LN w/ {:?}", scale),
            LookupOp::ReLU => "RELU".to_string(),
            LookupOp::LeakyReLU { slope: a } => format!("L_RELU /s {}", a),
            LookupOp::Sigmoid { scale } => format!("SIGMOID w/ {:?}", scale),
            LookupOp::Sqrt { scale } => format!("SQRT w/ {:?}", scale),
            LookupOp::Erf { scale } => format!("ERF w/ {:?}", scale),
            LookupOp::Rsqrt { scale } => format!("RSQRT w/ {:?}", scale),
            LookupOp::Exp { scale } => format!("EXP w/ {:?}", scale),
            LookupOp::Tan { scale } => format!("TAN w/ {:?}", scale),
            LookupOp::ATan { scale } => format!("ATAN w/ {:?}", scale),
            LookupOp::Tanh { scale } => format!("TANH w/ {:?}", scale),
            LookupOp::ATanh { scale } => format!("ATANH w/ {:?}", scale),
            LookupOp::Cos { scale } => format!("COS w/ {:?}", scale),
            LookupOp::ACos { scale } => format!("ACOS w/ {:?}", scale),
            LookupOp::Cosh { scale } => format!("COSH w/ {:?}", scale),
            LookupOp::ACosh { scale } => format!("ACOSH w/ {:?}", scale),
            LookupOp::Sin { scale } => format!("SIN w/ {:?}", scale),
            LookupOp::ASin { scale } => format!("ASIN w/ {:?}", scale),
            LookupOp::Sinh { scale } => format!("SINH w/ {:?}", scale),
            LookupOp::ASinh { scale } => format!("ASINH w/ {:?}", scale),
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
    fn out_scale(&self, inputs_scale: Vec<u32>) -> u32 {
        match self {
            LookupOp::Sign | LookupOp::GreaterThan { .. } | LookupOp::LessThan { .. } => 0,
            _ => inputs_scale[0],
        }
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        vec![self.clone()]
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}
