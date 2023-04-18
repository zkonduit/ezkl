use super::*;
use halo2_proofs::circuit::Region;
use halo2curves::FieldExt;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::{
    circuit::{layouts, utils},
    fieldutils::i128_to_felt,
    tensor::{self, Tensor, TensorError, TensorType},
};

use super::Op;

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize, Serialize)]
pub enum LookupOp {
    Div { denom: utils::F32 },
    ReLU { scale: usize },
    Sqrt { scales: (usize, usize) },
    LeakyReLU { scale: usize, slope: utils::F32 },
    Sigmoid { scales: (usize, usize) },
    Tanh { scales: (usize, usize) },
    Erf { scales: (usize, usize) },
}

impl LookupOp {
    /// a value which is always in the table
    pub fn default_pair<F: FieldExt + TensorType>(&self) -> (F, F) {
        let x = vec![0_i128].into_iter().into();
        (
            <F as TensorType>::zero().unwrap(),
            i128_to_felt(Op::<F>::f(self, &[x]).unwrap()[0]),
        )
    }
}

impl<F: FieldExt + TensorType> Op<F> for LookupOp {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, x: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        match &self {
            LookupOp::Div { denom } => Ok(tensor::ops::nonlinearities::const_div(
                &x[0],
                f32::from(*denom),
            )),
            LookupOp::ReLU { scale } => {
                Ok(tensor::ops::nonlinearities::leakyrelu(&x[0], *scale, 0_f32))
            }
            LookupOp::LeakyReLU { scale, slope } => Ok(tensor::ops::nonlinearities::leakyrelu(
                &x[0], *scale, slope.0,
            )),
            LookupOp::Sigmoid { scales } => Ok(tensor::ops::nonlinearities::sigmoid(
                &x[0], scales.0, scales.1,
            )),
            LookupOp::Sqrt { scales } => {
                Ok(tensor::ops::nonlinearities::sqrt(&x[0], scales.0, scales.1))
            }
            LookupOp::Tanh { scales } => {
                Ok(tensor::ops::nonlinearities::tanh(&x[0], scales.0, scales.1))
            }
            LookupOp::Erf { scales } => Ok(tensor::ops::nonlinearities::erffunc(
                &x[0], scales.0, scales.1,
            )),
        }
    }

    /// Returns the name of the operation
    fn as_str(&self) -> &'static str {
        match self {
            LookupOp::Div { .. } => "DIV",
            LookupOp::ReLU { .. } => "RELU",
            LookupOp::LeakyReLU { .. } => "LEAKY_RELU",
            LookupOp::Sigmoid { .. } => "SIGMOID",
            LookupOp::Sqrt { .. } => "SQRT",
            LookupOp::Tanh { .. } => "TANH",
            LookupOp::Erf { .. } => "ERF",
        }
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut Region<F>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(Some(layouts::nonlinearity(
            config,
            region,
            values[..].try_into()?,
            self,
            offset,
        )?))
    }
}
