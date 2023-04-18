use std::error::Error;

use halo2_proofs::circuit::Region;
use halo2curves::FieldExt;
use serde::Serialize;

use crate::tensor::{Tensor, TensorError, TensorType, ValTensor};

///
pub mod base;
///
pub mod hybrid;
/// Layouts for specific functions (composed of base ops)
pub mod layouts;
///
pub mod lookup;
///
pub mod poly;

///
pub trait Op<F: FieldExt + TensorType>:
    erased_serde::Serialize + std::fmt::Debug + Send + Sync
{
    ///
    fn f(self: &Self, x: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError>;
    ///
    fn as_str(self: &Self) -> &'static str;

    ///
    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut Region<F>,
        values: &[ValTensor<F>],
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>>;
}

///
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Input;

impl<F: FieldExt + TensorType> Op<F> for Input {
    fn f(self: &Self, x: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        Ok(x[0].clone())
    }

    fn as_str(self: &Self) -> &'static str {
        "Input"
    }
    fn layout(
        &self,
        _config: &mut crate::circuit::BaseConfig<F>,
        _region: &mut Region<F>,
        _values: &[ValTensor<F>],
        _offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(None)
    }
}

///
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Const;

impl<F: FieldExt + TensorType> Op<F> for Const {
    fn f(self: &Self, x: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        Ok(x[0].clone())
    }

    fn as_str(self: &Self) -> &'static str {
        "Const"
    }
    fn layout(
        &self,
        _config: &mut crate::circuit::BaseConfig<F>,
        _region: &mut Region<F>,
        _values: &[ValTensor<F>],
        _offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(None)
    }
}

///
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Unknown(pub String);

impl<F: FieldExt + TensorType> Op<F> for Unknown {
    fn f(self: &Self, _: &[Tensor<i128>]) -> Result<Tensor<i128>, TensorError> {
        Err(TensorError::WrongMethod)
    }

    fn as_str(self: &Self) -> &'static str {
        "UnknownOp: "
    }
    fn layout(
        &self,
        _config: &mut crate::circuit::BaseConfig<F>,
        _region: &mut Region<F>,
        _values: &[ValTensor<F>],
        _offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Err(Box::new(super::CircuitError::UnsupportedOp))
    }
}
