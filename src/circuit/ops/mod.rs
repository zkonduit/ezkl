use std::any::Any;

use serde::{Deserialize, Serialize};

use crate::{
    graph::quantize_tensor,
    tensor::{self, Tensor, TensorType, ValTensor},
};
use halo2curves::ff::PrimeField;

use self::{lookup::LookupOp, region::RegionCtx};

///
pub mod base;
///
pub mod chip;
///
pub mod errors;
///
pub mod hybrid;
/// Layouts for specific functions (composed of base ops)
pub mod layouts;
///
pub mod lookup;
///
pub mod poly;
///
pub mod region;

pub use errors::CircuitError;

/// A struct representing the result of a forward pass.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ForwardResult<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
    pub(crate) output: Tensor<F>,
}

/// A trait representing operations that can be represented as constraints in a circuit.
pub trait Op<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>:
    std::fmt::Debug + Send + Sync + Any
{
    /// Returns a string representation of the operation.
    fn as_string(&self) -> String;

    /// Layouts the operation in a circuit.
    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError>;

    /// Returns the scale of the output of the operation.
    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError>;

    /// Do any of the inputs to this op require homogenous input scales?
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        vec![]
    }

    /// Returns true if the operation is an input.
    fn is_input(&self) -> bool {
        false
    }

    /// Returns true if the operation is a constant.
    fn is_constant(&self) -> bool {
        false
    }

    /// Boxes and clones
    fn clone_dyn(&self) -> Box<dyn Op<F>>;

    /// Returns a reference to the Any trait.
    fn as_any(&self) -> &dyn Any;
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Clone for Box<dyn Op<F>> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

///
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InputType {
    ///
    Bool,
    ///
    F16,
    ///
    F32,
    ///
    F64,
    ///
    Int,
    ///
    TDim,
}

impl InputType {
    ///
    pub fn is_integer(&self) -> bool {
        matches!(self, InputType::Int | InputType::TDim | InputType::Bool)
    }

    ///
    pub fn roundtrip<T: num::ToPrimitive + num::FromPrimitive + Clone + std::fmt::Debug>(
        &self,
        input: &mut T,
    ) {
        match self {
            InputType::Bool => {
                let boolean_input = input.clone().to_i64().unwrap();
                assert!(boolean_input == 0 || boolean_input == 1);
                *input = T::from_i64(boolean_input).unwrap();
            }
            InputType::F16 => {
                // TODO: implement f16
                let f32_input = input.clone().to_f32().unwrap();
                *input = T::from_f32(f32_input).unwrap();
            }
            InputType::F32 => {
                let f32_input: f32 = input.clone().to_f32().unwrap();
                *input = T::from_f32(f32_input).unwrap();
            }
            InputType::F64 => {
                let f64_input = input.clone().to_f64().unwrap();
                *input = T::from_f64(f64_input).unwrap();
            }
            InputType::Int | InputType::TDim => {
                let int_input = input.clone().to_i64().unwrap();
                *input = T::from_i64(int_input).unwrap();
            }
        }
    }
}

impl std::str::FromStr for InputType {
    type Err = CircuitError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bool" => Ok(InputType::Bool),
            "f16" => Ok(InputType::F16),
            "f32" => Ok(InputType::F32),
            "f64" => Ok(InputType::F64),
            "int" => Ok(InputType::Int),
            "tdim" => Ok(InputType::TDim),
            e => Err(CircuitError::InvalidInputType(e.to_string())),
        }
    }
}

///
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Input {
    ///
    pub scale: crate::Scale,
    ///
    pub datum_type: InputType,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Op<F> for Input {
    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        Ok(self.scale)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        "Input".into()
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        let value = values[0].clone();
        if !value.all_prev_assigned() {
            match self.datum_type {
                InputType::Bool => {
                    log::debug!("constraining input to be boolean");
                    Ok(Some(super::layouts::boolean_identity(
                        config,
                        region,
                        values[..].try_into()?,
                        true,
                    )?))
                }
                _ => Ok(Some(super::layouts::identity(
                    config,
                    region,
                    values[..].try_into()?,
                )?)),
            }
        } else {
            Ok(Some(value))
        }
    }

    fn is_input(&self) -> bool {
        true
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// An unknown operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Unknown;

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Op<F> for Unknown {
    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        Ok(0)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        "Unknown".into()
    }
    fn layout(
        &self,
        _: &mut crate::circuit::BaseConfig<F>,
        _: &mut RegionCtx<F>,
        _: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        Err(super::CircuitError::UnsupportedOp)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constant<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
    ///
    pub quantized_values: Tensor<F>,
    ///
    pub raw_values: Tensor<f32>,
    ///
    #[serde(skip)]
    pub pre_assigned_val: Option<ValTensor<F>>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Constant<F> {
    ///
    pub fn new(quantized_values: Tensor<F>, raw_values: Tensor<f32>) -> Self {
        Self {
            quantized_values,
            raw_values,
            pre_assigned_val: None,
        }
    }
    /// Rebase the scale of the constant
    pub fn rebase_scale(&mut self, new_scale: crate::Scale) -> Result<(), CircuitError> {
        let visibility = match self.quantized_values.visibility() {
            Some(v) => v,
            None => return Err(CircuitError::UnsetVisibility),
        };
        self.quantized_values = quantize_tensor(self.raw_values.clone(), new_scale, &visibility)?;
        Ok(())
    }

    /// Empty raw value
    pub fn empty_raw_value(&mut self) {
        self.raw_values = Tensor::new(None, &[0]).unwrap();
    }

    /// Pre-assign a value
    pub fn pre_assign(&mut self, val: ValTensor<F>) {
        self.pre_assigned_val = Some(val)
    }
}

impl<
        F: PrimeField
            + TensorType
            + PartialOrd
            + std::hash::Hash
            + Serialize
            + for<'de> Deserialize<'de>,
    > Op<F> for Constant<F>
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_string(&self) -> String {
        format!("CONST (scale={})", self.quantized_values.scale().unwrap())
    }
    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        _: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        let value = if let Some(value) = &self.pre_assigned_val {
            value.clone()
        } else {
            self.quantized_values.clone().try_into()?
        };
        // we gotta constrain it once if its used multiple times
        Ok(Some(layouts::identity(config, region, &[value])?))
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }

    fn out_scale(&self, _: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        Ok(self.quantized_values.scale().unwrap())
    }

    fn is_constant(&self) -> bool {
        true
    }
}
