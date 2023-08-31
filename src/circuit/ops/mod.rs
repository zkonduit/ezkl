use std::{any::Any, error::Error};

use serde::{Deserialize, Serialize};

use crate::tensor::{self, Tensor, TensorError, TensorType, ValTensor};
use halo2curves::ff::PrimeField;

use self::{lookup::LookupOp, region::RegionCtx};

///
pub mod base;
///
pub mod chip;
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

/// A struct representing the result of a forward pass.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ForwardResult<F: PrimeField + TensorType + PartialOrd> {
    pub(crate) output: Tensor<F>,
    pub(crate) intermediate_lookups: Vec<Tensor<i128>>,
}

/// An enum representing operations that can be represented as constraints in a circuit.
pub trait Op<F: PrimeField + TensorType + PartialOrd>: std::fmt::Debug + Send + Sync + Any {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, x: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError>;
    /// Returns a string representation of the operation.
    fn as_string(&self) -> String;

    /// Layouts the operation in a circuit.
    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>>;

    /// Returns the scale of the output of the operation.
    fn out_scale(&self, _: Vec<u32>) -> u32;

    /// Do any of the inputs to this op require homogenous input scales?
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        vec![]
    }

    /// Returns the lookups required by the operation.
    fn required_lookups(&self) -> Vec<LookupOp> {
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

impl<F: PrimeField + TensorType + PartialOrd> Clone for Box<dyn Op<F>> {
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
    Num,
}

///
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Input {
    ///
    pub scale: u32,
    ///
    pub datum_type: InputType,
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for Input {
    fn out_scale(&self, _: Vec<u32>) -> u32 {
        self.scale
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, x: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        Ok(ForwardResult {
            output: x[0].clone(),
            intermediate_lookups: vec![],
        })
    }

    fn as_string(&self) -> String {
        "Input".into()
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        let value = values[0].clone();
        if !value.all_prev_assigned() {
            match self.datum_type {
                InputType::Num => Ok(Some(super::layouts::identity(
                    config,
                    region,
                    values[..].try_into()?,
                )?)),
                InputType::Bool => {
                    log::debug!("constraining input to be boolean");
                    Ok(Some(super::layouts::boolean_identity(
                        config,
                        region,
                        values[..].try_into()?,
                    )?))
                }
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

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for Unknown {
    fn out_scale(&self, _: Vec<u32>) -> u32 {
        0
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn f(&self, _: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        Err(TensorError::WrongMethod)
    }

    fn as_string(&self) -> String {
        "Unknown".into()
    }
    fn layout(
        &self,
        _: &mut crate::circuit::BaseConfig<F>,
        _: &mut RegionCtx<F>,
        _: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Err(Box::new(super::CircuitError::UnsupportedOp))
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constant<F: PrimeField + TensorType + PartialOrd> {
    ///
    pub quantized_values: Tensor<F>,
    ///
    pub raw_values: Tensor<f32>,
    ///
    #[serde(skip)]
    pub pre_assigned_val: Option<ValTensor<F>>,
    ///
    pub num_uses: usize,
}

impl<F: PrimeField + TensorType + PartialOrd> Constant<F> {
    ///
    pub fn new(quantized_values: Tensor<F>, raw_values: Tensor<f32>) -> Self {
        Self {
            quantized_values,
            raw_values,
            pre_assigned_val: None,
            num_uses: 0,
        }
    }

    /// Empty raw value
    pub fn empty_raw_value(&mut self) {
        self.raw_values = Tensor::new(None, &[0]).unwrap();
    }

    /// Returns true if the constant is only used once.
    pub fn is_single_use(&self) -> bool {
        self.num_uses == 1
    }

    ///
    pub fn pre_assign(&mut self, val: ValTensor<F>) {
        self.pre_assigned_val = Some(val)
    }
}

impl<F: PrimeField + TensorType + PartialOrd + Serialize + for<'de> Deserialize<'de>> Op<F>
    for Constant<F>
{
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn f(&self, _: &[Tensor<F>]) -> Result<ForwardResult<F>, TensorError> {
        let output = self.quantized_values.clone();

        Ok(ForwardResult {
            output,
            intermediate_lookups: vec![],
        })
    }

    fn as_string(&self) -> String {
        "CONST".into()
    }
    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        _: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        if let Some(value) = &self.pre_assigned_val {
            Ok(Some(value.clone()))
        } else if self.is_single_use() {
            // we can just assign it within the op
            Ok(Some(self.quantized_values.clone().into()))
        } else {
            // we gotta constrain it once if its used multiple times
            Ok(Some(layouts::identity(
                config,
                region,
                &[self.quantized_values.clone().into()],
            )?))
        }
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }

    fn out_scale(&self, _: Vec<u32>) -> u32 {
        self.quantized_values.scale().unwrap()
    }

    fn is_constant(&self) -> bool {
        true
    }
}
