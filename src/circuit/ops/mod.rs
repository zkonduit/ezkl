use std::{any::Any, error::Error};

use itertools::Itertools;
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
pub struct ForwardResult {
    pub(crate) output: Tensor<i128>,
    pub(crate) intermediate_lookups: Vec<Tensor<i128>>,
}

/// An enum representing operations that can be represented as constraints in a circuit.
pub trait Op<F: PrimeField + TensorType + PartialOrd>: std::fmt::Debug + Send + Sync + Any {
    /// Matches a [Op] to an operation in the `tensor::ops` module.
    fn f(&self, x: &[Tensor<i128>]) -> Result<ForwardResult, TensorError>;
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
    fn out_scale(&self, _: Vec<u32>, global_scale: u32) -> u32 {
        global_scale
    }

    /// Do any of the inputs to this op require homogenous input scales?
    fn requires_homogenous_input_scales(&self) -> Vec<usize> {
        vec![]
    }

    /// Returns the lookups required by the operation.
    fn required_lookups(&self) -> Vec<LookupOp> {
        vec![]
    }

    /// Rescales the operation given a vector of input scales and a global (circuit) scale.
    fn rescale(&self, inputs_scale: Vec<u32>, global_scale: u32) -> Box<dyn Op<F>>;

    /// Returns true if the operation is an input.
    fn is_input(&self) -> bool {
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
pub struct Input {
    ///
    pub scale: u32,
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for Input {
    fn out_scale(&self, _: Vec<u32>, _: u32) -> u32 {
        self.scale
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, x: &[Tensor<i128>]) -> Result<ForwardResult, TensorError> {
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
        _: &mut crate::circuit::BaseConfig<F>,
        _: &mut RegionCtx<F>,
        _: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(None)
    }

    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        Box::new(self.clone())
    }

    fn is_input(&self) -> bool {
        true
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// A wrapper for an operation that has been rescaled.
#[derive(Clone, Debug)]
pub struct Rescaled<F: PrimeField + TensorType + PartialOrd> {
    /// The operation to be rescaled.
    pub inner: Box<dyn Op<F>>,
    /// The scale of the operation's inputs.
    pub scale: Vec<(usize, u128)>,
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for Rescaled<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn f(&self, x: &[Tensor<i128>]) -> Result<ForwardResult, TensorError> {
        if self.scale.len() != x.len() {
            return Err(TensorError::DimMismatch("rescaled inputs".to_string()));
        }

        let mut rescaled_inputs = vec![];
        let inputs = &mut x.to_vec();
        for (i, ri) in inputs.iter_mut().enumerate() {
            rescaled_inputs.push(tensor::ops::nonlinearities::const_div(
                ri,
                self.scale[i].1 as f64,
            ));
        }
        Op::<F>::f(&*self.inner, &rescaled_inputs)
    }

    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        Box::new(self.clone())
    }

    fn as_string(&self) -> String {
        format!("RESCALED {}", self.inner.as_string())
    }

    fn out_scale(&self, in_scales: Vec<u32>, _g: u32) -> u32 {
        let in_scales = in_scales
            .into_iter()
            .zip(self.scale.iter())
            .map(|(a, b)| a - crate::graph::mult_to_scale(b.1 as f64))
            .collect();

        Op::<F>::out_scale(&*self.inner, in_scales, _g)
    }

    fn required_lookups(&self) -> Vec<LookupOp> {
        let mut required_lookups = vec![];
        for scale in &self.scale {
            if scale.1 != 0 {
                required_lookups.push(LookupOp::Div {
                    denom: (scale.1 as f32).into(),
                });
            }
        }
        required_lookups
    }

    fn layout(
        &self,
        config: &mut crate::circuit::BaseConfig<F>,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        if self.scale.len() != values.len() {
            return Err(Box::new(TensorError::DimMismatch(
                "rescaled inputs".to_string(),
            )));
        }

        let res = &layouts::rescale(config, region, values[..].try_into()?, &self.scale)?[..];
        self.inner.layout(config, region, res)
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

/// An unknown operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Unknown;

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for Unknown {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn f(&self, _: &[Tensor<i128>]) -> Result<ForwardResult, TensorError> {
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
    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        Box::new(self.clone())
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

///
#[derive(Clone, Debug)]
pub struct Constant<F: PrimeField + TensorType + PartialOrd> {
    ///
    pub quantized_values: ValTensor<F>,
    ///
    pub raw_values: Tensor<f32>,
}

impl<F: PrimeField + TensorType + PartialOrd> Constant<F> {
    ///
    pub fn new(quantized_values: ValTensor<F>, raw_values: Tensor<f32>) -> Self {
        Self {
            quantized_values,
            raw_values,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> Op<F> for Constant<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn f(&self, _: &[Tensor<i128>]) -> Result<ForwardResult, TensorError> {
        let values = self.quantized_values.clone();
        let int_values = values.get_int_evals().unwrap();
        let output = Tensor::new(Some(&int_values), values.dims().clone())?;

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
        _: &mut crate::circuit::BaseConfig<F>,
        _: &mut RegionCtx<F>,
        _: &[ValTensor<F>],
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        Ok(Some(self.quantized_values.clone()))
    }
    fn rescale(&self, _: Vec<u32>, _: u32) -> Box<dyn Op<F>> {
        Box::new(self.clone())
    }

    fn clone_dyn(&self) -> Box<dyn Op<F>> {
        Box::new(self.clone()) // Forward to the derive(Clone) impl
    }
}

fn homogenize_input_scales<F: PrimeField + TensorType + PartialOrd>(
    op: impl Op<F> + Clone,
    input_scales: Vec<u32>,
    inputs_to_scale: Vec<usize>,
) -> Result<Box<dyn Op<F>>, Box<dyn Error>> {
    if inputs_to_scale.is_empty() {
        return Ok(Box::new(op));
    }

    let mut dividers: Vec<u128> = vec![1; input_scales.len()];
    if !input_scales.windows(2).all(|w| w[0] == w[1]) {
        let min_scale = input_scales.iter().min().unwrap();
        let _ = input_scales
            .iter()
            .enumerate()
            .map(|(idx, input_scale)| {
                if !inputs_to_scale.contains(&idx) {
                    return;
                }
                let scale_diff = input_scale - min_scale;
                if scale_diff > 0 {
                    let mult = crate::graph::scale_to_multiplier(scale_diff);
                    dividers[idx] = mult as u128;
                }
            })
            .collect_vec();
    }

    // only rescale if need to
    if dividers.iter().any(|&x| x > 1) {
        Ok(Box::new(crate::circuit::Rescaled {
            inner: Box::new(op),
            scale: (0..input_scales.len()).zip(dividers).collect_vec(),
        }))
    } else {
        Ok(Box::new(op))
    }
}