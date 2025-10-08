use std::{collections::HashMap, marker::PhantomData};

use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;

use crate::{
    circuit::CircuitError,
    tensor::{Tensor, TensorError, TensorType},
};

/// Circuit parameter for a single einsum equation
#[derive(Clone, Debug, Default)]
pub struct SingleEinsumParams<F: PrimeField + TensorType + PartialOrd> {
    ///
    pub equation: String,
    /// Map from input axes to dimensions
    pub input_axes_to_dims: HashMap<char, usize>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> SingleEinsumParams<F> {
    ///
    pub fn new(equation: &str, inputs: &[&Tensor<Value<F>>]) -> Result<Self, CircuitError> {
        let mut eq = equation.split("->");
        let inputs_eq = eq.next().ok_or(CircuitError::InvalidEinsum)?;
        let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();

        // Check that the number of inputs matches the number of inputs in the equation
        if inputs.len() != inputs_eq.len() {
            return Err(TensorError::DimMismatch("einsum".to_string()).into());
        }

        let mut input_axes_to_dims = HashMap::new();
        for (i, input) in inputs.iter().enumerate() {
            for j in 0..inputs_eq[i].len() {
                let c = inputs_eq[i]
                    .chars()
                    .nth(j)
                    .ok_or(CircuitError::InvalidEinsum)?;
                if let std::collections::hash_map::Entry::Vacant(e) = input_axes_to_dims.entry(c) {
                    e.insert(input.dims()[j]);
                } else if input_axes_to_dims[&c] != input.dims()[j] {
                    return Err(TensorError::DimMismatch("einsum".to_string()).into());
                }
            }
        }

        Ok(Self {
            equation: equation.to_owned(),
            input_axes_to_dims,
            _marker: PhantomData,
        })
    }
}
