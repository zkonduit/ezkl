use super::*;
use crate::tensor::TensorType;
use halo2_proofs::{arithmetic::FieldExt, circuit::Value};

/// A macro to abort concisely.
#[macro_export]
macro_rules! abort {
    ($msg:literal $(, $ex:expr)*) => {
        error!($msg, $($ex,)*);
        panic!();
    };
}

/// Used to deal with unknown values at proof time arising from `Fixed` variables.
/// In such a scenario swaps the assigned value for another value (presumably the value assigned to the `Fixed` variable).
pub fn value_muxer<F: FieldExt + TensorType>(
    variable: &VarTensor,
    assigned: &Tensor<Value<F>>,
    input: &ValTensor<F>,
) -> Tensor<Value<F>> {
    match variable {
        VarTensor::Advice { inner: _, dims: _ } => assigned.clone(),
        VarTensor::Fixed { inner: _, dims: _ } => match input {
            ValTensor::Value {
                inner: val,
                dims: _,
            } => val.clone(),
            _ => unimplemented!(),
        },
    }
}
