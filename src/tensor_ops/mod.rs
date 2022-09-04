pub mod eltwise;
pub mod utils;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
pub use std::ops::{Add, Mul};

// for now assumes a batch size of 1
pub fn vec_matmul_field<F: FieldExt>(
    a: Tensor<AssignedCell<Assigned<F>, F>>,
    b: Tensor<AssignedCell<Assigned<F>, F>>,
    biases: Option<Tensor<AssignedCell<Assigned<F>, F>>>,
) -> Tensor<Value<Assigned<F>>> {
    // calculate value of output
    assert!(a.dims().len() == 1);
    assert!(b.dims().len() == 2);
    assert!(a.dims()[0] == b.dims()[0]);
    let out_dim = b.dims()[1];
    let in_dim = a.dims();
    // calculate value of output
    let mut output: Tensor<Value<Assigned<F>>> = Tensor::new(None, &[out_dim]).unwrap();

    for (i, o) in output.iter_mut().enumerate() {
        for (j, x) in a.iter().enumerate() {
            *o = *o + b.get(&[i, j]).value_field() * x.value_field();
        }
        // add bias
        match biases {
            Some(ref bias) => *o = *o + bias.get(&[i]).value_field(),
            None => {}
        }
    }
    output
}
