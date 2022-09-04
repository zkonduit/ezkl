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
pub fn vec_matmul<T: TensorType + Clone + Add<Output = T> + Mul<Output = T>>(
    a: Tensor<T>,
    b: Tensor<T>,
    biases: Option<Tensor<T>>,
) -> Tensor<T> {
    // calculate value of output
    assert!(a.dims().len() == 1);
    assert!(b.dims().len() == 2);
    assert!(a.dims()[0] == b.dims()[0]);
    let out_dim = b.dims();
    let in_dim = a.dims();
    let mut output = Tensor::new(None, out_dim).unwrap();

    for (i, o) in output.iter_mut().enumerate() {
        for (j, x) in a.iter().enumerate() {
            *o = *o + b.get(&[i, j]) * *x;
        }
        // add bias
        match biases {
            Some(bias) => *o = *o + bias.get(&[i]),
            None => {}
        }
    }
    output
}

// for now assumes a batch size of 1
pub fn vec_matmul_field<T: TensorType + Clone + Add<Output = T> + Mul<Output = T>>(
    a: Tensor<AssignedCell<Assigned<F>, F>>,
    b: Tensor<AssignedCell<Assigned<F>, F>>,
    biases: Option<Tensor<Tensor<AssignedCell<Assigned<F>, F>>>>,
) -> Tensor<T> {
    // calculate value of output
    assert!(a.dims().len() == 1);
    assert!(b.dims().len() == 2);
    assert!(a.dims()[0] == b.dims()[0]);
    let out_dim = b.dims();
    let in_dim = a.dims();
    let mut output = Tensor::new(None, out_dim).unwrap();

    for (i, o) in output.iter_mut().enumerate() {
        for (j, x) in a.iter().enumerate() {
            *o = *o + b.get(&[i, j]) * *x;
        }
        // add bias
        match biases {
            Some(bias) => *o = *o + bias.get(&[i]),
            None => {}
        }
    }
    output
}
