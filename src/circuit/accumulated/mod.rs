use crate::tensor::TensorType;
use std::{
    fmt,
    ops::{Add, Mul, Sub},
};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BaseOp {
    Dot,
    InitDot,
    Add,
}

/// Matches a [Op] to an operation in the `tensor::ops` module.
impl BaseOp {
    /// forward func
    pub fn f<T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
        &self,
        inputs: (T, T, T),
    ) -> T {
        let (a, b, m) = inputs;
        match &self {
            BaseOp::InitDot => a * b,
            BaseOp::Dot => a * b + m,
            BaseOp::Add => b + m,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            BaseOp::InitDot => "INITDOT",
            BaseOp::Dot => "DOT",
            BaseOp::Add => "ADD",
        }
    }
}

impl fmt::Display for BaseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BaseOp::InitDot => write!(f, "base accum init dot"),
            BaseOp::Dot => write!(f, "base accum dot"),
            BaseOp::Add => write!(f, "base accum add"),
        }
    }
}

/// Accumulated dot product
pub mod dot;
/// Accumulated matmul op
pub mod matmul;
