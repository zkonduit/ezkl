use crate::tensor::TensorType;
use std::{
    fmt,
    ops::{Add, Mul, Neg, Sub},
};

#[allow(missing_docs)]
/// An enum representing the operations that can be used to express more complex operations via accumulation
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BaseOp {
    Dot,
    Identity,
    Add,
    Mult,
    Sub,
    Sum,
    Neg,
    Range { tol: i32 },
    IsZero,
    IsBoolean,
}

/// Matches a [BaseOp] to an operation over inputs
impl BaseOp {
    /// forward func
    pub fn f<
        T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    >(
        &self,
        inputs: (T, T, T),
    ) -> T {
        let (a, b, m) = inputs;
        match &self {
            BaseOp::Dot => a * b + m,
            BaseOp::Add => a + b,
            BaseOp::Identity => b,
            BaseOp::Sum => b + m,
            BaseOp::Neg => -b,
            BaseOp::Sub => a - b,
            BaseOp::Mult => a * b,
            BaseOp::Range { .. } => b,
            BaseOp::IsZero => b,
            BaseOp::IsBoolean => b,
        }
    }

    ///
    pub fn as_str(&self) -> &'static str {
        match self {
            BaseOp::Identity => "IDENTITY",
            BaseOp::Dot => "DOT",
            BaseOp::Add => "ADD",
            BaseOp::Neg => "NEG",
            BaseOp::Sub => "SUB",
            BaseOp::Mult => "MULT",
            BaseOp::Sum => "SUM",
            BaseOp::Range { .. } => "RANGE",
            BaseOp::IsZero => "ISZERO",
            BaseOp::IsBoolean => "ISBOOLEAN",
        }
    }

    ///
    pub fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::Identity => (0, 1),
            BaseOp::Neg => (0, 1),
            BaseOp::Dot => (-1, 2),
            BaseOp::Add => (0, 1),
            BaseOp::Sub => (0, 1),
            BaseOp::Mult => (0, 1),
            BaseOp::Sum => (-1, 2),
            BaseOp::Range { .. } => (0, 1),
            BaseOp::IsZero => (0, 1),
            BaseOp::IsBoolean => (0, 1),
        }
    }

    ///
    pub fn num_inputs(&self) -> usize {
        match self {
            BaseOp::Identity => 1,
            BaseOp::Neg => 1,
            BaseOp::Dot => 2,
            BaseOp::Add => 2,
            BaseOp::Sub => 2,
            BaseOp::Mult => 2,
            BaseOp::Sum => 1,
            BaseOp::Range { .. } => 1,
            BaseOp::IsZero => 1,
            BaseOp::IsBoolean => 1,
        }
    }

    ///
    pub fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::Identity => 0,
            BaseOp::Neg => 0,
            BaseOp::Dot => 1,
            BaseOp::Add => 0,
            BaseOp::Sub => 0,
            BaseOp::Mult => 0,
            BaseOp::Range { .. } => 0,
            BaseOp::Sum => 1,
            BaseOp::IsZero => 0,
            BaseOp::IsBoolean => 0,
        }
    }
}

impl fmt::Display for BaseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
