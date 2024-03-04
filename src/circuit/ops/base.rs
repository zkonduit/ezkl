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
    DotInit,
    CumProdInit,
    CumProd,
    Add,
    Mult,
    Sub,
    SumInit,
    Sum,
    IsZero,
    IsBoolean,
}

/// Matches a [BaseOp] to an operation over inputs
impl BaseOp {
    /// forward func
    pub fn nonaccum_f<
        T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    >(
        &self,
        inputs: (T, T),
    ) -> T {
        let (a, b) = inputs;
        match &self {
            BaseOp::Add => a + b,
            BaseOp::Sub => a - b,
            BaseOp::Mult => a * b,
            BaseOp::IsZero => b,
            BaseOp::IsBoolean => b,
            _ => panic!("nonaccum_f called on accumulating operation"),
        }
    }

    /// forward func
    pub fn accum_f<
        T: TensorType + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Neg<Output = T>,
    >(
        &self,
        prev_output: T,
        a: Vec<T>,
        b: Vec<T>,
    ) -> T {
        let zero = T::zero().unwrap();
        let one = T::one().unwrap();

        match &self {
            BaseOp::DotInit => a.into_iter().zip(b).fold(zero, |acc, (a, b)| acc + a * b),
            BaseOp::Dot => prev_output + a.into_iter().zip(b).fold(zero, |acc, (a, b)| acc + a * b),
            BaseOp::CumProdInit => b.into_iter().fold(one, |acc, b| acc * b),
            BaseOp::CumProd => prev_output * b.into_iter().fold(one, |acc, b| acc * b),
            BaseOp::SumInit => b.into_iter().fold(zero, |acc, b| acc + b),
            BaseOp::Sum => prev_output + b.into_iter().fold(zero, |acc, b| acc + b),
            _ => panic!("accum_f called on non-accumulating operation"),
        }
    }

    /// display func
    pub fn as_str(&self) -> &'static str {
        match self {
            BaseOp::Dot => "DOT",
            BaseOp::DotInit => "DOTINIT",
            BaseOp::CumProdInit => "CUMPRODINIT",
            BaseOp::CumProd => "CUMPROD",
            BaseOp::Add => "ADD",
            BaseOp::Sub => "SUB",
            BaseOp::Mult => "MULT",
            BaseOp::Sum => "SUM",
            BaseOp::SumInit => "SUMINIT",
            BaseOp::IsZero => "ISZERO",
            BaseOp::IsBoolean => "ISBOOLEAN",
        }
    }

    /// Returns the range of the query offset for this operation.
    pub fn query_offset_rng(&self) -> (i32, usize) {
        match self {
            BaseOp::DotInit => (0, 1),
            BaseOp::Dot => (-1, 2),
            BaseOp::CumProd => (-1, 2),
            BaseOp::CumProdInit => (0, 1),
            BaseOp::Add => (0, 1),
            BaseOp::Sub => (0, 1),
            BaseOp::Mult => (0, 1),
            BaseOp::Sum => (-1, 2),
            BaseOp::SumInit => (0, 1),
            BaseOp::IsZero => (0, 1),
            BaseOp::IsBoolean => (0, 1),
        }
    }

    /// Returns the number of inputs for this operation.
    pub fn num_inputs(&self) -> usize {
        match self {
            BaseOp::DotInit => 2,
            BaseOp::Dot => 2,
            BaseOp::CumProdInit => 1,
            BaseOp::CumProd => 1,
            BaseOp::Add => 2,
            BaseOp::Sub => 2,
            BaseOp::Mult => 2,
            BaseOp::Sum => 1,
            BaseOp::SumInit => 1,
            BaseOp::IsZero => 0,
            BaseOp::IsBoolean => 0,
        }
    }

    /// Returns the number of outputs for this operation.
    pub fn constraint_idx(&self) -> usize {
        match self {
            BaseOp::DotInit => 0,
            BaseOp::Dot => 1,
            BaseOp::Add => 0,
            BaseOp::Sub => 0,
            BaseOp::Mult => 0,
            BaseOp::Sum => 1,
            BaseOp::SumInit => 0,
            BaseOp::CumProd => 1,
            BaseOp::CumProdInit => 0,
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
