use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Fixed},
};
use std::ops::Range;

pub mod affine;
pub mod cnvrl;
pub mod io;

#[derive(Debug, Clone)]
pub enum IOType<F: FieldExt + TensorType> {
    Value(Tensor<Value<F>>),
    AssignedValue(Tensor<Value<Assigned<F>>>),
    PrevAssigned(Tensor<AssignedCell<Assigned<F>, F>>),
}

impl<F: FieldExt + TensorType> IOType<F> {
    pub fn get_slice(&self, indices: &[Range<usize>]) -> IOType<F> {
        match self {
            IOType::Value(v) => IOType::Value(v.get_slice(indices)),
            IOType::AssignedValue(v) => IOType::AssignedValue(v.get_slice(indices)),
            IOType::PrevAssigned(v) => IOType::PrevAssigned(v.get_slice(indices)),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            IOType::Value(v) => v.dims(),
            IOType::AssignedValue(v) => v.dims(),
            IOType::PrevAssigned(v) => v.dims(),
        }
    }

    pub fn reshape(&mut self, new_dims: &[usize]) {
        match self {
            IOType::Value(v) => v.reshape(new_dims),
            IOType::AssignedValue(v) => v.reshape(new_dims),
            IOType::PrevAssigned(v) => v.reshape(new_dims),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ParamType {
    Advice(Tensor<Column<Advice>>),
    Fixed(Tensor<Column<Fixed>>),
}

impl ParamType {
    pub fn get_slice(&self, indices: &[Range<usize>]) -> ParamType {
        match self {
            ParamType::Advice(v) => ParamType::Advice(v.get_slice(indices)),
            ParamType::Fixed(v) => ParamType::Fixed(v.get_slice(indices)),
        }
    }

    pub fn enable_equality<F: FieldExt>(&self, meta: &mut ConstraintSystem<F>) {
        match self {
            ParamType::Advice(advices) => {
                for advice in advices.iter() {
                    meta.enable_equality(*advice);
                }
            }
            ParamType::Fixed(_) => {}
        }
    }
}

pub trait LayerConfig<F: FieldExt + TensorType> {
    fn configure(
        _meta: &mut ConstraintSystem<F>,
        params: &[ParamType],
        input: ParamType,
        output: ParamType,
    ) -> Self;
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: IOType<F>,
        params: &[IOType<F>],
    ) -> IOType<F>;
    fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        input: IOType<F>,
        params: &[IOType<F>],
    ) -> Tensor<AssignedCell<Assigned<F>, F>>;
}
