use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Fixed, VirtualCells},
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
            ParamType::Advice(advices) => for advice in advices.iter() {
                meta.enable_equality(*advice);
            },
            ParamType::Fixed(_) => {}
        }
    }
}

pub trait LayerConfig<F: FieldExt + TensorType> {
    fn configure(_meta: &mut ConstraintSystem<F>, params: ParamType, dims: &[usize]) -> Self;
    fn query(&self, meta: &mut VirtualCells<'_, F>, offset: usize) -> Tensor<Expression<F>>;
    fn query_idx(&self, meta: &mut VirtualCells<'_, F>, idx: usize, offset: usize)
        -> Expression<F>;
    fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        input: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>>;
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        raw_input: Tensor<i32>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error>;
}
