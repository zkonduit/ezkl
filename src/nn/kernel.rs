use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Fixed, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub enum ParamType {
    Advice(Column<Advice>),
    Fixed(Column<Fixed>),
}

impl ParamType {
    fn assign<F: FieldExt>(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        elem: Value<F>,
    ) -> AssignedCell<Assigned<F>, F> {
        match self {
            ParamType::Fixed(c) => region
                .assign_fixed(
                    || format!("w"),
                    // row indices
                    *c,
                    // columns indices
                    offset,
                    || elem.into(),
                )
                .unwrap(),
            ParamType::Advice(c) => region
                .assign_advice(
                    || format!("w"),
                    // row indices
                    *c,
                    // columns indices
                    offset,
                    || elem.into(),
                )
                .unwrap(),
        }
    }
    pub fn query<F: FieldExt>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation: Rotation,
    ) -> Expression<F> {
        match self {
            ParamType::Fixed(c) => meta.query_fixed(*c, rotation),
            ParamType::Advice(c) => meta.query_advice(*c, rotation),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelConfig<F: FieldExt> {
    pub params: Tensor<ParamType>,
    dims: Vec<usize>,
    marker: PhantomData<F>,
}

impl<F: FieldExt> KernelConfig<F>
where
    Value<F>: TensorType,
{
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        params: Tensor<ParamType>,
        dims: &[usize],
    ) -> Self {
        Self {
            params,
            dims: dims.to_vec(),
            marker: PhantomData,
        }
    }

    pub fn query(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation: Rotation,
    ) -> Tensor<Expression<F>> {
        let mut t = self.params.map(|col| col.query(meta, rotation));
        match self.params[0] {
            ParamType::Advice(_) => {}
            ParamType::Fixed(_) => t.reshape(&self.dims),
        }
        t
    }

    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        kernel: Tensor<Value<F>>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        let dims = kernel.dims();
        assert!(dims.len() == 2);
        kernel.enum_map(|i, k| {
            let coord = [i / dims[1], i % dims[1]];
            match self.params[0] {
                ParamType::Fixed(_) => self.params.get(&coord).assign(region, offset, k),
                ParamType::Advice(_) => {
                    self.params
                        .get(&[coord[0]])
                        .assign(region, offset + coord[1], k)
                }
            }
        })
    }
}
