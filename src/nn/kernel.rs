use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Fixed, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;
use std::ops::Range;

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
}

#[derive(Debug, Clone)]
pub struct KernelConfig<F: FieldExt> {
    pub params: ParamType,
    dims: Vec<usize>,
    marker: PhantomData<F>,
}

impl<F: FieldExt> KernelConfig<F>
where
    Value<F>: TensorType,
{
    pub fn configure(_meta: &mut ConstraintSystem<F>, params: ParamType, dims: &[usize]) -> Self {
        Self {
            params,
            dims: dims.to_vec(),
            marker: PhantomData,
        }
    }

    pub fn query(&self, meta: &mut VirtualCells<'_, F>, offset: usize) -> Tensor<Expression<F>> {
        match &self.params {
            ParamType::Fixed(f) => {
                let mut t = f.map(|c| meta.query_fixed(c, Rotation(offset as i32)));
                t.reshape(&self.dims);
                t
            }
            ParamType::Advice(a) => a.map(|c| meta.query_advice(c, Rotation(offset as i32))),
        }
    }

    pub fn query_idx(
        &self,
        meta: &mut VirtualCells<'_, F>,
        idx: usize,
        offset: usize,
    ) -> Expression<F> {
        match &self.params {
            ParamType::Fixed(f) => meta.query_fixed(f[idx], Rotation(offset as i32)),
            ParamType::Advice(a) => meta.query_advice(a[idx], Rotation(offset as i32)),
        }
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
            match &self.params {
                ParamType::Fixed(f) => region
                    .assign_fixed(
                        || format!("w"),
                        // row indices
                        f.get(&coord),
                        // columns indices
                        offset,
                        || k.into(),
                    )
                    .unwrap(),
                ParamType::Advice(a) => region
                    .assign_advice(
                        || format!("w"),
                        // row indices
                        a.get(&[coord[0]]),
                        // columns indices
                        offset + coord[1],
                        || k.into(),
                    )
                    .unwrap(),
            }
        })
    }
}
