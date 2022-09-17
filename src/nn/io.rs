use super::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region},
    plonk::{Assigned, ConstraintSystem, Expression, Selector, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct IOConfig<F: FieldExt + TensorType> {
    pub values: ParamType,
    selector: Selector,
    dims: Vec<usize>,
    marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> IOConfig<F> {
    pub fn configure(meta: &mut ConstraintSystem<F>, values: ParamType, dims: &[usize]) -> Self {
        assert!(dims.len() == 2);
        Self {
            values,
            selector: meta.selector(),
            dims: dims.to_vec(),
            marker: PhantomData,
        }
    }

    pub fn query(&self, meta: &mut VirtualCells<'_, F>, offset: usize) -> Tensor<Expression<F>> {
        let mut t = match &self.values {
            // when fixed we have 1 col per param
            ParamType::Fixed(f) => f.map(|c| meta.query_fixed(c, Rotation(offset as i32))),
            // when advice we have 1 col per row
            ParamType::Advice(a) => a
                .map(|column| {
                    Tensor::from(
                        (0..self.dims[1])
                            .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                    )
                })
                .flatten(),
        };
        t.reshape(&self.dims);
        t
    }

    pub fn query_idx(
        &self,
        meta: &mut VirtualCells<'_, F>,
        idx: usize,
        offset: usize,
    ) -> Expression<F> {
        match &self.values {
            ParamType::Fixed(f) => meta.query_fixed(f[idx], Rotation(offset as i32)),
            ParamType::Advice(a) => meta.query_advice(a[idx], Rotation(offset as i32)),
        }
    }

    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        kernel: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        match kernel {
            IOType::Value(mut v) => {
                v.reshape(&self.dims);
                v.enum_map(|i, k| {
                    let coord = [i / self.dims[1], i % self.dims[1]];
                    match &self.values {
                        ParamType::Fixed(f) => region
                            .assign_fixed(|| "k", f.get(&coord), offset, || k.into())
                            .unwrap(),
                        ParamType::Advice(a) => region
                            .assign_advice(
                                || "k",
                                a.get(&[coord[0]]),
                                offset + coord[1],
                                || k.into(),
                            )
                            .unwrap(),
                    }
                })
            }
            IOType::PrevAssigned(mut v) => {
                v.reshape(&self.dims);
                v.enum_map(|i, x| {
                    let coord = [i / self.dims[1], i % self.dims[1]];
                    match &self.values {
                        ParamType::Fixed(_) => panic!("not implemented"),
                        ParamType::Advice(a) => x
                            .copy_advice(|| "k", region, a.get(&[coord[0]]), offset + coord[1])
                            .unwrap(),
                    }
                })
            }
            IOType::AssignedValue(mut v) => {
                v.reshape(&self.dims);
                v.enum_map(|i, k| {
                    let coord = [i / self.dims[1], i % self.dims[1]];
                    match &self.values {
                        ParamType::Fixed(f) => region
                            .assign_fixed(|| "k", f.get(&coord), offset, || k)
                            .unwrap(),
                        ParamType::Advice(a) => region
                            .assign_advice(|| "k", a.get(&[coord[0]]), offset + coord[1], || k)
                            .unwrap(),
                    }
                })
            }
        }
    }

    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        raw_input: Tensor<i32>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Input",
            |mut region| {
                let offset = 0;
                self.selector.enable(&mut region, offset)?;
                Ok(self.assign(&mut region, offset, IOType::Value(raw_input.clone().into())))
            },
        )
    }
}
