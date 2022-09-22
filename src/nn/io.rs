use super::*;
use crate::tensor::{ValTensor, VarTensor};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Assigned, ConstraintSystem, Expression, Selector, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct IOConfig<F: FieldExt + TensorType> {
    pub values: VarTensor,
    selector: Selector,
    marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> IOConfig<F> {
    pub fn configure(meta: &mut ConstraintSystem<F>, mut values: VarTensor) -> Self {
        match values.dims().len() {
            1 => values.reshape(&[1, values.dims()[0]]),
            2 => {}
            _ => panic!(
                "input of dims {:?} should be 1 or 2 dimensional",
                values.dims()
            ),
        }
        assert!(values.dims().len() == 2);
        Self {
            values,
            selector: meta.selector(),
            marker: PhantomData,
        }
    }

    pub fn query(&self, meta: &mut VirtualCells<'_, F>, offset: usize) -> Tensor<Expression<F>> {
        let mut t = match &self.values {
            // when fixed we have 1 col per param
            VarTensor::Fixed { inner: f, dims: _ } => {
                f.map(|c| meta.query_fixed(c, Rotation(offset as i32)))
            }
            // when advice we have 1 col per row
            VarTensor::Advice { inner: a, dims: d } => a
                .map(|column| {
                    Tensor::from(
                        (0..d[1])
                            .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                    )
                })
                .combine(),
        };
        t.reshape(self.values.dims());
        t
    }

    pub fn query_idx(
        &self,
        meta: &mut VirtualCells<'_, F>,
        idx: usize,
        offset: usize,
    ) -> Expression<F> {
        match &self.values {
            VarTensor::Fixed { inner: f, dims: _ } => {
                meta.query_fixed(f[idx], Rotation(offset as i32))
            }
            VarTensor::Advice { inner: a, dims: _ } => {
                meta.query_advice(a[idx], Rotation(offset as i32))
            }
        }
    }

    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        mut kernel: ValTensor<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        match kernel.dims().len() {
            1 => kernel.reshape(&[1, kernel.dims()[0]]),
            2 => {}
            _ => panic!(
                "kernel of dims {:?} should be 1 or 2 dimensional",
                kernel.dims()
            ),
        }

        match kernel {
            ValTensor::Value { inner: v, dims: d } => v.enum_map(|i, k| {
                let coord = [i / d[1], i % d[1]];
                match &self.values {
                    VarTensor::Fixed { inner: f, dims: _ } => region
                        .assign_fixed(|| "k", f.get(&coord), offset, || k.into())
                        .unwrap(),
                    VarTensor::Advice { inner: a, dims: _ } => region
                        .assign_advice(|| "k", a.get(&[coord[0]]), offset + coord[1], || k.into())
                        .unwrap(),
                }
            }),
            ValTensor::PrevAssigned { inner: v, dims: d } => v.enum_map(|i, x| {
                let coord = [i / d[1], i % d[1]];
                match &self.values {
                    VarTensor::Fixed { inner: _, dims: _ } => panic!("not implemented"),
                    VarTensor::Advice { inner: a, dims: _ } => x
                        .copy_advice(|| "k", region, a.get(&[coord[0]]), offset + coord[1])
                        .unwrap(),
                }
            }),
            ValTensor::AssignedValue { inner: v, dims: d } => v.enum_map(|i, k| {
                let coord = [i / d[1], i % d[1]];
                match &self.values {
                    VarTensor::Fixed { inner: f, dims: _ } => region
                        .assign_fixed(|| "k", f.get(&coord), offset, || k)
                        .unwrap(),
                    VarTensor::Advice { inner: a, dims: _ } => region
                        .assign_advice(|| "k", a.get(&[coord[0]]), offset + coord[1], || k)
                        .unwrap(),
                }
            }),
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
                Ok(self.assign(
                    &mut region,
                    offset,
                    ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(
                        raw_input.clone(),
                    )),
                ))
            },
        )
    }
}
