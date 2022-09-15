use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Selector, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;
use std::ops::Range;

// Takes input data provided as raw data type, e.g. i32, and sets it up to be passed into a pipeline,
// including laying it out in a column and outputting Vec<AssignedCell<Assigned<F>, F>> suitable for copying
// Can also have a variant to check a signature, check that input matches a hash, etc.
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

#[derive(Debug, Clone)]
pub struct IOConfig<F: FieldExt> {
    pub advices: Tensor<Column<Advice>>,
    pub dims: Vec<usize>,
    pub q: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> IOConfig<F>
where
    Value<F>: TensorType,
{
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        mut advices: Tensor<Column<Advice>>,
        dims: &[usize],
    ) -> IOConfig<F> {
        let qs = cs.selector();
        // could put additional constraints on input here
        // 1D input will have dims [1,LEN] for example
        assert!(dims.len() == 2);
        assert!(advices.len() == dims[0]);
        IOConfig {
            advices,
            dims: dims.to_vec(),
            q: qs,
            _marker: PhantomData,
        }
    }
    pub fn query(
        &mut self,
        meta: &mut VirtualCells<'_, F>,
        offset: usize,
    ) -> Tensor<Expression<F>> {
        let mut t: Tensor<Expression<F>> = self
            .advices
            .map(|column| {
                Tensor::from(
                    (0..self.dims[1])
                        .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                )
            })
            .flatten();
        // we assume every column is of the same length
        t.reshape(&self.dims);
        t
    }

    pub fn query_idx(
        &self,
        meta: &mut VirtualCells<'_, F>,
        idx: usize,
        offset: usize,
    ) -> Expression<F> {
        meta.query_advice(self.advices[idx], Rotation(offset as i32))
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
                self.q.enable(&mut region, offset)?;
                Ok(self.assign(&mut region, offset, IOType::Value(raw_input.clone().into())))
            },
        )
    }

    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        input: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        match input {
            IOType::Value(mut v) => {
                v.reshape(&self.dims);
                v.enum_map(|i, x| {
                    let dims = v.dims();
                    let coord = [i / dims[1], i % dims[1]];
                    region
                        .assign_advice(
                            || format!("input at row: {:?}, column: {:?}", coord[1], coord[0]),
                            self.advices[coord[0]],
                            offset + coord[1],
                            || x.into(),
                        )
                        .unwrap()
                })
            }
            IOType::AssignedValue(mut v) => {
                v.reshape(&self.dims);
                v.enum_map(|i, x| {
                    let coord = [i / self.dims[1], i % self.dims[1]];
                    region
                        .assign_advice(
                            || format!("input at row: {:?}, column: {:?}", coord[1], coord[0]),
                            self.advices[coord[0]],
                            offset + coord[1],
                            || x.into(),
                        )
                        .unwrap()
                })
            }
            IOType::PrevAssigned(mut a) => {
                a.reshape(&self.dims);
                a.enum_map(|i, x| {
                    let coord = [i / self.dims[1], i % self.dims[1]];
                    x.copy_advice(
                        || format!("input at row: {:?}, column: {:?}", coord[1], coord[0]),
                        region,
                        self.advices[coord[0]],
                        offset + coord[1],
                    )
                    .unwrap()
                })
            }
        }
    }
}
