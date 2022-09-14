use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Selector, VirtualCells},
    poly::Rotation,
};
use std::marker::PhantomData;

// Takes input data provided as raw data type, e.g. i32, and sets it up to be passed into a pipeline,
// including laying it out in a column and outputting Vec<AssignedCell<Assigned<F>, F>> suitable for copying
// Can also have a variant to check a signature, check that input matches a hash, etc.
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
                Ok(self.assign(&mut region, offset, raw_input.clone().into()))
            },
        )
    }

    pub fn assign(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        input: Tensor<Value<F>>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        let dims = input.dims();
        assert!(dims == self.dims);
        input.enum_map(|i, x| {
            let coord = [i / dims[1], i % dims[1]];
            region
                .assign_advice(
                    || format!("input at row: {:?}, column: {:?}", coord[1], coord[0]),
                    // row indices
                    self.advices[coord[0]],
                    // columns indices
                    offset + coord[1],
                    || x.into(),
                )
                .unwrap()
        })
    }
}
