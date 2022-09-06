use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;

#[derive(Debug, Clone)]
pub struct ImageConfig<F: FieldExt> {
    advices: Tensor<Column<Advice>>,
    height: usize,
    marker: PhantomData<F>,
}

impl<F: FieldExt> ImageConfig<F>
where
    Value<F>: TensorType,
{
    pub fn configure(advices: Tensor<Column<Advice>>, height: usize) -> Self {
        Self {
            advices,
            height,
            marker: PhantomData,
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
                    (0..self.height)
                        .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                )
            })
            .flatten();
        t.reshape(&[self.advices.len(), self.height]);
        t
    }

    pub fn assign_image_2d(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        image: Tensor<Value<F>>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        println!("{:?}", image.dims());
        println!("{:?}", self.advices.dims());
        image.enum_map(|i, w| {
            let row = i / image.dims()[1];
            let col = i % image.dims()[0];
            println!("row: {:?}, col: {:?},", row, col);
            region
                .assign_advice(
                    || format!("pixel at row: {:?}, column: {:?}", row, col),
                    // row indices
                    self.advices[col],
                    // columns indices
                    offset + row,
                    || w.into(),
                )
                .unwrap()
        })
    }

    pub fn flatten(&self) -> Vec<Column<Advice>> {
        self.advices.to_vec()
    }
}
