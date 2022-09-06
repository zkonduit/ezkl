use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;

pub type Image<T, const HEIGHT: usize, const WIDTH: usize> = Tensor<T>;

#[derive(Debug, Clone)]
pub struct ImageConfig<F: FieldExt, const HEIGHT: usize, const WIDTH: usize>(
    Tensor<Column<Advice>>,
    PhantomData<F>,
);

impl<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> ImageConfig<F, HEIGHT, WIDTH>
where
    Value<F>: TensorType,
{
    pub fn configure(advices: Tensor<Column<Advice>>) -> Self {
        Self(advices, PhantomData)
    }

    pub fn query(&self, meta: &mut VirtualCells<'_, F>, offset: usize) -> Tensor<Expression<F>> {
        let mut t: Tensor<Expression<F>> = self
            .0
            .map(|column| {
                Tensor::from(
                    (0..HEIGHT)
                        .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                )
            })
            .flatten();
        t.reshape(&[HEIGHT, WIDTH]);
        t
    }

    pub fn assign_image_2d(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        image: Image<Value<F>, HEIGHT, WIDTH>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        let mut res = Vec::new();
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                res.push(
                    region
                        .assign_advice(
                            || format!("pixel at row: {:?}, column: {:?}", j, i),
                            self.0[i],
                            offset + j,
                            || image.get(&[i, j]).into(),
                        )
                        .unwrap(),
                )
            }
        }
        let mut t = Tensor::from(res.into_iter());
        t.reshape(&[HEIGHT, WIDTH]);
        t
    }

    pub fn flatten(&self) -> Vec<Column<Advice>> {
        self.0.to_vec()
    }
}
