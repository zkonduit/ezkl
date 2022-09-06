use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;
use crate::tensor::{Tensor, TensorType};

pub type Kernel<T, const HEIGHT: usize, const WIDTH: usize> = Tensor<T>;

#[derive(Debug, Clone)]
pub struct KernelConfig<F: FieldExt, const HEIGHT: usize, const WIDTH: usize>(
    Kernel<Column<Fixed>, HEIGHT, WIDTH>,
    PhantomData<F>,
);

impl<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> KernelConfig<F, HEIGHT, WIDTH>
where
    Value<F>: TensorType,
{
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Self {
        let mut vec = Vec::new();
        for _ in 0..WIDTH {
            for _ in 0..HEIGHT {
                vec.push(meta.fixed_column())
            }
        }
        let mut t = Tensor::from(vec.into_iter());
        t.reshape(&[WIDTH, HEIGHT]);
        Self(t, PhantomData)
    }

    pub fn query(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation: Rotation,
    ) -> Tensor<Expression<F>> {
        let mut t = Tensor::from(self.0.iter().map(|&col| meta.query_fixed(col, rotation)));
        t.reshape(&[WIDTH, HEIGHT]);
        t
    }

    pub fn assign_kernel_2d(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        kernel: Kernel<Value<F>, HEIGHT, WIDTH>,
    ) -> Kernel<AssignedCell<Assigned<F>, F>, HEIGHT, WIDTH> {
        let mut res = Vec::new();
        println!("self {:?}", self.0);
        for i in 0..WIDTH {
            for j in 0..HEIGHT {
                res.push(
                    region
                        .assign_fixed(
                            || format!("kernel at row: {:?}, column: {:?}", j, i),
                            self.0.get(&[i, j]),
                            offset,
                            || kernel.get(&[i, j]).into(),
                        )
                        .unwrap(),
                )
            }
        }
        let mut t = Tensor::from(res.into_iter());
        t.reshape(&[WIDTH, HEIGHT]);
        t
    }
}
