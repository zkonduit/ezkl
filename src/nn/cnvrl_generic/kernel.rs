use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;
use crate::tensor::{Tensor, TensorType};

#[derive(Debug, Clone)]
pub struct KernelConfig<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> {
    fixed: Tensor<Column<Fixed>>,
    marker: PhantomData<F>,
}

impl<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> KernelConfig<F, HEIGHT, WIDTH>
where
    Value<F>: TensorType,
{
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Self {
        let mut fixed = Tensor::from((0..WIDTH * HEIGHT).map(|_| meta.fixed_column()));
        fixed.reshape(&[WIDTH, HEIGHT]);
        Self {
            fixed,
            marker: PhantomData,
        }
    }

    pub fn query(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation: Rotation,
    ) -> Tensor<Expression<F>> {
        let mut t = self.fixed.map(|col| meta.query_fixed(col, rotation));
        t.reshape(&[WIDTH, HEIGHT]);
        t
    }

    pub fn assign_kernel_2d(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        kernel: Tensor<Value<F>>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        let dims = kernel.dims();
        assert!(dims.len() == 2);
        kernel.enum_map(|i, k| {
            let row = i % dims[1];
            let col = i / dims[1];
            println! {"row {:?} col {:?} w {:?}" , row, col, k};
            region
                .assign_fixed(
                    || format!("kernel at row: {:?}, column: {:?}", row, col),
                    // row indices
                    self.fixed.get(&[col, row]),
                    // columns indices
                    offset,
                    || k.into(),
                )
                .unwrap()
        })
    }
}
