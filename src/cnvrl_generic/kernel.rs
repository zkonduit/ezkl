use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;

pub type Kernel<T, const HEIGHT: usize, const WIDTH: usize> = [[T; HEIGHT]; WIDTH];

#[derive(Debug, Clone)]
pub struct KernelConfig<F: FieldExt, const HEIGHT: usize, const WIDTH: usize>(
    Kernel<Column<Fixed>, HEIGHT, WIDTH>,
    PhantomData<F>,
);

impl<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> KernelConfig<F, HEIGHT, WIDTH> {
    pub fn configure(meta: &mut ConstraintSystem<F>) -> Self {
        Self(
            (0..WIDTH)
                .map(|_| {
                    (0..HEIGHT)
                        .map(|_| meta.fixed_column())
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            PhantomData,
        )
    }

    pub fn query(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation: Rotation,
    ) -> [[Expression<F>; HEIGHT]; WIDTH] {
        self.0
            .iter()
            .map(|cols| {
                cols.iter()
                    .map(|&column| meta.query_fixed(column, rotation))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn assign_kernel_2d(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        kernel: Kernel<Value<F>, HEIGHT, WIDTH>,
    ) -> Result<Kernel<AssignedCell<Assigned<F>, F>, HEIGHT, WIDTH>, Error> {
        let res = kernel
            .iter()
            .enumerate()
            .map(
                |(col_idx, column)| -> Result<[AssignedCell<Assigned<F>, F>; HEIGHT], Error> {
                    let res = column
                        .iter()
                        .enumerate()
                        .map(
                            |(row_idx, &cell)| -> Result<AssignedCell<Assigned<F>, F>, Error> {
                                region.assign_fixed(
                                    || {
                                        format!(
                                            "kernel value at row: {:?}, column: {:?}",
                                            row_idx, col_idx
                                        )
                                    },
                                    self.0[col_idx][row_idx],
                                    offset,
                                    || cell.into(),
                                )
                            },
                        )
                        .collect::<Vec<_>>();
                    let res: Result<Vec<_>, Error> = res.into_iter().collect();
                    res.map(|vec| vec.try_into().unwrap())
                },
            )
            .collect::<Vec<_>>();

        let res: Result<Vec<_>, Error> = res.into_iter().collect();
        res.map(|v| v.try_into().unwrap())
    }
}
