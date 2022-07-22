use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;

pub type Image<T, const HEIGHT: usize, const WIDTH: usize> = [[T; HEIGHT]; WIDTH];

#[derive(Debug, Clone)]
pub struct ImageConfig<F: FieldExt, const HEIGHT: usize, const WIDTH: usize>(
    [Column<Advice>; WIDTH],
    PhantomData<F>,
);

impl<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> ImageConfig<F, HEIGHT, WIDTH> {
    pub fn configure(
        advices: [Column<Advice>; WIDTH],
    ) -> Self {
        Self(
            advices,
            PhantomData,
        )
    }

    pub fn query(
        &self,
        meta: &mut VirtualCells<'_, F>,
        offset: usize
    ) -> [[Expression<F>; HEIGHT]; WIDTH] {
        self.0
            .iter()
            .map(|&column| {
                (0..HEIGHT).map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32)))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn assign_image_2d(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        image: Image<Value<F>, HEIGHT, WIDTH>,
    ) -> Result<Image<AssignedCell<Assigned<F>, F>, HEIGHT, WIDTH>, Error> {
        let res = image
            .iter()
            .enumerate()
            .map(
                |(col_idx, column)| -> Result<[AssignedCell<Assigned<F>, F>; HEIGHT], Error> {
                    let res = column
                        .iter()
                        .enumerate()
                        .map(
                            |(row_idx, &cell)| -> Result<AssignedCell<Assigned<F>, F>, Error> {
                                region.assign_advice(
                                    || {
                                        format!(
                                            "pixel at row: {:?}, column: {:?}",
                                            row_idx, col_idx
                                        )
                                    },
                                    self.0[col_idx],
                                    offset + row_idx,
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

    pub fn flatten(&self) -> Vec<Column<Advice>> {
        self.0.to_vec()
    }
}
