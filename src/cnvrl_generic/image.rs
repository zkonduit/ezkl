use halo2_proofs::{circuit::AssignedCell, plonk::VirtualCells};

use super::*;

pub type Image<T, const HEIGHT: usize, const WIDTH: usize> = [[T; HEIGHT]; WIDTH];

#[derive(Debug, Clone)]
pub struct ImageConfig<F: FieldExt, const HEIGHT: usize, const WIDTH: usize>(
    Image<Column<Advice>, HEIGHT, WIDTH>,
    PhantomData<F>,
);

impl<F: FieldExt, const HEIGHT: usize, const WIDTH: usize> ImageConfig<F, HEIGHT, WIDTH> {
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        advices: [Column<Advice>; HEIGHT * WIDTH],
    ) -> Self {
        Self(
            (0..WIDTH)
                .map(|w| {
                    (0..HEIGHT)
                        .map(|h| advices[h * WIDTH + w])
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
                    .map(|&column| meta.query_advice(column, rotation))
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
