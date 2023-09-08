use log::{error, warn};

use crate::circuit::CheckMode;

use super::*;
/// A wrapper around Halo2's `Column<Fixed>` or `Column<Advice>`.
/// Typically assign [ValTensor]s to [VarTensor]s when laying out a circuit.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub enum VarTensor {
    /// A VarTensor for holding Advice values, which are assigned at proving time.
    Advice {
        /// Vec of Advice columns
        inner: Vec<Column<Advice>>,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
    },
    /// Dummy var
    Dummy {
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
    },
    /// Empty var
    #[default]
    Empty,
}

impl VarTensor {
    /// Create a new VarTensor::Advice
    /// Arguments
    /// * `cs` - The constraint system
    /// * `logrows` - log2 number of rows in the matrix, including any system and blinding rows.
    /// * `capacity` - The number of advice cells to allocate
    pub fn new_advice<F: PrimeField>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        capacity: usize,
    ) -> Self {
        let base = 2u32;
        let max_rows = base.pow(logrows as u32) as usize - cs.blinding_factors() - 1;

        let mut modulo = (capacity / max_rows) + 1;
        // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
        modulo = ((capacity + modulo) / max_rows) + 1;
        let mut advices = vec![];

        if modulo > 1 {
            warn!(
                "will be using column duplication for {} columns",
                modulo - 1
            );
        }

        for _ in 0..modulo {
            let col = cs.advice_column();
            cs.enable_equality(col);
            advices.push(col);
        }

        VarTensor::Advice {
            inner: advices,
            col_size: max_rows,
        }
    }

    /// Create a new VarTensor::Dummy
    pub fn dummy(logrows: usize) -> Self {
        let base = 2u32;
        let max_rows = base.pow(logrows as u32) as usize - 6;
        VarTensor::Dummy { col_size: max_rows }
    }

    /// Gets the dims of the object the VarTensor represents
    pub fn num_cols(&self) -> usize {
        match self {
            VarTensor::Advice { inner, .. } => inner.len(),
            _ => 0,
        }
    }

    /// Gets the size of each column
    pub fn col_size(&self) -> usize {
        match self {
            VarTensor::Advice { col_size, .. } | VarTensor::Dummy { col_size } => *col_size,
            _ => 0,
        }
    }

    /// Take a linear coordinate and output the (column, row) position in the storage block.
    pub fn cartesian_coord(&self, linear_coord: usize) -> (usize, usize) {
        match self {
            VarTensor::Advice { col_size, .. } => {
                let x = linear_coord / col_size;
                let y = linear_coord % col_size;
                (x, y)
            }
            _ => (0, 0),
        }
    }
}

impl VarTensor {
    /// Retrieve the value of a specific cell in the tensor.
    pub fn query_rng<F: PrimeField>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation_offset: i32,
        idx_offset: usize,
        rng: usize,
    ) -> Result<Tensor<Expression<F>>, halo2_proofs::plonk::Error> {
        match &self {
            // when advice we have 1 col per row
            VarTensor::Advice { inner: advices, .. } => {
                let c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..rng).map(|i| {
                        let (x, y) = self.cartesian_coord(idx_offset + i);
                        meta.query_advice(advices[x], Rotation(rotation_offset + y as i32))
                    }),
                );
                Ok(c)
            }
            _ => {
                error!("VarTensor was not initialized");
                Err(halo2_proofs::plonk::Error::Synthesis)
            }
        }
    }

    /// Assigns a constant value to a specific cell in the tensor.
    pub fn assign_constant<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        constant: F,
    ) -> Result<AssignedCell<F, F>, halo2_proofs::plonk::Error> {
        let (x, y) = self.cartesian_coord(offset);

        match &self {
            VarTensor::Advice { inner: advices, .. } => {
                region.assign_advice_from_constant(|| "constant", advices[x], y, constant)
            }
            _ => panic!(),
        }
    }

    /// Assigns [ValTensor] to the columns of the inner tensor.
    pub fn assign<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, halo2_proofs::plonk::Error> {
        let mut res: ValTensor<F> = match values {
            ValTensor::Instance {
                inner: instance,
                dims,
                ..
            } => match &self {
                VarTensor::Advice { inner: v, .. } => {
                    // this should never ever fail
                    let t: Tensor<i32> = Tensor::new(None, dims).unwrap();
                    Ok(t.enum_map(|coord, _| {
                        let (x, y) = self.cartesian_coord(offset + coord);
                        region.assign_advice_from_instance(
                            || "pub input anchor",
                            *instance,
                            coord,
                            v[x],
                            y,
                        )
                    })?
                    .into())
                }
                _ => {
                    error!("Instance is only supported for advice columns");
                    Err(halo2_proofs::plonk::Error::Synthesis)
                }
            },
            ValTensor::Value { inner: v, .. } => Ok(v
                .enum_map(|coord, k| {
                    let (x, y) = self.cartesian_coord(offset + coord);
                    let cell = self.assign_value(region, offset, k.clone(), x, y, coord)?;

                    match k {
                        ValType::Constant(f) => Ok::<ValType<F>, halo2_proofs::plonk::Error>(
                            ValType::AssignedConstant(cell, f),
                        ),
                        ValType::AssignedConstant(_, f) => Ok(ValType::AssignedConstant(cell, f)),
                        _ => Ok(ValType::PrevAssigned(cell)),
                    }
                })?
                .into()),
        }?;
        res.set_scale(values.scale());
        Ok(res)
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor but allows for column wrapping for accumulated operations.
    /// Duplication occurs by copying the last cell of the column to the first cell next column and creating a copy constraint between the two.
    pub fn dummy_assign_with_duplication<F: PrimeField + TensorType + PartialOrd>(
        &self,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {
                // duplicates every nth element to adjust for column overflow
                let mut res: ValTensor<F> = v.duplicate_every_n(self.col_size(), offset).unwrap().into();
                let total_used_len = res.len();
                res.remove_every_n(self.col_size(), offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                Ok((res, total_used_len))
            }
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor but allows for column wrapping for accumulated operations.
    /// Duplication occurs by copying the last cell of the column to the first cell next column and creating a copy constraint between the two.
    pub fn assign_with_duplication<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
        check_mode: &CheckMode,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
        let mut prev_cell = None;

        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {
                // duplicates every nth element to adjust for column overflow
                let v = v.duplicate_every_n(self.col_size(), offset).unwrap();
                let mut res: ValTensor<F> = {
                    v.enum_map(|coord, k| {

                    let (x, y) = self.cartesian_coord(offset + coord);
                    if matches!(check_mode, CheckMode::SAFE) && coord > 0 && y == 0 {
                        // assert that duplication occurred correctly
                        assert_eq!(Into::<i32>::into(k.clone()), Into::<i32>::into(v[coord - 1].clone()));
                    };

                    let cell = self.assign_value(region, offset, k.clone(), x, y, coord)?;

                    if y == (self.col_size() - 1)   {
                        // if we are at the end of the column, we need to copy the cell to the next column
                        prev_cell = Some(cell.clone());
                    } else if coord > 0 && y == 0 {
                        if let Some(prev_cell) = prev_cell.as_ref() {
                            region.constrain_equal(prev_cell.cell(),cell.cell())?;
                        } else {
                            error!("Error assigning copy-constraining previous value: {:?}", (x,y));
                            return Err(halo2_proofs::plonk::Error::Synthesis);
                        }
                    }

                    match k {
                        ValType::Constant(f) => {
                            Ok(ValType::AssignedConstant(cell, f))
                        },
                        ValType::AssignedConstant(_, f) => {
                            Ok(ValType::AssignedConstant(cell, f))
                        },
                        _ => {
                            Ok(ValType::PrevAssigned(cell))
                        }
                    }

                })?.into()};
                let total_used_len = res.len();
                res.remove_every_n(self.col_size(), offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                if matches!(check_mode, CheckMode::SAFE) {
                     // during key generation this will be 0 so we use this as a flag to check
                     // TODO: this isn't very safe and would be better to get the phase directly
                    let is_assigned = !Into::<Tensor<i32>>::into(res.clone().get_inner().unwrap())
                    .iter()
                    .all(|&x| x == 0);
                    if is_assigned {
                        assert_eq!(
                            Into::<Tensor<i32>>::into(values.get_inner().unwrap()),
                            Into::<Tensor<i32>>::into(res.get_inner().unwrap())
                    )};
                }

                Ok((res, total_used_len))
            }
        }
    }

    fn assign_value<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        k: ValType<F>,
        x: usize,
        y: usize,
        coord: usize,
    ) -> Result<AssignedCell<F, F>, halo2_proofs::plonk::Error> {
        match k {
            ValType::Value(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    region.assign_advice(|| "k", advices[x], y, || v)
                }
                _ => unimplemented!(),
            },
            ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    v.copy_advice(|| "k", region, advices[x], y)
                }
                _ => {
                    error!("PrevAssigned is only supported for advice columns");
                    Err(halo2_proofs::plonk::Error::Synthesis)
                }
            },
            ValType::AssignedValue(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => region
                    .assign_advice(|| "k", advices[x], y, || v)
                    .map(|a| a.evaluate()),
                _ => unimplemented!(),
            },
            ValType::Constant(v) => self.assign_constant(region, offset + coord, v),
        }
    }
}
