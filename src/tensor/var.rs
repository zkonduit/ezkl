use std::collections::HashSet;

use log::{error, warn};

use crate::circuit::CheckMode;

use super::*;
/// A wrapper around Halo2's `Column<Fixed>` or `Column<Advice>`.
/// Typically assign [ValTensor]s to [VarTensor]s when laying out a circuit.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub enum VarTensor {
    /// A VarTensor for holding Advice values, which are assigned at proving time.
    Advice {
        /// Vec of Advice columns, we have [[xx][xx][xx]...] where each inner vec is xx columns
        inner: Vec<Vec<Column<Advice>>>,
        ///
        num_inner_cols: usize,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
    },
    /// Dummy var
    Dummy {
        ///
        num_inner_cols: usize,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
    },
    /// Empty var
    #[default]
    Empty,
}

impl VarTensor {
    ///
    pub fn is_advice(&self) -> bool {
        match self {
            VarTensor::Advice { .. } => true,
            _ => false,
        }
    }

    ///
    pub fn max_rows<F: PrimeField>(cs: &ConstraintSystem<F>, logrows: usize) -> usize {
        let base = 2u32;
        base.pow(logrows as u32) as usize - cs.blinding_factors() - 1
    }

    /// Create a new VarTensor::Advice that is unblinded
    /// Arguments
    /// * `cs` - The constraint system
    /// * `logrows` - log2 number of rows in the matrix, including any system and blinding rows.
    /// * `capacity` - The number of advice cells to allocate
    pub fn new_unblinded_advice<F: PrimeField>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        num_inner_cols: usize,
        capacity: usize,
    ) -> Self {
        let max_rows = Self::max_rows(cs, logrows) * num_inner_cols;

        let mut modulo = (capacity / max_rows) + 1;
        // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
        modulo = ((capacity + modulo) / max_rows) + 1;
        let mut advices = vec![];

        if modulo > 1 {
            warn!(
                "using column duplication for {} unblinded advice blocks",
                modulo - 1
            );
        }

        for _ in 0..modulo {
            let mut inner = vec![];
            for _ in 0..num_inner_cols {
                let col = cs.unblinded_advice_column();
                cs.enable_equality(col);
                inner.push(col);
            }
            advices.push(inner);
        }

        VarTensor::Advice {
            inner: advices,
            num_inner_cols,
            col_size: max_rows,
        }
    }

    /// Create a new VarTensor::Advice
    /// Arguments
    /// * `cs` - The constraint system
    /// * `logrows` - log2 number of rows in the matrix, including any system and blinding rows.
    /// * `capacity` - The number of advice cells to allocate
    pub fn new_advice<F: PrimeField>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        num_inner_cols: usize,
        capacity: usize,
    ) -> Self {
        let max_rows = Self::max_rows(cs, logrows);
        let max_assignments = Self::max_rows(cs, logrows) * num_inner_cols;

        let mut modulo = (capacity / max_assignments) + 1;
        // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
        modulo = ((capacity + modulo) / max_assignments) + 1;
        let mut advices = vec![];

        if modulo > 1 {
            warn!("using column duplication for {} advice blocks", modulo - 1);
        }

        for _ in 0..modulo {
            let mut inner = vec![];
            for _ in 0..num_inner_cols {
                let col = cs.advice_column();
                cs.enable_equality(col);
                inner.push(col);
            }
            advices.push(inner);
        }

        VarTensor::Advice {
            inner: advices,
            num_inner_cols,
            col_size: max_rows,
        }
    }

    /// Initializes fixed columns to support the VarTensor::Advice
    /// Arguments
    /// * `cs` - The constraint system
    /// * `logrows` - log2 number of rows in the matrix, including any system and blinding rows.
    /// * `capacity` - The number of advice cells to allocate
    pub fn constant_cols<F: PrimeField>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        num_constants: usize,
        uses_modules: bool,
    ) -> usize {
        if num_constants == 0 && !uses_modules {
            return 0;
        } else if num_constants == 0 && uses_modules {
            let col = cs.fixed_column();
            cs.enable_constant(col);
            return 1;
        }

        let max_rows = Self::max_rows(cs, logrows);

        let mut modulo = num_constants / max_rows + 1;
        // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
        modulo = (num_constants + modulo) / max_rows + 1;

        if modulo > 1 {
            warn!("using column duplication for {} fixed columns", modulo - 1);
        }

        for _ in 0..modulo {
            let col = cs.fixed_column();
            cs.enable_constant(col);
        }
        modulo
    }

    /// Create a new VarTensor::Dummy
    pub fn dummy(logrows: usize, num_inner_cols: usize) -> Self {
        let base = 2u32;
        let max_rows = base.pow(logrows as u32) as usize - 6;
        VarTensor::Dummy {
            col_size: max_rows,
            num_inner_cols,
        }
    }

    /// Gets the dims of the object the VarTensor represents
    pub fn num_blocks(&self) -> usize {
        match self {
            VarTensor::Advice { inner, .. } => inner.len(),
            _ => 0,
        }
    }

    /// Num inner cols
    pub fn num_inner_cols(&self) -> usize {
        match self {
            VarTensor::Advice { num_inner_cols, .. } | VarTensor::Dummy { num_inner_cols, .. } => {
                *num_inner_cols
            }
            _ => 0,
        }
    }

    /// Total number of columns
    pub fn num_cols(&self) -> usize {
        match self {
            VarTensor::Advice { inner, .. } => inner[0].len() * inner.len(),
            _ => 0,
        }
    }

    /// Gets the size of each column
    pub fn col_size(&self) -> usize {
        match self {
            VarTensor::Advice { col_size, .. } | VarTensor::Dummy { col_size, .. } => *col_size,
            _ => 0,
        }
    }

    /// Gets the size of each column
    pub fn block_size(&self) -> usize {
        match self {
            VarTensor::Advice {
                num_inner_cols,
                col_size,
                ..
            }
            | VarTensor::Dummy {
                col_size,
                num_inner_cols,
                ..
            } => *col_size * num_inner_cols,
            _ => 0,
        }
    }

    /// Take a linear coordinate and output the (column, row) position in the storage block.
    pub fn cartesian_coord(&self, linear_coord: usize) -> (usize, usize, usize) {
        // x indexes over blocks of size num_inner_cols
        let x = linear_coord / self.block_size();
        // y indexes over the cols inside a block
        let y = linear_coord % self.num_inner_cols();
        // z indexes over the rows inside a col
        let z = (linear_coord - x * self.block_size()) / self.num_inner_cols();
        (x, y, z)
    }
}

impl VarTensor {
    /// Retrieve the value of a specific cell in the tensor.
    pub fn query_rng<F: PrimeField>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        x: usize,
        y: usize,
        z: i32,
        rng: usize,
    ) -> Result<Tensor<Expression<F>>, halo2_proofs::plonk::Error> {
        match &self {
            // when advice we have 1 col per row
            VarTensor::Advice { inner: advices, .. } => {
                let c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..rng).map(|i| meta.query_advice(advices[x][y], Rotation(z + i as i32))),
                );
                Ok(c)
            }
            _ => {
                error!("VarTensor was not initialized");
                Err(halo2_proofs::plonk::Error::Synthesis)
            }
        }
    }

    /// Retrieve the value of a specific block at an offset in the tensor.
    pub fn query_whole_block<F: PrimeField>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        x: usize,
        z: i32,
        rng: usize,
    ) -> Result<Tensor<Expression<F>>, halo2_proofs::plonk::Error> {
        match &self {
            // when advice we have 1 col per row
            VarTensor::Advice { inner: advices, .. } => {
                let c = Tensor::from({
                    // this should fail if dims is empty, should be impossible
                    let cartesian = (0..rng).cartesian_product(0..self.num_inner_cols());
                    cartesian.map(|(i, y)| meta.query_advice(advices[x][y], Rotation(z + i as i32)))
                });
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
        let (x, y, z) = self.cartesian_coord(offset);
        match &self {
            VarTensor::Advice { inner: advices, .. } => {
                region.assign_advice_from_constant(|| "constant", advices[x][y], z, constant)
            }
            _ => {
                error!("VarTensor was not initialized");
                Err(halo2_proofs::plonk::Error::Synthesis)
            }
        }
    }

    /// Assigns [ValTensor] to the columns of the inner tensor.
    pub fn assign_with_omissions<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
        omissions: &HashSet<&usize>,
    ) -> Result<ValTensor<F>, halo2_proofs::plonk::Error> {
        let mut assigned_coord = 0;
        let mut res: ValTensor<F> = match values {
            ValTensor::Instance { .. } => {
                unimplemented!("cannot assign instance to advice columns with omissions")
            }
            ValTensor::Value { inner: v, .. } => Ok::<_, halo2_proofs::plonk::Error>(
                v.enum_map(|coord, k| {
                    if omissions.contains(&coord) {
                        return Ok(k);
                    }
                    let cell = self.assign_value(region, offset, k.clone(), assigned_coord)?;
                    assigned_coord += 1;

                    match k {
                        ValType::Constant(f) => Ok::<ValType<F>, halo2_proofs::plonk::Error>(
                            ValType::AssignedConstant(cell, f),
                        ),
                        ValType::AssignedConstant(_, f) => Ok(ValType::AssignedConstant(cell, f)),
                        _ => Ok(ValType::PrevAssigned(cell)),
                    }
                })?
                .into(),
            ),
        }?;
        res.set_scale(values.scale());
        Ok(res)
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
                idx,
                initial_offset,
                ..
            } => match &self {
                VarTensor::Advice { inner: v, .. } => {
                    let total_offset: usize = initial_offset
                        + dims[..*idx]
                            .iter()
                            .map(|x| x.iter().product::<usize>())
                            .sum::<usize>();
                    let dims = &dims[*idx];
                    // this should never ever fail
                    let t: Tensor<i32> = Tensor::new(None, dims).unwrap();
                    Ok(t.enum_map(|coord, _| {
                        let (x, y, z) = self.cartesian_coord(offset + coord);
                        region.assign_advice_from_instance(
                            || "pub input anchor",
                            *instance,
                            coord + total_offset,
                            v[x][y],
                            z,
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
                    let cell = self.assign_value(region, offset, k.clone(), coord)?;
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
        row: usize,
        offset: usize,
        values: &ValTensor<F>,
        single_inner_col: bool,
    ) -> Result<(ValTensor<F>, usize, usize), halo2_proofs::plonk::Error> {
        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {
                let duplication_freq = if single_inner_col {
                    self.col_size()
                } else {
                    self.block_size()
                };

                let num_repeats = if single_inner_col {
                    1
                } else {
                    self.num_inner_cols()
                };

                let duplication_offset = if single_inner_col {
                    row
                } else {
                    offset
                };


                // duplicates every nth element to adjust for column overflow
                let mut res: ValTensor<F> = v.duplicate_every_n(duplication_freq, num_repeats, duplication_offset).unwrap().into();
                let total_used_len = res.len();
                let total_constants = res.num_constants();
                res.remove_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                Ok((res, total_used_len, total_constants))
            }
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor but allows for column wrapping for accumulated operations.
    /// Duplication occurs by copying the last cell of the column to the first cell next column and creating a copy constraint between the two.
    pub fn assign_with_duplication<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        row: usize,
        offset: usize,
        values: &ValTensor<F>,
        check_mode: &CheckMode,
        single_inner_col: bool,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
        let mut prev_cell = None;

        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {

                let duplication_freq = if single_inner_col {
                    self.col_size()
                } else {
                    self.block_size()
                };

                let num_repeats = if single_inner_col {
                    1
                } else {
                    self.num_inner_cols()
                };

                let duplication_offset = if single_inner_col {
                    row
                } else {
                    offset
                };

                // duplicates every nth element to adjust for column overflow
                let v = v.duplicate_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();
                let mut res: ValTensor<F> = {
                    v.enum_map(|coord, k| {

                    let step = if !single_inner_col {
                        1
                    } else {
                        self.num_inner_cols()
                    };

                    let (x, y, z) = self.cartesian_coord(offset + coord * step);
                    if matches!(check_mode, CheckMode::SAFE) && coord > 0 && z == 0 && y == 0 {
                        // assert that duplication occurred correctly
                        assert_eq!(Into::<i32>::into(k.clone()), Into::<i32>::into(v[coord - 1].clone()));
                    };

                    let cell = self.assign_value(region, offset, k.clone(), coord * step)?;

                    if single_inner_col {
                    if z == 0 {
                        // if we are at the end of the column, we need to copy the cell to the next column
                        prev_cell = Some(cell.clone());
                    } else if coord > 0 && z == 0 && single_inner_col {
                        if let Some(prev_cell) = prev_cell.as_ref() {
                            region.constrain_equal(prev_cell.cell(),cell.cell())?;
                        } else {
                            error!("Error copy-constraining previous value: {:?}", (x,y));
                            return Err(halo2_proofs::plonk::Error::Synthesis);
                        }
                    }}

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
                res.remove_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();

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
        coord: usize,
    ) -> Result<AssignedCell<F, F>, halo2_proofs::plonk::Error> {
        let (x, y, z) = self.cartesian_coord(offset + coord);
        match k {
            ValType::Value(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    region.assign_advice(|| "k", advices[x][y], z, || v)
                }
                _ => unimplemented!(),
            },
            ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    v.copy_advice(|| "k", region, advices[x][y], z)
                }
                _ => {
                    error!("PrevAssigned is only supported for advice columns");
                    Err(halo2_proofs::plonk::Error::Synthesis)
                }
            },
            ValType::AssignedValue(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => region
                    .assign_advice(|| "k", advices[x][y], z, || v)
                    .map(|a| a.evaluate()),
                _ => unimplemented!(),
            },
            ValType::Constant(v) => self.assign_constant(region, offset + coord, v),
        }
    }
}
