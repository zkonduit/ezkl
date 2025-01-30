use std::collections::HashSet;

use log::{debug, error, warn};

use crate::circuit::{region::ConstantsMap, CheckMode};

use super::*;
/// A wrapper around Halo2's Column types that represents a tensor of variables in the circuit.
/// VarTensors are used to store and manage circuit columns, typically for assigning ValTensor
/// values during circuit layout. The tensor organizes storage into blocks of columns, where each
/// block contains multiple columns and each column contains multiple rows.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub enum VarTensor {
    /// A VarTensor for holding Advice values, which are assigned at proving time.
    Advice {
        /// Vec of Advice columns, we have [[xx][xx][xx]...] where each inner vec is xx columns
        inner: Vec<Vec<Column<Advice>>>,
        /// The number of columns in each inner block
        num_inner_cols: usize,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
    },
    /// A placeholder tensor used for testing or temporary storage
    Dummy {
        /// The number of columns in each inner block
        num_inner_cols: usize,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
    },
    /// An empty tensor with no storage
    #[default]
    Empty,
}

impl VarTensor {
    /// Returns the name of the tensor variant as a static string
    pub fn name(&self) -> &'static str {
        match self {
            VarTensor::Advice { .. } => "Advice",
            VarTensor::Dummy { .. } => "Dummy",
            VarTensor::Empty => "Empty",
        }
    }

    /// Returns true if the tensor is an Advice variant
    pub fn is_advice(&self) -> bool {
        matches!(self, VarTensor::Advice { .. })
    }

    /// Calculates the maximum number of usable rows in the constraint system
    ///
    /// # Arguments
    /// * `cs` - The constraint system
    /// * `logrows` - Log base 2 of the total number of rows (including system and blinding rows)
    ///
    /// # Returns
    /// The maximum number of usable rows after accounting for blinding factors
    pub fn max_rows<F: PrimeField>(cs: &ConstraintSystem<F>, logrows: usize) -> usize {
        let base = 2u32;
        base.pow(logrows as u32) as usize - cs.blinding_factors() - 1
    }

    /// Creates a new VarTensor::Advice with unblinded columns. Unblinded columns are used when
    /// the values do not need to be hidden in the proof.
    ///
    /// # Arguments
    /// * `cs` - The constraint system to create columns in
    /// * `logrows` - Log base 2 of the total number of rows
    /// * `num_inner_cols` - Number of columns in each inner block
    /// * `capacity` - Total number of advice cells to allocate
    ///
    /// # Returns
    /// A new VarTensor::Advice with unblinded columns enabled for equality constraints
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

    /// Creates a new VarTensor::Advice with standard (blinded) columns, used when
    /// the values need to be hidden in the proof.
    ///
    /// # Arguments
    /// * `cs` - The constraint system to create columns in
    /// * `logrows` - Log base 2 of the total number of rows
    /// * `num_inner_cols` - Number of columns in each inner block
    /// * `capacity` - Total number of advice cells to allocate
    ///
    /// # Returns
    /// A new VarTensor::Advice with blinded columns enabled for equality constraints
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
            debug!("using column duplication for {} advice blocks", modulo - 1);
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

    /// Initializes fixed columns in the constraint system to support the VarTensor::Advice
    /// Fixed columns are used for constant values that are known at circuit creation time.
    ///
    /// # Arguments
    /// * `cs` - The constraint system to create columns in
    /// * `logrows` - Log base 2 of the total number of rows
    /// * `num_constants` - Number of constant values needed
    /// * `module_requires_fixed` - Whether the module requires at least one fixed column
    ///
    /// # Returns
    /// The number of fixed columns created
    pub fn constant_cols<F: PrimeField>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        num_constants: usize,
        module_requires_fixed: bool,
    ) -> usize {
        if num_constants == 0 && !module_requires_fixed {
            return 0;
        } else if num_constants == 0 && module_requires_fixed {
            let col = cs.fixed_column();
            cs.enable_constant(col);
            return 1;
        }

        let max_rows = Self::max_rows(cs, logrows);

        let mut modulo = num_constants / max_rows + 1;
        // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
        modulo = (num_constants + modulo) / max_rows + 1;

        if modulo > 1 {
            debug!("using column duplication for {} fixed columns", modulo - 1);
        }

        for _ in 0..modulo {
            let col = cs.fixed_column();
            cs.enable_constant(col);
        }
        modulo
    }

    /// Creates a new dummy VarTensor for testing or temporary storage
    ///
    /// # Arguments
    /// * `logrows` - Log base 2 of the total number of rows
    /// * `num_inner_cols` - Number of columns in each inner block
    ///
    /// # Returns
    /// A new VarTensor::Dummy with the specified dimensions
    pub fn dummy(logrows: usize, num_inner_cols: usize) -> Self {
        let base = 2u32;
        let max_rows = base.pow(logrows as u32) as usize - 6;
        VarTensor::Dummy {
            col_size: max_rows,
            num_inner_cols,
        }
    }

    /// Returns the number of blocks in the tensor
    pub fn num_blocks(&self) -> usize {
        match self {
            VarTensor::Advice { inner, .. } => inner.len(),
            _ => 0,
        }
    }

    /// Returns the number of columns in each inner block
    pub fn num_inner_cols(&self) -> usize {
        match self {
            VarTensor::Advice { num_inner_cols, .. } | VarTensor::Dummy { num_inner_cols, .. } => {
                *num_inner_cols
            }
            _ => 0,
        }
    }

    /// Returns the total number of columns across all blocks
    pub fn num_cols(&self) -> usize {
        match self {
            VarTensor::Advice { inner, .. } => inner[0].len() * inner.len(),
            _ => 0,
        }
    }

    /// Returns the maximum number of rows in each column
    pub fn col_size(&self) -> usize {
        match self {
            VarTensor::Advice { col_size, .. } | VarTensor::Dummy { col_size, .. } => *col_size,
            _ => 0,
        }
    }

    /// Returns the total size of each block (num_inner_cols * col_size)
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

    /// Converts a linear coordinate to (block, column, row) coordinates in the storage
    ///
    /// # Arguments
    /// * `linear_coord` - The linear index to convert
    ///
    /// # Returns
    /// A tuple of (block_index, column_index, row_index)
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
    /// Queries a range of cells in the tensor during circuit synthesis
    ///
    /// # Arguments
    /// * `meta` - Virtual cells accessor
    /// * `x` - Block index
    /// * `y` - Column index within block
    /// * `z` - Starting row offset
    /// * `rng` - Number of consecutive rows to query
    ///
    /// # Returns
    /// A tensor of expressions representing the queried cells
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

    /// Queries an entire block of cells at a given offset
    ///
    /// # Arguments
    /// * `meta` - Virtual cells accessor
    /// * `x` - Block index
    /// * `z` - Row offset
    /// * `rng` - Number of consecutive rows to query
    ///
    /// # Returns
    /// A tensor of expressions representing the queried block
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

    /// Assigns a constant value to a specific cell in the tensor
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `offset` - Base offset for the assignment
    /// * `coord` - Coordinate within the tensor
    /// * `constant` - The constant value to assign
    ///
    /// # Returns
    /// The assigned cell or an error if assignment fails
    pub fn assign_constant<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        coord: usize,
        constant: F,
    ) -> Result<AssignedCell<F, F>, halo2_proofs::plonk::Error> {
        let (x, y, z) = self.cartesian_coord(offset + coord);
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

    /// Assigns values from a ValTensor to this tensor, excluding specified positions
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `offset` - Base offset for assignments
    /// * `values` - The ValTensor containing values to assign
    /// * `omissions` - Set of positions to skip during assignment
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// The assigned ValTensor or an error if assignment fails
    pub fn assign_with_omissions<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
        omissions: &HashSet<usize>,
        constants: &mut ConstantsMap<F>,
    ) -> Result<ValTensor<F>, halo2_proofs::plonk::Error> {
        let mut assigned_coord = 0;
        let mut res: ValTensor<F> = match values {
            ValTensor::Instance { .. } => {
                unimplemented!("cannot assign instance to advice columns with omissions")
            }
            ValTensor::Value { inner: v, .. } => Ok::<ValTensor<F>, halo2_proofs::plonk::Error>(
                v.enum_map(|coord, k| {
                    if omissions.contains(&coord) {
                        return Ok::<_, halo2_proofs::plonk::Error>(k);
                    }
                    let cell =
                        self.assign_value(region, offset, k.clone(), assigned_coord, constants)?;
                    assigned_coord += 1;
                    Ok::<_, halo2_proofs::plonk::Error>(cell)
                })?
                .into(),
            ),
        }?;
        res.set_scale(values.scale());
        Ok(res)
    }

    /// Assigns values from a ValTensor to this tensor
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `offset` - Base offset for assignments
    /// * `values` - The ValTensor containing values to assign
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// The assigned ValTensor or an error if assignment fails
    pub fn assign<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
        constants: &mut ConstantsMap<F>,
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
                    let t: Tensor<IntegerRep> = Tensor::new(None, dims).unwrap();
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
                    self.assign_value(region, offset, k.clone(), coord, constants)
                })?
                .into()),
        }?;
        res.set_scale(values.scale());
        Ok(res)
    }

    /// Returns the remaining available space in a column for assignments
    ///
    /// # Arguments
    /// * `offset` - Current offset in the column
    /// * `values` - The ValTensor to check space for
    ///
    /// # Returns
    /// The number of rows that need to be flushed or an error if space is insufficient
    pub fn get_column_flush<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        &self,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Result<usize, halo2_proofs::plonk::Error> {
        if values.len() > self.col_size() {
            error!(
                "There are too many values to flush for this column size, try setting the logrows to a higher value (eg. --logrows 22 on the cli)"
            );
            return Err(halo2_proofs::plonk::Error::Synthesis);
        }

        // this can only be called on columns that have a single inner column
        if self.num_inner_cols() != 1 {
            error!("This function can only be called on columns with a single inner column");
            return Err(halo2_proofs::plonk::Error::Synthesis);
        }

        // check if the values fit in the remaining space of the column
        let current_cartesian = self.cartesian_coord(offset);
        let final_cartesian = self.cartesian_coord(offset + values.len());

        let mut flush_len = 0;
        if current_cartesian.0 != final_cartesian.0 {
            debug!("Values overflow the column, flushing to next column");
            // diff is the number of values that overflow the column
            flush_len += self.col_size() - current_cartesian.2;
        }

        Ok(flush_len)
    }

    /// Assigns values to a single column, avoiding column overflow by flushing to the next column if needed
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `offset` - Base offset for assignments
    /// * `values` - The ValTensor containing values to assign
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// A tuple of (assigned ValTensor, number of rows flushed) or an error if assignment fails
    pub fn assign_exact_column<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
        constants: &mut ConstantsMap<F>,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
        let flush_len = self.get_column_flush(offset, values)?;

        let assigned_vals = self.assign(region, offset + flush_len, values, constants)?;

        Ok((assigned_vals, flush_len))
    }

    /// Assigns values with duplication in dummy mode, used for testing and simulation
    ///
    /// # Arguments
    /// * `row` - Starting row for assignment
    /// * `offset` - Base offset for assignments
    /// * `values` - The ValTensor containing values to assign
    /// * `single_inner_col` - Whether to treat as a single column
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// A tuple of (assigned ValTensor, total length used) or an error if assignment fails
    pub fn dummy_assign_with_duplication<
        F: PrimeField + TensorType + PartialOrd + std::hash::Hash,
    >(
        &self,
        row: usize,
        offset: usize,
        values: &ValTensor<F>,
        single_inner_col: bool,
        constants: &mut ConstantsMap<F>,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
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

                let constants_map = res.create_constants_map();
                constants.extend(constants_map);

                let total_used_len = res.len();
                res.remove_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                Ok((res, total_used_len))
            }
        }
    }

    /// Assigns values with duplication but without enforcing constraints between duplicated values
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `offset` - Base offset for assignments
    /// * `values` - The ValTensor containing values to assign
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// A tuple of (assigned ValTensor, total length used) or an error if assignment fails
    pub fn assign_with_duplication_unconstrained<
        F: PrimeField + TensorType + PartialOrd + std::hash::Hash,
    >(
        &self,
        region: &mut Region<F>,
        offset: usize,
        values: &ValTensor<F>,
        constants: &mut ConstantsMap<F>,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {

                let duplication_freq = self.block_size();

                let num_repeats = self.num_inner_cols();

                let duplication_offset = offset;

                // duplicates every nth element to adjust for column overflow
                let v = v.duplicate_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();
                let mut res: ValTensor<F> = {
                    v.enum_map(|coord, k| {
                    let cell = self.assign_value(region, offset, k.clone(), coord, constants)?;
                    Ok::<_, halo2_proofs::plonk::Error>(cell)

                })?.into()};
                let total_used_len = res.len();
                res.remove_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                Ok((res, total_used_len))
            }
        }
    }

    /// Assigns values with duplication and enforces equality constraints between duplicated values
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `row` - Starting row for assignment
    /// * `offset` - Base offset for assignments
    /// * `values` - The ValTensor containing values to assign
    /// * `check_mode` - Mode for checking equality constraints
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// A tuple of (assigned ValTensor, total length used) or an error if assignment fails
    pub fn assign_with_duplication_constrained<
        F: PrimeField + TensorType + PartialOrd + std::hash::Hash,
    >(
        &self,
        region: &mut Region<F>,
        row: usize,
        offset: usize,
        values: &ValTensor<F>,
        check_mode: &CheckMode,
        constants: &mut ConstantsMap<F>,
    ) -> Result<(ValTensor<F>, usize), halo2_proofs::plonk::Error> {
        let mut prev_cell = None;

        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {

                let duplication_freq = self.col_size();
                let num_repeats = 1;
                let duplication_offset = row;

                // duplicates every nth element to adjust for column overflow
                let v = v.duplicate_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();
                let mut res: ValTensor<F> =
                    v.enum_map(|coord, k| {

                    let step = self.num_inner_cols();

                    let (x, y, z) = self.cartesian_coord(offset + coord * step);
                    if matches!(check_mode, CheckMode::SAFE) && coord > 0 && z == 0 && y == 0 {
                        // assert that duplication occurred correctly
                        assert_eq!(Into::<IntegerRep>::into(k.clone()), Into::<IntegerRep>::into(v[coord - 1].clone()));
                    };

                    let cell = self.assign_value(region, offset, k.clone(), coord * step, constants)?;

                    let at_end_of_column = z == duplication_freq - 1;
                    let at_beginning_of_column = z == 0;

                    if at_end_of_column {
                        // if we are at the end of the column, we need to copy the cell to the next column
                        prev_cell = Some(cell.clone());
                    } else if coord > 0 && at_beginning_of_column  {
                        if let Some(prev_cell) = prev_cell.as_ref() {
                            let cell = if let Some(cell) = cell.cell() {
                                cell
                            } else {
                                error!("Error getting cell: {:?}", (x,y));
                                return Err(halo2_proofs::plonk::Error::Synthesis);
                            };
                            let prev_cell = if let Some(prev_cell) = prev_cell.cell() {
                                prev_cell
                            } else {
                                error!("Error getting prev cell: {:?}", (x,y));
                                return Err(halo2_proofs::plonk::Error::Synthesis);
                            };
                            region.constrain_equal(prev_cell,cell)?;
                        } else {
                            error!("Previous cell was not set");
                            return Err(halo2_proofs::plonk::Error::Synthesis);
                        }
                    }

                    Ok(cell)

                })?.into();

                let total_used_len = res.len();
                res.remove_every_n(duplication_freq, num_repeats, duplication_offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                Ok((res, total_used_len))
            }
        }
    }

    /// Assigns a single value to the tensor. This is a helper function used by other assignment methods.
    ///
    /// # Arguments
    /// * `region` - The region to assign values in
    /// * `offset` - Base offset for the assignment
    /// * `k` - The value to assign
    /// * `coord` - The coordinate where to assign the value
    /// * `constants` - Map for tracking constant assignments
    ///
    /// # Returns
    /// The assigned value or an error if assignment fails
    fn assign_value<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
        &self,
        region: &mut Region<F>,
        offset: usize,
        k: ValType<F>,
        coord: usize,
        constants: &mut ConstantsMap<F>,
    ) -> Result<ValType<F>, halo2_proofs::plonk::Error> {
        let (x, y, z) = self.cartesian_coord(offset + coord);
        let res = match k {
            // Handle direct value assignment
            ValType::Value(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    ValType::PrevAssigned(region.assign_advice(|| "k", advices[x][y], z, || v)?)
                }
                _ => unimplemented!(),
            },
            // Handle copying previously assigned value
            ValType::PrevAssigned(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    ValType::PrevAssigned(v.copy_advice(|| "k", region, advices[x][y], z)?)
                }
                _ => unimplemented!(),
            },
            // Handle copying previously assigned constant
            ValType::AssignedConstant(v, val) => match &self {
                VarTensor::Advice { inner: advices, .. } => {
                    ValType::AssignedConstant(v.copy_advice(|| "k", region, advices[x][y], z)?, val)
                }
                _ => unimplemented!(),
            },
            // Handle assigning evaluated value
            ValType::AssignedValue(v) => match &self {
                VarTensor::Advice { inner: advices, .. } => ValType::PrevAssigned(
                    region
                        .assign_advice(|| "k", advices[x][y], z, || v)?
                        .evaluate(),
                ),
                _ => unimplemented!(),
            },
            // Handle constant value assignment with caching
            ValType::Constant(v) => {
                if let std::collections::hash_map::Entry::Vacant(e) = constants.entry(v) {
                    let value = ValType::AssignedConstant(
                        self.assign_constant(region, offset, coord, v)?,
                        v,
                    );
                    e.insert(value.clone());
                    value
                } else {
                    let cell = constants.get(&v).unwrap();
                    self.assign_value(region, offset, cell.clone(), coord, constants)?
                }
            }
        };
        Ok(res)
    }
}
