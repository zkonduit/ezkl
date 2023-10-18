use log::{error, warn};

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
    ///
    pub fn max_rows<F: PrimeField>(cs: &ConstraintSystem<F>, logrows: usize) -> usize {
        let base = 2u32;
        base.pow(logrows as u32) as usize - cs.blinding_factors() - 1
    }

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
        let max_rows = Self::max_rows(cs, logrows);

        let mut modulo = (capacity / max_rows) + 1;
        // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
        modulo = ((capacity + modulo) / max_rows) + 1;
        let mut advices = vec![];

        if modulo > 1 {
            warn!(
                "will be using column duplication for {} advice columns",
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
            warn!(
                "will be using column duplication for {} fixed columns",
                modulo - 1
            );
        }

        for _ in 0..modulo {
            let col = cs.fixed_column();
            cs.enable_constant(col);
        }
        modulo
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
        x: usize,
        y: usize,
        constant: F,
    ) -> Result<AssignedCell<F, F>, halo2_proofs::plonk::Error> {
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
                        let (x, y) = self.cartesian_coord(offset + coord);
                        region.assign_advice_from_instance(
                            || "pub input anchor",
                            *instance,
                            coord + total_offset,
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
                    let cell = self.assign_value(region, k.clone(), x, y)?;

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

    /// Number of times it crosses into a new columns
    pub fn duplicated_len(&self, len: usize, offset: usize) -> usize {
        match self {
            VarTensor::Advice { col_size, .. } => {
                len + (offset + len) / col_size - offset / col_size
            }
            _ => 0,
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor but allows for column wrapping for accumulated operations.
    /// Duplication occurs by copying the last cell of the column to the first cell next column and creating a copy constraint between the two.
    pub fn dummy_assign_with_duplication<F: PrimeField + TensorType + PartialOrd>(
        &self,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Result<(ValTensor<F>, usize, usize), halo2_proofs::plonk::Error> {
        match values {
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims , ..} => {
                // duplicates every nth element to adjust for column overflow
                let mut res: ValTensor<F> = v.duplicate_every_n(self.col_size(), offset).unwrap().into();
                let total_used_len = res.len();
                let total_constants = res.num_constants();
                res.remove_every_n(self.col_size(), offset).unwrap();

                res.reshape(dims).unwrap();
                res.set_scale(values.scale());

                Ok((res, total_used_len, total_constants))
            }
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor but allows for column wrapping for accumulated operations.
    pub fn assign_value<F: PrimeField + TensorType + PartialOrd>(
        &self,
        region: &mut Region<F>,
        k: ValType<F>,
        x: usize,
        y: usize,
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
            ValType::Constant(v) => self.assign_constant(region, x, y, v),
        }
    }
}
