use std::{error::Error, marker::PhantomData};

use halo2curves::ff::PrimeField;

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Expression, TableColumn},
};
use log::warn;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    circuit::CircuitError,
    fieldutils::i128_to_felt,
    tensor::{Tensor, TensorType},
};

use crate::circuit::lookup::LookupOp;

use super::Op;

/// The safety factor for the range of the lookup table.
pub const RANGE_MULTIPLIER: i128 = 2;
/// The safety factor offset for the number of rows in the lookup table.
pub const RESERVED_BLINDING_ROWS_PAD: usize = 3;

#[derive(Debug, Clone)]
///
pub struct SelectorConstructor<F: PrimeField> {
    ///
    pub degree: usize,
    ///
    _marker: PhantomData<F>,
}

impl<F: PrimeField> SelectorConstructor<F> {
    ///
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            _marker: PhantomData,
        }
    }

    ///
    pub fn get_expr_at_idx(&self, i: usize, expr: Expression<F>) -> Expression<F> {
        let indices = 0..self.degree;
        indices
            .into_par_iter()
            .filter(|x| *x != i)
            .map(|i| {
                if i == 0 {
                    expr.clone()
                } else {
                    (Expression::Constant(F::from(i as u64))) - expr.clone()
                }
            })
            .reduce(|| Expression::Constant(F::from(1_u64)), |acc, x| acc * x)
    }

    ///
    pub fn get_selector_val_at_idx(&self, i: usize) -> F {
        let indices = 0..self.degree;
        indices
            .into_par_iter()
            .filter(|x| *x != i)
            .map(|x| {
                if x == 0 {
                    F::from(i as u64)
                } else {
                    F::from(x as u64) - F::from(i as u64)
                }
            })
            .product()
    }
}

/// Halo2 lookup table for element wise non-linearities.
#[derive(Clone, Debug)]
pub struct Table<F: PrimeField> {
    /// Non-linearity to be used in table.
    pub nonlinearity: LookupOp,
    /// Input to table.
    pub table_inputs: Vec<TableColumn>,
    /// col size
    pub col_size: usize,
    /// Output of table
    pub table_outputs: Vec<TableColumn>,
    /// selector cn
    pub selector_constructor: SelectorConstructor<F>,
    /// Flags if table has been previously assigned to.
    pub is_assigned: bool,
    /// Number of bits used in lookup table.
    pub range: (i128, i128),
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> Table<F> {
    /// get column index given input
    pub fn get_col_index(&self, input: F) -> F {
        //    range is split up into chunks of size col_size, find the chunk that input is in
        let chunk =
            (crate::fieldutils::felt_to_i128(input) - self.range.0).abs() / (self.col_size as i128);

        i128_to_felt(chunk)
    }

    /// get first_element of column
    pub fn get_first_element(&self, chunk: usize) -> (F, F) {
        let chunk = chunk as i128;
        // we index from 1 to prevent soundness issues
        let first_element = i128_to_felt(chunk * (self.col_size as i128) + self.range.0);
        let op_f = Op::<F>::f(
            &self.nonlinearity,
            &[Tensor::from(vec![first_element].into_iter())],
        )
        .unwrap();
        (first_element, op_f.output[0])
    }

    ///
    pub fn cal_col_size(logrows: usize, reserved_blinding_rows: usize) -> usize {
        2usize.pow(logrows as u32) - reserved_blinding_rows
    }

    ///
    pub fn cal_bit_range(bits: usize, reserved_blinding_rows: usize) -> usize {
        2usize.pow(bits as u32) - reserved_blinding_rows
    }

    ///
    pub fn num_cols_required(range: (i128, i128), col_size: usize) -> usize {
        // double it to be safe
        let range_len = range.1 - range.0;
        // number of cols needed to store the range
        (range_len / (col_size as i128)) as usize + 1
    }
}

impl<F: PrimeField + TensorType + PartialOrd> Table<F> {
    /// Configures the table.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        range: (i128, i128),
        logrows: usize,
        nonlinearity: &LookupOp,
        preexisting_inputs: Option<Vec<TableColumn>>,
    ) -> Table<F> {
        let factors = cs.blinding_factors() + RESERVED_BLINDING_ROWS_PAD;
        let col_size = Self::cal_col_size(logrows, factors);
        // number of cols needed to store the range
        let num_cols = Self::num_cols_required(range, col_size);

        log::debug!("table range: {:?}", range);

        let table_inputs = preexisting_inputs.unwrap_or_else(|| {
            let mut cols = vec![];
            for _ in 0..num_cols {
                cols.push(cs.lookup_table_column());
            }
            cols
        });

        let num_cols = table_inputs.len();

        if num_cols > 1 {
            warn!("Using {} columns for non-linearity table.", num_cols);
        }

        let table_outputs = table_inputs
            .iter()
            .map(|_| cs.lookup_table_column())
            .collect::<Vec<_>>();

        Table {
            nonlinearity: nonlinearity.clone(),
            table_inputs,
            table_outputs,
            is_assigned: false,
            selector_constructor: SelectorConstructor::new(num_cols),
            col_size,
            range,
            _marker: PhantomData,
        }
    }

    /// Take a linear coordinate and output the (column, row) position in the storage block.
    pub fn cartesian_coord(&self, linear_coord: usize) -> (usize, usize) {
        let x = linear_coord / self.col_size;
        let y = linear_coord % self.col_size;
        (x, y)
    }

    /// Assigns values to the constraints generated when calling `configure`.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        preassigned_input: bool,
    ) -> Result<(), Box<dyn Error>> {
        if self.is_assigned {
            return Err(Box::new(CircuitError::TableAlreadyAssigned));
        }

        let smallest = self.range.0;
        let largest = self.range.1;

        let inputs = Tensor::from(smallest..=largest).map(|x| i128_to_felt(x));
        let evals = Op::<F>::f(&self.nonlinearity, &[inputs.clone()])?;
        let chunked_inputs = inputs.chunks(self.col_size);

        self.is_assigned = true;

        let col_multipliers: Vec<F> = (0..chunked_inputs.len())
            .map(|x| self.selector_constructor.get_selector_val_at_idx(x))
            .collect();

        let _ = chunked_inputs
            .enumerate()
            .map(|(chunk_idx, inputs)| {
                layouter.assign_table(
                    || "nl table",
                    |mut table| {
                        let _ = inputs
                            .iter()
                            .enumerate()
                            .map(|(mut row_offset, input)| {
                                let col_multiplier = col_multipliers[chunk_idx];

                                row_offset += chunk_idx * self.col_size;
                                let (x, y) = self.cartesian_coord(row_offset);
                                if !preassigned_input {
                                    table.assign_cell(
                                        || format!("nl_i_col row {}", row_offset),
                                        self.table_inputs[x],
                                        y,
                                        || Value::known(*input * col_multiplier),
                                    )?;
                                }

                                let output = evals.output[row_offset];

                                table.assign_cell(
                                    || format!("nl_o_col row {}", row_offset),
                                    self.table_outputs[x],
                                    y,
                                    || Value::known(output * col_multiplier),
                                )?;

                                Ok(())
                            })
                            .collect::<Result<Vec<()>, halo2_proofs::plonk::Error>>()?;
                        Ok(())
                    },
                )
            })
            .collect::<Result<Vec<()>, halo2_proofs::plonk::Error>>()?;
        Ok(())
    }
}
