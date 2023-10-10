use std::{error::Error, marker::PhantomData};

use halo2curves::ff::PrimeField;

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, TableColumn},
};
use log::warn;

use crate::{
    circuit::CircuitError,
    fieldutils::i128_to_felt,
    tensor::{Tensor, TensorType, VarTensor},
};

use crate::circuit::lookup::LookupOp;

use super::Op;

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
}

impl<F: PrimeField + TensorType + PartialOrd> Table<F> {
    /// Configures the table.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        bits: usize,
        logrows: usize,
        nonlinearity: &LookupOp,
        preexisting_inputs: Option<Vec<TableColumn>>,
    ) -> Table<F> {
        let num_cols = std::cmp::max(1, 1 + bits as i128 - logrows as i128) as usize;
        let col_size = VarTensor::max_rows(cs, logrows) - 2;
        let range = nonlinearity.bit_range(num_cols * col_size);

        log::debug!("table range: {:?}", range);

        let table_inputs = preexisting_inputs.unwrap_or_else(|| {
            let mut cols = vec![];
            for _ in 0..num_cols {
                cols.push(cs.lookup_table_column());
            }
            cols
        });

        if table_inputs.len() > 1 {
            warn!(
                "Using {} columns for non-linearity table.",
                table_inputs.len()
            );
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

        println!("chunked_inputs: {:?}", chunked_inputs.len());

        self.is_assigned = true;

        let _ = chunked_inputs
            .enumerate()
            .map(|(chunk_idx, inputs)| {
                // println!("chunk_idx: {}", chunk_idx);
                layouter.assign_table(
                    || "nl table",
                    |mut table| {
                        let _ = inputs
                            .iter()
                            .enumerate()
                            .map(|(mut row_offset, input)| {
                                row_offset += chunk_idx * self.col_size;
                                let (x, y) = self.cartesian_coord(row_offset);
                                if !preassigned_input {
                                    table.assign_cell(
                                        || format!("nl_i_col row {}", row_offset),
                                        self.table_inputs[x],
                                        y,
                                        || Value::known(*input),
                                    )?;
                                }

                                let output = evals.output[row_offset];

                                table.assign_cell(
                                    || format!("nl_o_col row {}", row_offset),
                                    self.table_outputs[x],
                                    y,
                                    || Value::known(output),
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
