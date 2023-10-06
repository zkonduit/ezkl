use std::{error::Error, marker::PhantomData};

use halo2curves::ff::PrimeField;

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, TableColumn},
};

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
    /// Configures the table.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        bits: usize,
        logrows: usize,
        nonlinearity: &LookupOp,
        preexisting_inputs: Option<Vec<TableColumn>>,
    ) -> Table<F> {
        let range = nonlinearity.bit_range(bits, cs.blinding_factors());
        let max_rows = VarTensor::max_rows(&cs, logrows) as i128;

        let table_inputs = preexisting_inputs.unwrap_or_else(|| {
            let capacity = range.1 - range.0;

            let mut modulo = (capacity / max_rows) + 1;
            // we add a buffer for duplicated rows (we get at most 1 duplicated row per column)
            modulo = ((capacity + modulo) / max_rows) + 1;
            let mut cols = vec![];
            for _ in 0..modulo {
                cols.push(cs.lookup_table_column());
            }
            cols
        });

        let table_outputs = table_inputs
            .iter()
            .map(|_| cs.lookup_table_column())
            .collect::<Vec<_>>();

        Table {
            nonlinearity: nonlinearity.clone(),
            table_inputs,
            table_outputs,
            is_assigned: false,
            col_size: max_rows as usize,
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

        self.is_assigned = true;
        layouter
            .assign_table(
                || "nl table",
                |mut table| {
                    let _ = inputs
                        .iter()
                        .enumerate()
                        .map(|(row_offset, input)| {
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
            .map_err(Box::<dyn Error>::from)
    }
}
