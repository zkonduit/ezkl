use std::{error::Error, marker::PhantomData};

use halo2curves::ff::PrimeField;

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, TableColumn},
};

use crate::{
    circuit::CircuitError,
    fieldutils::i128_to_felt,
    tensor::{Tensor, TensorType},
};

use crate::circuit::lookup::LookupOp;

use super::Op;

/// Halo2 lookup table for element wise non-linearities.
#[derive(Clone, Debug)]
pub struct Table<F: PrimeField> {
    /// Non-linearity to be used in table.
    pub nonlinearity: LookupOp,
    /// Input to table.
    pub table_input: TableColumn,
    /// Output of table
    pub table_output: TableColumn,
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
        nonlinearity: &LookupOp,
        preexisting_input: Option<TableColumn>,
    ) -> Table<F> {
        let table_input = preexisting_input.unwrap_or_else(|| cs.lookup_table_column());
        let range = nonlinearity.bit_range(bits);

        Table {
            nonlinearity: nonlinearity.clone(),
            table_input,
            table_output: cs.lookup_table_column(),
            is_assigned: false,
            range,
            _marker: PhantomData,
        }
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
                            if !preassigned_input {
                                table.assign_cell(
                                    || format!("nl_i_col row {}", row_offset),
                                    self.table_input,
                                    row_offset,
                                    || Value::known(*input),
                                )?;
                            }

                            let output = evals.output[row_offset];

                            table.assign_cell(
                                || format!("nl_o_col row {}", row_offset),
                                self.table_output,
                                row_offset,
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
