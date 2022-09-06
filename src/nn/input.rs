use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Selector},
};
use std::marker::PhantomData;


// Takes input data provided as raw data type, e.g. i32, and sets it up to be passed into a pipeline,
// including laying it out in a column and outputting Vec<AssignedCell<Assigned<F>, F>> suitable for copying
// Can also have a variant to check a signature, check that input matches a hash, etc.
#[derive(Clone)]
pub struct InputConfig<F: FieldExt, const IN: usize> {
    pub input: Column<Advice>,
    pub q: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const IN: usize> InputConfig<F, IN> {
    pub fn configure(cs: &mut ConstraintSystem<F>, col: Column<Advice>) -> InputConfig<F, IN> {
        let qs = cs.selector();
        // could put additional constraints on input here
        InputConfig {
            input: col,
            q: qs,
            _marker: PhantomData,
        }
    }

    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        raw_input: Tensor<i32>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Input",
            |mut region| {
                let offset = 0;
                self.q.enable(&mut region, offset)?;

                let output: Tensor<Value<Assigned<F>>> = raw_input.clone().into();
                Ok(output.enum_map(|i, o| {
                    region
                        .assign_advice(|| "o".to_string(), self.input, offset + i, || o)
                        .unwrap()
                }))
            },
        )
    }
}
