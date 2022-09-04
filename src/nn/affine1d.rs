use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;

use crate::fieldutils::i32tofelt;
use crate::tensor::{Tensor, TensorType};
use crate::tensor_ops::utils::map2;
use crate::tensor_ops::vec_matmul_field;

// We layout in two phases: first we load any parameters (returning parameters, used only in case of a tied weight model),
// then we load the input, perform the forward pass, and layout the input and output, returning the output

#[derive(Clone)]
pub struct RawParameters<const IN: usize, const OUT: usize> {
    pub weights: Tensor<i32>,
    pub biases: Tensor<i32>,
}

pub struct Parameters<F: FieldExt, const IN: usize, const OUT: usize> {
    weights: Tensor<AssignedCell<Assigned<F>, F>>,
    biases: Tensor<AssignedCell<Assigned<F>, F>>,
    pub _marker: PhantomData<F>,
}

pub struct Affine1dFullyAssigned<F: FieldExt, const IN: usize, const OUT: usize> {
    parameters: Parameters<F, IN, OUT>,
    input: Tensor<AssignedCell<Assigned<F>, F>>,
    output: Tensor<AssignedCell<Assigned<F>, F>>,
}

#[derive(Clone)]
pub struct Affine1dConfig<F: FieldExt, const IN: usize, const OUT: usize>
// where
//     [(); IN + 3]:,
{
    pub weights: [Column<Advice>; IN],
    pub input: Column<Advice>,
    pub output: Column<Advice>,
    pub bias: Column<Advice>,
    pub q: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const IN: usize, const OUT: usize> Affine1dConfig<F, IN, OUT>
// where
//     [(); IN + 3]:,
{
    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        weights: Tensor<i32>,
        biases: Tensor<i32>,
        input: Tensor<AssignedCell<Assigned<F>, F>>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Both",
            |mut region| {
                let offset = 0;
                self.q.enable(&mut region, offset)?;

                let params =
                    self.assign_parameters(&mut region, offset, weights.clone(), biases.clone())?;
                let output = self.forward(&mut region, offset, input.clone(), params)?;
                Ok(output)
            },
        )
    }

    pub fn assign_parameters(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        weights: Tensor<i32>,
        biases: Tensor<i32>,
    ) -> Result<Parameters<F, IN, OUT>, halo2_proofs::plonk::Error> {
        let biases: Tensor<Value<Assigned<F>>> = biases.into();
        let weights: Tensor<Value<Assigned<F>>> = weights.into();

        let mut biases_for_equality = biases.assign_cell(region, "b", &[self.bias], offset)?;

        let weights_for_equality = weights.assign_cell(region, "w", &self.weights, offset)?;

        let params = Parameters {
            biases: biases_for_equality,
            weights: weights_for_equality,
            _marker: PhantomData,
        };

        Ok(params)
    }

    pub fn forward(
        &self, // just advice
        region: &mut Region<'_, F>,
        offset: usize,
        input: Tensor<AssignedCell<Assigned<F>, F>>,
        params: Parameters<F, IN, OUT>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        // copy the input
        for (j, x) in input.iter().enumerate().take(IN) {
            x.copy_advice(|| "input", region, self.input, offset + j)?;
        }
        // calculate value of output
        let mut output = vec_matmul_field(input, params.weights, Some(params.biases));

        // assign that value and return it
        let mut output_for_equality =
            output.assign_cell(region, "o", &[self.output], offset)?;

        Ok(output_for_equality)
    }

    // composable_configure takes the input tensor as an argument, and completes the advice by generating new for the rest
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        weights: [Column<Advice>; IN],
        input: Column<Advice>,
        output: Column<Advice>,
        bias: Column<Advice>,
    ) -> Self {
        let qs = cs.selector();

        cs.create_gate("affine", |virtual_cells| {
            let q = virtual_cells.query_selector(qs);

            // We put the negation of the claimed output in the constraint tensor.
            let mut constraints: Vec<Expression<F>> = (0..OUT)
                .map(|i| -virtual_cells.query_advice(output, Rotation(i as i32)))
                .collect();

            // Now we compute the linear expression,  and add it to constraints
            for (i, c) in constraints.iter_mut().enumerate().take(OUT) {
                for j in 0..IN {
                    *c = c.clone()
                        + virtual_cells.query_advice(weights[i], Rotation(j as i32))
                            * virtual_cells.query_advice(input, Rotation(j as i32));
                }
                // add the bias
                *c = c.clone() + virtual_cells.query_advice(bias, Rotation(i as i32));
            }

            let constraints = (0..OUT).map(|_| "c").zip(constraints);
            Constraints::with_selector(q, constraints)
        });

        Self {
            weights,
            input,
            output,
            bias,
            q: qs,
            _marker: PhantomData,
        }
    }
}
