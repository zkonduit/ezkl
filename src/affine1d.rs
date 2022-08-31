use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
use std::marker::PhantomData;

use crate::fieldutils::i32tofelt;
use crate::tensorutils::map2;

// We layout in two phases: first we load any parameters (returning parameters, used only in case of a tied weight model),
// then we load the input, perform the forward pass, and layout the input and output, returning the output

#[derive(Clone)]
pub struct RawParameters<const IN: usize, const OUT: usize> {
    pub weights: Vec<Vec<i32>>,
    pub biases: Vec<i32>,
}

pub struct Parameters<F: FieldExt, const IN: usize, const OUT: usize> {
    weights: Vec<Vec<AssignedCell<Assigned<F>, F>>>,
    biases: Vec<AssignedCell<Assigned<F>, F>>,
    pub _marker: PhantomData<F>,
}

pub struct Affine1dFullyAssigned<F: FieldExt, const IN: usize, const OUT: usize> {
    parameters: Parameters<F, IN, OUT>,
    input: Vec<AssignedCell<Assigned<F>, F>>,
    output: Vec<AssignedCell<Assigned<F>, F>>,
}

#[derive(Clone)]
pub struct Affine1dConfig<F: FieldExt, const IN: usize, const OUT: usize>
where
    [(); IN + 3]:,
{
    pub weights: [Column<Advice>; IN],
    pub input: Column<Advice>,
    pub output: Column<Advice>,
    pub bias: Column<Advice>,
    pub q: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const IN: usize, const OUT: usize> Affine1dConfig<F, IN, OUT>
where
    [(); IN + 3]:,
{
    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        weights: Vec<Vec<i32>>,
        biases: Vec<i32>,
        input: Vec<AssignedCell<Assigned<F>, F>>,
    ) -> Result<Vec<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
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
        weights: Vec<Vec<i32>>,
        biases: Vec<i32>,
    ) -> Result<Parameters<F, IN, OUT>, halo2_proofs::plonk::Error> {
        let biases: Vec<Value<Assigned<F>>> = (0..OUT)
            .map(|i| Value::known(i32tofelt::<F>(biases[i]).into()))
            .collect();
        let weights: Vec<Vec<Value<Assigned<F>>>> =
            map2::<_, _, OUT, IN>(|i, j| Value::known(i32tofelt::<F>(weights[i][j]).into()));

        let mut biases_for_equality = Vec::new();
        for (i, bias_i) in biases.iter().enumerate().take(OUT) {
            let bias =
                region.assign_advice(|| "b".to_string(), self.bias, offset + i, || *bias_i)?;
            biases_for_equality.push(bias);
        }

        let mut weights_for_equality = Vec::new();
        for (i, w_i) in weights.iter().enumerate().take(OUT) {
            let mut row = Vec::new();
            for (j, w_i_j) in w_i.iter().enumerate().take(IN) {
                let weight = region.assign_advice(
                    || "w".to_string(),
                    self.weights[i],
                    offset + j,
                    || *w_i_j,
                )?;
                row.push(weight);
            }
            weights_for_equality.push(row);
        }

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
        input: Vec<AssignedCell<Assigned<F>, F>>,
        params: Parameters<F, IN, OUT>,
    ) -> Result<Vec<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        // copy the input
        for (j, x) in input.iter().enumerate().take(IN) {
            x.copy_advice(|| "input", region, self.input, offset + j)?;
        }

        // calculate value of output
        let mut output: Vec<Value<Assigned<F>>> =
            (0..OUT).map(|_| Value::known(F::zero().into())).collect();

        for (i, o) in output.iter_mut().enumerate().take(OUT) {
            for (j, x) in input.iter().enumerate().take(IN) {
                *o = *o + params.weights[i][j].value_field() * x.value_field();
            }
            // add bias
            *o = *o + params.biases[i].value_field();
        }

        // assign that value and return it
        let mut output_for_equality = Vec::new();
        for (i, o) in output.iter_mut().enumerate().take(OUT) {
            let ofe = region.assign_advice(
                || "o".to_string(),
                self.output, //advice
                offset + i,
                || *o, //value
            )?;
            output_for_equality.push(ofe);
        }
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

// impl<F: FieldExt, const IN: usize, const OUT: usize> Affine1dAC<F, IN, OUT> {}

#[derive(Clone)]
pub struct Affine1d<F: FieldExt, Inner, const IN: usize, const OUT: usize> {
    pub input: Vec<Inner>,        //  IN
    pub output: Vec<Inner>,       //  IN
    pub weights: Vec<Vec<Inner>>, // OUT x IN
    pub biases: Vec<Inner>,       // OUT
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt, Inner, const IN: usize, const OUT: usize> Affine1d<F, Inner, IN, OUT> {
    pub fn fill<Func1, Func2>(mut f: Func1, w: Func2) -> Self
    where
        Func1: FnMut(usize) -> Inner,
        Func2: FnMut(usize, usize) -> Inner,
    {
        Affine1d {
            input: (0..IN).map(&mut f).collect(),
            output: (0..OUT).map(&mut f).collect(),
            weights: map2::<_, _, OUT, IN>(w),
            biases: (0..OUT).map(f).collect(),

            _marker: PhantomData,
        }
    }
    pub fn without_witnesses() -> Affine1d<F, Value<Assigned<F>>, IN, OUT> {
        Affine1d::<F, Value<Assigned<F>>, IN, OUT>::fill(
            |_| Value::default(),
            |_, _| Value::default(),
        )
    }

    pub fn from_i32(
        input: Vec<i32>,
        output: Vec<i32>,
        weights: Vec<Vec<i32>>,
        biases: Vec<i32>,
    ) -> Affine1d<F, Value<Assigned<F>>, IN, OUT> {
        let input: Vec<Value<Assigned<F>>> = (0..IN)
            .map(|i| Value::known(i32tofelt::<F>(input[i]).into()))
            .collect();
        let output: Vec<Value<Assigned<F>>> = (0..OUT)
            .map(|i| Value::known(i32tofelt::<F>(output[i]).into()))
            .collect();
        let biases: Vec<Value<Assigned<F>>> = (0..OUT)
            .map(|i| Value::known(i32tofelt::<F>(biases[i]).into()))
            .collect();
        let weights: Vec<Vec<Value<Assigned<F>>>> =
            map2::<_, _, OUT, IN>(|i, j| Value::known(i32tofelt::<F>(weights[i][j]).into()));

        Affine1d {
            input,
            output,
            weights,
            biases,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt, const IN: usize, const OUT: usize> Affine1d<F, Value<Assigned<F>>, IN, OUT> {
    /// Assign parameters, leaving input and output as unknown Values.
    pub fn from_parameters(weights: Vec<Vec<i32>>, biases: Vec<i32>) -> Self {
        let biases: Vec<Value<Assigned<F>>> = (0..OUT)
            .map(|i| Value::known(i32tofelt::<F>(biases[i]).into()))
            .collect();
        let weights: Vec<Vec<Value<Assigned<F>>>> =
            map2::<_, _, OUT, IN>(|i, j| Value::known(i32tofelt::<F>(weights[i][j]).into()));

        let input: Vec<Value<Assigned<F>>> = (0..IN).map(|_| Value::default()).collect();
        let output: Vec<Value<Assigned<F>>> = (0..OUT).map(|_| Value::default()).collect();

        Affine1d {
            input,
            output,
            weights,
            biases,
            _marker: PhantomData,
        }
    }

    /// Take a layer with set parameters, accept an input, perform forward pass, and return output.
    /// Mutates self to assign the input and computed output.
    pub fn forward(&mut self, input: Vec<Value<Assigned<F>>>) -> Vec<Value<Assigned<F>>> {
        self.input = input.clone();

        let mut output: Vec<Value<Assigned<F>>> =
            (0..OUT).map(|_| Value::known(F::zero().into())).collect();

        for (i, o) in output.iter_mut().enumerate().take(OUT) {
            for (j, x) in input.iter().enumerate().take(IN) {
                *o = *o + self.weights[i][j] * x;
            }
            // add the bias
            *o = *o + self.biases[i];
        }

        self.output = output.clone();
        output
    }
}
