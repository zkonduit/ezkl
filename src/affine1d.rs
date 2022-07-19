use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Constraints, Error, Expression, Selector,
    },
    poly::Rotation,
};
use std::marker::PhantomData;

use crate::fieldutils::i32tofelt;
use crate::tensorutils::map2;

#[derive(Clone)]
pub struct Affine1d<F: FieldExt, Inner, const IN: usize, const OUT: usize> {
    pub input: Vec<Inner>,        //  IN
    pub output: Vec<Inner>,       //  IN
    pub weights: Vec<Vec<Inner>>, // OUT x IN
    pub biases: Vec<Inner>,       // OUT
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt, Inner, const IN: usize, const OUT: usize> Affine1d<F, Inner, IN, OUT> {
    pub fn fill<Func1, Func2>(mut f: Func1, mut w: Func2) -> Self
    where
        Func1: FnMut(usize) -> Inner,
        Func2: FnMut(usize, usize) -> Inner,
    {
        Affine1d {
            input: (0..IN).map(|i| f(i)).collect(),
            output: (0..OUT).map(|i| f(i)).collect(),
            weights: map2::<_, _, OUT, IN>(|i, j| w(i, j)),
            biases: (0..OUT).map(|i| f(i)).collect(),

            _marker: PhantomData,
        }
    }
    pub fn without_witnesses() -> Affine1d<F, Value<Assigned<F>>, IN, OUT> {
        Affine1d::<F, Value<Assigned<F>>, IN, OUT>::fill(
            |_| Value::default(),
            |_, _| Value::default(),
        )
    }

    // pub fn from_values<T>(
    //     input: Vec<Vec<T>>,
    //     output: Vec<T>,
    //     weights: Vec<Vec<T>>,
    //     biases: Vec<T>,
    // ) -> Affine1d<F, Value<Assigned<F>>, IN, OUT>
    // where
    //     T: Into<F> + Copy,
    // {
    // 	let input: Vec<Value<Assigned<F>>> =  (0..IN)
    //             .map(|i| Value::known(input[i].into().into()))
    //             .collect();
    //     let output: Vec<Value<Assigned<F>>> = (0..OUT)
    //             .map(|i| Value::known(output[i].into().into()))
    //             .collect();
    //     let weights: Vec<Vec<Value<Assigned<F>>>> = map2::<_, _, OUT, IN>(|i, j| weights[i][j].into().into()),
    //     let biases: Vec<Value<Assigned<F>>> = (0..OUT)
    //             .map(|i| Value::known(biases[i].into().into()))
    //             .collect();

    //     Self {
    // 	    input, output, weights, biases,
    //         _marker: PhantomData,
    //     }
    // }

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

#[derive(Clone)]
pub struct Affine1dConfig<F: FieldExt, const IN: usize, const OUT: usize> {
    pub advice: Affine1d<F, Column<Advice>, IN, OUT>,
    pub q: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const IN: usize, const OUT: usize> Affine1dConfig<F, IN, OUT> {
    fn define_advice(cs: &mut ConstraintSystem<F>) -> Affine1d<F, Column<Advice>, IN, OUT> {
        Affine1d {
            input: (0..IN).map(|i| cs.advice_column()).collect(),
            output: (0..OUT).map(|i| cs.advice_column()).collect(),
            weights: map2::<_, _, OUT, IN>(|i, j| cs.advice_column()),
            biases: (0..OUT).map(|i| cs.advice_column()).collect(),
            _marker: PhantomData,
        }

        // Affine1d::<F, Column<Advice>, IN, OUT>::fill(
        //     |_| cs.advice_column(),
        //     |_, _| cs.advice_column(),
        // )
    }

    // composable_configure takes the input tensor as an argument, and completes the advice by generating new for the rest
    pub fn composable_configure(
        advice: Affine1d<F, Column<Advice>, IN, OUT>,
        cs: &mut ConstraintSystem<F>,
    ) -> Affine1dConfig<F, IN, OUT> {
        let qs = cs.selector();

        cs.create_gate("affine", |virtual_cells| {
            let q = virtual_cells.query_selector(qs);

            //	    let input = advice.input.map(|a| virtual_cells.query_advice(a, Rotation::cur()));
            let input: Vec<Expression<F>> = (0..IN)
                .map(|i| virtual_cells.query_advice(advice.input[i], Rotation::cur()))
                .collect();

            let output: Vec<Expression<F>> = (0..OUT)
                .map(|i| virtual_cells.query_advice(advice.output[i], Rotation::cur()))
                .collect();

            let biases: Vec<Expression<F>> = (0..OUT)
                .map(|i| virtual_cells.query_advice(advice.biases[i], Rotation::cur()))
                .collect();

            let weights: Vec<Vec<Expression<F>>> = map2::<_, _, OUT, IN>(|i, j| {
                virtual_cells.query_advice(advice.weights[i][j], Rotation::cur())
            });

            // We put the negation of the claimed output in the constraint tensor.
            let mut constraints: Vec<Expression<F>> =
                (0..OUT).map(|i| -output[i].clone()).collect();

            // Now we compute the linear expression,  and add it to constraints
            for i in 0..OUT {
                for j in 0..IN {
                    constraints[i] =
                        constraints[i].clone() + weights[i][j].clone() * input[j].clone();
                }
            }

            // add the bias
            for i in 0..OUT {
                constraints[i] = constraints[i].clone() + biases[i].clone();
            }

            let constraints = (0..OUT).map(|i| "c").zip(constraints);
            Constraints::with_selector(q, constraints)
        });

        Self {
            advice,
            q: qs,
            _marker: PhantomData,
        }
    }

    // configure creates fresh advice, while composable_configure uses previously-created advice.
    pub fn configure(cs: &mut ConstraintSystem<F>) -> Affine1dConfig<F, IN, OUT> {
        let advice = Self::define_advice(cs);
        Self::composable_configure(advice, cs)
    }

    pub fn layout(
        //here
        &self,
        assigned: &Affine1d<F, Value<Assigned<F>>, IN, OUT>,
        layouter: &mut impl Layouter<F>,
    ) -> Result<(), halo2_proofs::plonk::Error> {
        layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                self.q.enable(&mut region, offset)?;

                for i in 0..OUT {
                    region.assign_advice(
                        || format!("b"),
                        self.advice.biases[i],
                        offset,
                        || assigned.biases[i],
                    )?;
                }
                for i in 0..OUT {
                    region.assign_advice(
                        || format!("o"),
                        self.advice.output[i],
                        offset,
                        || assigned.output[i],
                    )?;
                }
                for j in 0..IN {
                    region.assign_advice(
                        || format!("i"),
                        self.advice.input[j], // Column<Advice>
                        offset,
                        || assigned.input[j], //Assigned<F>
                    )?;
                }

                for i in 0..OUT {
                    for j in 0..IN {
                        region.assign_advice(
                            || format!("w"),
                            self.advice.weights[i][j],
                            offset,
                            || assigned.weights[i][j],
                        )?;
                    }
                }

                Ok(())
            },
        )
    }
}

#[cfg(test)]
use halo2_proofs::{
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{pallas, vesta};
use rand::rngs::OsRng;
