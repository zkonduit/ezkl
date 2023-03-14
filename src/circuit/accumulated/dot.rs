use super::*;
use crate::circuit::{utils, CircuitError};
use crate::tensor::{ops::*, ValTensor, VarTensor};
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use std::collections::BTreeMap;
use std::error::Error;
use std::marker::PhantomData;

/// Configuration for an accumulated arg.
#[derive(Clone, Debug)]
pub struct Config<F: FieldExt + TensorType> {
    /// the inputs to the fused operations.
    pub inputs: Vec<VarTensor>,
    /// the (currently singular) output of the fused operations.
    pub output: VarTensor,
    /// [Selectors] generated when configuring the layer. We use a BTreeMap as we expect to configure many base gates.
    pub selectors: BTreeMap<(BaseOp, usize), Selector>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Config<F> {
    /// Configures the sequence of operations into a circuit gate.
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations.
    /// * `output` - The variable representing the (currently singular) output of the operations.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor],
        output: &VarTensor,
    ) -> Self {
        // setup a selector per base op
        let mut selectors = BTreeMap::new();
        for i in 0..inputs[0].num_cols() {
            selectors.insert((BaseOp::Dot, i), meta.selector());
        }
        let config = Self {
            selectors,
            inputs: inputs.to_vec(),
            output: output.clone(),
            _marker: PhantomData,
        };

        for ((base_op, i), selector) in config.selectors.iter() {
            meta.create_gate("accum dot", |meta| {
                let selector = meta.query_selector(*selector);

                let qis = config
                    .inputs
                    .iter()
                    .map(|input| {
                        input
                            .query_rng(meta, i * input.col_size(), 1)
                            .expect("accum: input query failed")[0]
                            .clone()
                    })
                    .collect::<Vec<_>>();

                // Get output expressions for each input channel
                let expected_output: Tensor<Expression<F>> = config
                    .output
                    .query_rng(meta, 0, 2)
                    .expect("poly: output query failed");

                let res = base_op.f((qis[0].clone(), qis[1].clone(), expected_output[0].clone()));

                let constraints = vec![expected_output[1].clone() - res];

                Constraints::with_selector(selector, constraints)
            });
        }

        config
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout(
        &mut self,
        layouter: &mut impl Layouter<F>,
        values: &[ValTensor<F>],
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        if values.len() != self.inputs.len() {
            return Err(Box::new(CircuitError::DimMismatch(
                "accum dot layout".to_string(),
            )));
        }

        let t = match layouter.assign_region(
            || "assign inputs",
            |mut region| {
                let offset = 0;

                let mut inputs = vec![];
                for (i, input) in values.iter().enumerate() {
                    let inp = utils::value_muxer(
                        &self.inputs[i],
                        &{
                            let res = self.inputs[i].assign(&mut region, offset, input)?;
                            res.map(|e| e.value_field().evaluate())
                        },
                        input,
                    );
                    inputs.push(inp);
                }

                // Now we can assign the dot product
                let accumulated_dot = accumulated::dot(&inputs)
                    .expect("accum poly: dot op failed")
                    .into();
                let output = self.output.assign(&mut region, offset, &accumulated_dot)?;

                for i in 0..inputs[0].len() {
                    let (x, y) = self.inputs[0].cartesian_coord(i);
                    self.selectors
                        .get(&(BaseOp::Dot, x))
                        .unwrap()
                        .enable(&mut region, y)?;
                }

                // last element is the result
                Ok(output
                    .get_slice(&[output.len() - 1..output.len()])
                    .expect("accum poly: failed to fetch last elem"))
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                return Err(Box::new(e));
            }
        };

        Ok(ValTensor::from(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        arithmetic::{Field, FieldExt},
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    use rand::rngs::OsRng;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = Config<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
            let output = VarTensor::new_advice(cs, K, LEN + 1, vec![LEN + 1], true, 512);

            Self::Config::configure(cs, &[a, b], &output)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _ = config.layout(&mut layouter, &self.inputs.clone());
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|_| Value::known(pallas::Base::random(OsRng))));

        let b = Tensor::from((0..LEN).map(|_| Value::known(pallas::Base::random(OsRng))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
