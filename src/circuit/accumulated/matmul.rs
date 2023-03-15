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
    /// the inputs to the operations.
    pub inputs: Vec<VarTensor>,
    /// the (currently singular) output of the operations.
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
    /// * `op` - The operation being represented
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor],
        output: &VarTensor,
    ) -> Self {
        // setup a selector per base op AND across columns
        // TODO: make this more robust as we expand the module
        let mut selectors = BTreeMap::new();
        for i in 0..inputs[0].num_cols() {
            selectors.insert((BaseOp::Dot, i), meta.selector());
            selectors.insert((BaseOp::InitDot, i), meta.selector());
        }

        let config = Self {
            selectors,
            inputs: inputs.to_vec(),
            output: output.clone(),
            _marker: PhantomData,
        };

        for ((base_op, i), selector) in config.selectors.iter() {
            meta.create_gate(base_op.as_str(), |meta| {
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
                    .query_rng(meta, i * config.output.col_size(), 2)
                    .expect("accum: output query failed");

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
        if values.len() != 2 {
            return Err(Box::new(CircuitError::DimMismatch(
                "accum matmul layout".to_string(),
            )));
        };

        let mut a = values[0].clone();
        let mut b = values[1].clone();
        b.transpose_2d()?;

        let num_a_repeats = b.dims()[0];
        let num_b_tiles = a.dims()[1];
        let b_row_len = b.dims()[1];

        a.repeat_rows(num_a_repeats)?;
        b.tile(num_b_tiles)?;

        let t = match layouter.assign_region(
            || "assign inputs",
            |mut region| {
                let offset = 0;

                let mut inputs = vec![];

                for (i, elem) in vec![a.clone(), b.clone()].iter().enumerate() {
                    let inp = utils::value_muxer(
                        &self.inputs[i],
                        &{
                            let res = self.inputs[i].assign(&mut region, offset, elem)?;
                            res.map(|e| e.value_field().evaluate())
                        },
                        elem,
                    );
                    inputs.push(inp);
                }

                // remove any repeats from the assignment
                if num_a_repeats > 1 {
                    let dims = inputs[0].dims().to_vec();
                    inputs[0].reshape(&[dims[0], dims[1..].iter().product()]);
                    let mut rm_dup = vec![];
                    for i in 0..dims[0] {
                        rm_dup.push(inputs[0].get_slice(&[i..i + 1, 0..dims[1]]).unwrap());
                    }
                    inputs[0] = Tensor::new(Some(&rm_dup), &[rm_dup.len()])
                        .unwrap()
                        .combine()
                        .unwrap();
                }

                inputs[0].reshape(values[0].dims());

                // transpose it back to its normal shape
                inputs[1] = inputs[1].get_slice(&[0..1]).unwrap();
                inputs[1].reshape(&[values[1].dims()[1], values[1].dims()[0]]);
                inputs[1].transpose_2d().unwrap();

                // now perform matrix multiplication on the processed tensors
                let accumulated_matmul =
                    accumulated::matmul(&vec![inputs[0].clone(), inputs[1].clone()])
                        .expect("accum poly: matmul op failed");

                // remove 0 elements bar the first
                let cleaned_matmul: Tensor<_> = accumulated_matmul
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| {
                        ((i % accumulated_matmul.dims().last().unwrap()) > 0) || i == 0
                    })
                    .map(|(_, v)| *v)
                    .into();

                let output = self
                    .output
                    .assign(&mut region, offset, &cleaned_matmul.into())?;

                // these selectors map from
                for i in 0..a.dims().iter().product::<usize>() {
                    let (x, y) = self.inputs[0].cartesian_coord(i);
                    if (i) % b_row_len > 0 || i == 0 {
                        self.selectors
                            .get(&(BaseOp::Dot, x))
                            .unwrap()
                            .enable(&mut region, y)?;
                    } else {
                        self.selectors
                            .get(&(BaseOp::InitDot, x))
                            .unwrap()
                            .enable(&mut region, y)?;
                    }
                }

                let dims = output.dims();
                let mut last_dims = vec![];

                for d in &dims[0..dims.len() - 1] {
                    last_dims.push(0..*d);
                }
                let script_len = dims.last().unwrap();
                last_dims.push(script_len - 1..*script_len);
                // Now we can assign the matmul op
                Ok({
                    output
                        .get_slice(&last_dims)
                        .expect("accum poly: failed to fetch last elem")
                })
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
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    // use halo2curves::pasta::pallas;
    use halo2curves::pasta::Fp as F;
    // use rand::rngs::OsRng;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct AffineCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for AffineCircuit<F> {
        type Config = Config<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true, 512);
            let output =
                VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true, 512);
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
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]);

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]);

        let circuit = AffineCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
