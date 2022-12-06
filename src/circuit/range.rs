use crate::abort;
use crate::fieldutils::i32_to_felt;
use crate::tensor::{TensorType, ValTensor, VarTensor};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use log::error;
use std::marker::PhantomData;

/// Configuration for a range check on the difference between `input` and `expected`.
#[derive(Debug, Clone)]
pub struct RangeCheckConfig<F: FieldExt + TensorType> {
    input: VarTensor,
    pub expected: VarTensor,
    selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> RangeCheckConfig<F> {
    /// Configures a range check on the difference between `input` and `expected`.
    /// # Arguments
    /// * `input` - the input
    /// * `expected` - the expected input we would have wanted to produce
    /// * `instance` - the public input we'll be assigning to `expected`
    /// * `tol` - the range (%2), effectively our tolerance for error between `input` and `expected`.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        expected: &VarTensor,
        tol: usize,
    ) -> Self {
        let config = Self {
            input: input.clone(),
            expected: expected.clone(),
            selector: cs.selector(),
            _marker: PhantomData,
        };

        cs.create_gate("range check", |cs| {
            //        value     |    q_range_check
            //       ------------------------------
            //          v       |         1

            let q = cs.query_selector(config.selector);
            let witnessed = match input.query(cs, 0) {
                Ok(q) => q,
                Err(e) => {
                    abort!("failed to query input {:?}", e);
                }
            };

            // Get output expressions for each input channel
            let expected = match expected.query(cs, 0) {
                Ok(q) => q,
                Err(e) => {
                    abort!("failed to query input {:?}", e);
                }
            };

            // Given a range R and a value v, returns the expression
            // (v) * (1 - v) * (2 - v) * ... * (R - 1 - v)
            let range_check = |tol: i32, value: Expression<F>| {
                (-tol..tol).fold(value.clone(), |expr, i| {
                    expr * (Expression::Constant(i32_to_felt(i)) - value.clone())
                })
            };

            let constraints = witnessed
                .enum_map(|i, o| range_check(tol as i32, o - expected[i].clone()))
                .unwrap();
            Constraints::with_selector(q, constraints)
        });

        config
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `input` - The input values we want to express an error tolerance for
    /// * `layouter` - A Halo2 Layouter.
    pub fn layout(
        &self,
        mut layouter: impl Layouter<F>,
        input: ValTensor<F>,
        output: ValTensor<F>,
    ) {
        match layouter.assign_region(
            || "range check layout",
            |mut region| {
                let offset = 0;

                // Enable q_range_check
                self.selector.enable(&mut region, offset)?;

                // assigns the instance to the advice.
                match self.input.assign(&mut region, offset, &input) {
                    Ok(_) => {}
                    Err(e) => {
                        abort!("failed to assign inputs during range layer layout {:?}", e);
                    }
                };

                match self.expected.assign(&mut region, offset, &output) {
                    Ok(_) => {}
                    Err(e) => {
                        abort!(
                            "failed to assign expected output during range layer layout {:?}",
                            e
                        );
                    }
                };

                Ok(())
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                abort!("failed to assign fused layer region {:?}", e);
            }
        };
        // ValTensor::from(t);
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp;
    use itertools::Itertools;

    const RANGE: usize = 8; // 3-bit value

    use super::*;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        input: ValTensor<F>,
        output: ValTensor<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = RangeCheckConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, 4, 1, vec![1], true))
                .collect_vec();
            let input = &advices[0];
            let expected = &advices[1];
            RangeCheckConfig::configure(cs, input, expected, RANGE)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout(
                layouter.namespace(|| "assign value"),
                self.input.clone(),
                self.output.clone(),
            );

            Ok(())
        }
    }

    #[test]
    fn test_range_check() {
        let k = 4;

        // Successful cases
        for i in 0..RANGE {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(i as u64))]), &[1]).unwrap();
            let out =
                Tensor::new(Some(&[Value::<Fp>::known(Fp::from(i as u64 + 1))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }
        {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(22_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(0_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            match prover.verify() {
                Ok(_) => {
                    assert!(false)
                }
                Err(_) => {
                    assert!(true)
                }
            }
        }
    }
}
