use crate::abort;
use crate::fieldutils::i32_to_felt;
use crate::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter},
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use log::error;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct RangeCheckConfig<F: FieldExt, const RANGE: usize> {
    input: VarTensor,
    pub output: VarTensor,
    selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const RANGE: usize> RangeCheckConfig<F, RANGE> {
    pub fn configure(meta: &mut ConstraintSystem<F>, input: &VarTensor,  output: &VarTensor) -> Self {
        let config = Self {
            selector: meta.selector(),
            input: input.clone(),
            output: output.clone(),
            _marker: PhantomData,
        };
       

        meta.create_gate("range check", |meta| {
            //        value     |    q_range_check
            //       ------------------------------
            //          v       |         1

            let q = meta.query_selector(config.selector);
            let witnessed = match input.query(meta, 0) {
                Ok(q) => q,
                Err(e) => {
                    abort!("failed to query input {:?}", e);
                }
            };

            // Get output expressions for each input channel
            let expected: Tensor<Expression<F>> = match config.output.query(meta, 0) {
                Ok(res) => res,
                Err(e) => {
                    abort!("failed to query output during fused layer layout {:?}", e);
                }
            };

            // Given a range R and a value v, returns the expression
            // (v) * (1 - v) * (2 - v) * ... * (R - 1 - v)
            let range_check = |range: i32, value: Expression<F>| {
                assert!(range > 0);
                (-range..range).fold(value.clone(), |expr, i| {
                    expr * (Expression::Constant(i32_to_felt(i)) - value.clone())
                })
            };

            let constraints = witnessed.enum_map(|i, o| range_check(RANGE as i32, o - expected[i].clone()));
            Constraints::with_selector(q, constraints)
        });

       config
    }

    pub fn layout(&self, mut layouter: impl Layouter<F>, input: ValTensor<F>, output: ValTensor<F>) -> ValTensor<F> {
        let t = match layouter.assign_region(
            || "Assign value",
            |mut region| {
                let offset = 0;

                // Enable q_range_check
                self.selector.enable(&mut region, offset)?;

                match self.input.assign(&mut region, offset, &input) {
                    Ok(res) => {res.map(|elem| elem.value_field().evaluate());},
                    Err(e) => {
                        abort!("failed to assign inputs during range layer layout {:?}", e);
                    }
                };
                match self.output.assign(&mut region, offset, &output) {
                    Ok(res) => Ok(res.map(|elem| elem.value_field().evaluate())),
                    Err(e) => {
                        abort!("failed to assign outputs during range layer layout {:?}", e);
                    }
                }
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                abort!("failed to assign fused layer region {:?}", e);
            }
        };

        ValTensor::from(t)
    }
}

#[cfg(test)]
mod tests {
    use halo2_proofs::{
        arithmetic::{FieldExt},
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::{MockProver},
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp;

    use super::*;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType, const RANGE: usize> {
        input: ValTensor<F>,
        output: ValTensor<F>,
    }

    impl<F: FieldExt + TensorType, const RANGE: usize> Circuit<F> for MyCircuit<F, RANGE> {
        type Config = RangeCheckConfig<F, RANGE>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = VarTensor::from(Tensor::from((0..2).map(|_| {
                let col = cs.advice_column();
                cs.enable_equality(col);
                col
            })));
            let input = advices.get_slice(&[0..1], &[1]);
            let output = advices.get_slice(&[1..2], &[1]);

            RangeCheckConfig::configure(cs, &input, &output)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout(layouter.namespace(|| "Assign value"), self.input.clone(), self.output.clone());

            Ok(())
        }
    }

    #[test]
    fn test_range_check() {
        let k = 4;
        const RANGE: usize = 8; // 3-bit value

        // Successful cases
        for i in 0..RANGE {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(i as u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(1 as u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp, RANGE> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
            };

            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }
        {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(22 as u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(1 as u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp, RANGE> {
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
