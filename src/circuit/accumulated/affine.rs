// use super::*;
use crate::circuit::accumulated::matmul::Config as MatmulConfig;
use crate::tensor::TensorType;
use crate::tensor::{ValTensor, VarTensor};
use halo2_proofs::plonk::ConstraintSystem;
use halo2_proofs::{arithmetic::FieldExt, circuit::Layouter};
use std::error::Error;

/// Configuration for an accumulated arg.
#[derive(Clone, Debug)]
pub struct Config<F: FieldExt + TensorType> {
    /// underlying matmul config.
    pub matmul_config: MatmulConfig<F>,
}

impl<F: FieldExt + TensorType> Config<F> {
    /// Configures the sequence of operations into a circuit gate.
    /// # Arguments
    /// * `inputs` - The explicit inputs to the operations.
    /// * `output` - The variable representing the (currently singular) output of the operations.
    /// * `op` - The operation being represented
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor; 2],
        output: &VarTensor,
    ) -> Self {
        Config {
            matmul_config: MatmulConfig::configure(meta, inputs, output),
        }
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
        let (kernel, bias, mut input) = (values[0].clone(), values[1].clone(), values[2].clone());

        input.pad_row_ones()?;
        let params = kernel.append_to_row(bias)?;

        self.matmul_config.layout(layouter, &[params, input])
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use super::*;
    use crate::tensor::Tensor;
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
        inputs: [ValTensor<F>; 3],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for AffineCircuit<F> {
        type Config = Config<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true, 512);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true, 512);
            let output =
                VarTensor::new_advice(cs, K, (LEN + 2) * LEN, vec![LEN + 2, 1, LEN], true, 512);
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
    fn affinecircuit() {
        // parameters
        let mut w = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, LEN]);

        let mut b = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        b.reshape(&[LEN, 1]);

        let mut x = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        x.reshape(&[LEN, 1]);

        let circuit = AffineCircuit::<F> {
            inputs: [ValTensor::from(w), ValTensor::from(b), ValTensor::from(x)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
