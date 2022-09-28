use super::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{Assigned, ConstraintSystem, Constraints, Expression, Selector},
};
use std::marker::PhantomData;

/// Configuration for an affine layer which (mat)multiplies a weight kernel to an input and adds
/// a bias vector to the result.
#[derive(Clone)]
pub struct Affine1dConfig<F: FieldExt + TensorType> {
    pub kernel: VarTensor,
    pub bias: VarTensor,
    pub input: VarTensor,
    pub output: VarTensor,
    pub selector: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> LayerConfig<F> for Affine1dConfig<F> {
    /// Configures and creates an affine gate within a circuit.
    /// Also constrains the output of the gate.
    fn configure(
        meta: &mut ConstraintSystem<F>,
        params: &[VarTensor],
        input: VarTensor,
        output: VarTensor,
    ) -> Self {
        assert!(params.len() == 2);

        let (kernel, bias) = (params[0].clone(), params[1].clone());

        assert_eq!(kernel.dims()[1], input.dims()[0]);
        assert_eq!(kernel.dims()[0], output.dims()[0]);
        assert_eq!(kernel.dims()[0], bias.dims()[0]);

        let in_dim = input.dims()[0];

        let config = Self {
            selector: meta.selector(),
            kernel,
            bias,
            input,
            output,
            _marker: PhantomData,
        };

        meta.create_gate("affine", |meta| {
            let selector = meta.query_selector(config.selector);
            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = config.output.query(meta, 0);
            // Now we compute the linear expression,  and add it to constraints
            let witnessed_output = expected_output.enum_map(|i, _| {
                let mut c = Expression::Constant(<F as TensorType>::zero().unwrap());
                for j in 0..in_dim {
                    c = c + config.kernel.query_idx(meta, i, j) * config.input.query_idx(meta, 0, j)
                }
                c + config.bias.query_idx(meta, 0, i)
                // add the bias
            });

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns values to the affine gate variables created when calling `configure`.
    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        params: &[ValTensor<F>],
    ) -> ValTensor<F> {
        assert_eq!(params.len(), 2);

        let (kernel, bias) = (params[0].clone(), params[1].clone());
        let t = layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    let offset = 0;
                    self.selector.enable(&mut region, offset)?;
                    let input = self.input.assign(&mut region, offset, input.clone());
                    let weights = self.kernel.assign(&mut region, offset, kernel.clone());
                    let bias = self.bias.assign(&mut region, offset, bias.clone());
                    // calculate value of output
                    let mut output: Tensor<Value<Assigned<F>>> =
                        Tensor::new(None, &[kernel.dims()[0]]).unwrap();

                    output = output.enum_map(|i, mut o| {
                        input.enum_map(|j, x| {
                            o = o + x.value_field() * weights.get(&[i, j]).value_field();
                        });

                        o + bias.get(&[i]).value_field()
                    });

                    Ok(self
                        .output
                        .assign(&mut region, offset, ValTensor::from(output)))
                },
            )
            .unwrap();

        ValTensor::from(t)
    }
}
