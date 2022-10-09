use super::*;
use crate::tensor::ops::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
};
use std::marker::PhantomData;

/// Configuration for an affine layer which (mat)multiplies a weight kernel to an input and adds
/// a bias vector to the result.
#[derive(Clone, Debug)]
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
    /// Variables are supplied as a 4-element array of `[weights, bias, input, output]` VarTensors.
    fn configure(
        meta: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        _: Option<&[usize]>,
    ) -> Self {
        assert_eq!(variables.len(), 4);

        let (kernel, bias, input, output) = (
            variables[0].clone(),
            variables[1].clone(),
            variables[2].clone(),
            variables[3].clone(),
        );

        assert_eq!(kernel.dims()[1], input.dims()[0]);
        assert_eq!(kernel.dims()[0], output.dims()[0]);
        assert_eq!(kernel.dims()[0], bias.dims()[0]);

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

            // Now we compute the linear expression,  and add it to constraints
            let input = config.input.query(meta, 0);
            let kernel = config.kernel.query(meta, 0);
            let bias = config.bias.query(meta, 0);

            let witnessed_output = affine(&vec![&input, &kernel, &bias]);

            // Get output expressions for each input channel
            let expected_output: Tensor<Expression<F>> = config.output.query(meta, 0);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns values to the affine gate variables created when calling `configure`.
    /// Values are supplied as a 3-element array of `[weights, bias, input]` VarTensors.
    fn layout(&self, layouter: &mut impl Layouter<F>, values: &[ValTensor<F>]) -> ValTensor<F> {
        assert_eq!(values.len(), 3);

        let (kernel, bias, input) = (values[0].clone(), values[1].clone(), values[2].clone());
        let t = layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    let offset = 0;
                    self.selector.enable(&mut region, offset)?;
                    let k = utils::value_muxer(
                        &self.kernel,
                        &self
                            .kernel
                            .assign(&mut region, offset, &kernel)
                            .map(|e| e.value_field().evaluate()),
                        &kernel,
                    );

                    let b = utils::value_muxer(
                        &self.bias,
                        &self
                            .bias
                            .assign(&mut region, offset, &bias)
                            .map(|e| e.value_field().evaluate()),
                        &bias,
                    );

                    let inp = utils::value_muxer(
                        &self.input,
                        &self
                            .input
                            .assign(&mut region, offset, &input)
                            .map(|e| e.value_field().evaluate()),
                        &input,
                    );

                    let mut output: ValTensor<F> = affine(&vec![&inp, &k, &b]).into();
                    output.flatten();

                    Ok(self.output.assign(&mut region, offset, &output))
                },
            )
            .unwrap();

        ValTensor::from(t)
    }
}
