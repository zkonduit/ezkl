use super::*;
use crate::tensor::ops::*;
use crate::tensor::TensorType;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Constraints, Selector},
};
use std::marker::PhantomData;

/// Configuration for a convolutional layer which convolves a kernel with an input (image).
#[derive(Debug, Clone)]
pub struct ConvConfig<F: FieldExt + TensorType>
where
    Value<F>: TensorType,
{
    selector: Selector,
    kernel: VarTensor,
    bias: VarTensor,
    input: VarTensor,
    pub output: VarTensor,
    padding: (usize, usize),
    stride: (usize, usize),
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> LayerConfig<F> for ConvConfig<F>
where
    Value<F>: TensorType,
{
    /// Configures and creates a convolution gate within a circuit.
    /// Variables are supplied as a 3-element array of `[kernel, input, output]` VarTensors.
    /// Takes in conv layer params as a 4-element array of `[padding_x, padding_y, stride_x, stride_y]` `usize` elements.
    fn configure(
        meta: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        conv_params: Option<&[usize]>,
    ) -> Self {
        assert_eq!(variables.len(), 4);
        let (kernel, bias, input, output) = (
            variables[0].clone(),
            variables[1].clone(),
            variables[2].clone(),
            variables[3].clone(),
        );
        assert_eq!(input.dims().len(), 3);
        assert_eq!(output.dims().len(), 3);
        assert_eq!(kernel.dims().len(), 4);
        assert_eq!(bias.dims().len(), 1);

        // should fail if None
        let conv_params = conv_params.unwrap();
        assert_eq!(conv_params.len(), 4);

        kernel.enable_equality(meta);
        input.enable_equality(meta);
        output.enable_equality(meta);

        let image_width = input.dims()[2];

        let config = Self {
            selector: meta.selector(),
            kernel,
            bias,
            input,
            output,
            padding: (conv_params[0], conv_params[1]),
            stride: (conv_params[2], conv_params[3]),
            _marker: PhantomData,
        };

        meta.create_gate("convolution", |meta| {
            let selector = meta.query_selector(config.selector);

            // Get output expressions for each input channel
            let image = config.input.query(meta, 0);
            let kernel = config.kernel.query(meta, 0);
            let bias = config.bias.query(meta, 0);

            let expected_output =
                convolution(&vec![&image, &kernel, &bias], config.padding, config.stride);

            let witnessed_output = config.output.query(meta, image_width);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns values to the convolution gate variables created when calling `configure`.
    /// Values are supplied as a 2-element array of `[kernel, input]` VarTensors.
    fn layout(&self, layouter: &mut impl Layouter<F>, values: &[ValTensor<F>]) -> ValTensor<F> {
        assert_eq!(values.len(), 3);

        let (kernel, bias, input) = (values[0].clone(), values[1].clone(), values[2].clone());
        let image_width = input.dims()[2];

        let t = layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    let offset = 0;
                    self.selector.enable(&mut region, 0)?;

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

                    let output: ValTensor<F> =
                        convolution(&vec![&inp, &k, &b], self.padding, self.stride).into();

                    Ok(self.output.assign(&mut region, image_width, &output))
                },
            )
            .unwrap();

        ValTensor::from(t)
    }
}
