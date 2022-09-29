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
pub struct ConvConfig<F: FieldExt + TensorType, const STRIDE: usize, const PADDING: usize>
where
    Value<F>: TensorType,
{
    selector: Selector,
    kernel: VarTensor,
    input: VarTensor,
    pub output: VarTensor,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const STRIDE: usize, const PADDING: usize> LayerConfig<F>
    for ConvConfig<F, STRIDE, PADDING>
where
    Value<F>: TensorType,
{
    /// Configures and creates a convolution gate within a circuit.
    /// Variables are supplied as a 3-element array of `[kernel, input, output]` VarTensors.
    fn configure(meta: &mut ConstraintSystem<F>, variables: &[VarTensor]) -> Self {
        assert_eq!(variables.len(), 3);
        let (kernel, input, output) = (
            variables[0].clone(),
            variables[1].clone(),
            variables[2].clone(),
        );
        assert_eq!(input.dims().len(), 3);
        assert_eq!(output.dims().len(), 3);
        assert_eq!(kernel.dims().len(), 4);

        kernel.enable_equality(meta);
        input.enable_equality(meta);
        output.enable_equality(meta);

        let image_width = input.dims()[2];

        let config = Self {
            selector: meta.selector(),
            kernel,
            input,
            output,
            _marker: PhantomData,
        };

        meta.create_gate("convolution", |meta| {
            let selector = meta.query_selector(config.selector);

            // Get output expressions for each input channel
            let image = config.input.query(meta, 0);
            let kernel = config.kernel.query(meta, 0);

            let expected_output = convolution::<_, PADDING, STRIDE>(kernel, image);

            let witnessed_output = config.output.query(meta, image_width);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    /// Assigns values to the convolution gate variables created when calling `configure`.
    /// Values are supplied as a 2-element array of `[kernel, input]` VarTensors.
    fn layout(&self, layouter: &mut impl Layouter<F>, values: &[ValTensor<F>]) -> ValTensor<F> {
        assert_eq!(values.len(), 2);

        let (kernel, input) = (values[0].clone(), values[1].clone());
        let (image_height, image_width) = (input.dims()[1], input.dims()[2]);
        let (out_channels, kernel_height, kernel_width) =
            (kernel.dims()[0], kernel.dims()[2], kernel.dims()[3]);

        let horz = (image_height + 2 * PADDING - kernel_height) / STRIDE + 1;
        let vert = (image_width + 2 * PADDING - kernel_width) / STRIDE + 1;

        let mut t = layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    self.selector.enable(&mut region, 0)?;

                    self.kernel.assign(&mut region, 0, kernel.clone());
                    self.input.assign(&mut region, 0, input.clone());

                    let output = match input.clone() {
                        ValTensor::Value {
                            inner: img,
                            dims: _,
                        } => match kernel.clone() {
                            ValTensor::Value { inner: k, dims: _ } => {
                                convolution::<_, PADDING, STRIDE>(k, img)
                            }
                            _ => panic!("not implemented"),
                        },
                        _ => panic!("not implemented"),
                    };

                    Ok(self
                        .output
                        .assign(&mut region, image_width, ValTensor::from(output)))
                },
            )
            .unwrap();
        t.reshape(&[out_channels, horz, vert]);
        ValTensor::from(t)
    }
}
