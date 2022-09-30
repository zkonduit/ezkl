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
    input: VarTensor,
    pub output: VarTensor,
    conv_params: [usize; 2],
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> LayerConfig<F> for ConvConfig<F>
where
    Value<F>: TensorType,
{
    /// Configures and creates a convolution gate within a circuit.
    /// Variables are supplied as a 3-element array of `[kernel, input, output]` VarTensors.
    /// Takes in conv layer params as a 2-element array of `[padding, stride]` `usize` elements.
    fn configure(
        meta: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        conv_params: Option<&[usize]>,
    ) -> Self {
        assert_eq!(variables.len(), 3);
        let (kernel, input, output) = (
            variables[0].clone(),
            variables[1].clone(),
            variables[2].clone(),
        );
        assert_eq!(input.dims().len(), 3);
        assert_eq!(output.dims().len(), 3);
        assert_eq!(kernel.dims().len(), 4);

        // should fail if None
        let conv_params = conv_params.unwrap();
        assert_eq!(conv_params.len(), 2);

        kernel.enable_equality(meta);
        input.enable_equality(meta);
        output.enable_equality(meta);

        let image_width = input.dims()[2];

        let config = Self {
            selector: meta.selector(),
            kernel,
            input,
            output,
            conv_params: conv_params[..2].try_into().unwrap(),
            _marker: PhantomData,
        };

        meta.create_gate("convolution", |meta| {
            let selector = meta.query_selector(config.selector);

            // Get output expressions for each input channel
            let image = config.input.query(meta, 0);
            let kernel = config.kernel.query(meta, 0);

            let expected_output = convolution(kernel, image, conv_params);

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
        let image_width = input.dims()[2];

        let t = layouter
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
                                convolution::<_>(k, img, &self.conv_params)
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    };

                    Ok(self
                        .output
                        .assign(&mut region, image_width, ValTensor::from(output)))
                },
            )
            .unwrap();

        println!("{:?}", t.dims());

        ValTensor::from(t)
    }
}
