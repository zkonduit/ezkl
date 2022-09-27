use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Assigned, ConstraintSystem, Constraints, Selector},
};

use super::*;
use crate::nn::io::*;
use crate::tensor_ops::*;

/// Configuration for a convolutional layer which convolves a kernel with an input (image).
#[derive(Debug, Clone)]
pub struct ConvConfig<F: FieldExt + TensorType, const STRIDE: usize, const PADDING: usize>
where
    Value<F>: TensorType,
{
    selector: Selector,
    kernel: IOConfig<F>,
    image: IOConfig<F>,
    pub output: IOConfig<F>,
}

impl<F: FieldExt + TensorType, const STRIDE: usize, const PADDING: usize> LayerConfig<F>
    for ConvConfig<F, STRIDE, PADDING>
where
    Value<F>: TensorType,
{
    fn configure(
        meta: &mut ConstraintSystem<F>,
        params: &[VarTensor],
        input: VarTensor,
        output: VarTensor,
    ) -> Self {
        assert_eq!(params.len(), 1);
        assert_eq!(input.dims().len(), 3);
        assert_eq!(output.dims().len(), 3);
        assert_eq!(params[0].dims().len(), 4);

        let kernel = params[0].clone();
        kernel.enable_equality(meta);
        input.enable_equality(meta);
        output.enable_equality(meta);

        let image_width = input.dims()[2];

        let config = Self {
            selector: meta.selector(),
            kernel: IOConfig::configure(meta, kernel),
            image: IOConfig::configure(meta, input),
            output: IOConfig::configure(meta, output),
        };

        meta.create_gate("convolution", |meta| {
            let selector = meta.query_selector(config.selector);

            // Get output expressions for each input channel
            let image = config.image.query(meta, 0);
            let kernel = config.kernel.query(meta, 0);

            let expected_output = convolution::<_, PADDING, STRIDE>(kernel, image);

            let witnessed_output = config.output.query(meta, image_width);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        params: &[ValTensor<F>],
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        assert_eq!(params.len(), 1);
        let kernel = params[0].clone();
        let image_width = input.dims()[2];
        layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    self.selector.enable(&mut region, 0)?;

                    self.kernel.assign(&mut region, 0, kernel.clone());
                    self.image.assign(&mut region, 0, input.clone());

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
            .unwrap()
    }

    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        params: &[ValTensor<F>],
    ) -> ValTensor<F> {
        assert_eq!(params.len(), 1);
        let kernel = params[0].clone();
        let (image_height, image_width) = (input.dims()[1], input.dims()[2]);
        let (out_channels, kernel_height, kernel_width) =
            (kernel.dims()[0], kernel.dims()[2], kernel.dims()[3]);

        let horz = (image_height + 2 * PADDING - kernel_height) / STRIDE + 1;
        let vert = (image_width + 2 * PADDING - kernel_width) / STRIDE + 1;

        let mut t = self.assign(
            &mut layouter.namespace(|| format!("filter")),
            input.clone(),
            params,
        );
        t.reshape(&[out_channels, horz, vert]);
        ValTensor::from(t)
    }
}
