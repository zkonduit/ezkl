use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Assigned, ConstraintSystem, Constraints, Selector},
};

use super::*;
use crate::nn::io::*;
use crate::tensor_ops::*;

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

#[cfg(test)]
mod tests {
    use std::cmp::max;

    use super::*;
    use halo2_proofs::{
        arithmetic::{Field, FieldExt},
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        plonk::{Circuit, Error},
    };
    use rand::rngs::OsRng;

    #[derive(Clone, Debug)]
    struct MyCircuit<
        F: FieldExt + TensorType,
        const KERNEL_HEIGHT: usize,
        const KERNEL_WIDTH: usize,
        const OUT_CHANNELS: usize,
        const STRIDE: usize,
        const IMAGE_HEIGHT: usize,
        const IMAGE_WIDTH: usize,
        const IN_CHANNELS: usize,
        const PADDING: usize,
    >
    where
        Value<F>: TensorType,
    {
        image: ValTensor<F>,
        kernels: ValTensor<F>,
    }

    impl<
            F: FieldExt + TensorType,
            const KERNEL_HEIGHT: usize,
            const KERNEL_WIDTH: usize,
            const OUT_CHANNELS: usize,
            const STRIDE: usize,
            const IMAGE_HEIGHT: usize,
            const IMAGE_WIDTH: usize,
            const IN_CHANNELS: usize,
            const PADDING: usize,
        > Circuit<F>
        for MyCircuit<
            F,
            KERNEL_HEIGHT,
            KERNEL_WIDTH,
            OUT_CHANNELS,
            STRIDE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IN_CHANNELS,
            PADDING,
        >
    where
        Value<F>: TensorType,
    {
        type Config = ConvConfig<F, STRIDE, PADDING>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
        // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
            let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

            let num_advices = max(output_height * OUT_CHANNELS, IMAGE_HEIGHT * IN_CHANNELS);

            let advices =
                VarTensor::from(Tensor::from((0..num_advices).map(|_| meta.advice_column())));

            let mut kernel = Tensor::from(
                (0..OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH)
                    .map(|_| meta.fixed_column()),
            );
            kernel.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]);

            Self::Config::configure(
                meta,
                &[VarTensor::from(kernel)],
                advices.get_slice(
                    &[0..IMAGE_HEIGHT * IN_CHANNELS],
                    &[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
                ),
                advices.get_slice(
                    &[0..output_height * OUT_CHANNELS],
                    &[OUT_CHANNELS, output_height, output_width],
                ),
            )
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _output = config.layout(&mut layouter, self.image.clone(), &[self.kernels.clone()]);
            Ok(())
        }
    }

    #[test]
    fn test_cnvrl() {
        use halo2curves::pasta::pallas;

        const KERNEL_HEIGHT: usize = 1;
        const KERNEL_WIDTH: usize = 3;
        const OUT_CHANNELS: usize = 2;
        const STRIDE: usize = 2;
        const IMAGE_HEIGHT: usize = 9;
        const IMAGE_WIDTH: usize = 7;
        const IN_CHANNELS: usize = 2;
        const PADDING: usize = 2;

        let mut image = Tensor::from(
            (0..IN_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        image.reshape(&[IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]);
        let mut kernels = Tensor::from(
            (0..{ OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH })
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        kernels.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH]);

        let circuit = MyCircuit::<
            pallas::Base,
            KERNEL_HEIGHT,
            KERNEL_WIDTH,
            OUT_CHANNELS,
            STRIDE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IN_CHANNELS,
            PADDING,
        > {
            image: ValTensor::from(image),
            kernels: ValTensor::from(kernels),
        };

        let k = 8;

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
