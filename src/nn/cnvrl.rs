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
pub struct ConvConfig<
    F: FieldExt + TensorType,
    const STRIDE: usize,
    const IN_CHANNELS: usize,
    const PADDING: usize,
> where
    Value<F>: TensorType,
{
    selector: Selector,
    kernel: IOConfig<F>,
    image: IOConfig<F>,
    pub output: IOConfig<F>,
}

impl<
        F: FieldExt + TensorType,
        const STRIDE: usize,
        const IN_CHANNELS: usize,
        const PADDING: usize,
    > LayerConfig<F> for ConvConfig<F, STRIDE, IN_CHANNELS, PADDING>
where
    Value<F>: TensorType,
{
    fn configure(
        meta: &mut ConstraintSystem<F>,
        params: &[VarTensor],
        input: VarTensor,
        output: VarTensor,
    ) -> Self {
        assert!(params.len() == 1);

        let kernel = params[0].clone();
        kernel.enable_equality(meta);
        input.enable_equality(meta);
        output.enable_equality(meta);

        let image_height = input.dims()[1];

        let config = Self {
            selector: meta.selector(),
            kernel: IOConfig::configure(meta, kernel),
            image: IOConfig::configure(meta, input),
            output: IOConfig::configure(meta, output),
        };

        meta.create_gate("convolution", |meta| {
            let selector = meta.query_selector(config.selector);

            // Get output expressions for each input channel
            let intermediate_outputs = (0..IN_CHANNELS)
                .map(|rotation| {
                    let image = config.image.query(meta, rotation * image_height);
                    let kernel = config.kernel.query(meta, rotation * image_height);
                    convolution::<_, PADDING, STRIDE>(kernel, image)
                })
                .collect();

            let witnessed_output = config.output.query(meta, IN_CHANNELS * image_height);
            let expected_output = op(intermediate_outputs, |a, b| a + b);

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
        assert!(params.len() == 1);
        let kernel = params[0].clone();
        let (in_channels, image_height) = (input.dims()[0], input.dims()[2]);
        layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    let mut offset = 0;
                    self.selector.enable(&mut region, offset)?;

                    let outputs = (0..in_channels)
                        .map(|i| {
                            self.kernel
                                .assign(&mut region, offset, kernel.get_slice(&[i..i + 1]));

                            self.image
                                .assign(&mut region, offset, input.get_slice(&[i..i + 1]));

                            let output = match input.clone() {
                                ValTensor::Value {
                                    inner: img,
                                    dims: _,
                                } => match kernel.clone() {
                                    ValTensor::Value { inner: k, dims: _ } => {
                                        convolution::<_, PADDING, STRIDE>(
                                            k.get_slice(&[i..i + 1]),
                                            img.get_slice(&[i..i + 1]),
                                        )
                                    }
                                    _ => panic!("not implemented"),
                                },
                                _ => panic!("not implemented"),
                            };

                            offset += image_height;
                            output
                        })
                        .collect();

                    let output = op(outputs, |a, b| a + b);
                    Ok(self
                        .output
                        .assign(&mut region, offset, ValTensor::from(output)))
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
        let (image_width, image_height) = (input.dims()[1], input.dims()[2]);
        let (out_channels, kernel_width, kernel_height) =
            (kernel.dims()[0], kernel.dims()[2], kernel.dims()[3]);
        let horz = (image_width + 2 * PADDING - kernel_width) / STRIDE + 1;
        let vert = (image_height + 2 * PADDING - kernel_height) / STRIDE + 1;

        let t = Tensor::from((0..out_channels).map(|i| {
            self.assign(
                &mut layouter.namespace(|| format!("filter: {:?}", i)),
                input.clone(),
                &[kernel.get_slice(&[i..i + 1])],
            )
        }));
        let mut t = t.flatten();
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
        type Config = ConvConfig<F, STRIDE, IN_CHANNELS, PADDING>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
        // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
            let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

            let num_advices = max(output_width, IMAGE_WIDTH);

            let advices =
                VarTensor::from(Tensor::from((0..num_advices).map(|_| meta.advice_column())));

            let mut kernel =
                Tensor::from((0..KERNEL_WIDTH * KERNEL_HEIGHT).map(|_| meta.fixed_column()));
            kernel.reshape(&[KERNEL_WIDTH, KERNEL_HEIGHT]);

            Self::Config::configure(
                meta,
                &[VarTensor::from(kernel)],
                advices.get_slice(&[0..IMAGE_WIDTH], &[IMAGE_WIDTH, IMAGE_HEIGHT]),
                advices.get_slice(&[0..output_width], &[output_width, output_height]),
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
        image.reshape(&[IN_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT]);
        let mut kernels = Tensor::from(
            (0..{ OUT_CHANNELS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH })
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        kernels.reshape(&[OUT_CHANNELS, IN_CHANNELS, KERNEL_WIDTH, KERNEL_HEIGHT]);

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
