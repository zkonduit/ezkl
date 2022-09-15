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
pub struct Config<
    F: FieldExt + TensorType,
    const KERNEL_HEIGHT: usize,
    const KERNEL_WIDTH: usize,
    const OUT_CHANNELS: usize,
    const STRIDE: usize,
    const IMAGE_HEIGHT: usize,
    const IMAGE_WIDTH: usize,
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
        const KERNEL_HEIGHT: usize,
        const KERNEL_WIDTH: usize,
        const OUT_CHANNELS: usize,
        const STRIDE: usize,
        const IMAGE_HEIGHT: usize,
        const IMAGE_WIDTH: usize,
        const IN_CHANNELS: usize,
        const PADDING: usize,
    > LayerConfig<F>
    for Config<
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
    fn configure(
        meta: &mut ConstraintSystem<F>,
        params: ParamType,
        input: ParamType,
        output: ParamType,
    ) -> Self {
        let output_height = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
        let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

        input.enable_equality(meta);
        params.enable_equality(meta);
        output.enable_equality(meta);

        let config = Self {
            selector: meta.selector(),
            kernel: IOConfig::configure(meta, params, &[KERNEL_WIDTH, KERNEL_HEIGHT]),
            image: IOConfig::configure(meta, input, &[IMAGE_WIDTH, IMAGE_HEIGHT]),
            output: IOConfig::configure(meta, output, &[output_width, output_height]),
        };

        meta.create_gate("convolution", |meta| {
            let selector = meta.query_selector(config.selector);

            // Get output expressions for each input channel
            let intermediate_outputs = (0..IN_CHANNELS)
                .map(|rotation| {
                    let image = config.image.query(meta, rotation * IMAGE_HEIGHT);
                    let kernel = config.kernel.query(meta, rotation * IMAGE_HEIGHT);
                    println!("image {:?} kernel {:?}", image, kernel);
                    convolution::<_, PADDING, STRIDE>(kernel, image)
                })
                .collect();

            let witnessed_output = config.output.query(meta, IN_CHANNELS * IMAGE_HEIGHT);
            let expected_output = op(intermediate_outputs, |a, b| a + b);

            let constraints = witnessed_output.enum_map(|i, o| o - expected_output[i].clone());

            Constraints::with_selector(selector, constraints)
        });

        config
    }

    fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        image: IOType<F>,
        kernel: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        layouter
            .assign_region(
                || "assign image and kernel",
                |mut region| {
                    let mut offset = 0;
                    self.selector.enable(&mut region, offset)?;

                    let outputs = (0..IN_CHANNELS)
                        .map(|i| {
                            self.kernel
                                .assign(&mut region, offset, kernel.get_slice(&[i..i + 1]));

                            self.image
                                .assign(&mut region, offset, image.get_slice(&[i..i + 1]));

                            let output = match image.clone() {
                                IOType::Value(img) => match kernel.clone() {
                                    IOType::Value(k) => convolution::<_, PADDING, STRIDE>(
                                        k.get_slice(&[i..i + 1]),
                                        img.get_slice(&[i..i + 1]),
                                    ),
                                    _ => panic!("not implemented"),
                                },
                                _ => panic!("not implemented"),
                            };

                            offset += IMAGE_HEIGHT;
                            output
                        })
                        .collect();

                    let output = op(outputs, |a, b| a + b);
                    Ok(self
                        .output
                        .assign(&mut region, offset, IOType::Value(output)))
                },
            )
            .unwrap()
    }

    fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: IOType<F>,
        kernels: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        let horz = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;
        let vert = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
        let t = Tensor::from((0..OUT_CHANNELS).map(|i| {
            self.assign(
                &mut layouter.namespace(|| format!("filter: {:?}", i)),
                input.clone(),
                kernels.get_slice(&[i..i + 1]),
            )
        }));
        let mut t = t.flatten();
        t.reshape(&[OUT_CHANNELS, horz, vert]);
        t
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
        image: IOType<F>,
        kernels: IOType<F>,
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
        type Config = Config<
            F,
            KERNEL_HEIGHT,
            KERNEL_WIDTH,
            OUT_CHANNELS,
            STRIDE,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IN_CHANNELS,
            PADDING,
        >;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
        // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            let output_width = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

            let num_advices = max(output_width, IMAGE_WIDTH);

            let advices =
                ParamType::Advice(Tensor::from((0..num_advices).map(|_| meta.advice_column())));

            let mut kernel =
                Tensor::from((0..KERNEL_WIDTH * KERNEL_HEIGHT).map(|_| meta.fixed_column()));
            kernel.reshape(&[KERNEL_WIDTH, KERNEL_HEIGHT]);

            Self::Config::configure(
                meta,
                ParamType::Fixed(kernel),
                advices.get_slice(&[0..IMAGE_WIDTH]),
                advices.get_slice(&[0..output_width]),
            )
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            let _output = config.layout(&mut layouter, self.image.clone(), self.kernels.clone());
            Ok(())
        }
    }

    #[test]
    fn test_cnvrl() {
        //        use halo2_proofs::pasta::pallas;
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
            image: IOType::Value(image),
            kernels: IOType::Value(kernels),
        };

        let k = 8;
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
