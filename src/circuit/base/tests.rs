use crate::circuit::base::*;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::pasta::pallas;
use halo2curves::pasta::Fp as F;
use rand::rngs::OsRng;
use std::marker::PhantomData;

#[cfg(test)]
mod matmul {
    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct MatmulCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, 1, LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut &mut 0, Op::Matmul)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a =
            Tensor::from((0..(LEN + 1) * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN + 1]);

        let mut w = Tensor::from((0..LEN + 1).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN + 1, 1]);

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_col_overflow {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;

    #[derive(Clone)]
    struct MatmulCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN * LEN * LEN, vec![LEN, LEN, LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN * LEN * LEN, vec![LEN, LEN, LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN * LEN * LEN, vec![LEN, LEN, LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Matmul)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]);

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]);

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dot {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Dot)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod dot_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Dot)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Sum)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sum_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Sum)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod batchnorm {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct BNCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 3],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for BNCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::BatchNorm)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn batchnormcircuit() {
        // parameters
        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN]);

        let mut b = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        b.reshape(&[LEN]);

        let mut x = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        x.reshape(&[LEN]);

        let circuit = BNCircuit::<F> {
            inputs: [ValTensor::from(x), ValTensor::from(w), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod affine {
    use std::marker::PhantomData;

    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct AffineCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 3],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for AffineCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let output = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Affine)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
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
            inputs: [ValTensor::from(x), ValTensor::from(w), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod affine_col_overflow {
    use std::marker::PhantomData;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct AffineCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 3],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for AffineCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let output = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Affine)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
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
            inputs: [ValTensor::from(x), ValTensor::from(w), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod composition {
    use super::*;

    const K: usize = 9;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // lots of stacked dot products
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        let mut offset = 0;
                        let _ = config
                            .layout(&mut region, &self.inputs.clone(), &mut offset, Op::Dot)
                            .unwrap();
                        let _ = config
                            .layout(&mut region, &self.inputs.clone(), &mut offset, Op::Dot)
                            .unwrap();
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut offset, Op::Dot)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod conv {
    use halo2_proofs::arithmetic::Field;

    use super::*;

    const K: usize = 22;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: FieldExt + TensorType> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let output = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                &mut 0,
                                Op::Conv {
                                    padding: (1, 1),
                                    stride: (2, 2),
                                },
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn convcircuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 3;
        let image_height = 5;
        let image_width = 7;
        let in_channels = 3;
        let out_channels = 2;

        let mut image = Tensor::from(
            (0..in_channels * image_height * image_width)
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        image.reshape(&[in_channels, image_height, image_width]);
        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        kernels.reshape(&[out_channels, in_channels, kernel_height, kernel_width]);

        let bias =
            Tensor::from((0..{ out_channels }).map(|_| Value::known(pallas::Base::random(OsRng))));

        let circuit = ConvCircuit::<F> {
            inputs: [
                ValTensor::from(image),
                ValTensor::from(kernels),
                ValTensor::from(bias),
            ]
            .to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn convcircuitnobias() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = 4;
        let image_width = 5;
        let in_channels = 3;
        let out_channels = 2;

        let mut image = Tensor::from(
            (0..in_channels * image_height * image_width).map(|i| Value::known(F::from(i as u64))),
        );
        image.reshape(&[in_channels, image_height, image_width]);
        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| Value::known(F::from(i as u64))),
        );
        kernels.reshape(&[out_channels, in_channels, kernel_height, kernel_width]);

        let circuit = ConvCircuit::<F> {
            inputs: [ValTensor::from(image), ValTensor::from(kernels)].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sumpool {
    use halo2_proofs::arithmetic::Field;

    use super::*;

    const K: usize = 20;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: FieldExt + TensorType> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, LEN], true);
            let output = VarTensor::new_advice(cs, K, (LEN + 1) * LEN, vec![LEN + 1, 1, LEN], true);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                &mut 0,
                                Op::SumPool {
                                    padding: (0, 0),
                                    stride: (1, 1),
                                    kernel_shape: (3, 3),
                                },
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumpoolcircuit() {
        let image_height = 5;
        let image_width = 5;
        let in_channels = 1;

        let mut image = Tensor::from(
            (0..in_channels * image_height * image_width)
                .map(|_| Value::known(pallas::Base::random(OsRng))),
        );
        image.reshape(&[in_channels, image_height, image_width]);

        let circuit = ConvCircuit::<F> {
            inputs: [ValTensor::from(image)].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod add {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Add)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod add_with_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Add)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod sub {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Sub)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn subcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod mult {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Mult)
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn multcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod pow {
    use super::*;

    const K: usize = 8;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Pow(5))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn powcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod pack {
    use super::*;

    const K: usize = 8;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(&mut region, &self.inputs.clone(), &mut 0, Op::Pack(2, 1))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn packcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod rescaled {
    use super::*;

    const K: usize = 8;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, 0)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                &mut 0,
                                Op::Rescaled {
                                    inner: Box::new(Op::Sum),
                                    scale: vec![(0, 5)],
                                },
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn rescaledcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));
        a.reshape(&[LEN, 1]);

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod rangecheck {

    use crate::tensor::Tensor;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp;

    const RANGE: usize = 8; // 3-bit value
    const K: usize = 8;
    const LEN: usize = 4;

    use super::*;

    #[derive(Clone)]
    struct MyCircuit<F: FieldExt + TensorType> {
        input: ValTensor<F>,
        output: ValTensor<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let b = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
            let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE, RANGE as i32)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(
                                &mut region,
                                &[self.input.clone(), self.output.clone()],
                                &mut 0,
                                Op::RangeCheck(RANGE as i32),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_range_check() {
        let k = 4;

        // Successful cases
        for i in 0..RANGE {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(i as u64))]), &[1]).unwrap();
            let out =
                Tensor::new(Some(&[Value::<Fp>::known(Fp::from(i as u64 + 1))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }
        {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(22_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(0_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
            };
            let prover = MockProver::run(k, &circuit, vec![]).unwrap();
            match prover.verify() {
                Ok(_) => {
                    assert!(false)
                }
                Err(_) => {
                    assert!(true)
                }
            }
        }
    }
}
