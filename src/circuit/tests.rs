use crate::circuit::ops::poly::PolyOp;
use crate::circuit::ops::hybrid::HybridOp;
use crate::circuit::*;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::ff::{Field, PrimeField};
use halo2curves::pasta::pallas;
use halo2curves::pasta::Fp as F;
use rand::rngs::OsRng;
use std::marker::PhantomData;

#[derive(Default)]
struct TestParams;

#[cfg(test)]
mod matmul {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN * LEN);
            let b = VarTensor::new_advice(cs, K, LEN * LEN);
            let output = VarTensor::new_advice(cs, K, LEN * LEN);
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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Matmul { a: None }),
                            )
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
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, LEN * LEN * LEN);
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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Matmul { a: None }),
                            )
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
    use ops::poly::PolyOp;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Dot),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Dot),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
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
mod composition {
    use super::*;

    const K: usize = 9;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                            .layout(
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut offset,
                                Box::new(PolyOp::Dot),
                            )
                            .unwrap();
                        let _ = config
                            .layout(
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut offset,
                                Box::new(PolyOp::Dot),
                            )
                            .unwrap();
                        config
                            .layout(
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut offset,
                                Box::new(PolyOp::Dot),
                            )
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

    use super::*;

    const K: usize = 22;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN);
            let output = VarTensor::new_advice(cs, K, (LEN + 1) * LEN);
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
                                &mut Some(&mut region),
                                &[self.inputs[0].clone()],
                                &mut 0,
                                Box::new(PolyOp::Conv {
                                    kernel: self.inputs[1].clone(),
                                    bias: None,
                                    padding: (1, 1),
                                    stride: (2, 2),
                                }),
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

    use super::*;

    const K: usize = 20;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: Vec<ValTensor<F>>,
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, (LEN + 1) * LEN);
            let b = VarTensor::new_advice(cs, K, (LEN + 1) * LEN);
            let output = VarTensor::new_advice(cs, K, (LEN + 1) * LEN);
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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::SumPool {
                                    padding: (0, 0),
                                    stride: (1, 1),
                                    kernel_shape: (3, 3),
                                }),
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
mod add_w_shape_casting {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Add { a: None }),
                            )
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

        let b = Tensor::from((0..1).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Add { a: None }),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Add { a: None }),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Sub),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Mult { a: None }),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Pow(5)),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(PolyOp::Pack(2, 1)),
                            )
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
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &self.inputs.clone(),
                                &mut 0,
                                Box::new(Rescaled {
                                    inner: Box::new(PolyOp::Sum { axes: vec![0] }),
                                    scale: vec![(0, 5)],
                                }),
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
mod matmul_relu {
    use super::*;

    const K: usize = 18;
    const LEN: usize = 32;
    use crate::circuit::lookup::LookupOp;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    // A columnar ReLu MLP
    #[derive(Clone)]
    struct MyConfig<F: PrimeField + TensorType + PartialOrd> {
        base_config: BaseConfig<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = MyConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

            let mut base_config =
                BaseConfig::configure(cs, &[a, b.clone()], &output, CheckMode::SAFE, 0);
            // sets up a new relu table
            base_config
                .configure_lookup(cs, &b, &output, 16, &LookupOp::ReLU { scale: 1 })
                .unwrap();

            MyConfig { base_config }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.base_config.layout_tables(&mut layouter).unwrap();
            layouter.assign_region(
                || "",
                |mut region| {
                    let op = PolyOp::Matmul { a: None };
                    let mut offset = 0;
                    let output = config
                        .base_config
                        .layout(
                            &mut Some(&mut region),
                            &self.inputs,
                            &mut offset,
                            Box::new(op),
                        )
                        .unwrap();
                    let _output = config
                        .base_config
                        .layout(
                            &mut Some(&mut region),
                            &[output.unwrap()],
                            &mut offset,
                            Box::new(LookupOp::ReLU { scale: 1 }),
                        )
                        .unwrap();
                    Ok(())
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn matmulrelucircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|_| Value::known(F::from(1))));
        a.reshape(&[LEN, LEN]);

        // parameters
        let mut b = Tensor::from((0..LEN).map(|_| Value::known(F::from(1))));
        b.reshape(&[LEN, 1]);

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
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
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp;
    use crate::circuit::Tolerance;

    const RANGE: usize = 8; // 3-bit value
    const K: usize = 8;
    const LEN: usize = 4;

    use super::*;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        input: ValTensor<F>,
        output: ValTensor<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);

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
                                &mut Some(&mut region),
                                &[self.input.clone(), self.output.clone()],
                                &mut 0,
                                Box::new(HybridOp::RangeCheck(Tolerance::Abs { val: RANGE})),
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

#[cfg(test)]
mod rangecheckpercent {
    use crate::{tensor::Tensor, circuit};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp;
    use crate::circuit::Tolerance;

    const RANGE: f32 = 1.0; // 1 percent error tolerance
    const K: usize = 18;
    const LEN: usize = 1;
    const SCALE: usize = i128::pow(2, 7) as usize;

    use super::*;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        input: ValTensor<F>,
        output: ValTensor<F>,
        _marker: PhantomData<F>
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let scale = SCALE.pow(2);
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);
            let mut config =  Self::Config::configure(cs, &[a, b.clone()], &output, CheckMode::SAFE, 0);
            // set up a new GreaterThan and Recip tables
            let nl = &LookupOp::GreaterThan { 
                a: circuit::utils::F32((RANGE * scale as f32)/100.0)
            };
            config
                .configure_lookup(cs, &b, &output, 16, nl)
                .unwrap();
            config
                .configure_lookup(cs, &b, &output, 16, &LookupOp::Recip { scale } )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(
                                &mut Some(&mut region),
                                &[self.output.clone(), self.input.clone()],
                                &mut 0,
                                Box::new(HybridOp::RangeCheck(Tolerance::Percentage { val: RANGE, scale: SCALE })),
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
    fn test_range_check_percent() {

        // Successful cases
        {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(100_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(101_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
                _marker: PhantomData
            };
            let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }
        {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(200_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(199_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
                _marker: PhantomData
            };
            let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
            prover.assert_satisfied();
        }

        // Unsuccessful case
        {
            let inp = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(100_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<Fp>::known(Fp::from(102_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<Fp> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
                _marker: PhantomData
            };
            let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
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

#[cfg(test)]
mod relu {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp as F;

    #[derive(Clone)]
    struct ReLUCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for ReLUCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, 4, 3))
                .collect::<Vec<_>>();

            let nl = LookupOp::ReLU { scale: 1 };

            let mut config = BaseConfig::default();

            config
                .configure_lookup(cs, &advices[0], &advices[1], 2, &nl)
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        config
                            .layout(
                                &mut Some(&mut region),
                                &[self.input.clone()],
                                &mut 0,
                                Box::new(LookupOp::ReLU { scale: 1 }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn relucircuit() {
        let input: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::<F>::known(F::from(1_u64)); 4]), &[4]).unwrap();

        let circuit = ReLUCircuit::<F> {
            input: ValTensor::from(input),
        };

        let prover = MockProver::run(4_u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod softmax {

    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp as F;

    const K: usize = 18;
    const LEN: usize = 3;
    const SCALE: usize = i128::pow(2, 7) as usize;

    #[derive(Clone)]
    struct SoftmaxCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for SoftmaxCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }
        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, LEN);
            let b = VarTensor::new_advice(cs, K, LEN);
            let output = VarTensor::new_advice(cs, K, LEN);
            let mut config =  Self::Config::configure(cs, &[a, b.clone()], &output, CheckMode::SAFE, 0);
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, K, LEN))
                .collect::<Vec<_>>();

            config
                .configure_lookup(cs, &advices[0], &advices[1], 16, &LookupOp::Exp { scales: (SCALE, SCALE) })
                .unwrap();
            config
                .configure_lookup(cs, &advices[0], &advices[1], 16, &LookupOp::Recip { scale: SCALE.pow(2) })
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |mut region| {
                        let mut offset = 0;
                        let _output = config
                            .layout(
                                &mut Some(&mut region),
                                &[self.input.clone()],
                                &mut offset,
                                Box::new(HybridOp::Softmax { scales: ( SCALE, SCALE ) })
                            )
                            .unwrap();
                        Ok(())
                        },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn softmax_circuit() {
        let input = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = SoftmaxCircuit::<F> {
            input: ValTensor::from(input),
            _marker: PhantomData
        };
        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
