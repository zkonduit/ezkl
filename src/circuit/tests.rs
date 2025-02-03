use crate::circuit::ops::poly::PolyOp;
use crate::circuit::*;
use crate::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::Fr as F;
use halo2curves::ff::{Field, PrimeField};
#[cfg(not(any(
    all(target_arch = "wasm32", target_os = "unknown"),
    not(feature = "ezkl")
)))]
use ops::lookup::LookupOp;
use ops::region::RegionCtx;
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

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
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
    fn matmulcircuit() {
        // parameters
        let mut a =
            Tensor::from((0..(LEN + 1) * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN + 1]).unwrap();

        let mut w = Tensor::from((0..LEN + 1).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN + 1, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
mod matmul_col_overflow_double_col {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
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
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

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

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
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
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod matmul_col_ultra_overflow_double_col {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 10;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
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
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            crate::Commitments::KZG,
            crate::pfsys::TranscriptType::EVM,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod matmul_col_ultra_overflow {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 10;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
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
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            MatmulCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            crate::Commitments::KZG,
            crate::pfsys::TranscriptType::EVM,
            None,
            None,
        );

        assert!(prover.is_ok());
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
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
mod dot_col_overflow_triple_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            // used for constants in the padding
            let _fixed = cs.fixed_column();
            cs.enable_constant(_fixed);

            let a = VarTensor::new_advice(cs, K, 3, LEN);
            let b = VarTensor::new_advice(cs, K, 3, LEN);
            let output = VarTensor::new_advice(cs, K, 3, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 3, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
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
mod sum_col_overflow_double_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
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
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
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

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let b = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let output = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);

            // column for constants
            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &self.inputs,
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
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

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::random(OsRng)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let image = ValTensor::try_from(image).unwrap();

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|_| F::random(OsRng)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let kernels = ValTensor::try_from(kernels).unwrap();
        let mut bias = Tensor::from((0..{ out_channels }).map(|_| F::random(OsRng)));
        bias.set_visibility(&crate::graph::Visibility::Private);

        let bias = ValTensor::try_from(bias).unwrap();

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels, bias].to_vec(),
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

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let image = ValTensor::try_from(image).unwrap();
        let kernels = ValTensor::try_from(kernels).unwrap();

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod conv_col_ultra_overflow {

    use halo2_proofs::poly::{
        kzg::strategy::SingleStrategy,
        kzg::{
            commitment::KZGCommitmentScheme,
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
        },
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 6;
    const LEN: usize = 10;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        image: ValTensor<F>,
        kernel: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * LEN);
            let _constant = VarTensor::constant_cols(cs, K, LEN * LEN * LEN * LEN, false);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &[self.image.clone(), self.kernel.clone()],
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
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
    #[ignore]
    fn conv_circuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = LEN;
        let image_width = LEN;
        let in_channels = 3;
        let out_channels = 2;

        // get some logs fam
        crate::logger::init_logger();
        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            image: ValTensor::try_from(image).unwrap(),
            kernel: ValTensor::try_from(kernels).unwrap(),
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            ConvCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            crate::Commitments::KZG,
            crate::pfsys::TranscriptType::EVM,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}

#[cfg(test)]
// not wasm 32 unknown
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod conv_relu_col_ultra_overflow {

    use halo2_proofs::poly::kzg::{
        commitment::KZGCommitmentScheme,
        multiopen::{ProverSHPLONK, VerifierSHPLONK},
        strategy::SingleStrategy,
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    use super::*;

    const K: usize = 8;
    const LEN: usize = 15;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        image: ValTensor<F>,
        kernel: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * 4);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * 4);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN * 4);
            let mut base_config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);
            // sets up a new relu table

            base_config
                .configure_range_check(cs, &a, &b, (-1, 1), K)
                .unwrap();

            base_config
                .configure_range_check(cs, &a, &b, (0, 1), K)
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            base_config.clone()
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 2, 2);
                        let output = config
                            .layout(
                                &mut region,
                                &[self.image.clone(), self.kernel.clone()],
                                Box::new(PolyOp::Conv {
                                    padding: vec![(1, 1); 2],
                                    stride: vec![2; 2],
                                    group: 1,
                                }),
                            )
                            .map_err(|_| Error::Synthesis);
                        let _output = config
                            .layout(
                                &mut region,
                                &[output.unwrap().unwrap()],
                                Box::new(PolyOp::LeakyReLU {
                                    slope: 0.0.into(),
                                    scale: 1,
                                }),
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
    #[ignore]
    fn conv_relu_circuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = LEN;
        let image_width = LEN;
        let in_channels = 3;
        let out_channels = 2;

        // get some logs fam
        crate::logger::init_logger();
        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::from(0)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width }).map(|_| F::from(0)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            image: ValTensor::try_from(image).unwrap(),
            kernel: ValTensor::try_from(kernels).unwrap(),
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            ConvCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            CheckMode::SAFE,
            crate::Commitments::KZG,
            crate::pfsys::TranscriptType::EVM,
            // use safe mode to verify that the proof is correct
            None,
            None,
        );

        assert!(prover.is_ok());
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Add))
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

        let b = Tensor::from((0..1).map(|i| Value::known(F::from(i + 1))));

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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Add))
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
mod dynamic_lookup {
    use super::*;

    const K: usize = 6;
    const LEN: usize = 4;
    const NUM_LOOP: usize = 5;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        tables: [[ValTensor<F>; 2]; NUM_LOOP],
        lookups: [[ValTensor<F>; 2]; NUM_LOOP],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 2, LEN);
            let b = VarTensor::new_advice(cs, K, 2, LEN);
            let c: VarTensor = VarTensor::new_advice(cs, K, 2, LEN);

            let d = VarTensor::new_advice(cs, K, 1, LEN);
            let e = VarTensor::new_advice(cs, K, 1, LEN);
            let f: VarTensor = VarTensor::new_advice(cs, K, 1, LEN);

            let _constant = VarTensor::constant_cols(cs, K, LEN * NUM_LOOP, false);

            let mut config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &c, CheckMode::SAFE);
            config
                .configure_dynamic_lookup(
                    cs,
                    &[a.clone(), b.clone(), c.clone()],
                    &[d.clone(), e.clone(), f.clone()],
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        for i in 0..NUM_LOOP {
                            layouts::dynamic_lookup(
                                &config,
                                &mut region,
                                &self.lookups[i],
                                &self.tables[i],
                            )
                            .map_err(|_| Error::Synthesis)?;
                        }
                        assert_eq!(
                            region.dynamic_lookup_col_coord(),
                            NUM_LOOP * self.tables[0][0].len()
                        );
                        assert_eq!(region.dynamic_lookup_index(), NUM_LOOP);

                        Ok(())
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dynamiclookupcircuit() {
        // parameters
        let tables = (0..NUM_LOOP)
            .map(|loop_idx| {
                [
                    ValTensor::from(Tensor::from(
                        (0..LEN).map(|i| Value::known(F::from((i * loop_idx) as u64 + 1))),
                    )),
                    ValTensor::from(Tensor::from(
                        (0..LEN).map(|i| Value::known(F::from((loop_idx * i * i) as u64 + 1))),
                    )),
                ]
            })
            .collect::<Vec<_>>();

        let lookups = (0..NUM_LOOP)
            .map(|loop_idx| {
                [
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((i * loop_idx) as u64 + 1))),
                    )),
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((loop_idx * i * i) as u64 + 1))),
                    )),
                ]
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            tables: tables.clone().try_into().unwrap(),
            lookups: lookups.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let lookups = (0..NUM_LOOP)
            .map(|loop_idx| {
                let prev_idx = if loop_idx == 0 {
                    NUM_LOOP - 1
                } else {
                    loop_idx - 1
                };
                [
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((i * prev_idx) as u64 + 1))),
                    )),
                    ValTensor::from(Tensor::from(
                        (0..3).map(|i| Value::known(F::from((prev_idx * i * i) as u64 + 1))),
                    )),
                ]
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            tables: tables.try_into().unwrap(),
            lookups: lookups.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}

#[cfg(test)]
mod shuffle {
    use super::*;

    const K: usize = 6;
    const LEN: usize = 4;
    const NUM_LOOP: usize = 5;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [[ValTensor<F>; 1]; NUM_LOOP],
        references: [[ValTensor<F>; 1]; NUM_LOOP],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 2, LEN);
            let b = VarTensor::new_advice(cs, K, 2, LEN);
            let c: VarTensor = VarTensor::new_advice(cs, K, 2, LEN);

            let d = VarTensor::new_advice(cs, K, 1, LEN);
            let e = VarTensor::new_advice(cs, K, 1, LEN);
            let f: VarTensor = VarTensor::new_advice(cs, K, 1, LEN);

            let _constant = VarTensor::constant_cols(cs, K, LEN * NUM_LOOP, false);

            let mut config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &c, CheckMode::SAFE);
            config
                .configure_shuffles(
                    cs,
                    &[a.clone(), b.clone(), c.clone()],
                    &[d.clone(), e.clone(), f.clone()],
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        for i in 0..NUM_LOOP {
                            layouts::shuffles(
                                &config,
                                &mut region,
                                &self.inputs[i],
                                &self.references[i],
                                layouts::SortCollisionMode::Unsorted,
                            )
                            .map_err(|_| Error::Synthesis)?;
                        }
                        assert_eq!(
                            region.shuffle_col_coord(),
                            NUM_LOOP * self.references[0][0].len()
                        );
                        assert_eq!(region.shuffle_index(), NUM_LOOP);

                        Ok(())
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn shufflecircuit() {
        // parameters
        let references = (0..NUM_LOOP)
            .map(|loop_idx| {
                [ValTensor::from(Tensor::from((0..LEN).map(|i| {
                    Value::known(F::from((i * loop_idx) as u64 + 1))
                })))]
            })
            .collect::<Vec<_>>();

        let inputs = (0..NUM_LOOP)
            .map(|loop_idx| {
                [ValTensor::from(Tensor::from((0..LEN).rev().map(|i| {
                    Value::known(F::from((i * loop_idx) as u64 + 1))
                })))]
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            references: references.clone().try_into().unwrap(),
            inputs: inputs.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        let inputs = (0..NUM_LOOP)
            .map(|loop_idx| {
                let prev_idx = if loop_idx == 0 {
                    NUM_LOOP - 1
                } else {
                    loop_idx - 1
                };
                [ValTensor::from(Tensor::from((0..LEN).rev().map(|i| {
                    Value::known(F::from((i * prev_idx) as u64 + 1))
                })))]
            })
            .collect::<Vec<_>>();

        let circuit = MyCircuit::<F> {
            references: references.try_into().unwrap(),
            inputs: inputs.try_into().unwrap(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Add))
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
mod add_with_overflow_and_poseidon {
    use std::collections::HashMap;

    use halo2curves::bn256::Fr;

    use crate::circuit::modules::{
        poseidon::{
            spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
            PoseidonChip, PoseidonConfig,
        },
        Module, ModulePlanner,
    };

    use super::*;

    const K: usize = 15;
    const LEN: usize = 50;
    const WIDTH: usize = POSEIDON_WIDTH;
    const RATE: usize = POSEIDON_RATE;

    #[derive(Debug, Clone)]
    struct MyCircuitConfig {
        base: BaseConfig<Fr>,
        poseidon: PoseidonConfig<WIDTH, RATE>,
    }

    #[derive(Clone)]
    struct MyCircuit {
        inputs: [ValTensor<Fr>; 2],
    }

    impl Circuit<Fr> for MyCircuit {
        type Config = MyCircuitConfig;
        type FloorPlanner = ModulePlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            let base = BaseConfig::configure(cs, &[a, b], &output, CheckMode::SAFE);
            VarTensor::constant_cols(cs, K, 2, false);

            let poseidon = PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::configure(cs, ());

            MyCircuitConfig { base, poseidon }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let poseidon_chip: PoseidonChip<PoseidonSpec, WIDTH, RATE, WIDTH> =
                PoseidonChip::new(config.poseidon.clone());

            let assigned_inputs_a =
                poseidon_chip.layout(&mut layouter, &self.inputs[0..1], 0, &mut HashMap::new())?;
            let assigned_inputs_b =
                poseidon_chip.layout(&mut layouter, &self.inputs[1..2], 1, &mut HashMap::new())?;

            layouter.assign_region(|| "_new_module", |_| Ok(()))?;

            let inputs = vec![assigned_inputs_a, assigned_inputs_b];

            layouter.assign_region(
                || "model",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                    config
                        .base
                        .layout(&mut region, &inputs, Box::new(PolyOp::Add))
                        .map_err(|_| Error::Synthesis)
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        let a = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let b = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let commitment_a =
            PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(a.clone()).unwrap()[0][0];

        let commitment_b =
            PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(b.clone()).unwrap()[0][0];

        // parameters
        let a = Tensor::from(a.into_iter().map(Value::known));
        let b = Tensor::from(b.into_iter().map(Value::known));
        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
        };

        let prover =
            MockProver::run(K as u32, &circuit, vec![vec![commitment_a, commitment_b]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn addcircuit_bad_hashes() {
        let a = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let b = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let commitment_a = PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(a.clone())
            .unwrap()[0][0]
            + Fr::one();

        let commitment_b = PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(b.clone())
            .unwrap()[0][0]
            + Fr::one();

        // parameters
        let a = Tensor::from(a.into_iter().map(Value::known));
        let b = Tensor::from(b.into_iter().map(Value::known));
        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
        };

        let prover =
            MockProver::run(K as u32, &circuit, vec![vec![commitment_a, commitment_b]]).unwrap();
        assert!(prover.verify().is_err());
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Sub))
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Mult))
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

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Pow(5)))
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
mod matmul_relu {
    use super::*;

    const K: usize = 18;
    const LEN: usize = 32;

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

    impl Circuit<F> for MyCircuit<F> {
        type Config = MyConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            let mut base_config =
                BaseConfig::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);

            base_config
                .configure_range_check(cs, &a, &b, (-1, 1), K)
                .unwrap();

            base_config
                .configure_range_check(cs, &a, &b, (0, 1023), K)
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K, 8, false);

            MyConfig { base_config }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config
                .base_config
                .layout_range_checks(&mut layouter)
                .unwrap();
            layouter.assign_region(
                || "",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1, 1024, 2);
                    let op = PolyOp::Einsum {
                        equation: "ij,jk->ik".to_string(),
                    };
                    let output = config
                        .base_config
                        .layout(&mut region, &self.inputs, Box::new(op))
                        .unwrap();
                    let _output = config
                        .base_config
                        .layout(
                            &mut region,
                            &[output.unwrap()],
                            Box::new(PolyOp::LeakyReLU {
                                slope: 0.0.into(),
                                scale: 1,
                            }),
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
        a.reshape(&[LEN, LEN]).unwrap();

        // parameters
        let mut b = Tensor::from((0..LEN).map(|_| Value::known(F::from(1))));
        b.reshape(&[LEN, 1]).unwrap();

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
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

    const K: u32 = 8;

    #[derive(Clone)]
    struct ReLUCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl Circuit<F> for ReLUCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, 8, 1, 3))
                .collect::<Vec<_>>();

            let mut config = BaseConfig::configure(
                cs,
                &[advices[0].clone(), advices[1].clone()],
                &advices[2],
                CheckMode::SAFE,
            );

            config
                .configure_range_check(cs, &advices[0], &advices[1], (-1, 1), K as usize)
                .unwrap();

            config
                .configure_range_check(cs, &advices[0], &advices[1], (0, 1), K as usize)
                .unwrap();

            let _constant = VarTensor::constant_cols(cs, K as usize, 8, false);

            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            config.layout_range_checks(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 2, 2);
                        Ok(config
                            .layout(
                                &mut region,
                                &[self.input.clone()],
                                Box::new(PolyOp::LeakyReLU {
                                    slope: 0.0.into(),
                                    scale: 1,
                                }),
                            )
                            .unwrap())
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

        let prover = MockProver::run(K, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

#[cfg(test)]
#[cfg(all(
    feature = "ezkl",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
mod lookup_ultra_overflow {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem, Error},
        poly::kzg::{
            commitment::KZGCommitmentScheme,
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
    };
    use snark_verifier::system::halo2::transcript::evm::EvmTranscript;

    #[derive(Clone)]
    struct SigmoidCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl Circuit<F> for SigmoidCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, 4, 1, 3))
                .collect::<Vec<_>>();

            let nl = LookupOp::Sigmoid { scale: 1.0.into() };

            let mut config = BaseConfig::default();

            config
                .configure_lookup(
                    cs,
                    &advices[0],
                    &advices[1],
                    &advices[2],
                    (-1024, 1024),
                    4,
                    &nl,
                )
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
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1, 128, 2);
                        config
                            .layout(
                                &mut region,
                                &[self.input.clone()],
                                Box::new(LookupOp::Sigmoid { scale: 1.0.into() }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn sigmoidcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let a = Tensor::from((0..4).map(|i| Value::known(F::from(i + 1))));

        let circuit = SigmoidCircuit::<F> {
            input: ValTensor::from(a),
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(4_u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            SigmoidCircuit<F>,
        >(&circuit, &params, true)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit::<
            KZGCommitmentScheme<_>,
            _,
            ProverSHPLONK<_>,
            VerifierSHPLONK<_>,
            SingleStrategy<_>,
            _,
            EvmTranscript<_, _, _, _>,
            EvmTranscript<_, _, _, _>,
        >(
            circuit.clone(),
            vec![],
            &params,
            &pk,
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            crate::Commitments::KZG,
            crate::pfsys::TranscriptType::EVM,
            None,
            None,
        );

        assert!(prover.is_ok());
    }
}
