#[cfg(feature = "onnx")]
mod loadonnx_example {
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
    };
    use halo2curves::pasta::Fp as F;
    use halo2deeplearning::fieldutils::i32_to_felt;
    use halo2deeplearning::nn::affine::Affine1dConfig;
    use halo2deeplearning::nn::*;
    use halo2deeplearning::onnx::OnnxCircuit;
    use halo2deeplearning::tensor::{Tensor, TensorType, ValTensor, VarTensor};
    use halo2deeplearning::tensor_ops::eltwise::{EltwiseConfig, ReLu};
    use std::env;
    use std::marker::PhantomData;

    pub fn run() {
        let args: Vec<String> = env::args().collect();
        println!("{:?}", args);
        let k = 15; //2^k rows
                    //        let input = Tensor::<i32>::new(Some(&[-30, -21, 11]), &[3]).unwrap();
        let input = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        let public_input: Vec<i32> = vec![148, 0, 139, 0];
        println!("public input {:?}", public_input);

        let circuit = OnnxCircuit::<F, 14> {
            input,
            _marker: PhantomData,
        };

        let prover = MockProver::run(
            k,
            &circuit,
            vec![public_input.iter().map(|x| i32_to_felt::<F>(*x)).collect()],
        )
        .unwrap();
        prover.assert_satisfied();
    }
}
#[cfg(feature = "onnx")]
pub fn main() {
    crate::loadonnx_example::run()
}
#[cfg(not(feature = "onnx"))]
pub fn main() {}
