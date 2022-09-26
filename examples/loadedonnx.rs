#[cfg(feature = "onnx")]
mod loadonnx_example {
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp as F;
    use halo2deeplearning::fieldutils::i32_to_felt;
    use halo2deeplearning::onnx::OnnxCircuit;
    use halo2deeplearning::tensor::Tensor;
    use std::marker::PhantomData;

    pub fn run() {
        let k = 15; //2^k rows
        let input = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        let public_input: Vec<i32> = vec![148, 0, 139, 0];
        println!("public input (network output) {:?}", public_input);

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
