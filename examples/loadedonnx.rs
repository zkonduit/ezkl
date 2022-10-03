#[cfg(feature = "onnx")]
mod loadonnx_example {
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp as F;
    use halo2deeplearning::fieldutils::i32_to_felt;
    use halo2deeplearning::onnx::OnnxCircuit;
    use halo2deeplearning::tensor::Tensor;
    use std::env;
    use std::marker::PhantomData;

    pub fn run() {
        let k = 16; //2^k rows
        let input = Tensor::<i32>::new(Some(&[100, 20, 30]), &[3]).unwrap();
        let args: Vec<String> = env::args().collect();
        let filename = args[1].clone().split('/').last().unwrap().to_string();

        let public_input = match filename.as_str() {
            "ff.onnx" => vec![120, 0, 0, 0],
            "three.onnx" => vec![19, 42],
            "3lffOOR.onnx" => vec![0, 11, 28, 50],
            _ => todo!(),
        };

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
