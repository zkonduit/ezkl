use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use halo2curves::pasta::Fp as F;
use halo2deeplearning::fieldutils::i32_to_felt;
use halo2deeplearning::nn::affine::Affine1dConfig;
use halo2deeplearning::nn::*;
use halo2deeplearning::onnx::OnnxModel;
use halo2deeplearning::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2deeplearning::tensor_ops::eltwise::{DivideBy, EltwiseConfig, ReLu};
use std::marker::PhantomData;

#[derive(Clone)]
struct MyConfig<F: FieldExt + TensorType, const IN: usize, const OUT: usize, const BITS: usize> {
    l0: Affine1dConfig<F, IN, OUT>,
    l1: EltwiseConfig<F, BITS, ReLu<F>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<F: FieldExt, const IN: usize, const OUT: usize, const BITS: usize> {
    input: Tensor<i32>,
    l0_params: [Tensor<i32>; 2],
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const IN: usize, const OUT: usize, const BITS: usize> Circuit<F>
    for MyCircuit<F, IN, OUT, BITS>
{
    type Config = MyConfig<F, IN, OUT, BITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let advices = VarTensor::Advice(Tensor::from((0..OUT + 3).map(|_| {
            let col = cs.advice_column();
            cs.enable_equality(col);
            col
        })));

        let kernel = advices.get_slice(&[0..OUT]);
        let bias = advices.get_slice(&[OUT + 2..OUT + 3]);

        let l0 = Affine1dConfig::<F, IN, OUT>::configure(
            cs,
            &[kernel.clone(), bias.clone()],
            advices.get_slice(&[OUT..OUT + 1]),
            advices.get_slice(&[OUT + 1..OUT + 2]),
        );

        let l1: EltwiseConfig<F, BITS, ReLu<F>> =
            EltwiseConfig::configure(cs, advices.get_slice(&[0..OUT]), None);

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        MyConfig {
            l0,
            l1,
            public_output,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let x = self.input.clone();
        let x = config.l0.layout(
            &mut layouter,
            ValTensor::Value(x.into()),
            &self
                .l0_params
                .iter()
                .map(|a| ValTensor::Value(a.clone().into()))
                .collect::<Vec<ValTensor<F>>>(),
        );
        let x = config.l1.layout(&mut layouter, x);
        println!("{:?}", x);
        match x {
            ValTensor::PrevAssigned(v) => v.enum_map(|i, x| {
                layouter
                    .constrain_instance(x.cell(), config.public_output, i)
                    .unwrap()
            }),
            _ => panic!("Should be assigned"),
        };
        Ok(())
    }
}

pub fn runmlp() {
    let k = 15; //2^k rows
                // parameters

    let onnx_model = OnnxModel::new("onnx_models/ff.onnx");

    let l0_kernel = onnx_model.get_tensor_by_node_name("fc1.weight", 0f32, 256f32);
    let l0_bias = onnx_model.get_tensor_by_node_name("fc1.bias", 0f32, 256f32);

    let input = Tensor::<i32>::new(Some(&[-30, -21, 11]), &[1, 3]).unwrap();

    let circuit = MyCircuit::<F, 3, 4, 14> {
        input,
        l0_params: [l0_kernel, l0_bias],
        _marker: PhantomData,
    };

    let public_input: Vec<i32> = vec![0, 0, 0, 1653];

    println!("public input {:?}", public_input);

    let prover = MockProver::run(
        k,
        &circuit,
        vec![public_input
            .iter()
            .map(|x| i32_to_felt::<F>(*x).into())
            .collect()],
        //            vec![vec![(4).into(), (1).into(), (35).into(), (22).into()]],
    )
    .unwrap();
    prover.assert_satisfied();
}

pub fn main() {
    runmlp()
}
