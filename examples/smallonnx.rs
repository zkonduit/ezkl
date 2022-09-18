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
use halo2deeplearning::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2deeplearning::tensor_ops::eltwise::{DivideBy, EltwiseConfig, ReLu};
use std::{fs, marker::PhantomData, path::Path};
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact};
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::InferenceOp;

// A columnar ReLu MLP
#[derive(Clone)]
struct MyConfig<F: FieldExt + TensorType, const LEN: usize, const BITS: usize> {
    l0: Affine1dConfig<F, LEN, LEN>,
    l1: EltwiseConfig<F, BITS, ReLu<F>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<F: FieldExt, const LEN: usize, const BITS: usize> {
    input: Tensor<i32>,
    l0_params: [Tensor<i32>; 2],
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const LEN: usize, const BITS: usize> Circuit<F>
    for MyCircuit<F, LEN, BITS>
{
    type Config = MyConfig<F, LEN, BITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let advices = VarTensor::Advice(Tensor::from((0..LEN + 3).map(|_| {
            let col = cs.advice_column();
            cs.enable_equality(col);
            col
        })));

        let kernel = advices.get_slice(&[0..LEN]);
        let bias = advices.get_slice(&[LEN + 2..LEN + 3]);

        let l0 = Affine1dConfig::<F, LEN, LEN>::configure(
            cs,
            &[kernel.clone(), bias.clone()],
            advices.get_slice(&[LEN..LEN + 1]),
            advices.get_slice(&[LEN + 1..LEN + 2]),
        );

        let l1: EltwiseConfig<F, BITS, ReLu<F>> =
            EltwiseConfig::configure(cs, advices.get_slice(&[0..LEN]), None);

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
        //println!("{:?}", x);
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

struct OnnxModel {
    model: Graph<InferenceFact, Box<dyn InferenceOp>>,
}

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();
        println!("loaded model {:?}", model);
        OnnxModel { model }
    }

    pub fn get_weights(
        &self,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> {
        //        let model = tract_onnx::onnx().model_for_path("data/ff.onnx").unwrap();

        let node = self.model.node_by_name("fc1.weight").unwrap();
        //        let fact = &self.model.nodes[1].outputs[0].fact;
        let fact = &node.outputs[0].fact;
        let shape = fact.shape.clone().as_concrete_finite().unwrap().unwrap();
        println!("{:?}", shape);
        let nav = fact
            // let nav = self.model.nodes[1].outputs[0]
            //     .fact
            .value
            .concretize()
            .unwrap()
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();
        nav
        //    println!("{:?}", nav);
    }

    pub fn get_biases(
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> {
        let model = tract_onnx::onnx().model_for_path("data/ff.onnx").unwrap();

        println!("model {:?}", model);

        // println!(
        //     "{:?}",
        let nav = model.nodes[1].outputs[0]
            .fact
            .value
            .concretize()
            .unwrap()
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();
        nav
        //    println!("{:?}", nav);
    }
}

pub fn runmlp() {
    let k = 15; //2^k rows
                // parameters

    let onnx_model = OnnxModel::new("data/ff.onnx");
    onnx_model.get_weights();

    let l0_kernel = Tensor::<i32>::new(
        Some(&[10, 0, 0, -1, 0, 10, 1, 0, 0, 1, 10, 0, 1, 0, 0, 10]),
        &[4, 4],
    )
    .unwrap();
    let l0_bias = Tensor::<i32>::new(Some(&[0, 0, 0, 1]), &[1, 4]).unwrap();

    let input = Tensor::<i32>::new(Some(&[-30, -21, 11, 40]), &[1, 4]).unwrap();

    let circuit = MyCircuit::<F, 4, 14> {
        input,
        l0_params: [l0_kernel, l0_bias],
        _marker: PhantomData,
    };

    let public_input: Vec<i32> = vec![0, 0, 89, 371];

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
