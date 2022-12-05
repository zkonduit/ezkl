use ezkl::circuit::eltwise::{DivideBy, EltwiseConfig, ReLU};
use ezkl::circuit::fused::*;
use ezkl::fieldutils::i32_to_felt;
use ezkl::tensor::*;
use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};
use halo2curves::pasta::Fp as F;
use std::marker::PhantomData;

const K: usize = 15;
// A columnar ReLu MLP
#[derive(Clone)]
struct MyConfig<F: FieldExt + TensorType> {
    l0: FusedConfig<F>,
    l1: EltwiseConfig<F, ReLU<F>>,
    l2: FusedConfig<F>,
    l3: EltwiseConfig<F, ReLU<F>>,
    l4: EltwiseConfig<F, DivideBy<F>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    F: FieldExt + TensorType,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const BITS: usize,
> {
    // Given the stateless MyConfig type information, a DNN trace is determined by its input and the parameters of its layers.
    // Computing the trace still requires a forward pass. The intermediate activations are stored only by the layouter.
    input: ValTensor<F>,
    l0_params: [ValTensor<F>; 2],
    l2_params: [ValTensor<F>; 2],
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const LEN: usize, const BITS: usize> Circuit<F>
    for MyCircuit<F, LEN, BITS>
{
    type Config = MyConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let input = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
        let kernel = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true, 512);
        let bias = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
        let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true, 512);
        // tells the config layer to add an affine op to the circuit gate
        let affine_node = FusedNode {
            op: FusedOp::Affine,
            input_order: vec![
                FusedInputType::Input(0),
                FusedInputType::Input(1),
                FusedInputType::Input(2),
            ],
        };

        let l0 = FusedConfig::<F>::configure(
            cs,
            &[input.clone(), kernel.clone(), bias.clone()],
            &output,
            &[affine_node.clone()],
        );

        let l2 = FusedConfig::<F>::configure(
            cs,
            &[input.clone(), kernel, bias],
            &output,
            &[affine_node],
        );

        // sets up a new ReLU table and resuses it for l1 and l3 non linearities
        let [l1, l3]: [EltwiseConfig<F, ReLU<F>>; 2] =
            EltwiseConfig::configure_multiple(cs, &input, &output, Some(&[BITS, 1]));

        // sets up a new Divide by table
        let l4: EltwiseConfig<F, DivideBy<F>> =
            EltwiseConfig::configure(cs, &input, &output, Some(&[BITS, 128]));

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        MyConfig {
            l0,
            l1,
            l2,
            l3,
            l4,
            public_output,
        }
    }

    fn synthesize(
        &self,
        mut config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let x = config.l0.layout(
            &mut layouter,
            &[
                self.input.clone(),
                self.l0_params[0].clone(),
                self.l0_params[1].clone(),
            ],
        );
        let x = config.l1.layout(&mut layouter, x);
        let x = config.l2.layout(
            &mut layouter,
            &[x, self.l2_params[0].clone(), self.l2_params[1].clone()],
        );
        let x = config.l3.layout(&mut layouter, x);
        let x = config.l4.layout(&mut layouter, x);
        match x {
            ValTensor::PrevAssigned { inner: v, dims: _ } => v
                .enum_map(|i, x| {
                    layouter
                        .constrain_instance(x.cell(), config.public_output, i)
                        .unwrap()
                })
                .unwrap(),
            _ => panic!("Should be assigned"),
        };
        Ok(())
    }
}

pub fn runmlp() {
    // parameters
    let l0_kernel: Tensor<Value<F>> = Tensor::<i32>::new(
        Some(&[10, 0, 0, -1, 0, 10, 1, 0, 0, 1, 10, 0, 1, 0, 0, 10]),
        &[4, 4],
    )
    .unwrap()
    .into();
    let l0_bias: Tensor<Value<F>> = Tensor::<i32>::new(Some(&[0, 0, 0, 1]), &[4])
        .unwrap()
        .into();

    let l2_kernel: Tensor<Value<F>> = Tensor::<i32>::new(
        Some(&[0, 3, 10, -1, 0, 10, 1, 0, 0, 1, 0, 12, 1, -2, 32, 0]),
        &[4, 4],
    )
    .unwrap()
    .into();
    // input data, with 1 padding to allow for bias
    let input: Tensor<Value<F>> = Tensor::<i32>::new(Some(&[-30, -21, 11, 40]), &[4])
        .unwrap()
        .into();
    let l2_bias: Tensor<Value<F>> = Tensor::<i32>::new(Some(&[0, 0, 0, 1]), &[4])
        .unwrap()
        .into();

    let circuit = MyCircuit::<F, 4, 14> {
        input: input.into(),
        l0_params: [l0_kernel.into(), l0_bias.into()],
        l2_params: [l2_kernel.into(), l2_bias.into()],
        _marker: PhantomData,
    };

    let public_input: Vec<i32> = unsafe {
        vec![
            (531f32 / 128f32).round().to_int_unchecked::<i32>(),
            (103f32 / 128f32).round().to_int_unchecked::<i32>(),
            (4469f32 / 128f32).round().to_int_unchecked::<i32>(),
            (2849f32 / 128f32).to_int_unchecked::<i32>(),
        ]
    };

    println!("public input {:?}", public_input);

    let prover = MockProver::run(
        K as u32,
        &circuit,
        vec![public_input.iter().map(|x| i32_to_felt::<F>(*x)).collect()],
    )
    .unwrap();
    prover.assert_satisfied();
}

pub fn main() {
    runmlp()
}
