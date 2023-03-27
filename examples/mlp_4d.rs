use eq_float::F32;
use ezkl_lib::circuit::base::{BaseConfig as PolyConfig, CheckMode, Op as PolyOp};
use ezkl_lib::circuit::lookup::{Config as LookupConfig, Op as LookupOp};
use ezkl_lib::fieldutils::i32_to_felt;
use ezkl_lib::tensor::*;
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
    layer_config: PolyConfig<F>,
    l1: LookupConfig<F>,
    l3: LookupConfig<F>,
    l4: LookupConfig<F>,
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
        let input = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
        let params = VarTensor::new_advice(cs, K, LEN * LEN, vec![LEN, LEN], true);
        let output = VarTensor::new_advice(cs, K, LEN, vec![LEN], true);
        // tells the config layer to add an affine op to the circuit gate

        let layer_config =
            PolyConfig::<F>::configure(cs, &[input.clone(), params], &output, CheckMode::SAFE, 0);

        // sets up a new ReLU table and resuses it for l1 and l3 non linearities
        let [l1, l3]: [LookupConfig<F>; 2] = LookupConfig::configure_multiple(
            cs,
            &input,
            &output,
            BITS,
            &[LookupOp::ReLU { scale: 1 }],
        )
        .unwrap();

        // sets up a new Divide by table
        let l4 = LookupConfig::configure(
            cs,
            &input,
            &output,
            BITS,
            &[LookupOp::Div {
                denom: F32::from(128.),
            }],
        );

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        MyConfig {
            layer_config,
            l1,
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
        config.l1.layout_table(&mut layouter).unwrap();
        config.l3.layout_table(&mut layouter).unwrap();
        config.l4.layout_table(&mut layouter).unwrap();

        let x = layouter
            .assign_region(
                || "mlp_4d",
                |mut region| {
                    let mut offset = 0;
                    let x = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[
                                self.input.clone(),
                                self.l0_params[0].clone(),
                                self.l0_params[1].clone(),
                            ],
                            &mut offset,
                            PolyOp::Affine,
                        )
                        .unwrap();

                    println!("offset: {}", offset);
                    let mut x = config.l1.layout(&mut region, &x, &mut offset).unwrap();
                    println!("offset: {}", offset);
                    x.reshape(&[x.dims()[0], 1]).unwrap();
                    let x = config
                        .layer_config
                        .layout(
                            &mut region,
                            &[x, self.l2_params[0].clone(), self.l2_params[1].clone()],
                            &mut offset,
                            PolyOp::Affine,
                        )
                        .unwrap();
                    println!("offset: {}", offset);
                    let x = config.l3.layout(&mut region, &x, &mut offset).unwrap();
                    println!("offset: {}", offset);
                    Ok(config.l4.layout(&mut region, &x, &mut offset).unwrap())
                },
            )
            .unwrap();
        match x {
            ValTensor::Value { inner: v, dims: _ } => v
                .enum_map(|i, x| match x {
                    ValType::PrevAssigned(v) => {
                        layouter.constrain_instance(v.cell(), config.public_output, i)
                    }
                    _ => panic!(),
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
    let l0_bias: Tensor<Value<F>> = Tensor::<i32>::new(Some(&[0, 0, 0, 1]), &[4, 1])
        .unwrap()
        .into();

    let l2_kernel: Tensor<Value<F>> = Tensor::<i32>::new(
        Some(&[0, 3, 10, -1, 0, 10, 1, 0, 0, 1, 0, 12, 1, -2, 32, 0]),
        &[4, 4],
    )
    .unwrap()
    .into();
    // input data, with 1 padding to allow for bias
    let input: Tensor<Value<F>> = Tensor::<i32>::new(Some(&[-30, -21, 11, 40]), &[4, 1])
        .unwrap()
        .into();
    let l2_bias: Tensor<Value<F>> = Tensor::<i32>::new(Some(&[0, 0, 0, 1]), &[4, 1])
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
