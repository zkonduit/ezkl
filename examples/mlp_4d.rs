use halo2_proofs::dev::MockProver;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        //create_proof, keygen_pk, keygen_vk, verify_proof, Advice,
        Circuit,
        Column,
        ConstraintSystem,
        Error,
        Instance,
    },
    // poly::{commitment::Params, Rotation},
    // transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use halo2curves::pasta::Fp as F;

use std::marker::PhantomData;
use std::rc::Rc;
//use crate::tensorutils::{dot3, flatten3, flatten4, map2, map3, map3r, map4, map4r};

use halo2deeplearning::fieldutils::i32tofelt;
use halo2deeplearning::nn::affine1d::{Affine1dConfig, RawParameters};
use halo2deeplearning::nn::io::IOConfig;
use halo2deeplearning::tensor::{Tensor, TensorType};
use halo2deeplearning::tensor_ops::eltwise::{DivideBy, EltwiseConfig, EltwiseTable, ReLu};
// A columnar ReLu MLP
#[derive(Clone)]
struct MyConfig<
    F: FieldExt + TensorType,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust
    const BITS: usize,
>
// where
//     [(); LEN + 3]:,
{
    relutable: Rc<EltwiseTable<F, BITS, ReLu<F>>>,
    divtable: Rc<EltwiseTable<F, BITS, DivideBy<F, 128>>>,
    input: IOConfig<F>,
    l0: Affine1dConfig<F, LEN>,
    l1: EltwiseConfig<F, LEN, BITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN>,
    l3: EltwiseConfig<F, LEN, BITS, ReLu<F>>,
    l4: EltwiseConfig<F, LEN, BITS, DivideBy<F, 128>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MyCircuit<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const BITS: usize,
> {
    // Given the stateless MyConfig type information, a DNN trace is determined by its input and the parameters of its layers.
    // Computing the trace still requires a forward pass. The intermediate activations are stored only by the layouter.
    input: Tensor<i32>,
    l0_params: RawParameters<LEN>,
    l2_params: RawParameters<LEN>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType, const LEN: usize, const BITS: usize> Circuit<F>
    for MyCircuit<F, LEN, BITS>
where
    Value<F>: TensorType,
    // where
    //     [(); LEN + 3]:,
{
    type Config = MyConfig<F, LEN, BITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let num_advices = LEN + 3;
        let advices = Tensor::from((0..num_advices).map(|_| {
            let col = cs.advice_column();
            cs.enable_equality(col);
            col
        }));

        let relutable_config = EltwiseTable::<F, BITS, ReLu<F>>::configure(cs);
        let divtable_config = EltwiseTable::<F, BITS, DivideBy<F, 128>>::configure(cs);

        let relutable = Rc::new(relutable_config);
        let divtable = Rc::new(divtable_config);

        let input = IOConfig::<F>::configure(cs, advices.get_slice(&[LEN..LEN + 1]), &[1, LEN]);

        let l0 = Affine1dConfig::<F, LEN>::configure(
            cs,
            advices.get_slice(&[0..LEN]), // wts gets several col, others get a column each
            advices[LEN],                 // input
            advices[LEN + 1],             // output
            advices[LEN + 2],             // bias
        );

        let l1: EltwiseConfig<F, LEN, BITS, ReLu<F>> = EltwiseConfig::configure(
            cs,
            (&advices[..LEN]).clone().try_into().unwrap(),
            relutable.clone(),
        );

        let l2 = Affine1dConfig::<F, LEN>::configure(
            cs,
            advices.get_slice(&[0..LEN]),
            advices[LEN],
            advices[LEN + 1],
            advices[LEN + 2],
        );

        let l3: EltwiseConfig<F, LEN, BITS, ReLu<F>> = EltwiseConfig::configure(
            cs,
            (&advices[..LEN]).clone().try_into().unwrap(),
            relutable.clone(),
        );

        let l4: EltwiseConfig<F, LEN, BITS, DivideBy<F, 128>> = EltwiseConfig::configure(
            cs,
            (&advices[..LEN]).clone().try_into().unwrap(),
            divtable.clone(),
        );

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        MyConfig {
            relutable,
            divtable,
            input,
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
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Layout the reused tables
        config.relutable.layout(&mut layouter)?;
        config.divtable.layout(&mut layouter)?;
        let mut x = self.input.clone();
        x.reshape(&[1, LEN]);
        let x = config.input.layout(&mut layouter, x)?;
        let x = config.l0.layout(
            &mut layouter,
            self.l0_params.weights.clone(),
            self.l0_params.biases.clone(),
            x,
        )?;
        let x = config.l1.layout(&mut layouter, x)?;
        let x = config.l2.layout(
            &mut layouter,
            self.l2_params.weights.clone(),
            self.l2_params.biases.clone(),
            x,
        )?;
        let x = config.l3.layout(&mut layouter, x)?;
        let x = config.l4.layout(&mut layouter, x)?;
        x.enum_map(|i, x| {
            layouter
                .constrain_instance(x.cell(), config.public_output, i)
                .unwrap()
        });
        Ok(())
    }
}

pub fn runmlp() {
    let k = 15; //2^k rows
                // parameters
    let l0weights = Tensor::<i32>::new(
        Some(&[10, 0, 0, -1, 0, 10, 1, 0, 0, 1, 10, 0, 1, 0, 0, 10]),
        &[4, 4],
    )
    .unwrap();
    let l0biases = Tensor::<i32>::new(Some(&[0, 0, 0, 1]), &[4]).unwrap();
    let l0_params = RawParameters {
        weights: l0weights,
        biases: l0biases,
    };
    let l2weights = Tensor::<i32>::new(
        Some(&[0, 3, 10, -1, 0, 10, 1, 0, 0, 1, 0, 12, 1, -2, 32, 0]),
        &[4, 4],
    )
    .unwrap();
    let l2biases = Tensor::<i32>::new(Some(&[12, 14, 17, 1]), &[4]).unwrap();
    let l2_params = RawParameters {
        weights: l2weights,
        biases: l2biases,
    };
    // input data
    let input = Tensor::<i32>::new(Some(&[-30, -21, 11, 40]), &[4]).unwrap();

    let circuit = MyCircuit::<F, 4, 14> {
        input,
        l0_params,
        l2_params,
        _marker: PhantomData,
    };

    let public_input: Vec<i32> = unsafe {
        vec![
            (531f32 / 128f32).round().to_int_unchecked::<i32>().into(),
            (103f32 / 128f32).round().to_int_unchecked::<i32>().into(),
            (4469f32 / 128f32).round().to_int_unchecked::<i32>().into(),
            (2849f32 / 128f32).to_int_unchecked::<i32>().into(),
        ]
    };

    println!("public input {:?}", public_input);

    let prover = MockProver::run(
        k,
        &circuit,
        vec![public_input
            .iter()
            .map(|x| i32tofelt::<F>(*x).into())
            .collect()],
        //            vec![vec![(4).into(), (1).into(), (35).into(), (22).into()]],
    )
    .unwrap();
    prover.assert_satisfied();
}

pub fn main() {
    runmlp()
}
