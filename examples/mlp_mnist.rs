use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner},
    plonk::{Circuit, Column, ConstraintSystem, Error, Instance},
};

use halo2deeplearning::{
    affine1d::{Affine1dConfig, RawParameters},
    eltwise::{DivideBy, NonlinConfig1d, ReLu},
    inputlayer::InputConfig,
};
use std::marker::PhantomData;

// A columnar ReLu MLP consisting of a stateless MLPConfig, and an MLPCircuit with parameters and input.

#[derive(Clone)]
struct MLPConfig<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> {
    input: InputConfig<F, LEN>,
    l0: Affine1dConfig<F, LEN, LEN>,
    l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN, LEN>,
    l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>>,
    public_output: Column<Instance>,
}

#[derive(Clone)]
struct MLPCircuit<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> {
    // Given the stateless MLPConfig type information, a DNN trace is determined by its input and the parameters of its layers.
    // Computing the trace still requires a forward pass. The intermediate activations are stored only by the layouter.
    input: Vec<i32>,
    l0_params: RawParameters<LEN, LEN>,
    l2_params: RawParameters<LEN, LEN>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> Circuit<F>
    for MLPCircuit<F, LEN, INBITS, OUTBITS>

{
    type Config = MLPConfig<F, LEN, INBITS, OUTBITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let num_advices = LEN + 3;
        let advices = (0..num_advices)
            .map(|_| {
                let col = cs.advice_column();
                cs.enable_equality(col);
                col
            })
            .collect::<Vec<_>>();

        let input = InputConfig::<F, LEN>::configure(cs, advices[LEN]);

        let l0 = Affine1dConfig::<F, LEN, LEN>::configure(
            cs,
            (&advices[..LEN]).try_into().unwrap(), // wts gets several col, others get a column each
            advices[LEN],                          // input
            advices[LEN + 1],                      // output
            advices[LEN + 2],                      // bias
        );

        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).try_into().unwrap());

        let l2 = Affine1dConfig::<F, LEN, LEN>::configure(
            cs,
            (&advices[..LEN]).try_into().unwrap(),
            advices[LEN],
            advices[LEN + 1],
            advices[LEN + 2],
        );

        let l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).try_into().unwrap());

        let l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>> =
            NonlinConfig1d::configure(cs, (&advices[..LEN]).try_into().unwrap());

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        MLPConfig {
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
        let x = config.input.layout(&mut layouter, self.input.clone())?;
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
        for i in 0..LEN {
            layouter.constrain_instance(x[i].cell(), config.public_output, i)?;
        }
        Ok(())
    }
}

pub fn mlprun() {
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp as F;
    use halo2deeplearning::fieldutils::i32tofelt;

    let k = 15; //2^k rows
                // parameters
    let l0weights: Vec<Vec<i32>> = vec![
        vec![10, 0, 0, -1],
        vec![0, 10, 1, 0],
        vec![0, 1, 10, 0],
        vec![1, 0, 0, 10],
    ];
    let l0biases: Vec<i32> = vec![0, 0, 0, 1];
    let l0_params = RawParameters {
        weights: l0weights,
        biases: l0biases,
    };
    let l2weights: Vec<Vec<i32>> = vec![
        vec![0, 3, 10, -1],
        vec![0, 10, 1, 0],
        vec![0, 1, 0, 12],
        vec![1, -2, 32, 0],
    ];
    let l2biases: Vec<i32> = vec![12, 14, 17, 1];
    let l2_params = RawParameters {
        weights: l2weights,
        biases: l2biases,
    };
    // input data
    let input: Vec<i32> = vec![-30, -21, 11, 40];

    let circuit = MLPCircuit::<F, 4, 14, 14> {
        input,
        l0_params,
        l2_params,
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
        k,
        &circuit,
        vec![public_input.iter().map(|x| i32tofelt::<F>(*x)).collect()],
        //            vec![vec![(4).into(), (1).into(), (35).into(), (22).into()]],
    )
    .unwrap();
    prover.assert_satisfied();
}

fn main() {
    mlprun()
}
