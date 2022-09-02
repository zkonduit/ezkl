use crate::inputlayer::InputConfig;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{layouter, AssignedCell, Layouter, Region, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Constraints, Error, Expression, Instance, Selector,
    },
};
use std::cmp::max;
use std::marker::PhantomData;

// enum Input {
//     Raw(Vec<i32>),
//     Assigned(Vec<AssignedCell<Assigned<F>, F>>),
// }

trait AcceptablePut {}
impl AcceptablePut for Vec<i32> {}
impl<F: FieldExt> AcceptablePut for Vec<AssignedCell<Assigned<F>, F>> {}
//various Into / From to convert layers ?

// Configs impl this
pub trait LayeredConfig<F: FieldExt> {
    type Input; //: AcceptablePut;
                //    type Output; //usually Vec<AssignedCell<Assigned<F>, F>> or i32 or () at the end
    type Parameters; // the type of the parameters, not their data
    fn num_advices() -> usize;
    fn configure(cs: &mut ConstraintSystem<F>, advices: Vec<Column<Advice>>) -> Self;
    fn layout_forward(
        layouter: &mut impl Layouter<F>,
        parameters: Self::Parameters,
        input: Self::Input,
    ) -> Vec<AssignedCell<Assigned<F>, F>>;
}

pub struct DNNConfig<F: FieldExt, const INPUTLEN: usize, const OUTPUTLEN: usize, Layers> {
    input: InputConfig<F, INPUTLEN>,
    layers: Layers,
    public_output: Column<Instance>,
}

impl<
        F: FieldExt,
        const INPUTLEN: usize,
        const OUTPUTLEN: usize,
        L0: LayeredConfig<F>,
        L1: LayeredConfig<F>,
    > LayeredConfig<F> for DNNConfig<F, INPUTLEN, OUTPUTLEN, (L0, L1)>
{
    type Input = Vec<i32>;
    type Parameters = (L0::Parameters, L1::Parameters);
    fn num_advices() -> usize {
        max(L0::num_advices(), L1::num_advices())
    }

    fn configure(
        cs: &mut ConstraintSystem<F>,
        advices: Vec<Column<Advice>>,
    ) -> DNNConfig<F, INPUTLEN, OUTPUTLEN, (L0, L1)> {
        let num_advices = Self::num_advices();
        let advices = (0..num_advices)
            .map(|_| {
                let col = cs.advice_column();
                cs.enable_equality(col);
                col
            })
            .collect::<Vec<_>>();

        // Each layer is responsible for choosing a subset of the advices to use
        let input = InputConfig::<F, INPUTLEN>::configure(cs, advices[0].clone());
        let l0 = L0::configure(cs, advices.clone());
        let l1 = L1::configure(cs, advices.clone());
        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);

        DNNConfig {
            input,
            layers: (l0, l1),
            public_output,
        }

        //	Self
    }

    // the input and parameters are all that is stored in a circuit's struct, they can also be passed directly.
    fn layout_forward(
        &self,
        layouter: &mut impl Layouter<F>,
        parameters: Self::Parameters,
        input: Vec<i32>,
    ) -> Vec<AssignedCell<Assigned<F>, F>> {
        let x = InputConfig::<F, INPUTLEN>::layout_forward(&mut layouter, input)?;
        let x = L0::layout_forward(&mut layouter, parameters.0, x)?;
        let x = L1::layout_forward(&mut layouter, parameters.1, x)?;
        for i in 0..OUTPUTLEN {
            layouter.constrain_instance(x[i].cell(), self.public_output, i)?;
        }
        Ok(Vec::new())
        //public
    }
}

// should be traits
pub struct DNNParams<T> {
    _marker: PhantomData<T>,
}

pub struct DNNCircuit<const INPUTLEN: usize, const OUTPUTLEN: usize, Layers> {
    parameters: DNNParams<Layers>,
    input: [i32; INPUTLEN],
    output: [i32; OUTPUTLEN],
    //    _marker: PhantomData<Layers>,
}

impl DNNCircuit {
    fn layout_forward(
        &self,
        layouter: &mut impl Layouter<F>,
        parameters: Self::Parameters,
        input: Vec<i32>,
    ) -> Vec<AssignedCell<Assigned<F>, F>>;
}

// // Each Li is a LayerConfig
// impl<F: FieldExt, L0, L1> LayeredCircuit<F> for (L0, L1) {
//     // compute the number of advice columns needed for the DNN
//     fn num_advices() -> usize {
// 	0
//     }

//     fn configure(cs: &mut ConstraintSystem<F>, advices: Vec<Column<Advice>>) -> Self {
// 	Self::l0_config()::configure()
//     }

//     fn l0_config() -> L0;
//}

pub trait TwoLayer {
    type L0;
    type L1;
}

pub trait ThreeLayer {
    type L0;
    type L1;
    type L2;
}

pub trait FourLayer {
    type L0;
    type L1;
    type L2;
    type L3;
}

pub trait FiveLayer {
    type L0;
    type L1;
    type L2;
    type L3;
    type L4;
}

pub trait SixLayer {
    type L0;
    type L1;
    type L2;
    type L3;
    type L4;
    type L5;
}

pub trait SevenLayer {
    type L0;
    type L1;
    type L2;
    type L3;
    type L4;
    type L5;
    type L6;
}
