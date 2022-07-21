use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Constraints, Error, Expression, Instance, Selector, SingleVerifier,
        TableColumn,
    },
    poly::{commitment::Params, Rotation},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{pallas, vesta};
use rand::rngs::OsRng;
use std::marker::PhantomData;

use crate::fieldutils::i32tofelt;
//use crate::tensorutils::{dot3, flatten3, flatten4, map2, map3, map3r, map4, map4r};

use crate::affine1d::{Affine1d, Affine1dConfig};
use crate::nonlin1d::{DivideBy, Nonlin1d, NonlinConfig1d, ReLu};

// A columnar ReLu MLP
#[derive(Clone)]
struct MyConfig<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust
    const INBITS: usize,
    const OUTBITS: usize,
> {
    l0: Affine1dConfig<F, LEN, LEN>,
    l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN, LEN>,
    l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>>,
    public_output: Column<Instance>,
}

struct MyCircuit<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const INBITS: usize,
    const OUTBITS: usize,
> {
    // circuit holds Values
    l0: Affine1d<F, Value<Assigned<F>>, LEN, LEN>,
    l1: Nonlin1d<F, Value<Assigned<F>>, LEN, ReLu<F>>,
    l2: Affine1d<F, Value<Assigned<F>>, LEN, LEN>,
    l3: Nonlin1d<F, Value<Assigned<F>>, LEN, ReLu<F>>,
    l4: Nonlin1d<F, Value<Assigned<F>>, LEN, DivideBy<F, 128>>,
}

impl<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> Circuit<F>
    for MyCircuit<F, LEN, INBITS, OUTBITS>
{
    type Config = MyConfig<F, LEN, INBITS, OUTBITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            l0: Affine1d::<F, Value<Assigned<F>>, LEN, LEN>::without_witnesses(),
            l1: Nonlin1d::<F, Value<Assigned<F>>, LEN, ReLu<F>>::without_witnesses(),
            l2: Affine1d::<F, Value<Assigned<F>>, LEN, LEN>::without_witnesses(),
            l3: Nonlin1d::<F, Value<Assigned<F>>, LEN, ReLu<F>>::without_witnesses(),
            l4: Nonlin1d::<F, Value<Assigned<F>>, LEN, DivideBy<F, 128>>::without_witnesses(),
        }
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let l0: Affine1dConfig<F, LEN, LEN> = Affine1dConfig::<F, LEN, LEN>::configure(cs);
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure_with_input(l0.advice.output.clone(), cs);
        //or	<Self::Config as FourLayer>::L1::configure_with_input(l0.advice.output.clone(), cs);
        let l2: Affine1dConfig<F, LEN, LEN> =
            Affine1dConfig::configure_with_input(l1.advice.output.clone(), cs); // or <Self::Config as FourLayer>::L2::
        let l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            NonlinConfig1d::configure_with_input(l2.advice.output.clone(), cs);
        let l4: NonlinConfig1d<F, LEN, INBITS, OUTBITS, DivideBy<F, 128>> =
            NonlinConfig1d::configure_with_input(l3.advice.output.clone(), cs);

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);
        for i in 0..LEN {
            cs.enable_equality(l4.advice.output[i]);
        }

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
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.l0.layout(&(self.l0), &mut layouter)?;
        config.l1.layout(&(self.l1), &mut layouter)?;
        config.l2.layout(&(self.l2), &mut layouter)?;
        config.l3.layout(&(self.l3), &mut layouter)?;
        // tie the last output to public inputs (instance column)
        let output_for_eq = config.l4.layout(&(self.l4), &mut layouter)?;
        for i in 0..LEN {
            layouter.constrain_instance(output_for_eq[i].cell(), config.public_output, i)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fieldutils::felt_to_i32;
    use halo2_proofs::{
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp as F,
        plonk::{Any, Circuit},
    };
    //     use nalgebra;
    use rand::prelude::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_rescale() {
        let k = 15; //2^k rows
                    // parameters
        let l0weights: Vec<Vec<i32>> = vec![
            vec![10, 0, 0, -1],
            vec![0, 10, 1, 0],
            vec![0, 1, 10, 0],
            vec![1, 0, 0, 10],
        ];
        let l0biases: Vec<i32> = vec![0, 0, 0, 1];
        let l2weights: Vec<Vec<i32>> = vec![
            vec![0, 3, 10, -1],
            vec![0, 10, 1, 0],
            vec![0, 1, 0, 12],
            vec![1, -2, 32, 0],
        ];
        let l2biases: Vec<i32> = vec![12, 14, 17, 1];

        // input data
        let l0input: Vec<i32> = vec![-30, -21, 11, 40];

        // Create the layers
        let mut l0 = Affine1d::<F, Value<Assigned<F>>, 4, 4>::from_parameters(l0weights, l0biases);
        let mut l1 = Nonlin1d::<F, Value<Assigned<F>>, 4, ReLu<F>>::from_parameters();
        let mut l2 = Affine1d::<F, Value<Assigned<F>>, 4, 4>::from_parameters(l2weights, l2biases);
        let mut l3 = Nonlin1d::<F, Value<Assigned<F>>, 4, ReLu<F>>::from_parameters();
        let mut l4 = Nonlin1d::<F, Value<Assigned<F>>, 4, DivideBy<F, 128>>::from_parameters();

        // Assign the input
        let l0input = l0input
            .iter()
            .map(|x| Value::known(i32tofelt::<F>(*x).into()))
            .collect();

        // Compute the forward pass and witness, assigning as we go
        let l0out = l0.forward(l0input);
        let l1out = l1.forward(l0out);
        let l2out = l2.forward(l1out);
        let l3out = l3.forward(l2out);
        let l4out = l4.forward(l3out);
        println!(
            "{:?}",
            l4out
                .iter()
                .map(|x| x.map(|y| felt_to_i32(y.evaluate())))
                .collect::<Vec<Value<i32>>>()
        );

        let circuit = MyCircuit::<F, 4, 14, 14> { l0, l1, l2, l3, l4 };

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
}
