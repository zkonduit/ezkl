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
use crate::tensorutils::{dot3, flatten3, flatten4, map2, map3, map3r, map4, map4r};

use crate::affine1d::{Affine1d, Affine1dConfig};
use crate::layertrait::FourLayer;
use crate::nl::{Nonlin1d, NonlinConfig1d, ReLu};

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
    public_output: Column<Instance>,
}

impl<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> FourLayer
    for MyConfig<F, LEN, INBITS, OUTBITS>
{
    type L0 = Affine1dConfig<F, LEN, LEN>;
    type L1 = NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>;
    type L2 = Affine1dConfig<F, LEN, LEN>;
    type L3 = NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>;
}

//#[derive(Default)]
struct MyCircuit<
    F: FieldExt,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const INBITS: usize,
    const OUTBITS: usize,
> {
    // circuit holds Values
    l0_assigned: Affine1d<F, Value<Assigned<F>>, LEN, LEN>,
    l1_assigned: Nonlin1d<F, Value<Assigned<F>>, LEN>,
    l2_assigned: Affine1d<F, Value<Assigned<F>>, LEN, LEN>,
    l3_assigned: Nonlin1d<F, Value<Assigned<F>>, LEN>,
}

impl<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> Circuit<F>
    for MyCircuit<F, LEN, INBITS, OUTBITS>
{
    type Config = MyConfig<F, LEN, INBITS, OUTBITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            l0_assigned: Affine1d::<F, Value<Assigned<F>>, LEN, LEN>::without_witnesses(),
            l1_assigned: Nonlin1d::<F, Value<Assigned<F>>, LEN>::without_witnesses(),
            l2_assigned: Affine1d::<F, Value<Assigned<F>>, LEN, LEN>::without_witnesses(),
            l3_assigned: Nonlin1d::<F, Value<Assigned<F>>, LEN>::without_witnesses(),
        }
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let l0 = <Self::Config as FourLayer>::L0::configure(cs);
        let l1_adv: Nonlin1d<F, Column<Advice>, LEN> = Nonlin1d {
            input: l0.advice.output.clone(),
            output: (0..LEN).map(|i| cs.advice_column()).collect(),
            _marker: PhantomData,
        };
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            <Self::Config as FourLayer>::L1::composable_configure(l1_adv, cs);
        let l2_adv: Affine1d<F, Column<Advice>, LEN, LEN> = Affine1d {
            input: l1.advice.output.clone(),
            output: (0..LEN).map(|i| cs.advice_column()).collect(),
            weights: map2::<_, _, LEN, LEN>(|i, j| cs.advice_column()),
            biases: (0..LEN).map(|i| cs.advice_column()).collect(),
            _marker: PhantomData,
        };
        let l2 = <Self::Config as FourLayer>::L2::composable_configure(l2_adv, cs);
        let l3_adv: Nonlin1d<F, Column<Advice>, LEN> = Nonlin1d {
            input: l2.advice.output.clone(),
            output: (0..LEN).map(|i| cs.advice_column()).collect(),
            _marker: PhantomData,
        };
        let l3: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            <Self::Config as FourLayer>::L1::composable_configure(l3_adv, cs);

        let public_output: Column<Instance> = cs.instance_column();
        cs.enable_equality(public_output);
        for i in 0..LEN {
            cs.enable_equality(l3.advice.output[i]);
        }

        MyConfig {
            l0,
            l1,
            l2,
            l3,
            public_output,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.l0.layout(&(self.l0_assigned), &mut layouter)?;
        config.l1.layout(&(self.l1_assigned), &mut layouter)?;
        config.l2.layout(&(self.l2_assigned), &mut layouter)?;
        let output_for_eq = config.l3.layout(&(self.l3_assigned), &mut layouter)?;
        for i in 0..LEN {
            layouter.constrain_instance(output_for_eq[i].cell(), config.public_output, i)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp as F,
        plonk::{Any, Circuit},
    };
    //     use nalgebra;
    use rand::prelude::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_mlp_with_public() {
        let k = 9; //2^k rows
        let l0input: Vec<i32> = vec![-3, -2, 1, 4];
        let l0weights: Vec<Vec<i32>> = vec![
            vec![0, 0, 0, -1],
            vec![0, 0, 1, 0],
            vec![0, 1, 0, 0],
            vec![1, 0, 0, 0],
        ];
        let l0biases: Vec<i32> = vec![0, 0, 0, 1];
        let l0output: Vec<i32> = vec![-4, 1, -2, -2]; //also l1input

        let l1output = vec![0, 1, 0, 0]; //also l2input

        let l2weights: Vec<Vec<i32>> = vec![
            vec![0, 3, 0, -1],
            vec![0, 0, 1, 0],
            vec![0, 1, 0, 0],
            vec![1, -2, 0, 0],
        ];
        let l2biases: Vec<i32> = vec![1, 1, 1, 1];
        let l2output: Vec<i32> = vec![4, 1, 2, -1]; // also l3input

        let l3output: Vec<i32> = vec![4, 1, 2, 0];

        let l0_assigned = Affine1d::<F, Value<Assigned<F>>, 4, 4>::from_i32(
            l0input, l0output, l0weights, l0biases,
        );
        let l1_assigned: Nonlin1d<F, Value<Assigned<F>>, 4> = Nonlin1d {
            input: l0_assigned.output.clone(),
            output: l1output
                .iter()
                .map(|x| Value::known(F::from(*x).into()))
                .collect(),

            _marker: PhantomData,
        };

        let l2_assigned = Affine1d::<F, Value<Assigned<F>>, 4, 4> {
            input: l1_assigned.output.clone(),
            output: l2output
                .iter()
                .map(|x| Value::known(i32tofelt::<F>(*x).into()))
                .collect(),
            weights: map2::<_, _, 4, 4>(|i, j| {
                Value::known(i32tofelt::<F>(l2weights[i][j]).into())
            }),
            biases: l2biases
                .iter()
                .map(|x| Value::known(i32tofelt::<F>(*x).into()))
                .collect(),
            _marker: PhantomData,
        };

        let l3_assigned: Nonlin1d<F, Value<Assigned<F>>, 4> = Nonlin1d {
            input: l2_assigned.output.clone(),
            output: l3output
                .iter()
                .map(|x| Value::known(i32tofelt::<F>(*x).into()))
                .collect(),

            _marker: PhantomData,
        };

        let circuit = MyCircuit::<F, 4, 8, 8> {
            l0_assigned,
            l1_assigned,
            l2_assigned,
            l3_assigned,
        };
        let prover = MockProver::run(
            k,
            &circuit,
            vec![vec![4u64.into(), 1u64.into(), 2u64.into(), 0u64.into()]],
        )
        .unwrap();
        prover.assert_satisfied();
    }
}
