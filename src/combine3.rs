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
use crate::cnvrl::{Conv2dAssigned, Conv2dConfig};
use crate::layertrait::ThreeLayer;
use crate::nl::{Nonlin1d, NonlinConfig1d, ReLu};

#[derive(Clone)]
struct MyConfig<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize, //= (IH - KH + 1); //not supported yet in rust
    const OW: usize, //= (IW - KW + 1); //not supported yet in rust
    const BITS: usize,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened //not supported yet in rust
    const INBITS: usize,
    const OUTBITS: usize,
> {
    l0: Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>,
    l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>,
    l2: Affine1dConfig<F, LEN, LEN>,
}

impl<
        F: FieldExt,
        const IH: usize,
        const IW: usize,
        const CHIN: usize,
        const CHOUT: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize, //= (IH - KH + 1); //not supported yet in rust
        const OW: usize, //= (IW - KW + 1); //not supported yet in rust
        const BITS: usize,
        const LEN: usize,
        const INBITS: usize,
        const OUTBITS: usize,
    > ThreeLayer for MyConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>
{
    type L0 = Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>;
    type L1 = NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>;
    type L2 = Affine1dConfig<F, LEN, LEN>;
}

//#[derive(Default)]
struct MyCircuit<
    F: FieldExt,
    const IH: usize,
    const IW: usize,
    const CHIN: usize,
    const CHOUT: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize, //= (IH - KH + 1); //not supported yet in rust
    const OW: usize, //= (IW - KW + 1); //not supported yet in rust
    const BITS: usize,
    const LEN: usize, //LEN = CHOUT x OH x OW flattened
    const INBITS: usize,
    const OUTBITS: usize,
> {
    // circuit holds Values
    c2d_assigned: Conv2dAssigned<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>,
    relu_assigned: Nonlin1d<F, Value<Assigned<F>>, LEN>,
    lin_assigned: Affine1d<F, Value<Assigned<F>>, LEN, LEN>,
    // Public input (from prover).
}

impl<
        F: FieldExt,
        const IH: usize,
        const IW: usize,
        const CHIN: usize,
        const CHOUT: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize, //= (IH - KH + 1); //not supported yet in rust
        const OW: usize, //= (IW - KW + 1); //not supported yet in rust
        const BITS: usize,
        const LEN: usize,
        const INBITS: usize,
        const OUTBITS: usize,
    > Circuit<F> for MyCircuit<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>
{
    type Config = MyConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let c2d_assigned = Conv2dAssigned::without_witnesses();
        let relu_assigned = Nonlin1d::<F, Value<Assigned<F>>, LEN>::without_witnesses();
        let lin_assigned = Affine1d::<F, Value<Assigned<F>>, LEN, LEN>::without_witnesses();
        Self {
            c2d_assigned,
            relu_assigned,
            lin_assigned,
        }
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let l0 = <Self::Config as ThreeLayer>::L0::configure(cs);
        let flattened = flatten3(l0.advice.lin_output.clone());
        let l1_in_adv = Nonlin1d::<F, Column<Advice>, LEN>::fill(|i| flattened[i]);
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            <Self::Config as ThreeLayer>::L1::composable_configure(l1_in_adv, cs);
        let l2_in_adv: Affine1d<F, Column<Advice>, LEN, LEN> = Affine1d {
            input: l1.advice.output.clone(),
            output: (0..LEN).map(|i| cs.advice_column()).collect(),
            weights: map2::<_, _, LEN, LEN>(|i, j| cs.advice_column()),
            biases: (0..LEN).map(|i| cs.advice_column()).collect(),
            _marker: PhantomData,
        };
        let l2 = <Self::Config as ThreeLayer>::L2::composable_configure(l2_in_adv, cs);
        MyConfig { l0, l1, l2 }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.l0.layout(&(self.c2d_assigned), &mut layouter)?;
        config.l1.layout(&(self.relu_assigned), &mut layouter)?;
        config.l2.layout(&(self.lin_assigned), &mut layouter)?;
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
    use std::time::{Duration, Instant};

    #[test]
    fn test_layers() {
        let k = 9; //2^k rows
        let input = vec![
            vec![
                vec![0u64, 1u64, 2u64],
                vec![3u64, 4u64, 5u64],
                vec![6u64, 7u64, 8u64],
            ],
            vec![
                vec![1u64, 2u64, 3u64],
                vec![4u64, 5u64, 6u64],
                vec![7u64, 8u64, 9u64],
            ],
        ];
        let kernel = vec![vec![
            vec![vec![0u64, 1u64], vec![2u64, 3u64]],
            vec![vec![1u64, 2u64], vec![3u64, 4u64]],
        ]];
        let l1output = vec![vec![vec![56u64, 72u64], vec![104u64, 120u64]]];
        let l2input = vec![56i32, 72i32, 104i32, 120i32];
        let l2weights: Vec<Vec<i32>> = vec![
            vec![0, 0, 0, -1],
            vec![0, 0, 1, 0],
            vec![0, 1, 0, 0],
            vec![1, 0, 0, 0],
        ];
        let l2biases: Vec<i32> = vec![0, 0, 0, 1];
        let l2output: Vec<i32> = vec![-120, 104, 72, 57];

        let c2d_assigned = Conv2dAssigned::<F, 3, 3, 2, 1, 2, 2, 2, 2, 8>::from_values(
            kernel,
            input,
            l1output.clone(),
        );
        let relu_v: Vec<Value<Assigned<F>>> = flatten3(l1output)
            .iter()
            .map(|x| Value::known(F::from(*x).into()))
            .collect();
        let relu_assigned: Nonlin1d<F, Value<Assigned<F>>, 4> = Nonlin1d {
            input: relu_v.clone(),
            output: relu_v,
            _marker: PhantomData,
        };
        let lin_assigned = Affine1d::<F, Value<Assigned<F>>, 4, 4>::from_i32(
            l2input, l2output, l2weights, l2biases,
        );

        let circuit = MyCircuit::<F, 3, 3, 2, 1, 2, 2, 2, 2, 8, 4, 8, 8> {
            c2d_assigned,
            relu_assigned,
            lin_assigned,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
