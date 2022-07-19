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
use crate::tensorutils::{dot3, flatten3, flatten4, map3, map3r, map4, map4r};

use crate::cnvrl::{Conv2dAssigned, Conv2dConfig};
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
}

trait ConfigContainer02 {
    type L0;
    type L1;
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
    > ConfigContainer02
    for MyConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>
{
    type L0 = Conv2dConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS>;
    type L1 = NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>>;
}

//#[derive(Default)]
struct Cnv2d_then_Relu_Circuit<
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
    > Circuit<F>
    for Cnv2d_then_Relu_Circuit<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>
{
    type Config = MyConfig<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>;
    //        Conv2d_then_Relu_Config<F, IH, IW, CHIN, CHOUT, KH, KW, OH, OW, BITS, LEN, INBITS, OUTBITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let c2d_assigned = Conv2dAssigned::without_witnesses();
        let relu_assigned = Nonlin1d::<F, Value<Assigned<F>>, LEN>::without_witnesses();
        Self {
            c2d_assigned,
            relu_assigned,
        }
    }

    // Here we wire together the layers by using the output advice in each layer as input advice in the next (not with copying / equality).
    // This can be automated but we will sometimes want skip connections, etc. so we need the flexibility.
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let l0 = <Self::Config as ConfigContainer02>::L0::configure(cs);
        let flattened = flatten3(l0.advice.lin_output.clone());
        let l1_in_adv = Nonlin1d::<F, Column<Advice>, LEN>::fill(|i| flattened[i]);
        let l1: NonlinConfig1d<F, LEN, INBITS, OUTBITS, ReLu<F>> =
            <Self::Config as ConfigContainer02>::L1::composable_configure(l1_in_adv, cs);
        MyConfig { l0, l1 }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        config.l0.layout(&(self.c2d_assigned), &mut layouter)?;
        config.l1.layout(&(self.relu_assigned), &mut layouter)?;
        Ok(())
    }
}
//
//
//
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
    fn test_cnv2d_succeed() {
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
        let output = vec![vec![vec![56u64, 72u64], vec![104u64, 120u64]]];
        let assigned =
            Conv2dAssigned::<F, 3, 3, 2, 1, 2, 2, 2, 2, 8>::from_values(kernel, input, output);

        let circuit = Conv2dCircuit::<F, 3, 3, 2, 1, 2, 2, 2, 2, 8> { assigned };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}

// const IH: usize,
// const IW: usize,
// const CHIN: usize,
// const CHOUT: usize,
// const KH: usize,
// const KW: usize,
// const OH: usize,
// const OW: usize,
// const BITS: usize,

//         let a_00: u64 = 1;
//         let a_01: u64 = 2;
//         let a_10: u64 = 3;
//         let a_11: u64 = 4;
//         let v_0: u64 = 5;
//         let v_1: u64 = 6;
//         let u_0: u64 = 17;
//         let u_1: u64 = 39;
//         let r_0: u64 = 17;
//         let r_1: u64 = 39;

//         let pub_inputs = vec![F::from(r_0), F::from(r_1), F::from(r_1)];
//         // Successful cases

//         let circuit = MvrlCircuit::<F, 2, 2, 8> {
//             a: vec![
//                 vec![
//                     Value::known(F::from(a_00).into()),
//                     Value::known(F::from(a_01).into()),
//                 ],
//                 vec![
//                     Value::known(F::from(a_10).into()),
//                     Value::known(F::from(a_11).into()),
//                 ],
//             ],
//             v: vec![
//                 Value::known(F::from(v_0).into()),
//                 Value::known(F::from(v_1).into()),
//             ],
//             u: vec![
//                 Value::known(F::from(u_0).into()),
//                 Value::known(F::from(u_1).into()),
//             ],
//             r: vec![
//                 Value::known(F::from(r_0).into()),
//                 Value::known(F::from(r_1).into()),
//             ],
//             //          wasa,
//             //         wasc,
//         };

//         // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
//         // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

//         let prover = MockProver::run(k, &circuit, vec![]).unwrap();
//         prover.assert_satisfied();
//     }

// }

// // move this into cnv2dconfig, layout(&mut layouter, cnv2dconfig, cnv2dassigned)
// // iterate layers
// layouter.assign_region(
//     || "Assign values", // the name of the region
//     |mut region| {
//         let offset = 0;

//         config.l0.q.enable(&mut region, offset)?;
//         // let kernel_res: Vec<Vec<Vec<Vec<()>>>> =
//         map4r::<_, _, halo2_proofs::plonk::Error, CHOUT, CHIN, KH, KW>(|i, j, k, l| {
//             region.assign_advice(
//                 || format!("kr_{i}_{j}_{k}_{l}"),
//                 config.l0.advice.kernel[i][j][k][l], // Column<Advice>
//                 offset,
//                 || self.c2d_assigned.kernel[i][j][k][l], //Assigned<F>
//             )
//         })?;

//         //                let input_res: Vec<Vec<Vec<_>>> =
//         map3r::<_, _, halo2_proofs::plonk::Error, CHIN, IH, IW>(|i, j, k| {
//             region.assign_advice(
//                 || format!("in_{i}_{j}_{k}"),
//                 config.l0.advice.input[i][j][k], // Column<Advice>
//                 offset,
//                 || self.c2d_assigned.input[i][j][k], //Assigned<F>
//             )
//         })?;

//         map3r::<_, _, halo2_proofs::plonk::Error, CHOUT, OH, OW>(|i, j, k| {
//             region.assign_advice(
//                 || format!("out_{i}_{j}_{k}"),
//                 config.l0.advice.lin_output[i][j][k], // Column<Advice>
//                 offset,
//                 || self.c2d_assigned.lin_output[i][j][k], //Assigned<F>
//             )
//         })?;

//         Ok(())
//     },
// )?;

//        config.alloc_table(&mut layouter)?;
