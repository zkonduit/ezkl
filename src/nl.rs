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

#[derive(Clone)]
struct Nonlin1d<F: FieldExt, Inner, const LEN: usize> {
    input: Vec<Inner>,
    output: Vec<Inner>,
    _marker: PhantomData<F>,
}
impl<F: FieldExt, Inner, const LEN: usize> Nonlin1d<F, Inner, LEN> {
    pub fn fill<Func>(mut f: Func) -> Self
    where
        Func: FnMut(usize) -> Inner,
    {
        Nonlin1d {
            input: (0..LEN).map(|i| f(i)).collect(),
            output: (0..LEN).map(|i| f(i)).collect(),
            _marker: PhantomData,
        }
    }
}

#[derive(Clone)]
struct NonlinTable<const INBITS: usize, const OUTBITS: usize> {
    table_input: TableColumn,
    table_output: TableColumn,
}

#[derive(Clone)]
struct NonlinConfig1d<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize> {
    advice: Nonlin1d<F, Column<Advice>, LEN>,
    table: NonlinTable<INBITS, OUTBITS>,
}

// trait NonlinFn<F> {
//     fn function() -> impl Fn(F) -> F {}
// }

impl<F: FieldExt, const LEN: usize, const INBITS: usize, const OUTBITS: usize>
    NonlinConfig1d<F, LEN, INBITS, OUTBITS>
{
    fn define_advice(cs: &mut ConstraintSystem<F>) -> Nonlin1d<F, Column<Advice>, LEN> {
        Nonlin1d::<F, Column<Advice>, LEN>::fill(|i| cs.advice_column())
    }
    fn configure(cs: &mut ConstraintSystem<F>) -> NonlinConfig1d<F, LEN, INBITS, OUTBITS> {
        let advice = Self::define_advice(cs);
        let table = NonlinTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
        };

        for i in 0..LEN {
            let _ = cs.lookup(|cs| {
                vec![
                    (
                        cs.query_advice(advice.input[i], Rotation::cur()),
                        table.table_input,
                    ),
                    (
                        cs.query_advice(advice.output[i], Rotation::cur()),
                        table.table_output,
                    ),
                ]
            });
        }

        Self { advice, table }
    }

    // Allocates all legal input-output tuples for the function in the first 2^k rows
    // of the constraint system.
    fn alloc_table(
        &self,
        layouter: &mut impl Layouter<F>,
        nonlinearity: Box<dyn Fn(i32) -> F>,
    ) -> Result<(), Error> {
        let base = 2i32;
        let smallest = -base.pow(INBITS as u32 - 1);
        let largest = base.pow(INBITS as u32 - 1);
        layouter.assign_table(
            || "nl table",
            |mut table| {
                let mut row_offset = 0;
                for int_input in smallest..largest {
                    let input: F = i32tofelt(int_input);
                    table.assign_cell(
                        || format!("nl_i_col row {}", row_offset),
                        self.table.table_input,
                        row_offset,
                        || Value::known(input),
                    )?;
                    table.assign_cell(
                        || format!("nl_o_col row {}", row_offset),
                        self.table.table_output,
                        row_offset,
                        || Value::known(nonlinearity(int_input)),
                    )?;
                    row_offset += 1;
                }
                Ok(())
            },
        )
    }
}

trait Nonlinearity<F: FieldExt> {
    fn nonlinearity(x: i32) -> F;
}

struct NLCircuit<
    F: FieldExt,
    const LEN: usize,
    const INBITS: usize,
    const OUTBITS: usize,
    NL: Nonlinearity<F>,
> {
    assigned: Nonlin1d<F, Value<Assigned<F>>, LEN>,
    _marker: PhantomData<NL>, //    nonlinearity: Box<dyn Fn(F) -> F>,
}

impl<
        F: FieldExt,
        const LEN: usize,
        const INBITS: usize,
        const OUTBITS: usize,
        NL: 'static + Nonlinearity<F>,
    > Circuit<F> for NLCircuit<F, LEN, INBITS, OUTBITS, NL>
{
    type Config = NonlinConfig1d<F, LEN, INBITS, OUTBITS>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let assigned = Nonlin1d::<F, Value<Assigned<F>>, LEN>::fill(|i| Value::default());
        Self {
            assigned,
            _marker: PhantomData,
        }
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        Self::Config::configure(cs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        // mvmul

        layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                for i in 0..LEN {
                    region.assign_advice(
                        || format!("nl_{i}"),
                        config.advice.input[i], // Column<Advice>
                        offset,
                        || self.assigned.input[i], //Assigned<F>
                    )?;
                }

                Ok(())
            },
        )?;

        config.alloc_table(&mut layouter, Box::new(NL::nonlinearity))?;

        Ok(())
    }
}

// Now implement nonlinearity functions like this
struct ReLu<F> {
    _marker: PhantomData<F>,
}
impl<F: FieldExt> Nonlinearity<F> for ReLu<F> {
    fn nonlinearity(x: i32) -> F {
        if x < 0 {
            F::zero()
        } else {
            i32tofelt(x)
        }
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
    use nalgebra;
    use std::time::{Duration, Instant};

    #[test]
    fn test_mvrl_succeed() {
        let k = 9; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 39;
        let r_0: u64 = 17;
        let r_1: u64 = 39;
        //        let wasa: Value<Assigned<F>> = Value::known((-F::from(3)).into());
        //        let wasc: Value<Assigned<F>> = Value::known(F::from(0).into());

        //        let pub_inputs = vec![F::from(r_0), F::from(r_1), F::from(r_1)];
        // Successful cases

        let circuit = MvrlCircuit::<F, 2, 2, 8> {
            a: vec![
                vec![
                    Value::known(F::from(a_00).into()),
                    Value::known(F::from(a_01).into()),
                ],
                vec![
                    Value::known(F::from(a_10).into()),
                    Value::known(F::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known(F::from(v_0).into()),
                Value::known(F::from(v_1).into()),
            ],
            u: vec![
                Value::known(F::from(u_0).into()),
                Value::known(F::from(u_1).into()),
            ],
            r: vec![
                Value::known(F::from(r_0).into()),
                Value::known(F::from(r_1).into()),
            ],
            //          wasa,
            //         wasc,
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_mvrl_withneg() {
        let k = 9; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 39;
        let r_0: u64 = 0;
        let r_1: u64 = 0;

        let circuit = MvrlCircuit::<F, 2, 2, 8> {
            a: vec![
                vec![
                    Value::known(F::from(a_00).into()),
                    Value::known(F::from(a_01).into()),
                ],
                vec![
                    Value::known(F::from(a_10).into()),
                    Value::known(F::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known((-F::from(v_0)).into()),
                Value::known((-F::from(v_1)).into()),
            ],
            u: vec![
                Value::known((-F::from(u_0)).into()),
                Value::known((-F::from(u_1)).into()),
            ],
            r: vec![
                Value::known(F::from(r_0).into()),
                Value::known(F::from(r_1).into()),
            ],
            //          wasa,
            //         wasc,
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    #[should_panic]
    fn test_mvrl_fail() {
        let k = 9; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 212;
        let r_0: u64 = 17;
        let r_1: u64 = 212;

        //        let wasa: Value<Assigned<F>> = Value::known((-F::from(3)).into());
        //       let wasc: Value<Assigned<F>> = Value::known(F::from(0).into());

        //   let pub_inputs = vec![F::from(r_0), F::from(r_1)];
        //        let pub_inputs = vec![F::from(0)];

        // Successful cases

        let circuit = MvrlCircuit::<F, 2, 2, 8> {
            a: vec![
                vec![
                    Value::known(F::from(a_00).into()),
                    Value::known(F::from(a_01).into()),
                ],
                vec![
                    Value::known(F::from(a_10).into()),
                    Value::known(F::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known(F::from(v_0).into()),
                Value::known(F::from(v_1).into()),
            ],
            u: vec![
                Value::known(F::from(u_0).into()),
                Value::known(F::from(u_1).into()),
            ],
            r: vec![
                Value::known(F::from(r_0).into()),
                Value::known(F::from(r_1).into()),
            ],
            //         wasa,
            //         wasc,
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    // #[test]
    // #[ignore]
    // fn test_mvmul_bigger() {
    //     const NROWS: usize = 128;
    //     const NCOLS: usize = 128; //32x52 overflows stack in verifying key creation
    //     let k = 4; //2^k rows
    //                //        let avals: [u64; 32] = rand.rand();
    //     let m_a: nalgebra::SMatrix<u64, NROWS, NCOLS> =
    //         nalgebra::SMatrix::from_iterator((0..NROWS * NCOLS).map(|x| x as u64));
    //     let m_v: nalgebra::SVector<u64, NCOLS> =
    //         nalgebra::SVector::from_iterator((0..NCOLS).map(|x| x as u64));
    //     let m_u = m_a * m_v;
    //     println!("a: {}", m_a);
    //     println!("v: {}", m_v);
    //     println!("u: {}", m_u);
    //     println!("{}", NROWS * NCOLS + NROWS + NCOLS);

    //     let mut a: Vec<Vec<Value<Assigned<F>>>> = Vec::new();
    //     for i in 0..NROWS {
    //         let mut row: Vec<Value<Assigned<F>>> = Vec::new();
    //         for j in 0..NCOLS {
    //             row.push(Value::known(F::from(m_a[(i, j)]).into()))
    //         }
    //         a.push(row);
    //     }

    //     let mut v: Vec<Value<Assigned<F>>> = Vec::new();
    //     let mut u: Vec<Value<Assigned<F>>> = Vec::new();
    //     for j in 0..NCOLS {
    //         v.push(Value::known(F::from(m_v[j]).into()));
    //     }

    //     for i in 0..NROWS {
    //         u.push(Value::known(F::from(m_u[i]).into()));
    //     }

    //     let circuit = MvrlCircuit::<F, NROWS, NCOLS, NBITS> { a, v, u, r };

    //     let params: Params<vesta::Affine> = Params::new(k);

    //     let empty_circuit = circuit.without_witnesses();

    //     // Initialize the proving key
    //     let now = Instant::now();
    //     let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
    //     println!("VK took {}", now.elapsed().as_secs());
    //     let now = Instant::now();
    //     let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
    //     println!("PK took {}", now.elapsed().as_secs());
    //     let now = Instant::now();
    //     //println!("{:?} {:?}", vk, pk);
    //     let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    //     let mut rng = OsRng;
    //     create_proof(&params, &pk, &[circuit], &[&[]], &mut rng, &mut transcript)
    //         .expect("proof generation should not fail");
    //     let proof = transcript.finalize();
    //     //println!("{:?}", proof);
    //     println!("Proof took {}", now.elapsed().as_secs());
    //     let now = Instant::now();
    //     let strategy = SingleVerifier::new(&params);
    //     let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    //     assert!(verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript).is_ok());
    //     println!("Verify took {}", now.elapsed().as_secs());
    // }
}
