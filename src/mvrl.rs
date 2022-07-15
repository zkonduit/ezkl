use std::marker::PhantomData; // Allows Rust to track types that do not otherwise appear in a struct's fields, here just the field element type

use halo2_proofs::{
    arithmetic::FieldExt, // the field element trait
    circuit::{
        floor_planner::V1,
        AssignedCell, // a value Value<V> together with its global location as a Cell with region_index, row_offset, and column
        Layouter,     // layout strategy and accepter struct, a bit like a Writer
        SimpleFloorPlanner,
        Value, // basically an Option<V>, where Some(v) is called known and None is unknown
    },
    plonk::{
        create_proof,
        keygen_pk,
        keygen_vk,
        verify_proof,
        Advice,           // empty struct to mark Advice columns
        Assigned, // enum Zero, Trivial(F) "does not require inversion to evaluate", or Rational(F, F) "stored as a fraction to enable batch inversion". This is an actual value (wrapped felt)
        Circuit,  // trait with without_witnesses, configure, and synthesize methods
        Column, // represents a pre-layout abstract Column. Fields are index: usize and column type.
        ConstraintSystem, // The container for the actual constraint system; much of the frontend code exists to make it easier to populate this container
        Constraints, // Iterable with a selector and Constraint s.  Constraints are single polynomial Expressions returned by create gate
        Error,       // Custom Error type
        Expression, // Polynomial expression enum, as binary tree, with 5 types of atomic variables v (Constant, Selector, Fixed, Advice, Instance) and combinations -v, v+v, a*v, or v*v.
        Selector, // (index: usize, simple: bool) column type, w/ index = index of this selector in the ConstraintSystem, simple = "can only be multiplied by Expressions not containing Selectors"
        SingleVerifier,
    },
    poly::commitment::Params,
    poly::Rotation, // i32 wrapper representing rotation in Lagrange basis
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};

use pasta_curves::{pallas, vesta};
use rand::rngs::OsRng;

#[derive(Clone)]
struct MvmulConfig<F: FieldExt, const NROWS: usize, const NCOLS: usize> {
    // Config holds labels
    a: Vec<Vec<Column<Advice>>>,
    v: Vec<Column<Advice>>,
    u: Vec<Column<Advice>>,
    q: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt, const NROWS: usize, const NCOLS: usize> MvmulConfig<F, NROWS, NCOLS> {
    //     fn alabels(&self) -> Vec<Vec<String>> {
    //         let mut out = Vec::new();
    //         for i in 1..NROWS {
    //             let mut row = Vec::new();
    //             for j in 1..NCOLS {
    //                 row.push(format!("a_{}_{}", i, j));
    //             }
    //             out.push(row);
    //         }
    //         out
    //     }

    //     fn vlabels(&self) -> Vec<String> {
    //         (1..NCOLS).map(|j| format!("v_{}", j)).collect()
    //     }
    //     fn ulabels(&self) -> Vec<String> {
    //         (1..NROWS).map(|j| format!("u_{}", j)).collect()
    //     }
}

//#[derive(Default)]
struct MvmulCircuit<F: FieldExt, const NROWS: usize, const NCOLS: usize> {
    // circuit holds Values
    a: Vec<Vec<Value<Assigned<F>>>>,
    v: Vec<Value<Assigned<F>>>,
    u: Vec<Value<Assigned<F>>>,
}
//impl<F: FieldExt, const RANGE: usize> MvmulCircuit<F, RANGE> {}

impl<F: FieldExt, const NROWS: usize, const NCOLS: usize> Circuit<F>
    for MvmulCircuit<F, NROWS, NCOLS>
{
    type Config = MvmulConfig<F, NROWS, NCOLS>;
    type FloorPlanner = SimpleFloorPlanner;
    //    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        // put Unknown in all the Value<Assigned>
        let mut a: Vec<Vec<Value<Assigned<F>>>> = Vec::new();
        let mut v: Vec<Value<Assigned<F>>> = Vec::new();
        let mut u: Vec<Value<Assigned<F>>> = Vec::new();

        for _i in 0..NROWS {
            let mut row: Vec<Value<Assigned<F>>> = Vec::new();
            for _j in 0..NCOLS {
                row.push(Value::default());
            }
            a.push(row);
        }

        for _j in 0..NCOLS {
            v.push(Value::default());
        }

        for _i in 0..NROWS {
            u.push(Value::default());
        }

        MvmulCircuit { a, v, u }
    }

    // define the constraints, mutate the provided ConstraintSystem, and output the resulting FrameType
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let (qs, aadv, uadv, vadv) = {
            let qs = cs.selector();
            let mut aadv: Vec<Vec<Column<Advice>>> = Vec::new();
            for _i in 0..NROWS {
                let mut row: Vec<Column<Advice>> = Vec::new();
                for _j in 0..NCOLS {
                    row.push(cs.advice_column());
                }
                aadv.push(row);
            }
            let uadv: Vec<Column<Advice>> = (0..NROWS).map(|_| cs.advice_column()).collect();
            let vadv: Vec<Column<Advice>> = (0..NCOLS).map(|_| cs.advice_column()).collect();
            (qs, aadv, uadv, vadv)
        };
        cs.create_gate("mvmul", |virtual_cells| {
            // 'allocate' all the advice  and selector cols by querying the cs with the labels
            let q = virtual_cells.query_selector(qs);
            let mut a: Vec<Vec<Expression<F>>> = Vec::new();
            for i in 0..NROWS {
                let mut row: Vec<Expression<F>> = Vec::new();
                for j in 0..NCOLS {
                    row.push(virtual_cells.query_advice(aadv[i][j], Rotation::cur()));
                }
                a.push(row);
            }

            let mut v: Vec<Expression<F>> = Vec::new();
            let mut u: Vec<Expression<F>> = Vec::new();
            for j in 0..NCOLS {
                v.push(virtual_cells.query_advice(vadv[j], Rotation::cur()));
            }

            for i in 0..NROWS {
                u.push(virtual_cells.query_advice(uadv[i], Rotation::cur()));
            }

            // build the constraints c[i] is -u_i + \sum_j a_{ij} v_j
            let mut c: Vec<Expression<F>> = Vec::new();
            // first c[i] = -u[i]
            for i in 0..NROWS {
                c.push(-u[i].clone());
            }

            for i in 0..NROWS {
                for j in 0..NCOLS {
                    c[i] = c[i].clone() + a[i][j].clone() * v[j].clone();
                }
            }

            let constraints = (0..NROWS).map(|j| "c").zip(c);
            Constraints::with_selector(q, constraints)
        });
        // The "FrameType"
        Self::Config {
            a: aadv,
            u: uadv,
            v: vadv,
            q: qs,
            _marker: PhantomData,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                config.q.enable(&mut region, offset)?;

                for i in 0..NROWS {
                    for j in 0..NCOLS {
                        region.assign_advice(
                            || format!("a_{i}_{j}"),
                            config.a[i][j], // Column<Advice>
                            offset,
                            || self.a[i][j],
                        )?;
                    }
                }

                for i in 0..NROWS {
                    region.assign_advice(|| format!("u_{i}"), config.u[i], offset, || self.u[i])?;
                }

                for j in 0..NCOLS {
                    region.assign_advice(|| format!("v_{j}"), config.v[j], offset, || self.v[j])?;
                }

                Ok(())
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp,
        plonk::{Any, Circuit},
    };
    use nalgebra;
    use std::time::{Duration, Instant};

    #[test]
    fn test_mvmul_succeed() {
        let k = 4; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 39;

        // Successful cases

        let circuit = MvmulCircuit::<Fp, 2, 2> {
            a: vec![
                vec![
                    Value::known(Fp::from(a_00).into()),
                    Value::known(Fp::from(a_01).into()),
                ],
                vec![
                    Value::known(Fp::from(a_10).into()),
                    Value::known(Fp::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known(Fp::from(v_0).into()),
                Value::known(Fp::from(v_1).into()),
            ],
            u: vec![
                Value::known(Fp::from(u_0).into()),
                Value::known(Fp::from(u_1).into()),
            ],
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    #[should_panic]
    fn test_mvmul_fail() {
        let k = 4; //2^k rows
        let a_00: u64 = 1;
        let a_01: u64 = 2;
        let a_10: u64 = 3;
        let a_11: u64 = 4;
        let v_0: u64 = 5;
        let v_1: u64 = 6;
        let u_0: u64 = 17;
        let u_1: u64 = 212;

        // Successful cases

        let circuit = MvmulCircuit::<Fp, 2, 2> {
            a: vec![
                vec![
                    Value::known(Fp::from(a_00).into()),
                    Value::known(Fp::from(a_01).into()),
                ],
                vec![
                    Value::known(Fp::from(a_10).into()),
                    Value::known(Fp::from(a_11).into()),
                ],
            ],
            v: vec![
                Value::known(Fp::from(v_0).into()),
                Value::known(Fp::from(v_1).into()),
            ],
            u: vec![
                Value::known(Fp::from(u_0).into()),
                Value::known(Fp::from(u_1).into()),
            ],
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_mvmul_bigger() {
        const NROWS: usize = 128;
        const NCOLS: usize = 128; //32x52 overflows stack in verifying key creation
        let k = 4; //2^k rows
                   //        let avals: [u64; 32] = rand.rand();
        let m_a: nalgebra::SMatrix<u64, NROWS, NCOLS> =
            nalgebra::SMatrix::from_iterator((0..NROWS * NCOLS).map(|x| x as u64));
        let m_v: nalgebra::SVector<u64, NCOLS> =
            nalgebra::SVector::from_iterator((0..NCOLS).map(|x| x as u64));
        let m_u = m_a * m_v;
        println!("a: {}", m_a);
        println!("v: {}", m_v);
        println!("u: {}", m_u);
        println!("{}", NROWS * NCOLS + NROWS + NCOLS);

        let mut a: Vec<Vec<Value<Assigned<Fp>>>> = Vec::new();
        for i in 0..NROWS {
            let mut row: Vec<Value<Assigned<Fp>>> = Vec::new();
            for j in 0..NCOLS {
                row.push(Value::known(Fp::from(m_a[(i, j)]).into()))
            }
            a.push(row);
        }

        let mut v: Vec<Value<Assigned<Fp>>> = Vec::new();
        let mut u: Vec<Value<Assigned<Fp>>> = Vec::new();
        for j in 0..NCOLS {
            v.push(Value::known(Fp::from(m_v[j]).into()));
        }

        for i in 0..NROWS {
            u.push(Value::known(Fp::from(m_u[i]).into()));
        }

        let circuit = MvmulCircuit::<Fp, NROWS, NCOLS> { a, v, u };

        //        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        //        prover.assert_satisfied();

        let params: Params<vesta::Affine> = Params::new(k);

        let empty_circuit = circuit.without_witnesses();

        // Initialize the proving key
        let now = Instant::now();
        let vk = keygen_vk(&params, &empty_circuit).expect("keygen_vk should not fail");
        println!("VK took {}", now.elapsed().as_secs());
        let now = Instant::now();
        let pk = keygen_pk(&params, vk.clone(), &empty_circuit).expect("keygen_pk should not fail");
        println!("PK took {}", now.elapsed().as_secs());
        let now = Instant::now();
        //println!("{:?} {:?}", vk, pk);
        let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
        let mut rng = OsRng;
        create_proof(&params, &pk, &[circuit], &[&[]], &mut rng, &mut transcript)
            .expect("proof generation should not fail");
        let proof = transcript.finalize();
        //println!("{:?}", proof);
        println!("Proof took {}", now.elapsed().as_secs());
        let now = Instant::now();
        let strategy = SingleVerifier::new(&params);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
        assert!(verify_proof(&params, pk.get_vk(), strategy, &[&[]], &mut transcript).is_ok());
        println!("Verify took {}", now.elapsed().as_secs());
    }
}
