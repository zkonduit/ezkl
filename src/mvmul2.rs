use std::marker::PhantomData; // Allows Rust to track types that do not otherwise appear in a struct's fields, here just the field element type

use halo2_proofs::{
    arithmetic::FieldExt, // the field element trait
    circuit::{
        floor_planner::V1,
        AssignedCell, // a value Value<V> together with its global location as a Cell with region_index, row_offset, and column
        Layouter,     // layout strategy and accepter struct, a bit like a Writer
        Value,        // basically an Option<V>, where Some(v) is called known and None is unknown
    },
    plonk::{
        Advice,           // empty struct to mark Advice columns
        Assigned, // enum Zero, Trivial(F) "does not require inversion to evaluate", or Rational(F, F) "stored as a fraction to enable batch inversion". This is an actual value (wrapped felt)
        Circuit,  // trait with without_witnesses, configure, and synthesize methods
        Column, // represents a pre-layout abstract Column. Fields are index: usize and column type.
        ConstraintSystem, // The container for the actual constraint system; much of the frontend code exists to make it easier to populate this container
        Constraints, // Iterable with a selector and Constraint s.  Constraints are single polynomial Expressions returned by create gate
        Error,       // Custom Error type
        Expression, // Polynomial expression enum, as binary tree, with 5 types of atomic variables v (Constant, Selector, Fixed, Advice, Instance) and combinations -v, v+v, a*v, or v*v.
        Selector, // (index: usize, simple: bool) column type, w/ index = index of this selector in the ConstraintSystem, simple = "can only be multiplied by Expressions not containing Selectors"
    },
    poly::Rotation, // i32 wrapper representing rotation in Lagrange basis
};

// A Config is an associated type of your custom circuit (required only to be Clone).  With no particular enforced structure, it stores whatever type information is needed
// to understand the constraint system (number and types of columns, their indices, some flags such as simple/complex selector, etc.).
// It is a bit like a morphism type in a Monoidal category (domain and codomain), or the row and column labels in a dataframe. Let's call it the FrameType
// It can be unstructured because it is the Circuit implementer's job to translate this information into the format needed for the Layouter.
#[derive(Clone)]
struct MvmulConfig<F: FieldExt, const NROWS: usize, const NCOLS: usize> {
    a: Vec<Vec<Column<Advice>>>,
    v: Vec<Column<Advice>>,
    u: Vec<Column<Advice>>,
    q: Selector, // do we need these?
    _marker: PhantomData<F>,
}
// By convention the Config gets a configure and assign method, which are delegated to by the configure and synthesize method of the Circuit.
impl<F: FieldExt, const NROWS: usize, const NCOLS: usize> MvmulConfig<F, NROWS, NCOLS> {
    fn alabels(&self) -> Vec<Vec<String>> {
        let mut out = Vec::new();
        for i in 1..NROWS {
            let mut row = Vec::new();
            for j in 1..NCOLS {
                row.push(format!("a_{}_{}", i, j));
            }
            out.push(row);
        }
        out
    }

    fn vlabels(&self) -> Vec<String> {
        (1..NCOLS).map(|j| format!("v_{}", j)).collect()
    }
    fn ulabels(&self) -> Vec<String> {
        (1..NROWS).map(|j| format!("u_{}", j)).collect()
    }

    // fn aadv(&self) -> Vec<Column<Advice>> {
    // }
    // fn vadv(&self) -> Vec<Column<Advice>> {
    //     //NCOLS
    //     (1..self.ncols).map(|j| format!("v_{}", j)).collect()
    // }
    // fn uadv(&self) -> Vec<Column<Advice>> {}
}

//#[derive(Default)] // todo derive the Circuit, Config/FrameType & methods from a small description of the variables (or circom code)
struct MvmulCircuit<F: FieldExt, const NROWS: usize, const NCOLS: usize> {
    //    config: MvmulConfig<F, NROWS, NCOLS>,
    // not totally convinced these are needed but
    a: Vec<Vec<Value<Assigned<F>>>>,
    u: Vec<Value<Assigned<F>>>,
    v: Vec<Value<Assigned<F>>>,
    // a_00: Value<Assigned<F>>,
    // a_01: Value<Assigned<F>>,
    // a_10: Value<Assigned<F>>,
    // a_11: Value<Assigned<F>>,
    // v_0: Value<Assigned<F>>,
    // v_1: Value<Assigned<F>>,
    // u_0: Value<Assigned<F>>,
    // u_1: Value<Assigned<F>>,
}
//impl<F: FieldExt, const RANGE: usize> MvmulCircuit<F, RANGE> {}

impl<F: FieldExt, const NROWS: usize, const NCOLS: usize> Circuit<F>
    for MvmulCircuit<F, NROWS, NCOLS>
{
    type Config = MvmulConfig<F, NROWS, NCOLS>;
    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        // put Unknown in all the advice
        //        Self::default()
        let mut a: Vec<Vec<Value<Assigned<F>>>> = Vec::new();
        let mut v: Vec<Value<Assigned<F>>> = Vec::new();
        let mut u: Vec<Value<Assigned<F>>> = Vec::new();

        for i in 0..NROWS {
            let mut row: Vec<Value<Assigned<F>>> = Vec::new();
            for j in 0..NCOLS {
                row.push(Value::default());
            }
            a.push(row);
        }

        for j in 0..NCOLS {
            v.push(Value::default());
        }

        for i in 0..NROWS {
            u.push(Value::default());
        }

        MvmulCircuit { a, u, v }
    }

    // define the constraints, mutate the provided ConstraintSystem, and output the resulting FrameType
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let (q, aadv, uadv, vadv) = {
            let q = cs.selector();
            let mut aadv: Vec<Vec<Column<Advice>>> = Vec::new();
            for i in 0..NROWS {
                let mut row: Vec<Column<Advice>> = Vec::new();
                for j in 0..NCOLS {
                    row.push(cs.advice_column());
                }
                aadv.push(row);
            }
            let uadv: Vec<Column<Advice>> = (0..NROWS).map(|_| cs.advice_column()).collect();
            let vadv: Vec<Column<Advice>> = (0..NCOLS).map(|_| cs.advice_column()).collect();
            (q, aadv, uadv, vadv)
        };
        cs.create_gate("mvmul", |virtual_cells| {
            // 'allocate' all the advice cols
            let q = virtual_cells.query_selector(q);
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

            let mut c: Vec<Expression<F>> = Vec::new();
            for i in 0..NROWS {
                c.push(-u[i].clone());
            }

            for i in 0..NROWS {
                for j in 0..NCOLS {
                    c[i] = c[i].clone() + a[i][j].clone() * v[j].clone();
                }
            }

            let constraints = (1..NROWS).map(|j| "c").zip(c);

            Constraints::with_selector(q, constraints)
        });
        // The "FrameType"
        Self::Config {
            a: aadv,
            u: uadv,
            v: vadv,
            q,
            _marker: PhantomData,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        // From the function docs:
        // Assign a region of gates to an absolute row number.
        // Inside the closure, the chip may freely use relative offsets; the `Layouter` will
        // treat these assignments as a single "region" within the circuit. Outside this
        // closure, the `Layouter` is allowed to optimise as it sees fit.

        layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                // Enable q_range_check. Remember that q_range_check is a label, a Selector.  Calling its enable
                // - calls region.enable_selector(_,q_range_check,offset)  which
                // - calls enable_selector on the region's RegionLayouter which
                // - calls enable_selector on its "CS" (actually an Assignment<F> (a trait), and whatever impls that
                // does the work, for example for MockProver the enable_selector function does some checks and then sets
                //   self.selectors[selector.0][row] = true;
                config.q.enable(&mut region, offset)?;

                // Similarly after indirection calls assign_advice in e.g. the MockProver, which
                // takes a Value-producing to() and does something like
                // CellValue::Assigned(to().into_field().evaluate().assign()?);

                for i in 0..NROWS {
                    for j in 0..NCOLS {
                        region.assign_advice(
                            || format!("a_{i}_{j}"),
                            config.a[i][j],
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
    use halo2_proofs::{
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp,
        plonk::{Any, Circuit},
    };

    use super::*;

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

        // Failing
    }

    // #[test]
    // #[should_panic]
    // fn test_mvmul_fail() {
    //     let k = 4; //2^k rows
    //     let a_00: u64 = 1;
    //     let a_01: u64 = 2;
    //     let a_10: u64 = 3;
    //     let a_11: u64 = 4;
    //     let v_0: u64 = 5;
    //     let v_1: u64 = 6;
    //     let u_0: u64 = 17;
    //     let u_1: u64 = 212;

    //     let circuit = MvmulCircuit::<Fp> {
    //         a_00: Value::known(Fp::from(a_00).into()),
    //         a_01: Value::known(Fp::from(a_01).into()),
    //         a_10: Value::known(Fp::from(a_10).into()),
    //         a_11: Value::known(Fp::from(a_11).into()),
    //         v_0: Value::known(Fp::from(v_0).into()),
    //         v_1: Value::known(Fp::from(v_1).into()),
    //         u_0: Value::known(Fp::from(u_0).into()),
    //         u_1: Value::known(Fp::from(u_1).into()),
    //     };

    //     // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
    //     // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

    //     let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    //     prover.assert_satisfied();
    // }
}
