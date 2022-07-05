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
struct MvmulConfig<F: FieldExt> {
    // Av = u
    a_00: Column<Advice>,
    a_01: Column<Advice>,
    a_10: Column<Advice>,
    a_11: Column<Advice>,
    v_0: Column<Advice>,
    v_1: Column<Advice>,
    u_0: Column<Advice>,
    u_1: Column<Advice>,
    q_range_check: Selector, // similarly a marker and index for a Selector
    _marker: PhantomData<F>,
}
// By convention the Config gets a configure and assign method, which are delegated to by the configure and synthesize method of the Circuit.
//impl<F: FieldExt, const RANGE: usize> MvmulConfig<F, RANGE> {}

#[derive(Default)] // todo derive the Circuit, Config/FrameType & methods from a small description of the variables (or circom code)
struct MvmulCircuit<F: FieldExt> {
    a_00: Value<Assigned<F>>,
    a_01: Value<Assigned<F>>,
    a_10: Value<Assigned<F>>,
    a_11: Value<Assigned<F>>,
    v_0: Value<Assigned<F>>,
    v_1: Value<Assigned<F>>,
    u_0: Value<Assigned<F>>,
    u_1: Value<Assigned<F>>,
}
//impl<F: FieldExt, const RANGE: usize> MvmulCircuit<F, RANGE> {}

impl<F: FieldExt> Circuit<F> for MvmulCircuit<F> {
    type Config = MvmulConfig<F>;
    type FloorPlanner = V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    // define the constraints, mutate the provided ConstraintSystem, and output the resulting FrameType
    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        // Create the column marker types. Requests the CS to allocate a new column (giving it a unique cs-global index and incrementing its
        // num_selectors, num_fixed_columns, num_advice_columns, or num_instance_columns).
        let a_00 = cs.advice_column();
        let a_01 = cs.advice_column();
        let a_10 = cs.advice_column();
        let a_11 = cs.advice_column();
        let v_0 = cs.advice_column();
        let v_1 = cs.advice_column();
        let u_0 = cs.advice_column();
        let u_1 = cs.advice_column();

        let q_range_check = cs.selector();

        // When we use cs.query_advice or cs.query_selector, we obtain an Expression which is a reference to a cell in the matrix.
        //         Expression::Advice {
        //    query_index: self.meta.query_advice_index(column, at),
        //    column_index: column.index,
        //    rotation: at,
        //}
        // Such an a_{ij} or a_{this_row + at, column} can be treated as a symbolic variable and put into a polynomial constraint.
        // More precisely, this is a relative reference wrt rows.

        // cs.create_gate takes a function from virtual_cells to contraints, pushing the constraints to the cs's accumulator.  So this puts
        // (value.clone()) * (1 - value.clone()) * (2 - value.clone()) * ... * (R - 1 - value.clone())
        // into the constraint list.
        cs.create_gate("mvmul", |virtual_cells| {
            let q = virtual_cells.query_selector(q_range_check);
            let a_00 = virtual_cells.query_advice(a_00, Rotation::cur());
            let a_01 = virtual_cells.query_advice(a_01, Rotation::cur());
            let a_10 = virtual_cells.query_advice(a_10, Rotation::cur());
            let a_11 = virtual_cells.query_advice(a_11, Rotation::cur());
            let v_0 = virtual_cells.query_advice(v_0, Rotation::cur());
            let v_1 = virtual_cells.query_advice(v_1, Rotation::cur());
            let u_0 = virtual_cells.query_advice(u_0, Rotation::cur());
            let u_1 = virtual_cells.query_advice(u_1, Rotation::cur());

            // Given a range R and a value v, returns the expression
            // (v) * (1 - v) * (2 - v) * ... * (R - 1 - v)
            let polynomial1 = -u_0 + a_00 * v_0.clone() + a_01 * v_1.clone();
            let polynomial2 = -u_1 + a_10 * v_0 + a_11 * v_1;

            Constraints::with_selector(q, [("p1", polynomial1), ("p2", polynomial2)])
        });

        // The "FrameType"
        Self::Config {
            a_00,
            a_01,
            a_10,
            a_11,
            v_0,
            v_1,
            u_0,
            u_1,
            q_range_check,
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
                config.q_range_check.enable(&mut region, offset)?;

                // Similarly after indirection calls assign_advice in e.g. the MockProver, which
                // takes a Value-producing to() and does something like
                // CellValue::Assigned(to().into_field().evaluate().assign()?);
                region.assign_advice(|| "a_00", config.a_00, offset, || self.a_00)?;
                region.assign_advice(|| "a_01", config.a_01, offset, || self.a_01)?;
                region.assign_advice(|| "a_10", config.a_10, offset, || self.a_10)?;
                region.assign_advice(|| "a_10", config.a_11, offset, || self.a_11)?;
                region.assign_advice(|| "v_0", config.v_0, offset, || self.v_0)?;
                region.assign_advice(|| "v_1", config.v_1, offset, || self.v_1)?;
                region.assign_advice(|| "u_0", config.u_0, offset, || self.u_0)?;
                region.assign_advice(|| "u_1", config.u_1, offset, || self.u_1)?;
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

        let circuit = MvmulCircuit::<Fp> {
            a_00: Value::known(Fp::from(a_00).into()),
            a_01: Value::known(Fp::from(a_01).into()),
            a_10: Value::known(Fp::from(a_10).into()),
            a_11: Value::known(Fp::from(a_11).into()),
            v_0: Value::known(Fp::from(v_0).into()),
            v_1: Value::known(Fp::from(v_1).into()),
            u_0: Value::known(Fp::from(u_0).into()),
            u_1: Value::known(Fp::from(u_1).into()),
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();

        // Failing
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

        let circuit = MvmulCircuit::<Fp> {
            a_00: Value::known(Fp::from(a_00).into()),
            a_01: Value::known(Fp::from(a_01).into()),
            a_10: Value::known(Fp::from(a_10).into()),
            a_11: Value::known(Fp::from(a_11).into()),
            v_0: Value::known(Fp::from(v_0).into()),
            v_1: Value::known(Fp::from(v_1).into()),
            u_0: Value::known(Fp::from(u_0).into()),
            u_1: Value::known(Fp::from(u_1).into()),
        };

        // The MockProver arguments are log_2(nrows), the circuit (with advice already assigned), and the instance variables.
        // The MockProver will need to internally supply a Layouter for the constraint system to be actually written.

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
