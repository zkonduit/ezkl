// This example uses a lookup table to outsource the computation of XOR-ing two 2-bit unsigned
// integers. The circuit proves knowledge of two integers `a` and `b` such that `xor(a, b) = c`
// for public input `c`. All legal input-output pairs for the XOR function are written in a public
// lookup table using fixed columns.
//
// The constraint system matrix is:
//
// |-----||-------|-------|-------|---------|-----------|-----------|-----------|
// | row || l_col | r_col | o_col | pub_col | xor_l_col | xor_r_col | xor_o_col |
// |-----||-------|-------|-------|---------|-----------|-----------|-----------|
// |  0  ||   a   |   b   |   c   |   PI    |     00    |     00    |     00    |
// |  1  ||       |       |       |         |     00    |     01    |     01    |
// |  2  ||       |       |       |         |     00    |     10    |     10    |
// |  3  ||       |       |       |         |     00    |     11    |     11    |
// |  4  ||       |       |       |         |     01    |     00    |     01    |
// |  5  ||       |       |       |         |     01    |     01    |     00    |
// |  6  ||       |       |       |         |     01    |     10    |     11    |
// |  7  ||       |       |       |         |     01    |     11    |     10    |
// |  8  ||       |       |       |         |     10    |     00    |     10    |
// |  9  ||       |       |       |         |     10    |     01    |     11    |
// |  10 ||       |       |       |         |     10    |     10    |     00    |
// |  11 ||       |       |       |         |     10    |     11    |     01    |
// |  12 ||       |       |       |         |     11    |     00    |     11    |
// |  13 ||       |       |       |         |     11    |     01    |     10    |
// |  14 ||       |       |       |         |     11    |     10    |     01    |
// |  15 ||       |       |       |         |     11    |     11    |     00    |
// |-----||-------|-------|-------|---------|-----------|-----------|-----------|
//
// where row 0 is used to allocate the prover's private inputs (`a` and `b`) and public input (`c`).
// Row 0 is also used to allocate the verifier's public input (`PI`) where consistency between the
// prover's and verifier's public inputs `c = PI` is checked using a public-input.
// Every legal input-output tuples for the 2-bit XOR function `xor(l, r) = o` is written
// into a row of the fixed columns `xor_r_col`, `xor_l_col`, and `xor_o_col`. Note that this circuit
// does not range check any of `a`, `b`, or `c` to be 2-bits, i.e. there are no polynomial
// constraints which explicity check `a, b, c ∈ {0, 1, 2, 3}`.

use std::io::Write;

use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Instance, TableColumn},
    poly::Rotation,
};

const XOR_BITS: usize = 2;

struct XorChip {
    config: XorChipConfig,
}

#[derive(Clone, Debug)]
struct XorChipConfig {
    l_col: Column<Advice>,
    r_col: Column<Advice>,
    o_col: Column<Advice>,
    xor_l_col: TableColumn,
    xor_r_col: TableColumn,
    xor_o_col: TableColumn,
    pub_col: Column<Instance>,
}

impl Chip<Fp> for XorChip {
    type Config = XorChipConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl XorChip {
    fn new(config: XorChipConfig) -> Self {
        XorChip { config }
    }

    fn configure(cs: &mut ConstraintSystem<Fp>) -> XorChipConfig {
        let l_col = cs.advice_column();
        let r_col = cs.advice_column();
        let o_col = cs.advice_column();
        let pub_col = cs.instance_column();
        // let s_pub = cs.selector();

        cs.enable_equality(l_col);
        cs.enable_equality(r_col);
        cs.enable_equality(o_col);
        cs.enable_equality(pub_col);

        let xor_l_col = cs.lookup_table_column();
        let xor_r_col = cs.lookup_table_column();
        let xor_o_col = cs.lookup_table_column();

        let _ = cs.lookup(|cs| {
            vec![
                (cs.query_advice(l_col, Rotation::cur()), xor_l_col),
                (cs.query_advice(r_col, Rotation::cur()), xor_r_col),
                (cs.query_advice(o_col, Rotation::cur()), xor_o_col),
            ]
        });

        XorChipConfig {
            l_col,
            r_col,
            o_col,
            xor_l_col,
            xor_r_col,
            xor_o_col,
            pub_col,
        }
    }

    // Allocates all legal input-output tuples for the XOR function in the first
    // `2^XOR_BITS * 2^XOR_BITS = 16` rows of the constraint system.
    fn alloc_table(&self, layouter: &mut impl Layouter<Fp>) -> Result<(), Error> {
        layouter.assign_table(
            || "xor table",
            |mut table| {
                let mut row_offset = 0;
                for l in 0..1 << XOR_BITS {
                    for r in 0..1 << XOR_BITS {
                        table.assign_cell(
                            || format!("xor_l_col row {}", row_offset),
                            self.config.xor_l_col,
                            row_offset,
                            || Value::known(Fp::from(l)),
                        )?;
                        table.assign_cell(
                            || format!("xor_r_col row {}", row_offset),
                            self.config.xor_r_col,
                            row_offset,
                            || Value::known(Fp::from(r)),
                        )?;
                        table.assign_cell(
                            || format!("xor_o_col row {}", row_offset),
                            self.config.xor_o_col,
                            row_offset,
                            || Value::known(Fp::from(l ^ r)),
                        )?;
                        row_offset += 1;
                    }
                }
                Ok(())
            },
        )
    }

    // Allocates `a`, `b`, and `c`.
    // The `l_col`, `r_col`, and `o_col` cells have not been alloacted.
    fn alloc_private_and_public_inputs(
        &self,
        layouter: &mut impl Layouter<Fp>,
        a: Value<Assigned<Fp>>,
        b: Value<Assigned<Fp>>,
        c: Value<Assigned<Fp>>,
        // a: Option<Fp>,
        // b: Option<Fp>,
        // c: Option<Fp>,
    ) -> Result<AssignedCell<Assigned<Fp>, Fp>, Error> {
        layouter.assign_region(
            || "private and public inputs",
            |mut region| {
                let row_offset = 0;
                region.assign_advice(
                    || "private input `a`",
                    self.config.l_col,
                    row_offset,
                    || a,
                )?;
                region.assign_advice(
                    || "private input `b`",
                    self.config.r_col,
                    row_offset,
                    || b,
                )?;
                let c = region.assign_advice(
                    || "public input `c`",
                    self.config.o_col,
                    row_offset,
                    || c,
                )?;
                Ok(c)
            },
        )
    }
}

// Proves knowledge of `a` and `b` such that `xor(a, b) == c` for public input `c`.
#[derive(Clone)]
struct XorCircuit {
    // Private inputs.
    a: Value<Assigned<Fp>>,
    b: Value<Assigned<Fp>>,
    // Public input (from prover).
    c: Value<Assigned<Fp>>,
}

impl Circuit<Fp> for XorCircuit {
    type Config = XorChipConfig;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(cs: &mut ConstraintSystem<Fp>) -> Self::Config {
        XorChip::configure(cs)
    }
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let xor_chip = XorChip::new(config);
        xor_chip.alloc_table(&mut layouter)?;
        let c = xor_chip.alloc_private_and_public_inputs(&mut layouter, self.a, self.b, self.c)?;

        layouter.constrain_instance(c.cell(), xor_chip.config().pub_col, 0) // equality for c and the pub_col? Why do we need the pub_col?
    }
    type FloorPlanner = SimpleFloorPlanner;
}

#[cfg(test)]
mod tests {
    use halo2_proofs::{
        circuit::floor_planner::V1,
        dev::{FailureLocation, MockProver, VerifyFailure},
        pasta::Fp,
        plonk::{Any, Circuit},
    };

    use super::*;

    #[test]
    fn test_xor1() {
        // The verifier's public input.
        const PUB_INPUT: u64 = 0;

        let k = 5;
        let pub_inputs = vec![Fp::from(PUB_INPUT)];

        // Assert that the lookup passes because `xor(2, 1) == PUB_INPUT`.
        let circuit = XorCircuit {
            a: Value::known(Fp::from(3).into()),
            b: Value::known(Fp::from(3).into()),
            c: Value::known(Fp::from(PUB_INPUT).into()),
        };

        //    use plotters::prelude::*;
        //    let root = BitMapBackend::new("layout.png", (1920, 1080)).into_drawing_area();
        //    root.fill(&WHITE).unwrap();
        //    let root = root
        //        .titled("Example Circuit Layout", ("sans-serif", 60))
        //        .unwrap();

        // halo2_proofs::dev::CircuitLayout::default()
        //     // The first argument is the size parameter for the circuit.
        //     .render(k, &circuit, &root)
        //     .unwrap();

        // let dot_string = halo2_proofs::dev::circuit_dot_graph(&circuit);
        // let mut dot_graph = std::fs::File::create("circuit.dot").unwrap();
        // dot_graph.write_all(dot_string.as_bytes()).unwrap();

        let prover = MockProver::run(k, &circuit, vec![pub_inputs.clone()]).unwrap();
        assert!(prover.verify().is_ok());

        // Assert that the public input gate is unsatisfied when `c != PUB_INPUT` (but when the lookup
        // passes).
        let bad_circuit = XorCircuit {
            a: Value::known(Fp::from(2).into()),
            b: Value::known(Fp::from(0).into()),
            c: Value::known(Fp::from(2).into()),
        };
        let prover = MockProver::run(k, &bad_circuit, vec![pub_inputs.clone()]).unwrap();
        match prover.verify() {
            Err(errors) => {
                match &errors[0] {
                    VerifyFailure::Permutation { .. } => {}
                    e => panic!("expected 'public input' gate error, found: {:?}", e),
                };
            }
            _ => panic!("expected `prover.verify()` to fail"),
        };

        // Assert that the lookup fails when `(a, b, c)` is not a row in the table; the lookup table is
        // for 2-bit XOR, using a 3-bit XOR input `a = 4` should result in a lookup failure.
        let mut bad_circuit = circuit;
        bad_circuit.c = Value::known(Fp::from(4).into());
        let prover = MockProver::run(k, &bad_circuit, vec![pub_inputs]).unwrap();
        match prover.verify() {
            Err(errors) => {
                match &errors[0] {
                    VerifyFailure::Lookup { .. } => {}
                    e => panic!("expected lookup error, found: {:?}", e),
                };
            }
            _ => panic!("expected `prover.verify()` to fail"),
        };
    }
}

// use proptest::prelude::*;
// proptest! {
//     #[test]
//     fn good_xor(a in 0..4u64, b in 0..4u64) {
//         let xored = a ^ b;
//         let circuit = XorCircuit {
//             a: Some(Fp::from(a)),
//             b: Some(Fp::from(b)),
//             c: Some(Fp::from(xored)),
//         };

//         let k = 5;
//         let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(xored)]]).unwrap();
//         assert!(prover.verify().is_ok());
//     }

//     #[test]
//     fn good_and_bad(a in 0..4u64, b in 0..4u64, c in 0..4u64) {
//         let xored = a ^ b;
//         let circuit = XorCircuit {
//             a: Some(Fp::from(a)),
//             b: Some(Fp::from(b)),
//             c: Some(Fp::from(c)),
//         };

//         let k = 5;
//         let prover = MockProver::run(k, &circuit, vec![vec![Fp::from(xored)]]).unwrap();
//         assert_eq!(prover.verify().is_ok(), c == xored);
//     }
// }
