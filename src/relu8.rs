use std::io::Write;

use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value},
    dev::{MockProver, VerifyFailure},
    pasta::Fp,
    plonk::{Advice, Assigned, Circuit, Column, ConstraintSystem, Error, Instance, TableColumn},
    poly::Rotation,
};

//const XOR_BITS: usize = 2;

struct ReluChip {
    config: ReluChipConfig,
}

#[derive(Clone, Debug)]
struct ReluChipConfig {
    i_col: Column<Advice>,
    o_col: Column<Advice>,
    relu_i_col: TableColumn,
    relu_o_col: TableColumn,
    pub_col: Column<Instance>,
}

impl Chip<Fp> for ReluChip {
    type Config = ReluChipConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl ReluChip {
    fn new(config: ReluChipConfig) -> Self {
        ReluChip { config }
    }

    fn configure(cs: &mut ConstraintSystem<Fp>) -> ReluChipConfig {
        let i_col = cs.advice_column();
        let o_col = cs.advice_column();
        let pub_col = cs.instance_column();
        // let s_pub = cs.selector();

        cs.enable_equality(i_col);
        cs.enable_equality(o_col);
        cs.enable_equality(pub_col);

        let relu_i_col = cs.lookup_table_column();
        let relu_o_col = cs.lookup_table_column();

        let _ = cs.lookup(|cs| {
            vec![
                (cs.query_advice(i_col, Rotation::cur()), relu_i_col),
                (cs.query_advice(o_col, Rotation::cur()), relu_o_col),
            ]
        });

        ReluChipConfig {
            i_col,
            o_col,
            relu_i_col,
            relu_o_col,
            pub_col,
        }
    }

    // Allocates all legal input-output tuples for the ReLu function in the first 2^16 rows
    // of the constraint system.
    fn alloc_table(&self, layouter: &mut impl Layouter<Fp>) -> Result<(), Error> {
        layouter.assign_table(
            || "relu table",
            |mut table| {
                let mut row_offset = 0;
                let shift = Fp::from(127);
                for input in 0..255 {
                    table.assign_cell(
                        || format!("relu_i_col row {}", row_offset),
                        self.config.relu_i_col,
                        row_offset,
                        || Value::known(Fp::from(input) - shift), //-127..127
                    )?;
                    table.assign_cell(
                        || format!("relu_o_col row {}", row_offset),
                        self.config.relu_o_col,
                        row_offset,
                        || Value::known(Fp::from(if input < 127 { 127 } else { input }) - shift),
                    )?;
                    row_offset += 1;
                }
                Ok(())
            },
        )
    }

    // Allocates `a` (private input) and `c` (public copy of output)
    fn alloc_private_and_public_inputs(
        &self,
        layouter: &mut impl Layouter<Fp>,
        a: Value<Assigned<Fp>>,
        c: Value<Assigned<Fp>>,
    ) -> Result<AssignedCell<Assigned<Fp>, Fp>, Error> {
        layouter.assign_region(
            || "private and public inputs",
            |mut region| {
                let row_offset = 0;
                region.assign_advice(
                    || "private input `a`",
                    self.config.i_col,
                    row_offset,
                    || a,
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

// Proves knowledge of `a` such that `relu(a) == c` for public input `c`.
#[derive(Clone)]
struct ReluCircuit {
    // Private inputs.
    a: Value<Assigned<Fp>>,
    // Public input (from prover).
    c: Value<Assigned<Fp>>,
}

impl Circuit<Fp> for ReluCircuit {
    type Config = ReluChipConfig;

    fn without_witnesses(&self) -> Self {
        todo!()
    }

    fn configure(cs: &mut ConstraintSystem<Fp>) -> Self::Config {
        ReluChip::configure(cs)
    }
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let relu_chip = ReluChip::new(config);
        relu_chip.alloc_table(&mut layouter)?;
        let c = relu_chip.alloc_private_and_public_inputs(&mut layouter, self.a, self.c)?;

        layouter.constrain_instance(c.cell(), relu_chip.config().pub_col, 0) // equality for c and the pub_col? Why do we need the pub_col?
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
    fn test_relu1() {
        // The verifier's public input.
        const PUB_INPUT: u64 = 0;

        let k = 9;
        let pub_inputs = vec![Fp::from(PUB_INPUT)];

        // Assert that the lookup passes because `relu(-3) == PUB_INPUT`.
        let circuit = ReluCircuit {
            a: Value::known((-Fp::from(3)).into()),
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
        let bad_circuit = ReluCircuit {
            a: Value::known(Fp::from(2).into()),
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

        // Assert that the lookup fails when `(a, c)` is not a row in the table;
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

    #[test]
    fn test_relu_all_good() {
        let shift = Fp::from(127);
        for ewe8 in 0..255 {
            let pi = if (ewe8 as i32) - 127 < 0 {
                Fp::from(0)
            } else {
                Fp::from(ewe8 - 127)
            };
            let k = 9;
            let pub_inputs = vec![pi];

            let circuit = ReluCircuit {
                a: Value::known((Fp::from(ewe8) - shift).into()),
                c: Value::known(pi.into()),
            };

            let prover = MockProver::run(k, &circuit, vec![pub_inputs.clone()]).unwrap();
            assert!(prover.verify().is_ok());
        }
    }
}
