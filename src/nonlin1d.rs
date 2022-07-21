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

use crate::fieldutils::{self, felt_to_i32, i32tofelt};
use crate::tensorutils::flatten3;

pub trait Nonlinearity<F: FieldExt> {
    fn nonlinearity(x: i32) -> F;
}

#[derive(Clone)]
pub struct Nonlin1d<F: FieldExt, Inner, const LEN: usize, NL: Nonlinearity<F>> {
    pub input: Vec<Inner>,
    pub output: Vec<Inner>,
    pub _marker: PhantomData<(F, NL)>,
}
impl<F: FieldExt, Inner, const LEN: usize, NL: Nonlinearity<F>> Nonlin1d<F, Inner, LEN, NL> {
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
    pub fn without_witnesses() -> Nonlin1d<F, Value<Assigned<F>>, LEN, NL> {
        Nonlin1d::<F, Value<Assigned<F>>, LEN, NL>::fill(|i| Value::default())
    }
}

impl<F: FieldExt, const LEN: usize, NL: Nonlinearity<F>> Nonlin1d<F, Value<Assigned<F>>, LEN, NL> {
    pub fn from_parameters() -> Self {
        Self::without_witnesses()
    }

    pub fn forward(&mut self, input: Vec<Value<Assigned<F>>>) -> Vec<Value<Assigned<F>>> {
        let output = input
            .iter()
            .map(|x| {
                (x.map(|x| <NL as Nonlinearity<F>>::nonlinearity(felt_to_i32(x.evaluate())))).into()
            })
            .collect::<Vec<Value<Assigned<F>>>>();
        self.input = input.clone();
        self.output = output.clone();

        output
    }
}

#[derive(Clone)]
struct NonlinTable<const INBITS: usize, const OUTBITS: usize> {
    table_input: TableColumn,
    table_output: TableColumn,
}

#[derive(Clone)]
pub struct NonlinConfig1d<
    F: FieldExt,
    const LEN: usize,
    const INBITS: usize,
    const OUTBITS: usize,
    NL: Nonlinearity<F>,
> {
    pub advice: Nonlin1d<F, Column<Advice>, LEN, NL>,
    table: NonlinTable<INBITS, OUTBITS>,
    qlookup: Selector,
    _marker: PhantomData<NL>,
}

// trait NonlinFn<F> {
//     fn function() -> impl Fn(F) -> F {}
// }

impl<
        F: FieldExt,
        const LEN: usize,
        const INBITS: usize,
        const OUTBITS: usize,
        NL: 'static + Nonlinearity<F>,
    > NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL>
{
    fn define_advice(cs: &mut ConstraintSystem<F>) -> Nonlin1d<F, Column<Advice>, LEN, NL> {
        Nonlin1d::<F, Column<Advice>, LEN, NL>::fill(|i| cs.advice_column())
    }

    pub fn configure_with_input(
        input: Vec<Column<Advice>>,
        cs: &mut ConstraintSystem<F>,
    ) -> NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL> {
        let mut advice = Self::define_advice(cs);
        advice.input = input;

        let table = NonlinTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
        };

        let qlookup = cs.complex_selector();

        for i in 0..LEN {
            let _ = cs.lookup(|cs| {
                let qlookup = cs.query_selector(qlookup);
                vec![
                    (
                        qlookup.clone() * cs.query_advice(advice.input[i], Rotation::cur()),
                        table.table_input,
                    ),
                    (
                        qlookup.clone() * cs.query_advice(advice.output[i], Rotation::cur()),
                        table.table_output,
                    ),
                ]
            });
        }

        Self {
            advice,
            table,
            qlookup,
            _marker: PhantomData,
        }
    }

    /// composable_configure takes the full advice as an argument, so parts can be filled in by caller
    pub fn composable_configure(
        advice: Nonlin1d<F, Column<Advice>, LEN, NL>,
        cs: &mut ConstraintSystem<F>,
    ) -> NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL> {
        let table = NonlinTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
        };

        let qlookup = cs.complex_selector();

        for i in 0..LEN {
            let _ = cs.lookup(|cs| {
                let qlookup = cs.query_selector(qlookup);
                vec![
                    (
                        qlookup.clone() * cs.query_advice(advice.input[i], Rotation::cur()),
                        table.table_input,
                    ),
                    (
                        qlookup.clone() * cs.query_advice(advice.output[i], Rotation::cur()),
                        table.table_output,
                    ),
                ]
            });
        }

        Self {
            advice,
            table,
            qlookup,
            _marker: PhantomData,
        }
    }

    /// configure generates the advice
    pub fn configure(cs: &mut ConstraintSystem<F>) -> NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL> {
        let advice = Self::define_advice(cs);
        Self::composable_configure(advice, cs)
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
                    //println!("{}->{:?}", int_input, nonlinearity(int_input));
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
                println!("Assigned Table");
                Ok(())
            },
        )
    }

    pub fn layout(
        &self,
        assigned: &Nonlin1d<F, Value<Assigned<F>>, LEN, NL>,
        layouter: &mut impl Layouter<F>,
    ) -> Result<Vec<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        let output_for_eq = layouter.assign_region(
            || "Assign values", // the name of the region
            |mut region| {
                let offset = 0;

                self.qlookup.enable(&mut region, offset)?;

                for i in 0..LEN {
                    region.assign_advice(
                        || format!("nl_{i}"),
                        self.advice.input[i], // Column<Advice>
                        offset,
                        || assigned.input[i], //Assigned<F>
                    )?;
                }

                let mut output_for_equality = Vec::new();
                for i in 0..LEN {
                    let ofe = region.assign_advice(
                        || format!("nl_{i}"),
                        self.advice.output[i], // Column<Advice>
                        offset,
                        || assigned.output[i], //Assigned<F>
                    )?;
                    output_for_equality.push(ofe);
                }

                Ok(output_for_equality)
            },
        )?;

        // layouter.assign_region(
        //     || "Assign values", // the name of the region
        //     |mut region| {
        //         let offset = 0;

        //         Ok(())
        //     },
        // )?;

        self.alloc_table(layouter, Box::new(NL::nonlinearity))?;

        Ok(output_for_eq)
    }
}

struct NLCircuit<
    F: FieldExt,
    const LEN: usize,
    const INBITS: usize,
    const OUTBITS: usize,
    NL: Nonlinearity<F>,
> {
    assigned: Nonlin1d<F, Value<Assigned<F>>, LEN, NL>,
    _marker: PhantomData<NL>, //    nonlinearity: Box<dyn Fn(F) -> F>,
}

impl<
        F: FieldExt,
        const LEN: usize,
        const INBITS: usize,
        const OUTBITS: usize,
        NL: 'static + Nonlinearity<F> + Clone,
    > Circuit<F> for NLCircuit<F, LEN, INBITS, OUTBITS, NL>
{
    type Config = NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        let assigned = Nonlin1d::<F, Value<Assigned<F>>, LEN, NL>::fill(|i| Value::default());
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

                for i in 0..LEN {
                    region.assign_advice(
                        || format!("nl_{i}"),
                        config.advice.output[i], // Column<Advice>
                        offset,
                        || self.assigned.output[i], //Assigned<F>
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
#[derive(Clone)]
pub struct ReLu<F> {
    _marker: PhantomData<F>,
}
impl<F: FieldExt> Nonlinearity<F> for ReLu<F> {
    fn nonlinearity(x: i32) -> F {
        let out = if x < 0 { F::zero() } else { i32tofelt(x) };
        //        println!("{}->{:?}", x, out);
        out
    }
}

#[derive(Clone)]
pub struct Sigmoid<F, const L: usize, const K: usize> {
    _marker: PhantomData<F>,
}
// L is our implicit or explicit denominator (fixed point d)
// Usually want K=L
impl<F: FieldExt, const L: usize, const K: usize> Nonlinearity<F> for Sigmoid<F, L, K> {
    fn nonlinearity(x: i32) -> F {
        let kix = (x as f32) / (K as f32);
        let fout = (L as f32) / (1.0 + (-kix).exp());
        let rounded = fout.round();
        let xi: i32 = unsafe { rounded.to_int_unchecked() };
        let felt = fieldutils::i32tofelt(xi);
        //        println!("{}->{}->{}->{}->{}", x, kix, fout, rounded, xi);
        felt
    }
}

#[derive(Clone)]
pub struct DivideBy<F, const D: usize> {
    _marker: PhantomData<F>,
}
impl<F: FieldExt, const D: usize> Nonlinearity<F> for DivideBy<F, D> {
    fn nonlinearity(x: i32) -> F {
        let d_inv_x = (x as f32) / (D as f32);
        let rounded = d_inv_x.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        let felt = fieldutils::i32tofelt(integral);
        //        println!("{}->{}->{}", x, d_inv_x, integral);
        felt
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
    fn test_relunl() {
        let k = 9; //2^k rows
        let output = vec![vec![vec![1u64, 2u64], vec![3u64, 4u64]]];
        let relu_v: Vec<Value<Assigned<F>>> = flatten3(output)
            .iter()
            .map(|x| Value::known(F::from(*x).into()))
            .collect();
        let assigned: Nonlin1d<F, Value<Assigned<F>>, 4, ReLu<F>> = Nonlin1d {
            input: relu_v.clone(),
            output: relu_v,
            _marker: PhantomData,
        };

        let circuit = NLCircuit::<F, 4, 8, 8, ReLu<F>> {
            assigned,
            _marker: PhantomData,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_sigmoid() {
        for i in -127..127 {
            let r = <Sigmoid<F, 128, 128> as Nonlinearity<F>>::nonlinearity(i);
            //            println!("{i}, {:?}", r);
        }
    }

    #[test]
    fn test_divide() {
        for i in -127..127 {
            let r = <DivideBy<F, 32> as Nonlinearity<F>>::nonlinearity(i);
            //            println!("{i}, {:?}", r);
        }
    }
}
