use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Assigned, Circuit, Column,
        ConstraintSystem, Constraints, Error, Expression, Instance, Selector, TableColumn,
    },
    poly::{commitment::Params, Rotation},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use halo2curves::pasta::{pallas, vesta};
use std::{fmt::format, marker::PhantomData};

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
    pub advice: [Column<Advice>; LEN],
    table: NonlinTable<INBITS, OUTBITS>,
    qlookup: Selector,
    _marker: PhantomData<(NL, F)>,
}

impl<
        F: FieldExt,
        const LEN: usize,
        const INBITS: usize,
        const OUTBITS: usize,
        NL: 'static + Nonlinearity<F>,
    > NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL>
{
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        advice: [Column<Advice>; LEN],
    ) -> NonlinConfig1d<F, LEN, INBITS, OUTBITS, NL> {
        // for col in advice.iter() {
        //     cs.enable_equality(*col);
        // }
        let table = NonlinTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
        };

        let qlookup = cs.complex_selector();

        for i in 0..LEN {
            let _ = cs.lookup("lk", |cs| {
                let qlookup = cs.query_selector(qlookup);
                vec![
                    (
                        qlookup.clone() * cs.query_advice(advice[i], Rotation::cur()),
                        table.table_input,
                    ),
                    (
                        qlookup.clone() * cs.query_advice(advice[i], Rotation::next()),
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
                Ok(())
            },
        )
    }

    pub fn witness(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<Value<Assigned<F>>>,
    ) -> Result<Vec<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        let output_for_eq = layouter.assign_region(
            || "Elementwise", // the name of the region
            |mut region| {
                let offset = 0;
                self.qlookup.enable(&mut region, offset)?;

                let mut input_vec = Vec::new();
                //witness the advice
                for i in 0..LEN {
                    let witnessed = region.assign_advice(
                        || format!("input {:?}", i),
                        self.advice[i],
                        offset,
                        || input[i],
                    )?;
                    input_vec.push(witnessed);
                }

                println!("input_vec {:?}", input_vec);

                self.layout_inner(&mut region, offset, input_vec)
            },
        )?;

        self.alloc_table(layouter, Box::new(NL::nonlinearity))?;

        Ok(output_for_eq)
    }

    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        input: Vec<AssignedCell<Assigned<F>, F>>,
    ) -> Result<Vec<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        let output_for_eq = layouter.assign_region(
            || "Elementwise", // the name of the region
            |mut region| {
                let offset = 0;
                self.qlookup.enable(&mut region, offset)?;

                //copy the advice
                for i in 0..LEN {
                    input[i].copy_advice(|| "input", &mut region, self.advice[i], offset)?;
                }

                self.layout_inner(&mut region, offset, input.clone())
            },
        )?;

        self.alloc_table(layouter, Box::new(NL::nonlinearity))?;

        Ok(output_for_eq)
    }

    fn layout_inner(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        input: Vec<AssignedCell<Assigned<F>, F>>,
    ) -> Result<Vec<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        //calculate the value of output

        let output = input
            .iter()
            .map(|acaf| acaf.value_field())
            .map(|vaf| {
                vaf.map(|f| <NL as Nonlinearity<F>>::nonlinearity(felt_to_i32(f.evaluate())).into())
            })
            .collect::<Vec<Value<Assigned<F>>>>();

        //        println!("output {:?}", output);

        let mut output_for_equality = Vec::new();
        for i in 0..LEN {
            let ofe = region.assign_advice(
                || format!("nl_{i}"),
                self.advice[i], // Column<Advice>
                offset + 1,
                || output[i], //Assigned<F>
            )?;
            output_for_equality.push(ofe);
        }

        Ok(output_for_equality)
    }
}

#[derive(Clone)]
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
        self.clone()
        // let assigned = Nonlin1d::<F, Value<Assigned<F>>, LEN, NL>::fill(|i| Value::default());
        // Self {
        //     assigned,
        //     _marker: PhantomData,
        // }
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let advices = (0..LEN)
            .map(|_| cs.advice_column())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Self::Config::configure(cs, advices)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        config.witness(&mut layouter, self.assigned.input.clone())?;

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
        plonk::{Any, Circuit},
    };
    use halo2curves::pasta::Fp as F;

    //     use nalgebra;
    use std::time::{Duration, Instant};

    #[test]
    fn test_eltrelunl() {
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
    fn test_eltsigmoid() {
        for i in -127..127 {
            let r = <Sigmoid<F, 128, 128> as Nonlinearity<F>>::nonlinearity(i);
            //            println!("{i}, {:?}", r);
        }
    }

    #[test]
    fn test_eltdivide() {
        for i in -127..127 {
            let r = <DivideBy<F, 32> as Nonlinearity<F>>::nonlinearity(i);
            //            println!("{i}, {:?}", r);
        }
    }
}
