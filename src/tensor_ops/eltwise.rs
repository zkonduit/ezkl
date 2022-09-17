use crate::fieldutils::{self, felt_to_i32, i32tofelt};
use crate::nn::*;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value},
    plonk::{Assigned, Circuit, ConstraintSystem, Error, Selector, TableColumn},
    poly::Rotation,
};
use std::{marker::PhantomData, rc::Rc};
pub trait Nonlinearity<F: FieldExt> {
    fn nonlinearity(x: i32) -> F;
}

#[derive(Clone)]
pub struct Nonlin1d<F: FieldExt + TensorType, const LEN: usize, NL: Nonlinearity<F>> {
    pub input: IOType<F>,
    pub output: IOType<F>,
    pub _marker: PhantomData<(F, NL)>,
}
impl<F: FieldExt + TensorType, const LEN: usize, NL: Nonlinearity<F>> Nonlin1d<F, LEN, NL> {
    pub fn fill<Func>(mut f: Func) -> Self
    where
        Func: FnMut(Tensor<usize>) -> IOType<F>,
    {
        Nonlin1d {
            input: f(Tensor::from(0..LEN)),
            output: f(Tensor::from(0..LEN)),
            _marker: PhantomData,
        }
    }
    pub fn without_witnesses() -> Nonlin1d<F, LEN, NL> {
        Nonlin1d::<F, LEN, NL>::fill(|x| IOType::Value(x.map(|_| Value::default())))
    }
}

// Table that should be reused across all lookups (so no Clone)
#[derive(Clone)]
pub struct EltwiseTable<F: FieldExt, const BITS: usize, NL: Nonlinearity<F>> {
    pub table_input: TableColumn,
    pub table_output: TableColumn,
    _marker: PhantomData<(F, NL)>,
}

impl<F: FieldExt, const BITS: usize, NL: Nonlinearity<F>> EltwiseTable<F, BITS, NL> {
    pub fn configure(cs: &mut ConstraintSystem<F>) -> EltwiseTable<F, BITS, NL> {
        EltwiseTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
            _marker: PhantomData,
        }
    }
    pub fn layout(&self, layouter: &mut impl Layouter<F>) -> Result<(), Error> {
        let base = 2i32;
        let smallest = -base.pow(BITS as u32 - 1);
        let largest = base.pow(BITS as u32 - 1);
        layouter.assign_table(
            || "nl table",
            |mut table| {
                for (row_offset, int_input) in (smallest..largest).enumerate() {
                    let input: F = i32tofelt(int_input);
                    table.assign_cell(
                        || format!("nl_i_col row {}", row_offset),
                        self.table_input,
                        row_offset,
                        || Value::known(input),
                    )?;
                    table.assign_cell(
                        || format!("nl_o_col row {}", row_offset),
                        self.table_output,
                        row_offset,
                        || Value::known(NL::nonlinearity(int_input)),
                    )?;
                }
                Ok(())
            },
        )
    }
}

#[derive(Clone)]
pub struct EltwiseConfig<F: FieldExt + TensorType, const BITS: usize, NL: Nonlinearity<F>> {
    pub input: ParamType,
    pub table: Rc<EltwiseTable<F, BITS, NL>>,
    qlookup: Selector,
    _marker: PhantomData<(NL, F)>,
}

impl<F: FieldExt + TensorType, const BITS: usize, NL: 'static + Nonlinearity<F>>
    EltwiseConfig<F, BITS, NL>
{
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        input: ParamType,
        table: Option<Rc<EltwiseTable<F, BITS, NL>>>,
    ) -> EltwiseConfig<F, BITS, NL> {
        let qlookup = cs.complex_selector();

        let table = match table {
            Some(t) => t,
            None => {
                Rc::new(EltwiseTable::<F, BITS, NL>::configure(cs))
            }
        };

        match &input {
            ParamType::Advice(advice) => {
                advice.map(|a| {
                    let _ = cs.lookup("lk", |cs| {
                        let qlookup = cs.query_selector(qlookup);
                        vec![
                            (
                                qlookup.clone() * cs.query_advice(a, Rotation::cur()),
                                table.table_input,
                            ),
                            (
                                qlookup * cs.query_advice(a, Rotation::next()),
                                table.table_output,
                            ),
                        ]
                    });
                });
            }
            _ => panic!("not yet implemented"),
        }

        Self {
            input,
            table,
            qlookup,
            _marker: PhantomData,
        }
    }

    fn assign(
        &self,
        layouter: &mut impl Layouter<F>,
        input: IOType<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        layouter
            .assign_region(
                || "Elementwise", // the name of the region
                |mut region| {
                    let offset = 0;
                    self.qlookup.enable(&mut region, offset)?;

                    let w = match &input {
                        IOType::AssignedValue(v) => match &self.input {
                            ParamType::Advice(advice) => v.enum_map(|i, x| {
                                // assign the advice
                                region
                                    .assign_advice(|| "input", advice[i], offset, || x)
                                    .unwrap()
                            }),
                            _ => panic!("not yet implemented"),
                        },
                        IOType::PrevAssigned(v) => match &self.input {
                            ParamType::Advice(advice) =>
                            //copy the advice
                            {
                                v.enum_map(|i, x| {
                                    x.copy_advice(|| "input", &mut region, advice[i], offset)
                                        .unwrap()
                                })
                            }
                            _ => panic!("not yet implemented"),
                        },
                        IOType::Value(v) => match &self.input {
                            ParamType::Advice(advice) => v.enum_map(|i, x| {
                                // assign the advice
                                region
                                    .assign_advice(|| "input", advice[i], offset, || x.into())
                                    .unwrap()
                            }),
                            _ => panic!("not yet implemented"),
                        },
                    };

                    let output = Tensor::from(w.iter().map(|acaf| acaf.value_field()).map(|vaf| {
                        vaf.map(|f| {
                            <NL as Nonlinearity<F>>::nonlinearity(felt_to_i32(f.evaluate())).into()
                        })
                    }));

                    match &self.input {
                        ParamType::Advice(advice) => Ok(output.enum_map(|i, o| {
                            region
                                .assign_advice(|| format!("nl_{i}"), advice[i], 1, || o)
                                .unwrap()
                        })),

                        _ => panic!("not yet implemented"),
                    }
                },
            )
            .unwrap()
    }

    pub fn layout(&self, layouter: &mut impl Layouter<F>, input: IOType<F>) -> IOType<F> {
        IOType::PrevAssigned(self.assign(layouter, input))
    }
}

#[derive(Clone)]
struct NLCircuit<F: FieldExt + TensorType, const LEN: usize, const BITS: usize, NL: Nonlinearity<F>>
{
    assigned: Nonlin1d<F, LEN, NL>,
    _marker: PhantomData<NL>, //    nonlinearity: Box<dyn Fn(F) -> F>,
}

impl<
        F: FieldExt + TensorType,
        const LEN: usize,
        const BITS: usize,
        NL: 'static + Nonlinearity<F> + Clone,
    > Circuit<F> for NLCircuit<F, LEN, BITS, NL>
{
    type Config = EltwiseConfig<F, BITS, NL>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let advices = ParamType::Advice((0..LEN).map(|_| cs.advice_column()).into());
        Self::Config::configure(cs, advices, None)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
    ) -> Result<(), Error> {
        config.table.layout(&mut layouter)?;
        config.layout(&mut layouter, self.assigned.input.clone());

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
        if x < 0 {
            F::zero()
        } else {
            i32tofelt(x)
        }
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
        fieldutils::i32tofelt(xi)
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
        fieldutils::i32tofelt(integral)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;
    use halo2curves::pasta::Fp as F;

    #[test]
    fn test_eltrelunl() {
        let k = 9; //2^k rows
        let output = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[4]).unwrap();
        let relu_v: Tensor<Value<F>> = output.into();
        let assigned: Nonlin1d<F, 4, ReLu<F>> = Nonlin1d {
            input: IOType::Value(relu_v.clone().into()),
            output: IOType::Value(relu_v.into()),
            _marker: PhantomData,
        };

        let circuit = NLCircuit::<F, 4, 8, ReLu<F>> {
            assigned,
            _marker: PhantomData,
        };
        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_eltsigmoid() {
        for i in -127..127 {
            let _r = <Sigmoid<F, 128, 128> as Nonlinearity<F>>::nonlinearity(i);
            //            println!("{i}, {:?}", r);
        }
    }

    #[test]
    fn test_eltdivide() {
        for i in -127..127 {
            let _r = <DivideBy<F, 32> as Nonlinearity<F>>::nonlinearity(i);
            //            println!("{i}, {:?}", r);
        }
    }
}
