use super::*;
use crate::fieldutils::{self, felt_to_i32, i32_to_felt};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Selector, TableColumn},
    poly::Rotation,
};
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

pub trait Nonlinearity<F: FieldExt> {
    fn nonlinearity(x: i32, scales: usize) -> F;
}

#[derive(Clone, Debug)]
pub struct Nonlin1d<F: FieldExt + TensorType, NL: Nonlinearity<F>> {
    pub input: ValTensor<F>,
    pub output: ValTensor<F>,
    pub _marker: PhantomData<(F, NL)>,
}

/// Halo2 lookup table for element wise non-linearities.
// Table that should be reused across all lookups (so no Clone)
#[derive(Clone, Debug)]
pub struct EltwiseTable<F: FieldExt, NL: Nonlinearity<F>> {
    pub table_input: TableColumn,
    pub table_output: TableColumn,
    pub is_assigned: bool,
    pub scale: usize,
    pub bits: usize,
    _marker: PhantomData<(F, NL)>,
}

impl<F: FieldExt, NL: Nonlinearity<F>> EltwiseTable<F, NL> {
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        bits: usize,
        scale: usize,
    ) -> EltwiseTable<F, NL> {
        EltwiseTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
            is_assigned: false,
            scale,
            bits,
            _marker: PhantomData,
        }
    }
    pub fn layout(&mut self, layouter: &mut impl Layouter<F>) {
        assert!(!self.is_assigned);
        let base = 2i32;
        let smallest = -base.pow(self.bits as u32 - 1);
        let largest = base.pow(self.bits as u32 - 1);
        layouter
            .assign_table(
                || "nl table",
                |mut table| {
                    for (row_offset, int_input) in (smallest..largest).enumerate() {
                        let input: F = i32_to_felt(int_input);
                        table
                            .assign_cell(
                                || format!("nl_i_col row {}", row_offset),
                                self.table_input,
                                row_offset,
                                || Value::known(input),
                            )
                            .unwrap();
                        table
                            .assign_cell(
                                || format!("nl_o_col row {}", row_offset),
                                self.table_output,
                                row_offset,
                                || Value::known(NL::nonlinearity(int_input, self.scale)),
                            )
                            .unwrap();
                    }
                    Ok(())
                },
            )
            .unwrap();
        self.is_assigned = true;
    }
}

/// Configuration for element-wise non-linearities.
#[derive(Clone, Debug)]
pub struct EltwiseConfig<F: FieldExt + TensorType, NL: Nonlinearity<F>> {
    pub input: VarTensor,
    pub table: Rc<RefCell<EltwiseTable<F, NL>>>,
    qlookup: Selector,
    _marker: PhantomData<(NL, F)>,
}

impl<F: FieldExt + TensorType, NL: 'static + Nonlinearity<F>> EltwiseConfig<F, NL> {
    /// Configures multiple element-wise non-linearities at once.
    pub fn configure_multiple<const NUM: usize>(
        cs: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        eltwise_params: Option<&[usize]>,
    ) -> [Self; NUM] {
        let mut table: Option<Rc<RefCell<EltwiseTable<F, NL>>>> = None;
        let configs = (0..NUM)
            .map(|_| {
                let l = match &table {
                    None => Self::configure(cs, variables, eltwise_params),
                    Some(t) => Self::configure_with_table(cs, variables, t.clone()),
                };
                table = Some(l.table.clone());
                l
            })
            .collect::<Vec<EltwiseConfig<F, NL>>>()
            .try_into();

        match configs {
            Ok(x) => x,
            Err(_) => panic!("failed to initialize layers"),
        }
    }

    /// Configures and creates an elementwise operation within a circuit using a supplied lookup table.
    fn configure_with_table(
        cs: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        table: Rc<RefCell<EltwiseTable<F, NL>>>,
    ) -> Self {
        let qlookup = cs.complex_selector();
        let input = variables[0].clone();
        match &input {
            VarTensor::Advice {
                inner: advice,
                dims: _,
            } => {
                advice.map(|a| {
                    let _ = cs.lookup("lk", |cs| {
                        let qlookup = cs.query_selector(qlookup);
                        vec![
                            (
                                qlookup.clone() * cs.query_advice(a, Rotation::cur()),
                                table.borrow().table_input,
                            ),
                            (
                                qlookup * cs.query_advice(a, Rotation::next()),
                                table.borrow().table_output,
                            ),
                        ]
                    });
                });
            }
            _ => todo!(),
        }

        Self {
            input,
            table,
            qlookup,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt + TensorType, NL: 'static + Nonlinearity<F>> LayerConfig<F>
    for EltwiseConfig<F, NL>
{
    /// Configures and creates an elementwise operation within a circuit.
    /// Variables are supplied as a 1-element array of `[input]` VarTensors.
    fn configure(
        cs: &mut ConstraintSystem<F>,
        variables: &[VarTensor],
        eltwise_params: Option<&[usize]>,
    ) -> Self {
        // will fail if not supplied
        let params = eltwise_params.unwrap();
        assert_eq!(params.len(), 2);
        let (bits, scale) = (params[0], params[1]);
        let table = Rc::new(RefCell::new(EltwiseTable::<F, NL>::configure(
            cs, bits, scale,
        )));
        Self::configure_with_table(cs, variables, table)
    }

    /// Assigns values to the variables created when calling `configure`.
    /// Values are supplied as a 1-element array of `[input]` VarTensors.
    fn layout(&self, layouter: &mut impl Layouter<F>, values: &[ValTensor<F>]) -> ValTensor<F> {
        if !self.table.borrow().is_assigned {
            self.table.borrow_mut().layout(layouter)
        }
        let mut t = ValTensor::from(
            layouter
                .assign_region(
                    || "Elementwise", // the name of the region
                    |mut region| {
                        let offset = 0;
                        self.qlookup.enable(&mut region, offset)?;

                        let w = match &values[0] {
                            ValTensor::AssignedValue { inner: v, dims: _ } => match &self.input {
                                VarTensor::Advice {
                                    inner: advice,
                                    dims: _,
                                } => v.enum_map(|i, x| {
                                    // assign the advice
                                    region
                                        .assign_advice(|| "input", advice[i], offset, || x)
                                        .unwrap()
                                }),
                                _ => todo!(),
                            },
                            ValTensor::PrevAssigned { inner: v, dims: _ } => match &self.input {
                                VarTensor::Advice {
                                    inner: advice,
                                    dims: _,
                                } =>
                                //copy the advice
                                {
                                    v.enum_map(|i, x| {
                                        x.copy_advice(|| "input", &mut region, advice[i], offset)
                                            .unwrap()
                                    })
                                }
                                _ => todo!(),
                            },
                            ValTensor::Value { inner: v, dims: _ } => match &self.input {
                                VarTensor::Advice {
                                    inner: advice,
                                    dims: _,
                                } => v.enum_map(|i, x| {
                                    // assign the advice
                                    region
                                        .assign_advice(|| "input", advice[i], offset, || x.into())
                                        .unwrap()
                                }),
                                _ => todo!(),
                            },
                        };

                        let output =
                            Tensor::from(w.iter().map(|acaf| acaf.value_field()).map(|vaf| {
                                vaf.map(|f| {
                                    <NL as Nonlinearity<F>>::nonlinearity(
                                        felt_to_i32(f.evaluate()),
                                        self.table.borrow().scale,
                                    )
                                    .into()
                                })
                            }));

                        match &self.input {
                            VarTensor::Advice {
                                inner: advice,
                                dims: _,
                            } => Ok(output.enum_map(|i, o| {
                                region
                                    .assign_advice(|| format!("nl_{i}"), advice[i], 1, || o)
                                    .unwrap()
                            })),

                            _ => todo!(),
                        }
                    },
                )
                .unwrap(),
        );
        t.reshape(values[0].dims());
        t
    }
}

// Now implement nonlinearity functions like this
#[derive(Clone, Debug)]
pub struct ReLu<F> {
    _marker: PhantomData<F>,
}
impl<F: FieldExt> Nonlinearity<F> for ReLu<F> {
    fn nonlinearity(x: i32, scale: usize) -> F {
        if x < 0 {
            F::zero()
        } else {
            let d_inv_x = (x as f32) / (scale as f32);
            let rounded = d_inv_x.round();
            let integral: i32 = unsafe { rounded.to_int_unchecked() };
            i32_to_felt(integral)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sigmoid<F> {
    _marker: PhantomData<F>,
}
// L is our implicit or explicit denominator (fixed point d)
// Usually want K=L
impl<F: FieldExt> Nonlinearity<F> for Sigmoid<F> {
    fn nonlinearity(x: i32, scale: usize) -> F {
        let kix = (x as f32) / (scale as f32);
        let fout = (128 as f32) / (1.0 + (-kix).exp());
        let rounded = fout.round();
        let xi: i32 = unsafe { rounded.to_int_unchecked() };
        fieldutils::i32_to_felt(xi)
    }
}

#[derive(Clone, Debug)]
pub struct DivideBy<F> {
    _marker: PhantomData<F>,
}
impl<F: FieldExt> Nonlinearity<F> for DivideBy<F> {
    fn nonlinearity(x: i32, scale: usize) -> F {
        let d_inv_x = (x as f32) / (scale as f32);
        let rounded = d_inv_x.round();
        let integral: i32 = unsafe { rounded.to_int_unchecked() };
        fieldutils::i32_to_felt(integral)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::pasta::Fp as F;

    #[test]
    fn test_eltrelunl() {
        for i in -127..127 {
            let _r = <ReLu<F> as Nonlinearity<F>>::nonlinearity(i, 32);
        }
    }

    #[test]
    fn test_eltsigmoid() {
        for i in -127..127 {
            let _r = <Sigmoid<F> as Nonlinearity<F>>::nonlinearity(i, 32);
        }
    }

    #[test]
    fn test_eltdivide() {
        for i in -127..127 {
            let _r = <DivideBy<F> as Nonlinearity<F>>::nonlinearity(i, 32);
        }
    }
}
