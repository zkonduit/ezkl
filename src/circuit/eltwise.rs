use super::*;
use crate::abort;
use crate::fieldutils::{self, felt_to_i32, i32_to_felt};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Expression, Selector, TableColumn},
    poly::Rotation,
};
use log::error;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

pub trait Nonlinearity<F: FieldExt> {
    fn nonlinearity(x: i32, scales: &[usize]) -> F;
    /// a value which is always in the table
    fn default_pair(scales: &[usize]) -> (F, F) {
        (F::zero(), Self::nonlinearity(0, scales))
    }
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
    pub scaling_params: Vec<usize>,
    pub bits: usize,
    _marker: PhantomData<(F, NL)>,
}

impl<F: FieldExt, NL: Nonlinearity<F>> EltwiseTable<F, NL> {
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        bits: usize,
        scaling_params: &[usize],
    ) -> EltwiseTable<F, NL> {
        EltwiseTable {
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
            is_assigned: false,
            scaling_params: scaling_params.to_vec(),
            bits,
            _marker: PhantomData,
        }
    }
    pub fn layout(&mut self, layouter: &mut impl Layouter<F>) {
        assert!(!self.is_assigned);
        let base = 2i32;
        let smallest = -base.pow(self.bits as u32 - 1);
        let largest = base.pow(self.bits as u32 - 1);
        match layouter.assign_table(
            || "nl table",
            |mut table| {
                for (row_offset, int_input) in (smallest..largest).enumerate() {
                    let input: F = i32_to_felt(int_input);
                    match table.assign_cell(
                        || format!("nl_i_col row {}", row_offset),
                        self.table_input,
                        row_offset,
                        || Value::known(input),
                    ) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign table cell {:?}", e);
                        }
                    }
                    match table.assign_cell(
                        || format!("nl_o_col row {}", row_offset),
                        self.table_output,
                        row_offset,
                        || Value::known(NL::nonlinearity(int_input, &self.scaling_params)),
                    ) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign table cell {:?}", e);
                        }
                    }
                }
                Ok(())
            },
        ) {
            Ok(a) => a,
            Err(e) => {
                abort!("failed to assign elt-wise table {:?}", e);
            }
        };
        self.is_assigned = true;
    }
}

/// Configuration for element-wise non-linearities.
#[derive(Clone, Debug)]
pub struct EltwiseConfig<F: FieldExt + TensorType, NL: Nonlinearity<F>> {
    pub input: VarTensor,
    pub output: VarTensor,
    pub table: Rc<RefCell<EltwiseTable<F, NL>>>,
    qlookup: Selector,
    _marker: PhantomData<(NL, F)>,
}

impl<F: FieldExt + TensorType, NL: 'static + Nonlinearity<F>> EltwiseConfig<F, NL> {
    /// Configures multiple element-wise non-linearities at once.
    pub fn configure_multiple<const NUM: usize>(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        eltwise_params: Option<&[usize]>,
    ) -> [Self; NUM] {
        let mut table: Option<Rc<RefCell<EltwiseTable<F, NL>>>> = None;
        let configs = (0..NUM)
            .map(|_| {
                let l = match &table {
                    None => Self::configure(cs, input, output, eltwise_params),
                    Some(t) => Self::configure_with_table(cs, input, output, t.clone()),
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
        input: &VarTensor,
        output: &VarTensor,
        table: Rc<RefCell<EltwiseTable<F, NL>>>,
    ) -> Self {
        let qlookup = cs.complex_selector();
        let input_inner = match &input {
            VarTensor::Advice { inner: advices, .. } => advices,
            _ => todo!(),
        };
        let output_inner = match &output {
            VarTensor::Advice { inner: advices, .. } => advices,
            _ => todo!(),
        };

        let _ = (0..input.dims().iter().product::<usize>())
            .map(|i| {
                let _ = cs.lookup("lk", |cs| {
                    let qlookup = cs.query_selector(qlookup);
                    let not_qlookup = Expression::Constant(F::one()) - qlookup.clone();
                    let (default_x, default_y) =
                        NL::default_pair(table.borrow().scaling_params.as_slice());
                    let (x, y) = input.cartesian_coord(i);
                    vec![
                        (
                            qlookup.clone() * cs.query_advice(input_inner[x], Rotation(y as i32))
                                + not_qlookup.clone() * default_x,
                            table.borrow().table_input,
                        ),
                        (
                            qlookup * cs.query_advice(output_inner[x], Rotation(y as i32))
                                + not_qlookup * default_y,
                            table.borrow().table_output,
                        ),
                    ]
                });
            })
            .collect::<Vec<_>>();

        Self {
            input: input.clone(),
            output: output.clone(),
            table,
            qlookup,
            _marker: PhantomData,
        }
    }
}

impl<F: FieldExt + TensorType, NL: 'static + Nonlinearity<F>> EltwiseConfig<F, NL> {
    /// Configures and creates an elementwise operation within a circuit.
    /// Variables are supplied as a single VarTensors.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        eltwise_params: Option<&[usize]>,
    ) -> Self {
        // will fail if not supplied
        let params = match eltwise_params {
            Some(p) => p,
            None => {
                panic!("failed to supply eltwise parameters")
            }
        };
        let bits = params[0];
        let table = Rc::new(RefCell::new(EltwiseTable::<F, NL>::configure(
            cs,
            bits,
            &params[1..],
        )));
        Self::configure_with_table(cs, input, output, table)
    }

    /// Assigns values to the variables created when calling `configure`.
    /// Values are supplied as a 1-element array of `[input]` VarTensors.
    pub fn layout(&self, layouter: &mut impl Layouter<F>, values: ValTensor<F>) -> ValTensor<F> {
        if !self.table.borrow().is_assigned {
            self.table.borrow_mut().layout(layouter)
        }
        let mut t = ValTensor::from(
            match layouter.assign_region(
                || "Elementwise", // the name of the region
                |mut region| {
                    self.qlookup.enable(&mut region, 0)?;

                    let w = self.input.assign(&mut region, 0, &values).unwrap();

                    let output: Tensor<Value<F>> =
                        Tensor::from(w.iter().map(|acaf| (*acaf).value_field()).map(|vaf| {
                            vaf.map(|f| {
                                <NL as Nonlinearity<F>>::nonlinearity(
                                    felt_to_i32(f.evaluate()),
                                    &self.table.borrow().scaling_params,
                                )
                            })
                        }));

                    Ok(self
                        .output
                        .assign(&mut region, 0, &ValTensor::from(output))
                        .unwrap())
                },
            ) {
                Ok(a) => a,
                Err(e) => {
                    abort!("failed to assign elt-wise region {:?}", e);
                }
            },
        );
        t.reshape(values.dims());
        t
    }
}

// Now implement nonlinearity functions like this
#[derive(Clone, Debug)]
pub struct ReLu<F> {
    _marker: PhantomData<F>,
}
impl<F: FieldExt> Nonlinearity<F> for ReLu<F> {
    fn nonlinearity(x: i32, scale: &[usize]) -> F {
        if x < 0 {
            F::zero()
        } else {
            let d_inv_x = (x as f32) / (scale[0] as f32);
            let rounded = d_inv_x.round();
            let integral: i32 = unsafe { rounded.to_int_unchecked() };
            i32_to_felt(integral)
        }
    }
}

#[derive(Clone, Debug)]
pub struct LeakyReLu<F> {
    _marker: PhantomData<F>,
}

impl<F: FieldExt> Nonlinearity<F> for LeakyReLu<F> {
    fn nonlinearity(x: i32, scale: &[usize]) -> F {
        if x < 0 {
            let d_inv_x = (0.05) * (x as f32) / (scale[0] as f32);
            let rounded = d_inv_x.round();
            let integral: i32 = unsafe { rounded.to_int_unchecked() };
            i32_to_felt(integral)
        } else {
            i32_to_felt(x)
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
    fn nonlinearity(x: i32, scale: &[usize]) -> F {
        let kix = (x as f32) / (scale[0] as f32);
        let fout = (scale[1] as f32) / (1.0 + (-kix).exp());
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
    fn nonlinearity(x: i32, scale: &[usize]) -> F {
        let d_inv_x = (x as f32) / (scale[0] as f32);
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
            let _r = <ReLu<F> as Nonlinearity<F>>::nonlinearity(i, &[1]);
        }
    }

    #[test]
    fn test_eltsigmoid() {
        for i in -127..127 {
            let _r = <Sigmoid<F> as Nonlinearity<F>>::nonlinearity(i, &[1, 1]);
        }
    }

    #[test]
    fn test_eltdivide() {
        for i in -127..127 {
            let _r = <DivideBy<F> as Nonlinearity<F>>::nonlinearity(i, &[1]);
        }
    }
}
