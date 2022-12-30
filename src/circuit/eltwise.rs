use super::*;
use crate::abort;
use crate::fieldutils::{felt_to_i32, i32_to_felt};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Expression, Selector, TableColumn},
    poly::Rotation,
};
use log::error;
use std::fmt;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

#[allow(missing_docs)]
/// An enum representing the operations that can be merged into a single circuit gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EltwiseOp {
    Sigmoid { scales: (usize, usize) },
    LeakyReLU { scale: usize, slope: eq_float::F32 },
    ReLU { scale: usize },
    Div { scale: usize },
}

impl EltwiseOp {
    fn f<F: FieldExt>(&self, x: i32) -> F {
        match &self {
            EltwiseOp::Sigmoid { scales } => {
                let kix = (x as f32) / (scales.0 as f32);
                let fout = (scales.1 as f32) / (1.0 + (-kix).exp());
                let rounded = fout.round();
                i32_to_felt(rounded as i32)
            }
            EltwiseOp::LeakyReLU { scale, slope } => {
                if x < 0 {
                    let d_inv_x = slope.0 * (x as f32) / (*scale as f32);
                    let rounded = d_inv_x.round();
                    i32_to_felt(rounded as i32)
                } else {
                    i32_to_felt(x)
                }
            }
            EltwiseOp::ReLU { scale } => {
                if x < 0 {
                    F::zero()
                } else {
                    let d_inv_x = (x as f32) / (*scale as f32);
                    let rounded = d_inv_x.round();
                    i32_to_felt(rounded as i32)
                }
            }
            EltwiseOp::Div { scale } => {
                let d_inv_x = (x as f32) / (*scale as f32);
                let rounded = d_inv_x.round();
                i32_to_felt(rounded as i32)
            }
        }
    }

    /// a value which is always in the table
    fn default_pair<F: FieldExt>(&self) -> (F, F) {
        (F::zero(), self.f(0))
    }
}

impl fmt::Display for EltwiseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EltwiseOp::ReLU { scale } => write!(f, "relu w/ scaling: {}", scale),
            EltwiseOp::LeakyReLU { scale, slope } => {
                write!(f, "leaky relu w/ scaling: {} and slope {}", scale, slope)
            }
            EltwiseOp::Div { scale } => write!(f, "div  w/ scaling: {}", scale),
            EltwiseOp::Sigmoid { scales } => write!(f, "sigmoid  w/ scaling: {}", scales.0),
        }
    }
}

/// Halo2 lookup table for element wise non-linearities.
// Table that should be reused across all lookups (so no Clone)
#[derive(Clone, Debug)]
pub struct EltwiseTable<F: FieldExt> {
    /// nonlinearity represented by the table
    pub nonlinearity: EltwiseOp,
    /// Input to table.
    pub table_input: TableColumn,
    /// Output of table
    pub table_output: TableColumn,
    /// Flags if table has been previously assigned to.
    pub is_assigned: bool,
    /// Number of bits used in lookup table.
    pub bits: usize,
    _marker: PhantomData<F>,
}

impl<F: FieldExt> EltwiseTable<F> {
    /// Configures the table.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        bits: usize,
        nonlinearity: EltwiseOp,
    ) -> EltwiseTable<F> {
        EltwiseTable {
            nonlinearity,
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
            is_assigned: false,
            bits,
            _marker: PhantomData,
        }
    }
    /// Assigns values to the constraints generated when calling `configure`.
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
                        || Value::known(self.nonlinearity.f::<F>(int_input)),
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
pub struct EltwiseConfig<F: FieldExt + TensorType> {
    /// [VarTensor] input to non-linearity.
    pub input: VarTensor,
    /// [VarTensor] input to non-linearity.
    pub output: VarTensor,
    /// Lookup table used to represent the non-linearity
    pub table: Rc<RefCell<EltwiseTable<F>>>,
    qlookup: Selector,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> EltwiseConfig<F> {
    /// Configures multiple element-wise non-linearities at once.
    pub fn configure_multiple<const NUM: usize>(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        bits: usize,
        nonlinearity: EltwiseOp,
    ) -> [Self; NUM] {
        let mut table: Option<Rc<RefCell<EltwiseTable<F>>>> = None;
        let configs = (0..NUM)
            .map(|_| {
                let l = match &table {
                    None => Self::configure(cs, input, output, bits, nonlinearity),
                    Some(t) => Self::configure_with_table(cs, input, output, t.clone()),
                };
                table = Some(l.table.clone());
                l
            })
            .collect::<Vec<EltwiseConfig<F>>>()
            .try_into();

        match configs {
            Ok(x) => x,
            Err(_) => panic!("failed to initialize layers"),
        }
    }

    /// Configures and creates an elementwise operation within a circuit using a supplied lookup table.
    pub fn configure_with_table(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        table: Rc<RefCell<EltwiseTable<F>>>,
    ) -> Self {
        let qlookup = cs.complex_selector();

        let _ = (0..input.dims().iter().product::<usize>())
            .map(|i| {
                let _ = cs.lookup("lk", |cs| {
                    let qlookup = cs.query_selector(qlookup);
                    let not_qlookup = Expression::Constant(F::one()) - qlookup.clone();
                    let (default_x, default_y) = table.borrow().nonlinearity.default_pair::<F>();
                    let (x, y) = input.cartesian_coord(i);
                    vec![
                        (
                            match &input {
                                VarTensor::Advice { inner: advices, .. } => {
                                    qlookup.clone()
                                        * cs.query_advice(advices[x], Rotation(y as i32))
                                        + not_qlookup.clone() * default_x
                                }
                                VarTensor::Fixed { inner: fixed, .. } => {
                                    qlookup.clone() * cs.query_fixed(fixed[x], Rotation(y as i32))
                                        + not_qlookup.clone() * default_x
                                }
                            },
                            table.borrow().table_input,
                        ),
                        (
                            match &output {
                                VarTensor::Advice { inner: advices, .. } => {
                                    qlookup * cs.query_advice(advices[x], Rotation(y as i32))
                                        + not_qlookup * default_y
                                }
                                VarTensor::Fixed { inner: fixed, .. } => {
                                    qlookup * cs.query_fixed(fixed[x], Rotation(y as i32))
                                        + not_qlookup * default_y
                                }
                            },
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

impl<F: FieldExt + TensorType> EltwiseConfig<F> {
    /// Configures and creates an elementwise operation within a circuit.
    /// Variables are supplied as a single VarTensors.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        bits: usize,
        nonlinearity: EltwiseOp,
    ) -> Self {
        let table = Rc::new(RefCell::new(EltwiseTable::<F>::configure(
            cs,
            bits,
            nonlinearity,
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
                                self.table
                                    .borrow()
                                    .nonlinearity
                                    .f(felt_to_i32(f.evaluate()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        arithmetic::FieldExt,
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };
    use halo2curves::pasta::Fp as F;

    #[derive(Clone)]
    struct ReLUCircuit<F: FieldExt + TensorType> {
        pub input: ValTensor<F>,
    }

    impl<F: FieldExt + TensorType> Circuit<F> for ReLUCircuit<F> {
        type Config = EltwiseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, 4, 1, vec![1], true, 512))
                .collect::<Vec<_>>();

            let nl = EltwiseOp::ReLU { scale: 1 };

            Self::Config::configure(cs, &advices[0], &advices[1], 2, nl)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            let _ = config.layout(&mut layouter, self.input.clone());

            Ok(())
        }
    }

    #[test]
    fn test_eltrelunl() {
        let nl = EltwiseOp::ReLU { scale: 1 };
        for i in -127..127 {
            let r: F = nl.f(i);
            if i <= 0 {
                assert_eq!(r, F::from(0_u64))
            } else {
                assert_eq!(r, F::from(i as u64))
            }
        }
    }

    #[test]
    fn test_eltleakyrelunl() {
        let nl = EltwiseOp::LeakyReLU {
            scale: 1,
            slope: eq_float::F32(0.05),
        };
        for i in -127..127 {
            let r: F = nl.f(i);
            if i <= 0 {
                println!("{:?}", (0.05 * i as f32));
                assert_eq!(r, -F::from(-(0.05 * i as f32).round() as u64))
            } else {
                assert_eq!(r, F::from(i as u64))
            }
        }
    }

    #[test]
    fn test_eltsigmoid() {
        let nl = EltwiseOp::Sigmoid { scales: (1, 1) };
        for i in -127..127 {
            let r: F = nl.f(i);
            let exp_sig = (1.0 / (1.0 + (-i as f32).exp())).round();
            assert_eq!(r, F::from(exp_sig as u64))
        }
    }

    #[test]
    fn test_eltdivide() {
        let nl = EltwiseOp::Div { scale: 1 };
        for i in -127..127 {
            let r: F = nl.f(i);
            println!("{:?}, {:?}, {:?}", i, r, F::from(-i as u64));
            if i <= 0 {
                assert_eq!(r, -F::from(-i as u64))
            } else {
                assert_eq!(r, F::from(i as u64))
            }
        }
    }

    #[test]
    fn relucircuit() {
        let input: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::<F>::known(F::from(1_u64))]), &[1]).unwrap();

        let circuit = ReLUCircuit::<F> {
            input: ValTensor::from(input.clone()),
        };

        let prover = MockProver::run(4_u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
