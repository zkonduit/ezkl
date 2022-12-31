use super::*;
use crate::tensor::ops::activations::*;
use crate::{abort, fieldutils::felt_to_i32, fieldutils::i32_to_felt};
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

impl fmt::Display for EltwiseOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EltwiseOp::Div { scale } => write!(f, "div  w/ scale: {}", scale),
            EltwiseOp::ReLU { scale } => write!(f, "relu w/ scale: {}", scale),
            EltwiseOp::LeakyReLU { scale, slope } => {
                write!(f, "leaky-relu w/ scale: {}, slope: {}", scale, slope)
            }
            EltwiseOp::Sigmoid { scales } => write!(f, "sigmoid  w/ scale: {}", scales.0),
        }
    }
}

impl EltwiseOp {
    fn f(&self, x: Tensor<i32>) -> Tensor<i32> {
        match &self {
            EltwiseOp::Div { scale } => const_div(&x, *scale as i32),
            EltwiseOp::ReLU { scale } => leakyrelu(&x, *scale, 0_f32),
            EltwiseOp::LeakyReLU { scale, slope } => leakyrelu(&x, *scale, slope.0),
            EltwiseOp::Sigmoid { scales } => sigmoid(&x, scales.0, scales.1),
        }
    }

    /// a value which is always in the table
    fn default_pair<F: FieldExt>(&self) -> (F, F) {
        (
            F::zero(),
            i32_to_felt(self.f(vec![0_i32].into_iter().into())[0]),
        )
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
        let inputs = Tensor::from(smallest..largest);
        let evals = self.nonlinearity.f(inputs.clone());
        layouter
            .assign_table(
                || "nl table",
                |mut table| {
                    inputs
                        .enum_map(|row_offset, input| {
                            table
                                .assign_cell(
                                    || format!("nl_i_col row {}", row_offset),
                                    self.table_input,
                                    row_offset,
                                    || Value::known(i32_to_felt::<F>(input)),
                                )
                                .expect("failed to assign table input cell");

                            table
                                .assign_cell(
                                    || format!("nl_o_col row {}", row_offset),
                                    self.table_output,
                                    row_offset,
                                    || Value::known(i32_to_felt::<F>(evals[row_offset])),
                                )
                                .expect("failed to assign table output cell");
                        })
                        .expect("failed to assign table");
                    Ok(())
                },
            )
            .expect("failed to layout table");
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
    pub fn layout(&self, layouter: &mut impl Layouter<F>, values: &ValTensor<F>) -> ValTensor<F> {
        if !self.table.borrow().is_assigned {
            self.table.borrow_mut().layout(layouter)
        }
        let mut t = ValTensor::from(
            match layouter.assign_region(
                || "Elementwise", // the name of the region
                |mut region| {
                    self.qlookup.enable(&mut region, 0)?;

                    let w = self.input.assign(&mut region, 0, &values).unwrap();

                    let mut res: Vec<i32> = vec![];
                    let _ = Tensor::from(w.iter().map(|acaf| (*acaf).value_field()).map(|vaf| {
                        vaf.map(|f| {
                            res.push(felt_to_i32(f.evaluate()));
                        })
                    }));

                    // for key generation res will be empty and we need to return a set of unassigned values
                    let output: Tensor<Value<F>> = match res.len() {
                        0 => w.map(|_| Value::unknown()),
                        _ => self
                            .table
                            .borrow()
                            .nonlinearity
                            .f(res.into_iter().into())
                            .map(|elem| Value::known(i32_to_felt(elem))),
                    };

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
            let _ = config.layout(&mut layouter, &self.input);

            Ok(())
        }
    }

    #[test]
    fn relucircuit() {
        let input: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::<F>::known(F::from(1_u64))]), &[1]).unwrap();

        let circuit = ReLUCircuit::<F> {
            input: ValTensor::from(input),
        };

        let prover = MockProver::run(4_u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }
}
