use super::*;
use crate::tensor::ops::activations::*;
use crate::{fieldutils::felt_to_i32, fieldutils::i32_to_felt};
use halo2_proofs::{
    arithmetic::{Field, FieldExt},
    circuit::{Layouter, Value},
    plonk::{ConstraintSystem, Expression, Selector, TableColumn},
    poly::Rotation,
};
use std::error::Error;
use std::fmt;
use std::{cell::RefCell, marker::PhantomData, rc::Rc};

#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Div {
        scale: usize,
    },
    ReLU {
        scale: usize,
    },
    LeakyReLU {
        scale: usize,
        slope: eq_float::F32,
    },
    PReLU {
        scale: usize,
        slopes: Vec<eq_float::F32>,
    },
    Sigmoid {
        scales: (usize, usize),
    },
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Div { scale } => write!(f, "div  w/ scale: {}", scale),
            Op::ReLU { scale } => write!(f, "relu w/ scale: {}", scale),
            Op::LeakyReLU { scale, slope } => {
                write!(f, "leaky-relu w/ scale: {}, slope: {}", scale, slope)
            }
            Op::PReLU { scale, slopes } => {
                write!(f, "leaky-relu w/ scale: {}, slopes: {:#?}", scale, slopes)
            }
            Op::Sigmoid { scales } => write!(f, "sigmoid  w/ scale: {}", scales.0),
        }
    }
}

impl Op {
    /// forward function
    pub fn f(&self, x: Tensor<i32>) -> Tensor<i32> {
        match &self {
            Op::Div { scale } => const_div(&x, *scale as i32),
            Op::ReLU { scale } => leakyrelu(&x, *scale, 0_f32),
            Op::LeakyReLU { scale, slope } => leakyrelu(&x, *scale, slope.0),
            Op::PReLU { scale, slopes } => leakyrelu(&x, *scale, slopes[0].0),
            Op::Sigmoid { scales } => sigmoid(&x, scales.0, scales.1),
        }
    }

    /// a value which is always in the table
    pub fn default_pair<F: FieldExt>(&self) -> (F, F) {
        let x = vec![0_i32].into_iter().into();
        (F::zero(), i32_to_felt(self.f(x)[0]))
    }
}

/// Halo2 lookup table for element wise non-linearities.
// Table that should be reused across all lookups (so no Clone)
#[derive(Clone, Debug)]
pub struct Table<F: FieldExt> {
    /// composed operations represented by the table
    pub nonlinearities: Vec<Op>,
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

impl<F: FieldExt> Table<F> {
    /// Configures the table.
    pub fn configure(cs: &mut ConstraintSystem<F>, bits: usize, nonlinearities: &[Op]) -> Table<F> {
        Table {
            nonlinearities: nonlinearities.to_vec(),
            table_input: cs.lookup_table_column(),
            table_output: cs.lookup_table_column(),
            is_assigned: false,
            bits,
            _marker: PhantomData,
        }
    }
    /// Assigns values to the constraints generated when calling `configure`.
    pub fn layout(&mut self, layouter: &mut impl Layouter<F>) -> Result<(), Box<dyn Error>> {
        if self.is_assigned {
            return Err(Box::new(CircuitError::TableAlreadyAssigned));
        }

        let base = 2i32;
        let smallest = -base.pow(self.bits as u32 - 1);
        let largest = base.pow(self.bits as u32 - 1);
        let inputs = Tensor::from(smallest..largest);
        let mut evals = inputs.clone();
        for nl in self.nonlinearities.clone() {
            evals = nl.f(inputs.clone());
        }
        self.is_assigned = true;
        layouter
            .assign_table(
                || "nl table",
                |mut table| {
                    let _ = inputs
                        .iter()
                        .enumerate()
                        .map(|(row_offset, input)| {
                            table.assign_cell(
                                || format!("nl_i_col row {}", row_offset),
                                self.table_input,
                                row_offset,
                                || Value::known(i32_to_felt::<F>(*input)),
                            )?;

                            table.assign_cell(
                                || format!("nl_o_col row {}", row_offset),
                                self.table_output,
                                row_offset,
                                || Value::known(i32_to_felt::<F>(evals[row_offset])),
                            )?;
                            Ok(())
                        })
                        .collect::<Result<Vec<()>, halo2_proofs::plonk::Error>>()?;
                    Ok(())
                },
            )
            .map_err(|e| Box::<dyn Error>::from(e))
    }
}

/// Configuration for a basic sequence of operations all fused together in a single gate.
#[derive(Clone, Debug)]
pub struct Config<F: FieldExt + TensorType> {
    /// the inputs to the lookup operations.
    pub input: VarTensor,
    /// the (currently singular) output of the fused operations.
    pub output: VarTensor,
    /// [Selector] generated when configuring the layer.
    pub qlookup: Selector,
    ///  table used to represent the non-linearity
    pub table: Rc<RefCell<Table<F>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Config<F> {
    /// Configures multiple element-wise non-linearities at once.
    pub fn configure_multiple<const NUM: usize>(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        bits: usize,
        nonlinearitities: &[Op],
    ) -> Result<[Self; NUM], Box<dyn Error>> {
        let mut table: Option<Rc<RefCell<Table<F>>>> = None;
        let mut configs: Vec<Config<F>> = vec![];
        for _ in 0..NUM {
            let l = match &table {
                None => Self::configure(cs, input, output, bits, &nonlinearitities),
                Some(t) => Self::configure_with_table(cs, input, output, t.clone()),
            };
            table = Some(l.table.clone());
            configs.push(l);
        }
        let res: [Self; NUM] = match configs.try_into() {
            Ok(a) => a,
            Err(_) => {
                return Err(Box::new(CircuitError::TableAlreadyAssigned));
            }
        };
        Ok(res)
    }

    /// Configures and creates an elementwise operation within a circuit using a supplied lookup table.
    pub fn configure_with_table(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        table: Rc<RefCell<Table<F>>>,
    ) -> Self {
        let qlookup = cs.complex_selector();

        let _ = (0..input.dims().iter().product::<usize>())
            .map(|i| {
                let _ = cs.lookup("lk", |cs| {
                    let qlookup = cs.query_selector(qlookup);
                    let not_qlookup = Expression::Constant(F::one()) - qlookup.clone();
                    let default_x = <F as Field>::zero();
                    let mut default_y = vec![0_i32].into_iter().into();
                    for nl in table.borrow().nonlinearities.clone() {
                        default_y = nl.f(default_y)
                    }
                    let default_y: F = i32_to_felt(default_y[0]);
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

impl<F: FieldExt + TensorType> Config<F> {
    /// Configures and creates an elementwise operation within a circuit.
    /// Variables are supplied as a single VarTensors.
    pub fn configure(
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        bits: usize,
        nonlinearitities: &[Op],
    ) -> Self {
        let table = Rc::new(RefCell::new(Table::<F>::configure(
            cs,
            bits,
            &nonlinearitities,
        )));
        Self::configure_with_table(cs, input, output, table)
    }

    /// Assigns values to the variables created when calling `configure`.
    /// Values are supplied as a 1-element array of `[input]` VarTensors.
    pub fn layout(
        &self,
        layouter: &mut impl Layouter<F>,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, Box<dyn Error>> {
        if !self.table.borrow().is_assigned {
            self.table.borrow_mut().layout(layouter)?
        }
        let mut t = ValTensor::from(
            match layouter.assign_region(
                || "Elementwise", // the name of the region
                |mut region| {
                    self.qlookup.enable(&mut region, 0)?;

                    let w = self.input.assign(&mut region, 0, &values)?;

                    let mut res: Vec<i32> = vec![];
                    let _ = Tensor::from(w.iter().map(|acaf| (*acaf).value_field()).map(|vaf| {
                        vaf.map(|f| {
                            res.push(felt_to_i32(f.evaluate()));
                        })
                    }));

                    // for key generation res will be empty and we need to return a set of unassigned values
                    let output: Tensor<Value<F>> = match res.len() {
                        0 => w.map(|_| Value::unknown()),
                        _ => {
                            let mut x = res.into_iter().into();
                            for nl in self.table.borrow().nonlinearities.clone() {
                                x = nl.f(x);
                            }
                            x.map(|elem| Value::known(i32_to_felt(elem)))
                        }
                    };

                    self.output.assign(&mut region, 0, &ValTensor::from(output))
                },
            ) {
                Ok(a) => a,
                Err(e) => {
                    return Err(Box::new(e));
                }
            },
        );
        t.reshape(values.dims())?;
        Ok(t)
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
        type Config = Config<F>;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..2)
                .map(|_| VarTensor::new_advice(cs, 4, 1, vec![1], true, 512))
                .collect::<Vec<_>>();

            let nl = Op::ReLU { scale: 1 };

            Self::Config::configure(cs, &advices[0], &advices[1], 2, &[nl])
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
