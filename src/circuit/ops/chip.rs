use std::str::FromStr;

use thiserror::Error;

use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};
use log::debug;
#[cfg(feature = "python-bindings")]
use pyo3::{
    conversion::{FromPyObject, PyTryFrom},
    exceptions::PyValueError,
    prelude::*,
    types::PyString,
};
use serde::{Deserialize, Serialize};
use tosubcommand::ToFlags;

use crate::{
    circuit::{
        ops::base::BaseOp,
        table::{Range, RangeCheck, Table},
        utils,
    },
    tensor::{Tensor, TensorType, ValTensor, VarTensor},
};
use std::{collections::BTreeMap, error::Error, marker::PhantomData};

use super::{lookup::LookupOp, region::RegionCtx, Op};
use halo2curves::ff::{Field, PrimeField};

/// circuit related errors.
#[derive(Debug, Error)]
pub enum CircuitError {
    /// Shape mismatch in circuit construction
    #[error("dimension mismatch in circuit construction for op: {0}")]
    DimMismatch(String),
    /// Error when instantiating lookup tables
    #[error("failed to instantiate lookup tables")]
    LookupInstantiation,
    /// A lookup table was was already assigned
    #[error("attempting to initialize an already instantiated lookup table")]
    TableAlreadyAssigned,
    /// This operation is unsupported
    #[error("unsupported operation in graph")]
    UnsupportedOp,
    ///
    #[error("invalid einsum expression")]
    InvalidEinsum,
}

#[allow(missing_docs)]
/// An enum representing activating the sanity checks we can perform on the accumulated arguments
#[derive(
    Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default, Copy,
)]
pub enum CheckMode {
    #[default]
    SAFE,
    UNSAFE,
}

impl std::fmt::Display for CheckMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckMode::SAFE => write!(f, "safe"),
            CheckMode::UNSAFE => write!(f, "unsafe"),
        }
    }
}

impl ToFlags for CheckMode {
    /// Convert the struct to a subcommand string
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl From<String> for CheckMode {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "safe" => CheckMode::SAFE,
            "unsafe" => CheckMode::UNSAFE,
            _ => {
                log::error!("Invalid value for CheckMode");
                log::warn!("defaulting to SAFE");
                CheckMode::SAFE
            }
        }
    }
}

#[allow(missing_docs)]
/// An enum representing the tolerance we can accept for the accumulated arguments, either absolute or percentage
#[derive(Clone, Default, Debug, PartialEq, PartialOrd, Serialize, Deserialize, Copy)]
pub struct Tolerance {
    pub val: f32,
    pub scale: utils::F32,
}

impl std::fmt::Display for Tolerance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.2}", self.val)
    }
}

impl ToFlags for Tolerance {
    /// Convert the struct to a subcommand string
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl FromStr for Tolerance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(val) = s.parse::<f32>() {
            Ok(Tolerance {
                val,
                scale: utils::F32(1.0),
            })
        } else {
            Err(
                "Invalid tolerance value provided. It should expressed as a percentage (f32)."
                    .to_string(),
            )
        }
    }
}

impl From<f32> for Tolerance {
    fn from(value: f32) -> Self {
        Tolerance {
            val: value,
            scale: utils::F32(1.0),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Converts CheckMode into a PyObject (Required for CheckMode to be compatible with Python)
impl IntoPy<PyObject> for CheckMode {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            CheckMode::SAFE => "safe".to_object(py),
            CheckMode::UNSAFE => "unsafe".to_object(py),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains CheckMode from PyObject (Required for CheckMode to be compatible with Python)
impl<'source> FromPyObject<'source> for CheckMode {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let trystr = <PyString as PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();
        match strval.to_lowercase().as_str() {
            "safe" => Ok(CheckMode::SAFE),
            "unsafe" => Ok(CheckMode::UNSAFE),
            _ => Err(PyValueError::new_err("Invalid value for CheckMode")),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Converts Tolerance into a PyObject (Required for Tolerance to be compatible with Python)
impl IntoPy<PyObject> for Tolerance {
    fn into_py(self, py: Python) -> PyObject {
        (self.val, self.scale.0).to_object(py)
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains Tolerance from PyObject (Required for Tolerance to be compatible with Python)
impl<'source> FromPyObject<'source> for Tolerance {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok((val, scale)) = ob.extract::<(f32, f32)>() {
            Ok(Tolerance {
                val,
                scale: utils::F32(scale),
            })
        } else {
            Err(PyValueError::new_err("Invalid tolerance value provided. "))
        }
    }
}

/// Configuration for an accumulated arg.
#[derive(Clone, Debug, Default)]
pub struct BaseConfig<F: PrimeField + TensorType + PartialOrd> {
    /// the inputs to the accumulated operations.
    pub inputs: Vec<VarTensor>,
    /// the VarTensor reserved for lookup operations (could be an element of inputs)
    /// Note that you should be careful to ensure that the lookup_input is not simultaneously assigned to by other non-lookup operations eg. in the case of composite ops.
    pub lookup_input: VarTensor,
    /// the (currently singular) output of the accumulated operations.
    pub output: VarTensor,
    /// The VarTensor reserved for dynamic lookup operations (could be an element of inputs or the same as output)
    /// Note that you should be careful to ensure that the lookup_output is not simultaneously assigned to by other non-lookup operations eg. in the case of composite ops.
    pub dynamic_lookup_tables: Vec<VarTensor>,
    /// the VarTensor reserved for lookup operations (could be an element of inputs or the same as output)
    /// Note that you should be careful to ensure that the lookup_output is not simultaneously assigned to by other non-lookup operations eg. in the case of composite ops.
    pub lookup_output: VarTensor,
    ///
    pub lookup_index: VarTensor,
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure [BaseOp].
    pub selectors: BTreeMap<(BaseOp, usize, usize), Selector>,
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many lookup ops.
    pub lookup_selectors: BTreeMap<(LookupOp, usize, usize), Selector>,
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many dynamic lookup ops.
    pub dynamic_lookup_selectors: BTreeMap<(usize, usize), Vec<Selector>>,
    ///
    pub dynamic_table_selectors: Vec<Selector>,
    ///
    pub tables: BTreeMap<LookupOp, Table<F>>,
    ///
    pub range_checks: BTreeMap<Range, RangeCheck<F>>,
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many lookup ops.
    pub range_check_selectors: BTreeMap<(Range, usize, usize), Selector>,
    /// Activate sanity checks
    pub check_mode: CheckMode,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> BaseConfig<F> {
    /// Returns a new [BaseConfig] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);

        Self {
            inputs: vec![dummy_var.clone(), dummy_var.clone()],
            lookup_input: dummy_var.clone(),
            output: dummy_var.clone(),
            lookup_output: dummy_var.clone(),
            dynamic_lookup_tables: vec![VarTensor::dummy(col_size, 2), dummy_var.clone()],
            lookup_index: dummy_var,
            selectors: BTreeMap::new(),
            lookup_selectors: BTreeMap::new(),
            dynamic_lookup_selectors: BTreeMap::new(),
            dynamic_table_selectors: vec![],
            range_check_selectors: BTreeMap::new(),
            tables: BTreeMap::new(),
            range_checks: BTreeMap::new(),
            check_mode: CheckMode::SAFE,
            _marker: PhantomData,
        }
    }

    /// Configures [BaseOp]s for a given [ConstraintSystem].
    /// # Arguments
    /// * `meta` - The [ConstraintSystem] to configure the operations in.
    /// * `inputs` - The explicit inputs to the operations.
    /// * `output` - The variable representing the (currently singular) output of the operations.
    /// * `check_mode` - The variable representing the (currently singular) output of the operations.
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        inputs: &[VarTensor; 2],
        output: &VarTensor,
        check_mode: CheckMode,
    ) -> Self {
        // setup a selector per base op
        let mut nonaccum_selectors = BTreeMap::new();
        let mut accum_selectors = BTreeMap::new();

        if inputs[0].num_cols() != inputs[1].num_cols() {
            log::warn!("input shapes do not match");
        }
        if inputs[0].num_cols() != output.num_cols() {
            log::warn!("input and output shapes do not match");
        }

        for i in 0..output.num_blocks() {
            for j in 0..output.num_inner_cols() {
                nonaccum_selectors.insert((BaseOp::Add, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::Sub, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::Neg, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::Mult, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::IsZero, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::Identity, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::IsBoolean, i, j), meta.selector());
            }
        }

        for i in 0..output.num_blocks() {
            accum_selectors.insert((BaseOp::DotInit, i, 0), meta.selector());
            accum_selectors.insert((BaseOp::Dot, i, 0), meta.selector());
            accum_selectors.insert((BaseOp::CumProd, i, 0), meta.selector());
            accum_selectors.insert((BaseOp::CumProdInit, i, 0), meta.selector());
            accum_selectors.insert((BaseOp::Sum, i, 0), meta.selector());
            accum_selectors.insert((BaseOp::SumInit, i, 0), meta.selector());
        }

        for ((base_op, block_idx, inner_col_idx), selector) in nonaccum_selectors.iter() {
            meta.create_gate(base_op.as_str(), |meta| {
                let selector = meta.query_selector(*selector);

                let zero = Expression::<F>::Constant(F::ZERO);
                let mut qis = vec![zero; 2];
                for (i, q_i) in qis
                    .iter_mut()
                    .enumerate()
                    .take(2)
                    .skip(2 - base_op.num_inputs())
                {
                    *q_i = inputs[i]
                        .query_rng(meta, *block_idx, *inner_col_idx, 0, 1)
                        .expect("non accum: input query failed")[0]
                        .clone()
                }

                // Get output expressions for each input channel
                let (rotation_offset, rng) = base_op.query_offset_rng();

                let constraints = match base_op {
                    BaseOp::IsBoolean => {
                        let expected_output: Tensor<Expression<F>> = output
                            .query_rng(meta, *block_idx, *inner_col_idx, 0, 1)
                            .expect("non accum: output query failed");

                        let output = expected_output[base_op.constraint_idx()].clone();

                        vec![(output.clone()) * (output.clone() - Expression::Constant(F::from(1)))]
                    }
                    BaseOp::IsZero => {
                        let expected_output: Tensor<Expression<F>> = output
                            .query_rng(meta, *block_idx, *inner_col_idx, 0, 1)
                            .expect("non accum: output query failed");
                        vec![expected_output[base_op.constraint_idx()].clone()]
                    }
                    _ => {
                        let expected_output: Tensor<Expression<F>> = output
                            .query_rng(meta, *block_idx, *inner_col_idx, rotation_offset, rng)
                            .expect("non accum: output query failed");

                        let res = base_op.nonaccum_f((qis[0].clone(), qis[1].clone()));
                        vec![expected_output[base_op.constraint_idx()].clone() - res]
                    }
                };

                Constraints::with_selector(selector, constraints)
            });
        }

        for ((base_op, block_idx, _), selector) in accum_selectors.iter() {
            meta.create_gate(base_op.as_str(), |meta| {
                let selector = meta.query_selector(*selector);
                let mut qis = vec![vec![]; 2];
                for (i, q_i) in qis
                    .iter_mut()
                    .enumerate()
                    .take(2)
                    .skip(2 - base_op.num_inputs())
                {
                    *q_i = inputs[i]
                        .query_whole_block(meta, *block_idx, 0, 1)
                        .expect("accum: input query failed")
                        .into_iter()
                        .collect()
                }

                // Get output expressions for each input channel
                let (rotation_offset, rng) = base_op.query_offset_rng();

                let expected_output: Tensor<Expression<F>> = output
                    .query_rng(meta, *block_idx, 0, rotation_offset, rng)
                    .expect("accum: output query failed");

                let res =
                    base_op.accum_f(expected_output[0].clone(), qis[0].clone(), qis[1].clone());
                let constraints = vec![expected_output[base_op.constraint_idx()].clone() - res];

                Constraints::with_selector(selector, constraints)
            });
        }

        // selectors is the merger of nonaccum and accum selectors
        let selectors = nonaccum_selectors
            .into_iter()
            .chain(accum_selectors)
            .collect();

        Self {
            selectors,
            lookup_selectors: BTreeMap::new(),
            range_check_selectors: BTreeMap::new(),
            dynamic_lookup_selectors: BTreeMap::new(),
            inputs: inputs.to_vec(),
            lookup_input: VarTensor::Empty,
            lookup_output: VarTensor::Empty,
            lookup_index: VarTensor::Empty,
            dynamic_table_selectors: vec![],
            dynamic_lookup_tables: vec![],
            tables: BTreeMap::new(),
            range_checks: BTreeMap::new(),
            output: output.clone(),
            check_mode,
            _marker: PhantomData,
        }
    }

    /// Configures and creates lookup selectors
    #[allow(clippy::too_many_arguments)]
    pub fn configure_lookup(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        output: &VarTensor,
        index: &VarTensor,
        lookup_range: Range,
        logrows: usize,
        nl: &LookupOp,
    ) -> Result<(), Box<dyn Error>>
    where
        F: Field,
    {
        if !index.is_advice() {
            return Err("wrong input type for lookup index".into());
        }
        if !input.is_advice() {
            return Err("wrong input type for lookup input".into());
        }
        if !output.is_advice() {
            return Err("wrong input type for lookup output".into());
        }

        // we borrow mutably twice so we need to do this dance

        let table = if !self.tables.contains_key(nl) {
            // as all tables have the same input we see if there's another table who's input we can reuse
            let table = if let Some(table) = self.tables.values().next() {
                Table::<F>::configure(
                    cs,
                    lookup_range,
                    logrows,
                    nl,
                    Some(table.table_inputs.clone()),
                )
            } else {
                Table::<F>::configure(cs, lookup_range, logrows, nl, None)
            };
            self.tables.insert(nl.clone(), table.clone());
            table
        } else {
            return Ok(());
        };

        for x in 0..input.num_blocks() {
            for y in 0..input.num_inner_cols() {
                let len = table.selector_constructor.degree;

                let multi_col_selector = cs.complex_selector();

                for ((col_idx, input_col), output_col) in table
                    .table_inputs
                    .iter()
                    .enumerate()
                    .zip(table.table_outputs.iter())
                {
                    cs.lookup("", |cs| {
                        let mut res = vec![];
                        let sel = cs.query_selector(multi_col_selector);

                        let synthetic_sel = match len {
                            1 => Expression::Constant(F::from(1)),
                            _ => match index {
                                VarTensor::Advice { inner: advices, .. } => {
                                    cs.query_advice(advices[x][y], Rotation(0))
                                }
                                _ => unreachable!(),
                            },
                        };

                        let input_query = match &input {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[x][y], Rotation(0))
                            }
                            _ => unreachable!(),
                        };

                        let output_query = match &output {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[x][y], Rotation(0))
                            }
                            _ => unreachable!(),
                        };

                        // we index from 1 to avoid the zero element creating soundness issues
                        // this is 0 if the index is the same as the column index (starting from 1)

                        let col_expr = sel.clone()
                            * table
                                .selector_constructor
                                .get_expr_at_idx(col_idx, synthetic_sel);

                        let multiplier =
                            table.selector_constructor.get_selector_val_at_idx(col_idx);

                        let not_expr = Expression::Constant(multiplier) - col_expr.clone();

                        let (default_x, default_y) = table.get_first_element(col_idx);

                        log::trace!("---------------- col {:?} ------------------", col_idx,);
                        log::trace!("expr: {:?}", col_expr,);
                        log::trace!("multiplier: {:?}", multiplier);
                        log::trace!("not_expr: {:?}", not_expr);
                        log::trace!("default x: {:?}", default_x);
                        log::trace!("default y: {:?}", default_y);

                        res.extend([
                            (
                                col_expr.clone() * input_query.clone()
                                    + not_expr.clone() * Expression::Constant(default_x),
                                *input_col,
                            ),
                            (
                                col_expr.clone() * output_query.clone()
                                    + not_expr.clone() * Expression::Constant(default_y),
                                *output_col,
                            ),
                        ]);

                        res
                    });
                }
                self.lookup_selectors
                    .insert((nl.clone(), x, y), multi_col_selector);
            }
        }
        // if we haven't previously initialized the input/output, do so now
        if let VarTensor::Empty = self.lookup_input {
            debug!("assigning lookup input");
            self.lookup_input = input.clone();
        }
        if let VarTensor::Empty = self.lookup_output {
            debug!("assigning lookup output");
            self.lookup_output = output.clone();
        }
        if let VarTensor::Empty = self.lookup_index {
            debug!("assigning lookup index");
            self.lookup_index = index.clone();
        }
        Ok(())
    }

    /// Configures and creates lookup selectors
    #[allow(clippy::too_many_arguments)]
    pub fn configure_dynamic_lookup(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        lookups: &[VarTensor; 2],
        tables: &[VarTensor; 2],
    ) -> Result<(), Box<dyn Error>>
    where
        F: Field,
    {
        for l in lookups.iter() {
            if !l.is_advice() {
                return Err("wrong input type for dynamic lookup".into());
            }
        }

        for t in tables.iter() {
            if !t.is_advice() || t.num_blocks() > 1 || t.num_inner_cols() > 1 {
                return Err("wrong table type for dynamic lookup".into());
            }
        }

        let one = Expression::Constant(F::ONE);

        let s_ltable = cs.complex_selector();

        for x in 0..lookups[0].num_blocks() {
            for y in 0..lookups[0].num_inner_cols() {
                let s_lookup = cs.complex_selector();

                cs.lookup_any("lookup", |cs| {
                    let s_lookupq = cs.query_selector(s_lookup);
                    let mut expression = vec![];
                    let s_ltableq = cs.query_selector(s_ltable);
                    let mut lookup_queries = vec![one.clone()];

                    for lookup in lookups {
                        lookup_queries.push(match lookup {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[x][y], Rotation(0))
                            }
                            _ => unreachable!(),
                        });
                    }

                    let mut table_queries = vec![one.clone()];
                    for table in tables {
                        table_queries.push(match table {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[0][0], Rotation(0))
                            }
                            _ => unreachable!(),
                        });
                    }

                    let lhs = lookup_queries.into_iter().map(|c| c * s_lookupq.clone());
                    let rhs = table_queries.into_iter().map(|c| c * s_ltableq.clone());
                    expression.extend(lhs.zip(rhs));

                    expression
                });
                self.dynamic_lookup_selectors
                    .entry((x, y))
                    .or_insert_with(Vec::new)
                    .push(s_lookup);
            }
        }
        self.dynamic_table_selectors.push(s_ltable);

        // if we haven't previously initialized the input/output, do so now
        if self.dynamic_lookup_tables.is_empty() {
            debug!("assigning dynamic lookup table");
            self.dynamic_lookup_tables = tables.to_vec();
        }

        Ok(())
    }

    /// Configures and creates lookup selectors
    #[allow(clippy::too_many_arguments)]
    pub fn configure_range_check(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        input: &VarTensor,
        index: &VarTensor,
        range: Range,
        logrows: usize,
    ) -> Result<(), Box<dyn Error>>
    where
        F: Field,
    {
        if !input.is_advice() {
            return Err("wrong input type for lookup input".into());
        }

        // we borrow mutably twice so we need to do this dance

        let range_check =
            if let std::collections::btree_map::Entry::Vacant(e) = self.range_checks.entry(range) {
                // as all tables have the same input we see if there's another table who's input we can reuse
                let range_check = RangeCheck::<F>::configure(cs, range, logrows);
                e.insert(range_check.clone());
                range_check
            } else {
                return Ok(());
            };

        for x in 0..input.num_blocks() {
            for y in 0..input.num_inner_cols() {
                let len = range_check.selector_constructor.degree;
                let multi_col_selector = cs.complex_selector();

                for (col_idx, input_col) in range_check.inputs.iter().enumerate() {
                    cs.lookup("", |cs| {
                        let mut res = vec![];
                        let sel = cs.query_selector(multi_col_selector);

                        let synthetic_sel = match len {
                            1 => Expression::Constant(F::from(1)),
                            _ => match index {
                                VarTensor::Advice { inner: advices, .. } => {
                                    cs.query_advice(advices[x][y], Rotation(0))
                                }
                                _ => unreachable!(),
                            },
                        };

                        let input_query = match &input {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[x][y], Rotation(0))
                            }
                            _ => unreachable!(),
                        };

                        let default_x = range_check.get_first_element(col_idx);

                        let col_expr = sel.clone()
                            * range_check
                                .selector_constructor
                                .get_expr_at_idx(col_idx, synthetic_sel);

                        let multiplier = range_check
                            .selector_constructor
                            .get_selector_val_at_idx(col_idx);

                        let not_expr = Expression::Constant(multiplier) - col_expr.clone();

                        res.extend([(
                            col_expr.clone() * input_query.clone()
                                + not_expr.clone() * Expression::Constant(default_x),
                            *input_col,
                        )]);

                        log::trace!("---------------- col {:?} ------------------", col_idx,);
                        log::trace!("expr: {:?}", col_expr,);
                        log::trace!("multiplier: {:?}", multiplier);
                        log::trace!("not_expr: {:?}", not_expr);
                        log::trace!("default x: {:?}", default_x);

                        res
                    });
                }
                self.range_check_selectors
                    .insert((range, x, y), multi_col_selector);
            }
        }
        // if we haven't previously initialized the input/output, do so now
        if let VarTensor::Empty = self.lookup_input {
            debug!("assigning lookup input");
            self.lookup_input = input.clone();
        }

        if let VarTensor::Empty = self.lookup_index {
            debug!("assigning lookup index");
            self.lookup_index = index.clone();
        }

        Ok(())
    }

    /// layout_tables must be called before layout.
    pub fn layout_tables(&mut self, layouter: &mut impl Layouter<F>) -> Result<(), Box<dyn Error>> {
        for (i, table) in self.tables.values_mut().enumerate() {
            if !table.is_assigned {
                debug!(
                    "laying out table for {}",
                    crate::circuit::ops::Op::<F>::as_string(&table.nonlinearity)
                );
                if i == 0 {
                    table.layout(layouter, false)?;
                } else {
                    table.layout(layouter, true)?;
                }
            }
        }
        Ok(())
    }

    /// layout_range_checks must be called before layout.
    pub fn layout_range_checks(
        &mut self,
        layouter: &mut impl Layouter<F>,
    ) -> Result<(), Box<dyn Error>> {
        for range_check in self.range_checks.values_mut() {
            if !range_check.is_assigned {
                debug!("laying out range check for {:?}", range_check.range);
                range_check.layout(layouter)?;
            }
        }
        Ok(())
    }

    /// Assigns variables to the regions created when calling `configure`.
    /// # Arguments
    /// * `values` - The explicit values to the operations.
    /// * `layouter` - A Halo2 Layouter.
    /// * `op` - The operation being represented.
    pub fn layout(
        &mut self,
        region: &mut RegionCtx<F>,
        values: &[ValTensor<F>],
        op: Box<dyn Op<F>>,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        let res = op.layout(self, region, values)?;

        if matches!(&self.check_mode, CheckMode::SAFE) && !region.is_dummy() {
            if let Some(claimed_output) = &res {
                // during key generation this will be unknown vals so we use this as a flag to check
                let mut is_assigned = !claimed_output.any_unknowns()?;
                for val in values.iter() {
                    is_assigned = is_assigned && !val.any_unknowns()?;
                }
                if is_assigned {
                    op.safe_mode_check(claimed_output, values)?;
                }
            }
        };
        Ok(res)
    }
}
