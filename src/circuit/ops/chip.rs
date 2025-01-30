use std::str::FromStr;

use halo2_proofs::{
    circuit::Layouter,
    plonk::{ConstraintSystem, Constraints, Expression, Selector, TableColumn},
    poly::Rotation,
};
use log::debug;
#[cfg(feature = "python-bindings")]
use pyo3::{
    conversion::{FromPyObject, IntoPy},
    exceptions::PyValueError,
    prelude::*,
};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToFlags;

use crate::{
    circuit::{
        ops::base::BaseOp,
        table::{Range, RangeCheck, Table},
        utils,
    },
    tensor::{Tensor, TensorType, ValTensor, VarTensor},
};
use std::{collections::BTreeMap, marker::PhantomData};

use super::{lookup::LookupOp, region::RegionCtx, CircuitError, Op};
use halo2curves::ff::{Field, PrimeField};

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

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for CheckMode {
    /// Convert the struct to a subcommand string
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl From<String> for CheckMode {
    fn from(value: String) -> Self {
        Self::from_str(value.as_str()).unwrap()
    }
}

impl FromStr for CheckMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safe" => Ok(CheckMode::SAFE),
            "unsafe" => Ok(CheckMode::UNSAFE),
            _ => Err("Invalid value for CheckMode".to_string()),
        }
    }
}

impl CheckMode {
    /// Returns the value of the check mode
    pub fn is_safe(&self) -> bool {
        match self {
            CheckMode::SAFE => true,
            CheckMode::UNSAFE => false,
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

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
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
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let trystr = String::extract_bound(ob)?;
        match trystr.to_lowercase().as_str() {
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
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok((val, scale)) = <(f32, f32)>::extract_bound(ob) {
            Ok(Tolerance {
                val,
                scale: utils::F32(scale),
            })
        } else {
            Err(PyValueError::new_err("Invalid tolerance value provided. "))
        }
    }
}

/// A struct representing the selectors for the dynamic lookup tables
#[derive(Clone, Debug, Default)]
pub struct DynamicLookups {
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many dynamic lookup ops.
    pub lookup_selectors: BTreeMap<(usize, (usize, usize)), Selector>,
    /// Selectors for the dynamic lookup tables
    pub table_selectors: Vec<Selector>,
    /// Inputs:
    pub inputs: Vec<VarTensor>,
    /// tables
    pub tables: Vec<VarTensor>,
}

impl DynamicLookups {
    /// Returns a new [DynamicLookups] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        let single_col_dummy_var = VarTensor::dummy(col_size, 1);

        Self {
            lookup_selectors: BTreeMap::new(),
            table_selectors: vec![],
            inputs: vec![dummy_var.clone(), dummy_var.clone(), dummy_var.clone()],
            tables: vec![
                single_col_dummy_var.clone(),
                single_col_dummy_var.clone(),
                single_col_dummy_var.clone(),
            ],
        }
    }
}

/// A struct representing the selectors for the dynamic lookup tables
#[derive(Clone, Debug, Default)]

pub struct Shuffles {
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many dynamic lookup ops.
    pub input_selectors: BTreeMap<(usize, (usize, usize)), Selector>,
    /// Selectors for the dynamic lookup tables
    pub output_selectors: Vec<Selector>,
    /// Inputs:
    pub inputs: Vec<VarTensor>,
    /// tables
    pub outputs: Vec<VarTensor>,
}

impl Shuffles {
    /// Returns a new [DynamicLookups] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        let single_col_dummy_var = VarTensor::dummy(col_size, 1);

        Self {
            input_selectors: BTreeMap::new(),
            output_selectors: vec![],
            inputs: vec![dummy_var.clone(), dummy_var.clone(), dummy_var.clone()],
            outputs: vec![
                single_col_dummy_var.clone(),
                single_col_dummy_var.clone(),
                single_col_dummy_var.clone(),
            ],
        }
    }
}

/// A struct representing the selectors for the static lookup tables
#[derive(Clone, Debug, Default)]
pub struct StaticLookups<F: PrimeField + TensorType + PartialOrd> {
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many dynamic lookup ops.
    pub selectors: BTreeMap<(LookupOp, usize, usize), Selector>,
    /// Selectors for the dynamic lookup tables
    pub tables: BTreeMap<LookupOp, Table<F>>,
    ///
    pub index: VarTensor,
    ///
    pub output: VarTensor,
    ///
    pub input: VarTensor,
}

impl<F: PrimeField + TensorType + PartialOrd> StaticLookups<F> {
    /// Returns a new [StaticLookups] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);

        Self {
            selectors: BTreeMap::new(),
            tables: BTreeMap::new(),
            index: dummy_var.clone(),
            output: dummy_var.clone(),
            input: dummy_var,
        }
    }
}

/// A struct representing the selectors for custom gates
#[derive(Clone, Debug, Default)]
pub struct CustomGates {
    /// the inputs to the accumulated operations.
    pub inputs: Vec<VarTensor>,
    /// the (currently singular) output of the accumulated operations.
    pub output: VarTensor,
    /// selector
    pub selectors: BTreeMap<(BaseOp, usize, usize), Selector>,
}

impl CustomGates {
    /// Returns a new [CustomGates] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        Self {
            inputs: vec![dummy_var.clone(), dummy_var.clone()],
            output: dummy_var,
            selectors: BTreeMap::new(),
        }
    }
}

/// A struct representing the selectors for the range checks
#[derive(Clone, Debug, Default)]
pub struct RangeChecks<F: PrimeField + TensorType + PartialOrd> {
    /// [Selector]s generated when configuring the layer. We use a [BTreeMap] as we expect to configure many dynamic lookup ops.
    pub selectors: BTreeMap<(Range, usize, usize), Selector>,
    /// Selectors for the dynamic lookup tables
    pub ranges: BTreeMap<Range, RangeCheck<F>>,
    ///
    pub index: VarTensor,
    ///
    pub input: VarTensor,
}

impl<F: PrimeField + TensorType + PartialOrd> RangeChecks<F> {
    /// Returns a new [RangeChecks] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        Self {
            selectors: BTreeMap::new(),
            ranges: BTreeMap::new(),
            index: dummy_var.clone(),
            input: dummy_var,
        }
    }
}

/// Configuration for an accumulated arg.
#[derive(Clone, Debug, Default)]
pub struct BaseConfig<F: PrimeField + TensorType + PartialOrd> {
    /// Custom gates
    pub custom_gates: CustomGates,
    /// StaticLookups
    pub static_lookups: StaticLookups<F>,
    /// [Selector]s for the dynamic lookup tables
    pub dynamic_lookups: DynamicLookups,
    /// [Selector]s for the range checks
    pub range_checks: RangeChecks<F>,
    /// [Selector]s for the shuffles
    pub shuffles: Shuffles,
    /// Activate sanity checks
    pub check_mode: CheckMode,
    _marker: PhantomData<F>,
    /// shared table inputs
    pub shared_table_inputs: Vec<TableColumn>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> BaseConfig<F> {
    /// Returns a new [BaseConfig] with no inputs, no selectors, and no tables.
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        Self {
            custom_gates: CustomGates::dummy(col_size, num_inner_cols),
            static_lookups: StaticLookups::dummy(col_size, num_inner_cols),
            dynamic_lookups: DynamicLookups::dummy(col_size, num_inner_cols),
            shuffles: Shuffles::dummy(col_size, num_inner_cols),
            range_checks: RangeChecks::dummy(col_size, num_inner_cols),
            check_mode: CheckMode::SAFE,
            shared_table_inputs: vec![],
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
        if inputs[0].num_inner_cols() != inputs[1].num_inner_cols() {
            log::warn!("input number of inner columns do not match");
        }
        if inputs[0].num_inner_cols() != output.num_inner_cols() {
            log::warn!("input and output number of inner columns do not match");
        }

        for i in 0..output.num_blocks() {
            for j in 0..output.num_inner_cols() {
                nonaccum_selectors.insert((BaseOp::Add, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::Sub, i, j), meta.selector());
                nonaccum_selectors.insert((BaseOp::Mult, i, j), meta.selector());
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
            custom_gates: CustomGates {
                inputs: inputs.to_vec(),
                output: output.clone(),
                selectors,
            },
            static_lookups: StaticLookups::default(),
            dynamic_lookups: DynamicLookups::default(),
            shuffles: Shuffles::default(),
            range_checks: RangeChecks::default(),
            shared_table_inputs: vec![],
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
    ) -> Result<(), CircuitError>
    where
        F: Field,
    {
        if !index.is_advice() {
            return Err(CircuitError::WrongColumnType(index.name().to_string()));
        }
        if !input.is_advice() {
            return Err(CircuitError::WrongColumnType(input.name().to_string()));
        }
        if !output.is_advice() {
            return Err(CircuitError::WrongColumnType(output.name().to_string()));
        }

        let table = if !self.static_lookups.tables.contains_key(nl) {
            let table =
                Table::<F>::configure(cs, lookup_range, logrows, nl, &mut self.shared_table_inputs);
            self.static_lookups.tables.insert(nl.clone(), table.clone());
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
                            * (table
                                .selector_constructor
                                .get_expr_at_idx(col_idx, synthetic_sel));

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

                // add a degree-k custom constraint of the following form to the range check and
                // static lookup configuration.
                // 𝑚𝑢𝑙𝑡𝑖𝑠𝑒𝑙 · ∏ (𝑠𝑒𝑙 − 𝑖) = 0 where 𝑠𝑒𝑙 is the synthetic_sel, and the product is over the set of overflowed columns
                // and 𝑚𝑢𝑙𝑡𝑖𝑠𝑒𝑙 is the selector value at the column index
                cs.create_gate("range_check_on_sel", |cs| {
                    let synthetic_sel = match len {
                        1 => Expression::Constant(F::from(1)),
                        _ => match index {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[x][y], Rotation(0))
                            }
                            _ => unreachable!(),
                        },
                    };

                    let range_check_on_synthetic_sel = match len {
                        1 => Expression::Constant(F::from(0)),
                        _ => {
                            let mut initial_expr = Expression::Constant(F::from(1));
                            for i in 0..len {
                                initial_expr = initial_expr
                                    * (synthetic_sel.clone()
                                        - Expression::Constant(F::from(i as u64)))
                            }
                            initial_expr
                        }
                    };

                    let sel = cs.query_selector(multi_col_selector);

                    Constraints::with_selector(sel, vec![range_check_on_synthetic_sel])
                });

                self.static_lookups
                    .selectors
                    .insert((nl.clone(), x, y), multi_col_selector);
            }
        }
        // if we haven't previously initialized the input/output, do so now
        if let VarTensor::Empty = self.static_lookups.input {
            debug!("assigning lookup input");
            self.static_lookups.input = input.clone();
        }
        if let VarTensor::Empty = self.static_lookups.output {
            debug!("assigning lookup output");
            self.static_lookups.output = output.clone();
        }
        if let VarTensor::Empty = self.static_lookups.index {
            debug!("assigning lookup index");
            self.static_lookups.index = index.clone();
        }
        Ok(())
    }

    /// Configures and creates lookup selectors
    #[allow(clippy::too_many_arguments)]
    pub fn configure_dynamic_lookup(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        lookups: &[VarTensor; 3],
        tables: &[VarTensor; 3],
    ) -> Result<(), CircuitError>
    where
        F: Field,
    {
        for l in lookups.iter() {
            if !l.is_advice() {
                return Err(CircuitError::WrongDynamicColumnType(l.name().to_string()));
            }
        }

        for t in tables.iter() {
            if !t.is_advice() || t.num_inner_cols() > 1 {
                return Err(CircuitError::WrongDynamicColumnType(t.name().to_string()));
            }
        }

        // assert all tables have the same number of inner columns
        if tables
            .iter()
            .map(|t| t.num_blocks())
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w[0] != w[1])
        {
            return Err(CircuitError::WrongDynamicColumnType(
                "tables inner cols".to_string(),
            ));
        }

        let one = Expression::Constant(F::ONE);

        for q in 0..tables[0].num_blocks() {
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
                                    cs.query_advice(advices[q][0], Rotation(0))
                                }
                                _ => unreachable!(),
                            });
                        }

                        let lhs = lookup_queries.into_iter().map(|c| c * s_lookupq.clone());
                        let rhs = table_queries.into_iter().map(|c| c * s_ltableq.clone());
                        expression.extend(lhs.zip(rhs));

                        expression
                    });
                    self.dynamic_lookups
                        .lookup_selectors
                        .entry((q, (x, y)))
                        .or_insert(s_lookup);
                }
            }

            self.dynamic_lookups.table_selectors.push(s_ltable);
        }

        // if we haven't previously initialized the input/output, do so now
        if self.dynamic_lookups.tables.is_empty() {
            debug!("assigning dynamic lookup table");
            self.dynamic_lookups.tables = tables.to_vec();
        }
        if self.dynamic_lookups.inputs.is_empty() {
            debug!("assigning dynamic lookup input");
            self.dynamic_lookups.inputs = lookups.to_vec();
        }

        Ok(())
    }

    /// Configures and creates lookup selectors
    #[allow(clippy::too_many_arguments)]
    pub fn configure_shuffles(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        inputs: &[VarTensor; 3],
        outputs: &[VarTensor; 3],
    ) -> Result<(), CircuitError>
    where
        F: Field,
    {
        for l in inputs.iter() {
            if !l.is_advice() {
                return Err(CircuitError::WrongDynamicColumnType(l.name().to_string()));
            }
        }

        for t in outputs.iter() {
            if !t.is_advice() || t.num_inner_cols() > 1 {
                return Err(CircuitError::WrongDynamicColumnType(t.name().to_string()));
            }
        }

        // assert all tables have the same number of blocks
        if outputs
            .iter()
            .map(|t| t.num_blocks())
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w[0] != w[1])
        {
            return Err(CircuitError::WrongDynamicColumnType(
                "outputs inner cols".to_string(),
            ));
        }

        let one = Expression::Constant(F::ONE);

        for q in 0..outputs[0].num_blocks() {
            let s_output = cs.complex_selector();

            for x in 0..inputs[0].num_blocks() {
                for y in 0..inputs[0].num_inner_cols() {
                    let s_input = cs.complex_selector();

                    cs.lookup_any("shuffle", |cs| {
                        let s_inputq = cs.query_selector(s_input);
                        let mut expression = vec![];
                        let s_outputq = cs.query_selector(s_output);
                        let mut input_queries = vec![one.clone()];

                        for input in inputs {
                            input_queries.push(match input {
                                VarTensor::Advice { inner: advices, .. } => {
                                    cs.query_advice(advices[x][y], Rotation(0))
                                }
                                _ => unreachable!(),
                            });
                        }

                        let mut output_queries = vec![one.clone()];
                        for output in outputs {
                            output_queries.push(match output {
                                VarTensor::Advice { inner: advices, .. } => {
                                    cs.query_advice(advices[q][0], Rotation(0))
                                }
                                _ => unreachable!(),
                            });
                        }

                        let lhs = input_queries.into_iter().map(|c| c * s_inputq.clone());
                        let rhs = output_queries.into_iter().map(|c| c * s_outputq.clone());
                        expression.extend(lhs.zip(rhs));

                        expression
                    });
                    self.shuffles
                        .input_selectors
                        .entry((q, (x, y)))
                        .or_insert(s_input);
                }
            }
            self.shuffles.output_selectors.push(s_output);
        }

        // if we haven't previously initialized the input/output, do so now
        if self.shuffles.outputs.is_empty() {
            debug!("assigning shuffles output");
            self.shuffles.outputs = outputs.to_vec();
        }
        if self.shuffles.inputs.is_empty() {
            debug!("assigning shuffles input");
            self.shuffles.inputs = inputs.to_vec();
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
    ) -> Result<(), CircuitError>
    where
        F: Field,
    {
        if !input.is_advice() {
            return Err(CircuitError::WrongColumnType(input.name().to_string()));
        }

        // we borrow mutably twice so we need to do this dance

        let range_check = if let std::collections::btree_map::Entry::Vacant(e) =
            self.range_checks.ranges.entry(range)
        {
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
                            * (range_check
                                .selector_constructor
                                .get_expr_at_idx(col_idx, synthetic_sel));

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

                // add a degree-k custom constraint of the following form to the range check and
                // static lookup configuration.
                // 𝑚𝑢𝑙𝑡𝑖𝑠𝑒𝑙 · ∏ (𝑠𝑒𝑙 − 𝑖) = 0 where 𝑠𝑒𝑙 is the synthetic_sel, and the product is over the set of overflowed columns
                // and 𝑚𝑢𝑙𝑡𝑖𝑠𝑒𝑙 is the selector value at the column index
                cs.create_gate("range_check_on_sel", |cs| {
                    let synthetic_sel = match len {
                        1 => Expression::Constant(F::from(1)),
                        _ => match index {
                            VarTensor::Advice { inner: advices, .. } => {
                                cs.query_advice(advices[x][y], Rotation(0))
                            }
                            _ => unreachable!(),
                        },
                    };

                    let range_check_on_synthetic_sel = match len {
                        1 => Expression::Constant(F::from(0)),
                        _ => {
                            let mut initial_expr = Expression::Constant(F::from(1));
                            for i in 0..len {
                                initial_expr = initial_expr
                                    * (synthetic_sel.clone()
                                        - Expression::Constant(F::from(i as u64)))
                            }
                            initial_expr
                        }
                    };

                    let sel = cs.query_selector(multi_col_selector);

                    Constraints::with_selector(sel, vec![range_check_on_synthetic_sel])
                });

                self.range_checks
                    .selectors
                    .insert((range, x, y), multi_col_selector);
            }
        }
        // if we haven't previously initialized the input/output, do so now
        if let VarTensor::Empty = self.range_checks.input {
            debug!("assigning range check input");
            self.range_checks.input = input.clone();
        }

        if let VarTensor::Empty = self.range_checks.index {
            debug!("assigning range check index");
            self.range_checks.index = index.clone();
        }

        Ok(())
    }

    /// layout_tables must be called before layout.
    pub fn layout_tables(&mut self, layouter: &mut impl Layouter<F>) -> Result<(), CircuitError> {
        for (i, table) in self.static_lookups.tables.values_mut().enumerate() {
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
    ) -> Result<(), CircuitError> {
        for range_check in self.range_checks.ranges.values_mut() {
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
    ) -> Result<Option<ValTensor<F>>, CircuitError> {
        op.layout(self, region, values)
    }
}
