use crate::{
    circuit::table::Range,
    fieldutils::IntegerRep,
    tensor::{Tensor, TensorType, ValTensor, ValType, VarTensor},
};
#[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
use colored::Colorize;
use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use maybe_rayon::iter::ParallelExtend;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use super::{lookup::LookupOp, CircuitError};

/// Constants map
pub type ConstantsMap<F> = HashMap<F, ValType<F>>;

/// Dynamic lookup index
#[derive(Clone, Debug, Default)]
pub struct DynamicLookupIndex {
    index: usize,
    col_coord: usize,
}

impl DynamicLookupIndex {
    /// Create a new dynamic lookup index
    pub fn new(index: usize, col_coord: usize) -> DynamicLookupIndex {
        DynamicLookupIndex { index, col_coord }
    }

    /// Get the lookup index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the column coord
    pub fn col_coord(&self) -> usize {
        self.col_coord
    }

    /// update with another dynamic lookup index
    pub fn update(&mut self, other: &DynamicLookupIndex) {
        self.index += other.index;
        self.col_coord += other.col_coord;
    }
}

/// Dynamic lookup index
#[derive(Clone, Debug, Default)]
pub struct ShuffleIndex {
    index: usize,
    col_coord: usize,
}

impl ShuffleIndex {
    /// Create a new dynamic lookup index
    pub fn new(index: usize, col_coord: usize) -> ShuffleIndex {
        ShuffleIndex { index, col_coord }
    }

    /// Get the lookup index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the column coord
    pub fn col_coord(&self) -> usize {
        self.col_coord
    }

    /// update with another shuffle index
    pub fn update(&mut self, other: &ShuffleIndex) {
        self.index += other.index;
        self.col_coord += other.col_coord;
    }
}

#[derive(Debug, Clone)]
/// Some settings for a region to differentiate it across the different phases of proof generation
pub struct RegionSettings {
    /// whether we are in witness generation mode
    pub witness_gen: bool,
    /// whether we should check range checks for validity
    pub check_range: bool,
    /// base for decompositions
    pub base: usize,
    /// number of legs for decompositions
    pub legs: usize,
}

#[allow(unsafe_code)]
unsafe impl Sync for RegionSettings {}
#[allow(unsafe_code)]
unsafe impl Send for RegionSettings {}

impl RegionSettings {
    /// Create a new region settings
    pub fn new(witness_gen: bool, check_range: bool, base: usize, legs: usize) -> RegionSettings {
        RegionSettings {
            witness_gen,
            check_range,
            base,
            legs,
        }
    }

    /// Create a new region settings with all true
    pub fn all_true(base: usize, legs: usize) -> RegionSettings {
        RegionSettings {
            witness_gen: true,
            check_range: true,
            base,
            legs,
        }
    }

    /// Create a new region settings with all false
    pub fn all_false(base: usize, legs: usize) -> RegionSettings {
        RegionSettings {
            witness_gen: false,
            check_range: false,
            base,
            legs,
        }
    }
}

#[derive(Debug, Default, Clone)]
/// Region statistics
pub struct RegionStatistics {
    /// the current maximum value of the lookup inputs
    pub max_lookup_inputs: IntegerRep,
    /// the current minimum value of the lookup inputs
    pub min_lookup_inputs: IntegerRep,
    /// the current maximum value of the range size
    pub max_range_size: IntegerRep,
    /// the current set of used lookups
    pub used_lookups: HashSet<LookupOp>,
    /// the current set of used range checks
    pub used_range_checks: HashSet<Range>,
}

impl RegionStatistics {
    /// update the statistics with another set of statistics
    pub fn update(&mut self, other: &RegionStatistics) {
        self.max_lookup_inputs = self.max_lookup_inputs.max(other.max_lookup_inputs);
        self.min_lookup_inputs = self.min_lookup_inputs.min(other.min_lookup_inputs);
        self.max_range_size = self.max_range_size.max(other.max_range_size);
        self.used_lookups.extend(other.used_lookups.clone());
        self.used_range_checks
            .extend(other.used_range_checks.clone());
    }
}

#[allow(unsafe_code)]
unsafe impl Sync for RegionStatistics {}
#[allow(unsafe_code)]
unsafe impl Send for RegionStatistics {}

#[derive(Debug)]
/// A context for a region
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
    region: Option<RefCell<Region<'a, F>>>,
    row: usize,
    linear_coord: usize,
    num_inner_cols: usize,
    dynamic_lookup_index: DynamicLookupIndex,
    shuffle_index: ShuffleIndex,
    statistics: RegionStatistics,
    settings: RegionSettings,
    assigned_constants: ConstantsMap<F>,
}

impl<'a, F: PrimeField + TensorType + PartialOrd + std::hash::Hash> RegionCtx<'a, F> {
    /// get the region's decomposition base
    pub fn base(&self) -> usize {
        self.settings.base
    }

    /// get the region's decomposition legs
    pub fn legs(&self) -> usize {
        self.settings.legs
    }

    #[cfg(not(any(target_os = "ios", target_arch = "wasm32")))]
    ///
    pub fn debug_report(&self) {
        log::debug!(
            "(rows={}, coord={}, constants={}, max_lookup_inputs={}, min_lookup_inputs={}, max_range_size={}, dynamic_lookup_col_coord={}, shuffle_col_coord={})",
            self.row().to_string().blue(),
            self.linear_coord().to_string().yellow(),
            self.total_constants().to_string().red(),
            self.max_lookup_inputs().to_string().green(),
            self.min_lookup_inputs().to_string().green(),
            self.max_range_size().to_string().green(),
            self.dynamic_lookup_col_coord().to_string().green(),
            self.shuffle_col_coord().to_string().green());
    }

    ///
    pub fn assigned_constants(&self) -> &ConstantsMap<F> {
        &self.assigned_constants
    }

    ///
    pub fn update_constants(&mut self, constants: ConstantsMap<F>) {
        self.assigned_constants.extend(constants);
    }

    ///
    pub fn increment_dynamic_lookup_index(&mut self, n: usize) {
        self.dynamic_lookup_index.index += n;
    }

    ///
    pub fn increment_dynamic_lookup_col_coord(&mut self, n: usize) {
        self.dynamic_lookup_index.col_coord += n;
    }

    ///
    pub fn increment_shuffle_index(&mut self, n: usize) {
        self.shuffle_index.index += n;
    }

    ///
    pub fn increment_shuffle_col_coord(&mut self, n: usize) {
        self.shuffle_index.col_coord += n;
    }

    ///
    pub fn witness_gen(&self) -> bool {
        self.settings.witness_gen
    }

    ///
    pub fn check_range(&self) -> bool {
        self.settings.check_range
    }

    ///
    pub fn statistics(&self) -> &RegionStatistics {
        &self.statistics
    }

    /// Create a new region context
    pub fn new(
        region: Region<'a, F>,
        row: usize,
        num_inner_cols: usize,
        decomp_base: usize,
        decomp_legs: usize,
    ) -> RegionCtx<'a, F> {
        let region = Some(RefCell::new(region));
        let linear_coord = row * num_inner_cols;

        RegionCtx {
            region,
            num_inner_cols,
            row,
            linear_coord,
            dynamic_lookup_index: DynamicLookupIndex::default(),
            shuffle_index: ShuffleIndex::default(),
            statistics: RegionStatistics::default(),
            settings: RegionSettings::all_true(decomp_base, decomp_legs),
            assigned_constants: HashMap::new(),
        }
    }

    /// Create a new region context
    pub fn new_with_constants(
        region: Region<'a, F>,
        row: usize,
        num_inner_cols: usize,
        decomp_base: usize,
        decomp_legs: usize,
        constants: ConstantsMap<F>,
    ) -> RegionCtx<'a, F> {
        let mut new_self = Self::new(region, row, num_inner_cols, decomp_base, decomp_legs);
        new_self.assigned_constants = constants;
        new_self
    }

    /// Create a new region context
    pub fn new_dummy(
        row: usize,
        num_inner_cols: usize,
        settings: RegionSettings,
    ) -> RegionCtx<'a, F> {
        let region = None;
        let linear_coord = row * num_inner_cols;

        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            dynamic_lookup_index: DynamicLookupIndex::default(),
            shuffle_index: ShuffleIndex::default(),
            statistics: RegionStatistics::default(),
            settings,
            assigned_constants: HashMap::new(),
        }
    }

    /// Create a new region context
    pub fn new_dummy_with_linear_coord(
        row: usize,
        linear_coord: usize,
        num_inner_cols: usize,
        settings: RegionSettings,
    ) -> RegionCtx<'a, F> {
        let region = None;
        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            dynamic_lookup_index: DynamicLookupIndex::default(),
            shuffle_index: ShuffleIndex::default(),
            statistics: RegionStatistics::default(),
            settings,
            assigned_constants: HashMap::new(),
        }
    }

    /// Apply a function in a loop to the region
    pub fn apply_in_loop<T: TensorType + Send + Sync>(
        &mut self,
        output: &mut Tensor<T>,
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> Result<T, CircuitError>
            + Send
            + Sync,
    ) -> Result<(), CircuitError> {
        if self.is_dummy() {
            self.dummy_loop(output, inner_loop_function)?;
        } else {
            self.real_loop(output, inner_loop_function)?;
        }
        Ok(())
    }

    /// Run a loop
    pub fn real_loop<T: TensorType + Send + Sync>(
        &mut self,
        output: &mut Tensor<T>,
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> Result<T, CircuitError>,
    ) -> Result<(), CircuitError> {
        output
            .iter_mut()
            .enumerate()
            .map(|(i, o)| {
                *o = inner_loop_function(i, self)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;

        Ok(())
    }

    /// Create a new region context per loop iteration
    /// hacky but it works

    pub fn dummy_loop<T: TensorType + Send + Sync>(
        &mut self,
        output: &mut Tensor<T>,
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> Result<T, CircuitError>
            + Send
            + Sync,
    ) -> Result<(), CircuitError> {
        let row = AtomicUsize::new(self.row());
        let linear_coord = AtomicUsize::new(self.linear_coord());
        let statistics = Arc::new(Mutex::new(self.statistics.clone()));
        let shuffle_index = Arc::new(Mutex::new(self.shuffle_index.clone()));
        let dynamic_lookup_index = Arc::new(Mutex::new(self.dynamic_lookup_index.clone()));
        let constants = Arc::new(Mutex::new(self.assigned_constants.clone()));

        *output = output.par_enum_map(|idx, _| {
            // we kick off the loop with the current offset
            let starting_offset = row.load(Ordering::SeqCst);
            let starting_linear_coord = linear_coord.load(Ordering::SeqCst);
            // get inner value of the locked lookups

            // we need to make sure that the region is not shared between threads
            let mut local_reg = Self::new_dummy_with_linear_coord(
                starting_offset,
                starting_linear_coord,
                self.num_inner_cols,
                self.settings.clone(),
            );
            let res = inner_loop_function(idx, &mut local_reg);
            // we update the offset and constants
            row.fetch_add(local_reg.row() - starting_offset, Ordering::SeqCst);
            linear_coord.fetch_add(
                local_reg.linear_coord() - starting_linear_coord,
                Ordering::SeqCst,
            );

            // update the lookups
            let mut statistics = statistics.lock().unwrap();
            statistics.update(local_reg.statistics());
            // update the dynamic lookup index
            let mut dynamic_lookup_index = dynamic_lookup_index.lock().unwrap();
            dynamic_lookup_index.update(&local_reg.dynamic_lookup_index);
            // update the shuffle index
            let mut shuffle_index = shuffle_index.lock().unwrap();
            shuffle_index.update(&local_reg.shuffle_index);
            // update the constants
            let mut constants = constants.lock().unwrap();
            constants.extend(local_reg.assigned_constants);

            res
        })?;
        self.linear_coord = linear_coord.into_inner();
        self.row = row.into_inner();
        self.statistics = Arc::try_unwrap(statistics)
            .map_err(|e| CircuitError::GetLookupsError(format!("{:?}", e)))?
            .into_inner()
            .map_err(|e| CircuitError::GetLookupsError(format!("{:?}", e)))?;
        self.dynamic_lookup_index = Arc::try_unwrap(dynamic_lookup_index)
            .map_err(|e| CircuitError::GetDynamicLookupError(format!("{:?}", e)))?
            .into_inner()
            .map_err(|e| CircuitError::GetDynamicLookupError(format!("{:?}", e)))?;
        self.shuffle_index = Arc::try_unwrap(shuffle_index)
            .map_err(|e| CircuitError::GetShuffleError(format!("{:?}", e)))?
            .into_inner()
            .map_err(|e| CircuitError::GetShuffleError(format!("{:?}", e)))?;
        self.assigned_constants = Arc::try_unwrap(constants)
            .map_err(|e| CircuitError::GetConstantsError(format!("{:?}", e)))?
            .into_inner()
            .map_err(|e| CircuitError::GetConstantsError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Update the max and min from inputs
    pub fn update_max_min_lookup_inputs(
        &mut self,
        inputs: &[ValTensor<F>],
    ) -> Result<(), CircuitError> {
        let (mut min, mut max) = (0, 0);
        for i in inputs {
            max = max.max(i.int_evals()?.into_iter().max().unwrap_or_default());
            min = min.min(i.int_evals()?.into_iter().min().unwrap_or_default());
        }
        self.statistics.max_lookup_inputs = self.statistics.max_lookup_inputs.max(max);
        self.statistics.min_lookup_inputs = self.statistics.min_lookup_inputs.min(min);
        Ok(())
    }

    /// Update the max and min from inputs
    pub fn update_max_min_lookup_range(&mut self, range: Range) -> Result<(), CircuitError> {
        if range.0 > range.1 {
            return Err(CircuitError::InvalidMinMaxRange(range.0, range.1));
        }

        let range_size = (range.1 - range.0).abs();

        self.statistics.max_range_size = self.statistics.max_range_size.max(range_size);
        Ok(())
    }

    /// Check if the region is dummy
    pub fn is_dummy(&self) -> bool {
        self.region.is_none()
    }

    /// add used lookup
    pub fn add_used_lookup(
        &mut self,
        lookup: LookupOp,
        inputs: &[ValTensor<F>],
    ) -> Result<(), CircuitError> {
        self.statistics.used_lookups.insert(lookup);
        self.update_max_min_lookup_inputs(inputs)
    }

    /// add used range check
    pub fn add_used_range_check(&mut self, range: Range) -> Result<(), CircuitError> {
        self.statistics.used_range_checks.insert(range);
        self.update_max_min_lookup_range(range)
    }

    /// Get the offset
    pub fn row(&self) -> usize {
        self.row
    }

    /// Linear coordinate
    pub fn linear_coord(&self) -> usize {
        self.linear_coord
    }

    /// Get the total number of constants
    pub fn total_constants(&self) -> usize {
        self.assigned_constants.len()
    }

    /// Get the dynamic lookup index
    pub fn dynamic_lookup_index(&self) -> usize {
        self.dynamic_lookup_index.index
    }

    /// Get the dynamic lookup column coordinate
    pub fn dynamic_lookup_col_coord(&self) -> usize {
        self.dynamic_lookup_index.col_coord
    }

    /// Get the shuffle index
    pub fn shuffle_index(&self) -> usize {
        self.shuffle_index.index
    }

    /// Get the shuffle column coordinate
    pub fn shuffle_col_coord(&self) -> usize {
        self.shuffle_index.col_coord
    }

    /// get used lookups
    pub fn used_lookups(&self) -> HashSet<LookupOp> {
        self.statistics.used_lookups.clone()
    }

    /// get used range checks
    pub fn used_range_checks(&self) -> HashSet<Range> {
        self.statistics.used_range_checks.clone()
    }

    /// max lookup inputs
    pub fn max_lookup_inputs(&self) -> IntegerRep {
        self.statistics.max_lookup_inputs
    }

    /// min lookup inputs
    pub fn min_lookup_inputs(&self) -> IntegerRep {
        self.statistics.min_lookup_inputs
    }

    /// max range check
    pub fn max_range_size(&self) -> IntegerRep {
        self.statistics.max_range_size
    }

    /// Assign a valtensor to a vartensor
    pub fn assign(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, CircuitError> {
        if let Some(region) = &self.region {
            Ok(var.assign(
                &mut region.borrow_mut(),
                self.linear_coord,
                values,
                &mut self.assigned_constants,
            )?)
        } else {
            if !values.is_instance() {
                let values_map = values.create_constants_map_iterator();
                self.assigned_constants.par_extend(values_map);
            }
            Ok(values.clone())
        }
    }

    ///
    pub fn combined_dynamic_shuffle_coord(&self) -> usize {
        self.dynamic_lookup_col_coord() + self.shuffle_col_coord()
    }

    /// Assign a valtensor to a vartensor
    pub fn assign_dynamic_lookup(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, CircuitError> {
        if let Some(region) = &self.region {
            Ok(var.assign(
                &mut region.borrow_mut(),
                self.combined_dynamic_shuffle_coord(),
                values,
                &mut self.assigned_constants,
            )?)
        } else {
            if !values.is_instance() {
                let values_map = values.create_constants_map_iterator();
                self.assigned_constants.par_extend(values_map);
            }
            Ok(values.clone())
        }
    }

    /// Assign a valtensor to a vartensor
    pub fn assign_shuffle(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, CircuitError> {
        self.assign_dynamic_lookup(var, values)
    }

    /// Assign a valtensor to a vartensor
    pub fn assign_with_omissions(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
        ommissions: &HashSet<usize>,
    ) -> Result<ValTensor<F>, CircuitError> {
        if let Some(region) = &self.region {
            Ok(var.assign_with_omissions(
                &mut region.borrow_mut(),
                self.linear_coord,
                values,
                ommissions,
                &mut self.assigned_constants,
            )?)
        } else {
            let mut values_clone = values.clone();
            let mut indices = ommissions.clone().into_iter().collect_vec();
            values_clone.remove_indices(&mut indices, false)?;

            let values_map = values.create_constants_map();

            self.assigned_constants.par_extend(values_map);

            Ok(values.clone())
        }
    }

    /// Assign a valtensor to a vartensor with duplication
    pub fn assign_with_duplication(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
        check_mode: &crate::circuit::CheckMode,
        single_inner_col: bool,
    ) -> Result<(ValTensor<F>, usize), Error> {
        if let Some(region) = &self.region {
            // duplicates every nth element to adjust for column overflow
            let (res, len) = var.assign_with_duplication(
                &mut region.borrow_mut(),
                self.row,
                self.linear_coord,
                values,
                check_mode,
                single_inner_col,
                &mut self.assigned_constants,
            )?;
            Ok((res, len))
        } else {
            let (_, len) = var.dummy_assign_with_duplication(
                self.row,
                self.linear_coord,
                values,
                single_inner_col,
                &mut self.assigned_constants,
            )?;
            Ok((values.clone(), len))
        }
    }

    /// Enable a selector
    pub fn enable(&mut self, selector: Option<&Selector>, offset: usize) -> Result<(), Error> {
        match &self.region {
            Some(region) => selector.unwrap().enable(&mut region.borrow_mut(), offset),
            None => Ok(()),
        }
    }

    /// constrain equal
    pub fn constrain_equal(
        &mut self,
        a: &ValTensor<F>,
        b: &ValTensor<F>,
    ) -> Result<(), CircuitError> {
        if let Some(region) = &self.region {
            let a = a.get_inner_tensor().unwrap();
            let b = b.get_inner_tensor().unwrap();
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).try_for_each(|(a, b)| {
                let a = a.get_prev_assigned();
                let b = b.get_prev_assigned();
                // if they're both assigned, we can constrain them
                if let (Some(a), Some(b)) = (&a, &b) {
                    region
                        .borrow_mut()
                        .constrain_equal(a.cell(), b.cell())
                        .map_err(|e| e.into())
                } else if a.is_some() || b.is_some() {
                    return Err(CircuitError::ConstrainError);
                } else {
                    Ok(())
                }
            })
        } else {
            Ok(())
        }
    }

    /// Increment the offset by 1
    pub fn next(&mut self) {
        self.linear_coord += 1;
        if self.linear_coord % self.num_inner_cols == 0 {
            self.row += 1;
        }
    }

    /// Increment the offset
    pub fn increment(&mut self, n: usize) {
        for _ in 0..n {
            self.next()
        }
    }

    /// flush row to the next row
    pub fn flush(&mut self) -> Result<(), CircuitError> {
        // increment by the difference between the current linear coord and the next row
        let remainder = self.linear_coord % self.num_inner_cols;
        if remainder != 0 {
            let diff = self.num_inner_cols - remainder;
            self.increment(diff);
        }
        if self.linear_coord % self.num_inner_cols != 0 {
            return Err(CircuitError::FlushError);
        }
        Ok(())
    }
}
