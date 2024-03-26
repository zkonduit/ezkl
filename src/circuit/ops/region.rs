use crate::{
    circuit::table::Range,
    tensor::{Tensor, TensorError, TensorType, ValTensor, ValType, VarTensor},
};
#[cfg(not(target_arch = "wasm32"))]
use colored::Colorize;
use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use halo2curves::ff::PrimeField;
use maybe_rayon::iter::ParallelExtend;
use portable_atomic::AtomicI128 as AtomicInt;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use super::lookup::LookupOp;

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

/// Region error
#[derive(Debug, thiserror::Error)]
pub enum RegionError {
    /// wrap other regions
    #[error("Wrapped region: {0}")]
    Wrapped(String),
}

impl From<String> for RegionError {
    fn from(e: String) -> Self {
        Self::Wrapped(e)
    }
}

impl From<&str> for RegionError {
    fn from(e: &str) -> Self {
        Self::Wrapped(e.to_string())
    }
}

impl From<TensorError> for RegionError {
    fn from(e: TensorError) -> Self {
        Self::Wrapped(format!("{:?}", e))
    }
}

impl From<Error> for RegionError {
    fn from(e: Error) -> Self {
        Self::Wrapped(format!("{:?}", e))
    }
}

impl From<Box<dyn std::error::Error>> for RegionError {
    fn from(e: Box<dyn std::error::Error>) -> Self {
        Self::Wrapped(format!("{:?}", e))
    }
}

#[derive(Debug)]
/// A context for a region
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd + std::hash::Hash> {
    region: Option<RefCell<Region<'a, F>>>,
    row: usize,
    linear_coord: usize,
    num_inner_cols: usize,
    dynamic_lookup_index: DynamicLookupIndex,
    shuffle_index: ShuffleIndex,
    used_lookups: HashSet<LookupOp>,
    used_range_checks: HashSet<Range>,
    max_lookup_inputs: i128,
    min_lookup_inputs: i128,
    max_range_size: i128,
    witness_gen: bool,
    assigned_constants: ConstantsMap<F>,
}

impl<'a, F: PrimeField + TensorType + PartialOrd + std::hash::Hash> RegionCtx<'a, F> {
    #[cfg(not(target_arch = "wasm32"))]
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
        self.assigned_constants.par_extend(constants);
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
        self.witness_gen
    }

    /// Create a new region context
    pub fn new(region: Region<'a, F>, row: usize, num_inner_cols: usize) -> RegionCtx<'a, F> {
        let region = Some(RefCell::new(region));
        let linear_coord = row * num_inner_cols;

        RegionCtx {
            region,
            num_inner_cols,
            row,
            linear_coord,
            dynamic_lookup_index: DynamicLookupIndex::default(),
            shuffle_index: ShuffleIndex::default(),
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
            max_range_size: 0,
            witness_gen: true,
            assigned_constants: HashMap::new(),
        }
    }

    /// Create a new region context
    pub fn new_with_constants(
        region: Region<'a, F>,
        row: usize,
        num_inner_cols: usize,
        constants: ConstantsMap<F>,
    ) -> RegionCtx<'a, F> {
        let mut new_self = Self::new(region, row, num_inner_cols);
        new_self.assigned_constants = constants;
        new_self
    }
    /// Create a new region context from a wrapped region
    pub fn from_wrapped_region(
        region: Option<RefCell<Region<'a, F>>>,
        row: usize,
        num_inner_cols: usize,
        dynamic_lookup_index: DynamicLookupIndex,
        shuffle_index: ShuffleIndex,
    ) -> RegionCtx<'a, F> {
        let linear_coord = row * num_inner_cols;
        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            dynamic_lookup_index,
            shuffle_index,
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
            max_range_size: 0,
            witness_gen: false,
            assigned_constants: HashMap::new(),
        }
    }

    /// Create a new region context
    pub fn new_dummy(row: usize, num_inner_cols: usize, witness_gen: bool) -> RegionCtx<'a, F> {
        let region = None;
        let linear_coord = row * num_inner_cols;

        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            dynamic_lookup_index: DynamicLookupIndex::default(),
            shuffle_index: ShuffleIndex::default(),
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
            max_range_size: 0,
            witness_gen,
            assigned_constants: HashMap::new(),
        }
    }

    /// Create a new region context
    pub fn new_dummy_with_linear_coord(
        row: usize,
        linear_coord: usize,
        num_inner_cols: usize,
        witness_gen: bool,
    ) -> RegionCtx<'a, F> {
        let region = None;
        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            dynamic_lookup_index: DynamicLookupIndex::default(),
            shuffle_index: ShuffleIndex::default(),
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
            max_range_size: 0,
            witness_gen,
            assigned_constants: HashMap::new(),
        }
    }

    /// Apply a function in a loop to the region
    pub fn apply_in_loop<T: TensorType + Send + Sync>(
        &mut self,
        output: &mut Tensor<T>,
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> Result<T, RegionError>
            + Send
            + Sync,
    ) -> Result<(), RegionError> {
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
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> Result<T, RegionError>,
    ) -> Result<(), RegionError> {
        output
            .iter_mut()
            .enumerate()
            .map(|(i, o)| {
                *o = inner_loop_function(i, self)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, RegionError>>()?;

        Ok(())
    }

    /// Create a new region context per loop iteration
    /// hacky but it works

    pub fn dummy_loop<T: TensorType + Send + Sync>(
        &mut self,
        output: &mut Tensor<T>,
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> Result<T, RegionError>
            + Send
            + Sync,
    ) -> Result<(), RegionError> {
        let row = AtomicUsize::new(self.row());
        let linear_coord = AtomicUsize::new(self.linear_coord());
        let max_lookup_inputs = AtomicInt::new(self.max_lookup_inputs());
        let min_lookup_inputs = AtomicInt::new(self.min_lookup_inputs());
        let lookups = Arc::new(Mutex::new(self.used_lookups.clone()));
        let range_checks = Arc::new(Mutex::new(self.used_range_checks.clone()));
        let dynamic_lookup_index = Arc::new(Mutex::new(self.dynamic_lookup_index.clone()));
        let shuffle_index = Arc::new(Mutex::new(self.shuffle_index.clone()));
        let constants = Arc::new(Mutex::new(self.assigned_constants.clone()));

        *output = output
            .par_enum_map(|idx, _| {
                // we kick off the loop with the current offset
                let starting_offset = row.load(Ordering::SeqCst);
                let starting_linear_coord = linear_coord.load(Ordering::SeqCst);
                // get inner value of the locked lookups

                // we need to make sure that the region is not shared between threads
                let mut local_reg = Self::new_dummy_with_linear_coord(
                    starting_offset,
                    starting_linear_coord,
                    self.num_inner_cols,
                    self.witness_gen,
                );
                let res = inner_loop_function(idx, &mut local_reg);
                // we update the offset and constants
                row.fetch_add(local_reg.row() - starting_offset, Ordering::SeqCst);
                linear_coord.fetch_add(
                    local_reg.linear_coord() - starting_linear_coord,
                    Ordering::SeqCst,
                );

                max_lookup_inputs.fetch_max(local_reg.max_lookup_inputs(), Ordering::SeqCst);
                min_lookup_inputs.fetch_min(local_reg.min_lookup_inputs(), Ordering::SeqCst);
                // update the lookups
                let mut lookups = lookups.lock().unwrap();
                lookups.par_extend(local_reg.used_lookups());
                // update the range checks
                let mut range_checks = range_checks.lock().unwrap();
                range_checks.par_extend(local_reg.used_range_checks());
                // update the dynamic lookup index
                let mut dynamic_lookup_index = dynamic_lookup_index.lock().unwrap();
                dynamic_lookup_index.update(&local_reg.dynamic_lookup_index);
                // update the shuffle index
                let mut shuffle_index = shuffle_index.lock().unwrap();
                shuffle_index.update(&local_reg.shuffle_index);
                // update the constants
                let mut constants = constants.lock().unwrap();
                constants.par_extend(local_reg.assigned_constants);

                res
            })
            .map_err(|e| RegionError::from(format!("dummy_loop: {:?}", e)))?;
        self.linear_coord = linear_coord.into_inner();
        #[allow(trivial_numeric_casts)]
        {
            self.max_lookup_inputs = max_lookup_inputs.into_inner();
            self.min_lookup_inputs = min_lookup_inputs.into_inner();
        }
        self.row = row.into_inner();
        self.used_lookups = Arc::try_unwrap(lookups)
            .map_err(|e| RegionError::from(format!("dummy_loop: failed to get lookups: {:?}", e)))?
            .into_inner()
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get lookups: {:?}", e))
            })?;
        self.used_range_checks = Arc::try_unwrap(range_checks)
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get range checks: {:?}", e))
            })?
            .into_inner()
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get range checks: {:?}", e))
            })?;
        self.dynamic_lookup_index = Arc::try_unwrap(dynamic_lookup_index)
            .map_err(|e| {
                RegionError::from(format!(
                    "dummy_loop: failed to get dynamic lookup index: {:?}",
                    e
                ))
            })?
            .into_inner()
            .map_err(|e| {
                RegionError::from(format!(
                    "dummy_loop: failed to get dynamic lookup index: {:?}",
                    e
                ))
            })?;
        self.shuffle_index = Arc::try_unwrap(shuffle_index)
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get shuffle index: {:?}", e))
            })?
            .into_inner()
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get shuffle index: {:?}", e))
            })?;
        self.assigned_constants = Arc::try_unwrap(constants)
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get constants: {:?}", e))
            })?
            .into_inner()
            .map_err(|e| {
                RegionError::from(format!("dummy_loop: failed to get constants: {:?}", e))
            })?;

        Ok(())
    }

    /// Update the max and min from inputs
    pub fn update_max_min_lookup_inputs(
        &mut self,
        inputs: &[ValTensor<F>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (mut min, mut max) = (0, 0);
        for i in inputs {
            max = max.max(i.get_int_evals()?.into_iter().max().unwrap_or_default());
            min = min.min(i.get_int_evals()?.into_iter().min().unwrap_or_default());
        }
        self.max_lookup_inputs = self.max_lookup_inputs.max(max);
        self.min_lookup_inputs = self.min_lookup_inputs.min(min);
        Ok(())
    }

    /// Update the max and min from inputs
    pub fn update_max_min_lookup_range(
        &mut self,
        range: Range,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if range.0 > range.1 {
            return Err(format!("update_max_min_lookup_range: invalid range {:?}", range).into());
        }

        let range_size = (range.1 - range.0).abs();

        self.max_range_size = self.max_range_size.max(range_size);
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
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.used_lookups.insert(lookup);
        self.update_max_min_lookup_inputs(inputs)
    }

    /// add used range check
    pub fn add_used_range_check(&mut self, range: Range) -> Result<(), Box<dyn std::error::Error>> {
        self.used_range_checks.insert(range);
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
        self.used_lookups.clone()
    }

    /// get used range checks
    pub fn used_range_checks(&self) -> HashSet<Range> {
        self.used_range_checks.clone()
    }

    /// max lookup inputs
    pub fn max_lookup_inputs(&self) -> i128 {
        self.max_lookup_inputs
    }

    /// min lookup inputs
    pub fn min_lookup_inputs(&self) -> i128 {
        self.min_lookup_inputs
    }

    /// max range check
    pub fn max_range_size(&self) -> i128 {
        self.max_range_size
    }

    /// Assign a valtensor to a vartensor
    pub fn assign(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, Error> {
        if let Some(region) = &self.region {
            var.assign(
                &mut region.borrow_mut(),
                self.linear_coord,
                values,
                &mut self.assigned_constants,
            )
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
    ) -> Result<ValTensor<F>, Error> {
        if let Some(region) = &self.region {
            var.assign(
                &mut region.borrow_mut(),
                self.combined_dynamic_shuffle_coord(),
                values,
                &mut self.assigned_constants,
            )
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
    ) -> Result<ValTensor<F>, Error> {
        self.assign_dynamic_lookup(var, values)
    }

    /// Assign a valtensor to a vartensor
    pub fn assign_with_omissions(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
        ommissions: &HashSet<&usize>,
    ) -> Result<ValTensor<F>, Error> {
        if let Some(region) = &self.region {
            var.assign_with_omissions(
                &mut region.borrow_mut(),
                self.linear_coord,
                values,
                ommissions,
                &mut self.assigned_constants,
            )
        } else {
            let inner_tensor = values.get_inner_tensor().unwrap();
            let mut values_map = values.create_constants_map();

            for o in ommissions {
                if let ValType::Constant(value) = inner_tensor.get_flat_index(**o) {
                    values_map.remove(&value);
                }
            }

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
    pub fn constrain_equal(&mut self, a: &ValTensor<F>, b: &ValTensor<F>) -> Result<(), Error> {
        if let Some(region) = &self.region {
            let a = a.get_inner_tensor().unwrap();
            let b = b.get_inner_tensor().unwrap();
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).try_for_each(|(a, b)| {
                let a = a.get_prev_assigned();
                let b = b.get_prev_assigned();
                // if they're both assigned, we can constrain them
                if let (Some(a), Some(b)) = (&a, &b) {
                    region.borrow_mut().constrain_equal(a.cell(), b.cell())
                } else if a.is_some() || b.is_some() {
                    log::error!(
                        "constrain_equal: one of the tensors is assigned and the other is not"
                    );
                    return Err(Error::Synthesis);
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
    pub fn flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // increment by the difference between the current linear coord and the next row
        let remainder = self.linear_coord % self.num_inner_cols;
        if remainder != 0 {
            let diff = self.num_inner_cols - remainder;
            self.increment(diff);
        }
        if self.linear_coord % self.num_inner_cols != 0 {
            return Err("flush: linear coord is not aligned with the next row".into());
        }
        Ok(())
    }
}
