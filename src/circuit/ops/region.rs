use crate::{
    circuit::table::Range,
    tensor::{Tensor, TensorError, TensorType, ValTensor, ValType, VarTensor},
};
use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use halo2curves::ff::PrimeField;
use std::{
    cell::RefCell,
    collections::HashSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use super::lookup::LookupOp;

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
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd> {
    region: Option<RefCell<Region<'a, F>>>,
    row: usize,
    linear_coord: usize,
    num_inner_cols: usize,
    total_constants: usize,
    used_lookups: HashSet<LookupOp>,
    used_range_checks: HashSet<Range>,
}

impl<'a, F: PrimeField + TensorType + PartialOrd> RegionCtx<'a, F> {
    ///
    pub fn increment_total_constants(&mut self, n: usize) {
        self.total_constants += n;
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
            total_constants: 0,
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
        }
    }
    /// Create a new region context from a wrapped region
    pub fn from_wrapped_region(
        region: Option<RefCell<Region<'a, F>>>,
        row: usize,
        num_inner_cols: usize,
    ) -> RegionCtx<'a, F> {
        let linear_coord = row * num_inner_cols;
        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            total_constants: 0,
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
        }
    }

    /// Create a new region context
    pub fn new_dummy(row: usize, num_inner_cols: usize) -> RegionCtx<'a, F> {
        let region = None;
        let linear_coord = row * num_inner_cols;

        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            total_constants: 0,
            used_lookups: HashSet::new(),
            used_range_checks: HashSet::new(),
        }
    }

    /// Create a new region context
    pub fn new_dummy_with_constants(
        row: usize,
        linear_coord: usize,
        total_constants: usize,
        num_inner_cols: usize,
        used_lookups: HashSet<LookupOp>,
        used_range_checks: HashSet<Range>,
    ) -> RegionCtx<'a, F> {
        let region = None;
        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            total_constants,
            used_lookups,
            used_range_checks,
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
        let constants = AtomicUsize::new(self.total_constants());
        let lookups = Arc::new(Mutex::new(self.used_lookups.clone()));
        let range_checks = Arc::new(Mutex::new(self.used_range_checks.clone()));

        *output = output
            .par_enum_map(|idx, _| {
                // we kick off the loop with the current offset
                let starting_offset = row.load(Ordering::SeqCst);
                let starting_linear_coord = linear_coord.load(Ordering::SeqCst);
                let starting_constants = constants.load(Ordering::SeqCst);
                // get inner value of the locked lookups

                // we need to make sure that the region is not shared between threads
                let mut local_reg = Self::new_dummy_with_constants(
                    starting_offset,
                    starting_linear_coord,
                    starting_constants,
                    self.num_inner_cols,
                    HashSet::new(),
                    HashSet::new(),
                );
                let res = inner_loop_function(idx, &mut local_reg);
                // we update the offset and constants
                row.fetch_add(local_reg.row() - starting_offset, Ordering::SeqCst);
                linear_coord.fetch_add(
                    local_reg.linear_coord() - starting_linear_coord,
                    Ordering::SeqCst,
                );
                constants.fetch_add(
                    local_reg.total_constants() - starting_constants,
                    Ordering::SeqCst,
                );
                // update the lookups
                let mut lookups = lookups.lock().unwrap();
                lookups.extend(local_reg.used_lookups());
                let mut range_checks = range_checks.lock().unwrap();
                range_checks.extend(local_reg.used_range_checks());
                res
            })
            .map_err(|e| {
                log::error!("dummy_loop: {:?}", e);
                Error::Synthesis
            })?;
        self.total_constants = constants.into_inner();
        self.linear_coord = linear_coord.into_inner();
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

        Ok(())
    }

    /// Check if the region is dummy
    pub fn is_dummy(&self) -> bool {
        self.region.is_none()
    }

    /// add used lookup
    pub fn add_used_lookup(&mut self, lookup: LookupOp) {
        self.used_lookups.insert(lookup);
    }

    /// add used range check
    pub fn add_used_range_check(&mut self, range: Range) {
        self.used_range_checks.insert(range);
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
        self.total_constants
    }

    /// get used lookups
    pub fn used_lookups(&self) -> HashSet<LookupOp> {
        self.used_lookups.clone()
    }

    /// get used range checks
    pub fn used_range_checks(&self) -> HashSet<Range> {
        self.used_range_checks.clone()
    }

    /// Assign a constant value
    pub fn assign_constant(&mut self, var: &VarTensor, value: F) -> Result<ValType<F>, Error> {
        self.total_constants += 1;
        if let Some(region) = &self.region {
            let cell = var.assign_constant(&mut region.borrow_mut(), self.linear_coord, value)?;
            Ok(cell.into())
        } else {
            Ok(value.into())
        }
    }
    /// Assign a valtensor to a vartensor
    pub fn assign(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
    ) -> Result<ValTensor<F>, Error> {
        self.total_constants += values.num_constants();
        if let Some(region) = &self.region {
            var.assign(&mut region.borrow_mut(), self.linear_coord, values)
        } else {
            Ok(values.clone())
        }
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
            )
        } else {
            self.total_constants += values.num_constants();
            let inner_tensor = values.get_inner_tensor().unwrap();
            for o in ommissions {
                self.total_constants -= inner_tensor.get_flat_index(**o).is_constant() as usize;
            }
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
            let (res, len, total_assigned_constants) = var.assign_with_duplication(
                &mut region.borrow_mut(),
                self.row,
                self.linear_coord,
                values,
                check_mode,
                single_inner_col,
            )?;
            self.total_constants += total_assigned_constants;
            Ok((res, len))
        } else {
            let (_, len, total_assigned_constants) = var.dummy_assign_with_duplication(
                self.row,
                self.linear_coord,
                values,
                single_inner_col,
            )?;
            self.total_constants += total_assigned_constants;
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

    /// increment constants
    pub fn increment_constants(&mut self, n: usize) {
        self.total_constants += n
    }
}
