use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use std::sync::{Arc, Mutex};

use crate::tensor::{TensorType, ValTensor, ValType, VarTensor};
use halo2curves::ff::PrimeField;

#[derive(Debug, Clone)]
/// A context for a region
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd> {
    region: Option<Arc<Mutex<Region<'a, F>>>>,
    offset: usize,
    total_constants: usize,
}

impl<'a, F: PrimeField + TensorType + PartialOrd> RegionCtx<'a, F> {
    /// Create a new region context
    pub fn new(region: Region<'a, F>, offset: usize) -> RegionCtx<'a, F> {
        let region = Some(Arc::new(Mutex::new(region)));

        RegionCtx {
            region,
            offset,
            total_constants: 0,
        }
    }
    /// Create a new region context from a wrapped region
    pub fn from_wrapped_region(
        region: Option<Arc<Mutex<Region<'a, F>>>>,
        offset: usize,
    ) -> RegionCtx<'a, F> {
        RegionCtx {
            region,
            offset,
            total_constants: 0,
        }
    }
    /// Get the region
    pub fn region(&self) -> Option<Arc<Mutex<Region<'a, F>>>> {
        self.region.clone()
    }

    /// Create a new region context
    pub fn new_dummy(offset: usize) -> RegionCtx<'a, F> {
        let region = None;

        RegionCtx {
            region,
            offset,
            total_constants: 0,
        }
    }

    /// Check if the region is dummy
    pub fn is_dummy(&self) -> bool {
        self.region.is_none()
    }

    /// Get the offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the total number of constants
    pub fn total_constants(&self) -> usize {
        self.total_constants
    }

    /// Assign a constant value
    pub fn assign_constant(&mut self, var: &VarTensor, value: F) -> Result<ValType<F>, Error> {
        self.total_constants += 1;
        if let Some(region) = &self.region {
            let mut lock = region.lock().unwrap();
            let cell = var.assign_constant(&mut lock, self.offset, value)?;
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
        if let Some(region) = &self.region {
            let mut lock = region.lock().unwrap();
            var.assign(&mut lock, self.offset, values)
        } else {
            self.total_constants += values.num_constants();
            Ok(values.clone())
        }
    }
    /// Assign a valtensor to a vartensor with duplication
    pub fn assign_with_duplication(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
        check_mode: &crate::circuit::CheckMode,
    ) -> Result<(ValTensor<F>, usize), Error> {
        if let Some(region) = &self.region {
            let mut lock = region.lock().unwrap();
            // duplicates every nth element to adjust for column overflow
            var.assign_with_duplication(&mut lock, self.offset, values, check_mode)
        } else {
            self.total_constants += values.num_constants();
            var.dummy_assign_with_duplication(self.offset, values)
        }
    }

    /// Enable a selector
    pub fn enable(&mut self, selector: Option<&Selector>, y: usize) -> Result<(), Error> {
        match self.region {
            Some(ref region) => {
                let mut lock = region.lock().unwrap();
                selector.unwrap().enable(&mut lock, y)
            }
            None => Ok(()),
        }
    }

    /// Increment the offset by 1
    pub fn next(&mut self) {
        self.offset += 1
    }

    /// Increment the offset
    pub fn increment(&mut self, n: usize) {
        self.offset += n
    }

    /// increment constants
    pub fn increment_constants(&mut self, n: usize) {
        self.total_constants += n
    }
}
