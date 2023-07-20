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
}

impl<'a, F: PrimeField + TensorType + PartialOrd> RegionCtx<'a, F> {
    /// Create a new region context
    pub fn new(region: Region<'a, F>, offset: usize) -> RegionCtx<'a, F> {
        let region = Some(Arc::new(Mutex::new(region)));

        RegionCtx { region, offset }
    }
    /// Create a new region context from a wrapped region
    pub fn from_wrapped_region(
        region: Option<Arc<Mutex<Region<'a, F>>>>,
        offset: usize,
    ) -> RegionCtx<'a, F> {
        RegionCtx { region, offset }
    }
    /// Get the region
    pub fn region(&self) -> Option<Arc<Mutex<Region<'a, F>>>> {
        self.region.clone()
    }

    /// Create a new region context
    pub fn new_dummy(offset: usize) -> RegionCtx<'a, F> {
        let region = None;

        RegionCtx { region, offset }
    }

    /// Check if the region is dummy
    pub fn is_dummy(&self) -> bool {
        self.region.is_none()
    }

    /// Get the offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Assign a constant value
    pub fn assign_constant(&mut self, var: &VarTensor, value: F) -> Result<ValType<F>, Error> {
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
}
