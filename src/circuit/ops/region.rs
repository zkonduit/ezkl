use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use std::sync::{Arc, Mutex};

use crate::tensor::{TensorType, ValTensor, ValType, VarTensor};
use halo2curves::ff::PrimeField;

#[derive(Debug)]
/// A context for a region
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd> {
    region: Arc<Mutex<Option<Region<'a, F>>>>,
    offset: usize,
}

impl<'a, F: PrimeField + TensorType + PartialOrd> RegionCtx<'a, F> {
    /// Create a new region context
    pub fn new(region: Region<'a, F>, offset: usize) -> RegionCtx<'a, F> {
        let region = Arc::new(Mutex::new(Some(region)));

        RegionCtx { region, offset }
    }
    /// Create a new region context from a wrapped region
    pub fn from_wrapped_region(
        region: Arc<Mutex<Option<Region<'a, F>>>>,
        offset: usize,
    ) -> RegionCtx<'a, F> {
        RegionCtx { region, offset }
    }
    /// Get the region
    pub fn region(&self) -> Arc<Mutex<Option<Region<'a, F>>>> {
        self.region.clone()
    }

    /// Create a new region context
    pub fn new_dummy(offset: usize) -> RegionCtx<'a, F> {
        let region = Arc::new(Mutex::new(None));

        RegionCtx { region, offset }
    }

    /// Get the offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Assign a constant value
    pub fn assign_constant(&mut self, var: &VarTensor, value: F) -> Result<ValType<F>, Error> {
        let mut lock = self.region.lock().unwrap();
        if let Some(region) = lock.as_mut() {
            let cell = var.assign_constant(region, self.offset, value)?;
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
        let mut lock = self.region.lock().unwrap();
        var.assign(&mut lock, self.offset, values)
    }
    /// Assign a valtensor to a vartensor with duplication
    pub fn assign_with_duplication(
        &mut self,
        var: &VarTensor,
        values: &ValTensor<F>,
        check_mode: &crate::circuit::CheckMode,
    ) -> Result<(ValTensor<F>, usize), Error> {
        let mut lock = self.region.lock().unwrap();
        var.assign_with_duplication(&mut lock, self.offset, values, check_mode)
    }

    /// Enable a selector
    pub fn enable(&mut self, selector: Option<&Selector>, y: usize) -> Result<(), Error> {
        let mut lock = self.region.lock().unwrap();
        if let Some(region) = lock.as_mut() {
            selector.unwrap().enable(region, y)
        } else {
            Ok(())
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
