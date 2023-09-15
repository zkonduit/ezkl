use crate::tensor::{TensorType, ValTensor, ValType, VarTensor};
use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use halo2curves::ff::PrimeField;
use std::cell::RefCell;

#[derive(Debug)]
/// A context for a region
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd> {
    region: Option<RefCell<Region<'a, F>>>,
    offset: usize,
    total_constants: usize,
}

impl<'a, F: PrimeField + TensorType + PartialOrd> RegionCtx<'a, F> {
    /// Create a new region context
    pub fn new(region: Region<'a, F>, offset: usize) -> RegionCtx<'a, F> {
        let region = Some(RefCell::new(region));

        RegionCtx {
            region,
            offset,
            total_constants: 0,
        }
    }
    /// Create a new region context from a wrapped region
    pub fn from_wrapped_region(
        region: Option<RefCell<Region<'a, F>>>,
        offset: usize,
    ) -> RegionCtx<'a, F> {
        RegionCtx {
            region,
            offset,
            total_constants: 0,
        }
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
            let cell = var.assign_constant(&mut region.borrow_mut(), self.offset, value)?;
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
            var.assign(&mut region.borrow_mut(), self.offset, values)
        } else {
            self.total_constants += values.num_constants();
            let assigned_tensor = crate::tensor::Tensor::new(
                Some(&vec![
                    halo2_proofs::circuit::Value::<F>::unknown();
                    values.len()
                ]),
                values.dims(),
            )
            .map_err(|_| Error::Synthesis)?;
            Ok(assigned_tensor.into())
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
            // duplicates every nth element to adjust for column overflow
            var.assign_with_duplication(&mut region.borrow_mut(), self.offset, values, check_mode)
        } else {
            let (dup, len, total_assigned_constants) =
                var.dummy_assign_with_duplication(self.offset, values)?;
            let assigned_tensor = crate::tensor::Tensor::new(
                Some(&vec![
                    halo2_proofs::circuit::Value::<F>::unknown();
                    dup.len()
                ]),
                dup.dims(),
            )
            .map_err(|_| Error::Synthesis)?;
            self.total_constants += total_assigned_constants;
            Ok((assigned_tensor.into(), len))
        }
    }

    /// Enable a selector
    pub fn enable(&mut self, selector: Option<&Selector>, y: usize) -> Result<(), Error> {
        match &self.region {
            Some(region) => selector.unwrap().enable(&mut region.borrow_mut(), y),
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
