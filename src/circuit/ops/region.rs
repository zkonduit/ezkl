use crate::tensor::{Tensor, TensorType, ValTensor, ValType, VarTensor};
use halo2_proofs::{
    circuit::Region,
    plonk::{Error, Selector},
};
use halo2curves::ff::PrimeField;
use std::{
    cell::RefCell,
    collections::HashSet,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug)]
/// A context for a region
pub struct RegionCtx<'a, F: PrimeField + TensorType + PartialOrd> {
    region: Option<RefCell<Region<'a, F>>>,
    row: usize,
    linear_coord: usize,
    num_inner_cols: usize,
    total_constants: usize,
}

impl<'a, F: PrimeField + TensorType + PartialOrd> RegionCtx<'a, F> {
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
        }
    }

    /// Create a new region context
    pub fn new_dummy_with_constants(
        row: usize,
        linear_coord: usize,
        constants: usize,
        num_inner_cols: usize,
    ) -> RegionCtx<'a, F> {
        let region = None;
        RegionCtx {
            region,
            num_inner_cols,
            linear_coord,
            row,
            total_constants: constants,
        }
    }

    /// Create a new region context per loop iteration
    /// hacky but it works
    pub fn dummy_loop<T: TensorType + Send + Sync>(
        &mut self,
        output: &mut Tensor<T>,
        inner_loop_function: impl Fn(usize, &mut RegionCtx<'a, F>) -> T + Sync + Send,
    ) -> Result<(), Error> {
        let row = AtomicUsize::new(self.row());
        let linear_coord = AtomicUsize::new(self.linear_coord());
        let constants = AtomicUsize::new(self.total_constants());
        *output = output.par_enum_map(|idx, _| {
            // we kick off the loop with the current offset
            let starting_offset = row.load(Ordering::SeqCst);
            let starting_linear_coord = linear_coord.load(Ordering::SeqCst);
            let starting_constants = constants.load(Ordering::SeqCst);
            // we need to make sure that the region is not shared between threads
            let mut local_reg = Self::new_dummy_with_constants(
                starting_offset,
                starting_linear_coord,
                starting_constants,
                self.num_inner_cols,
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
            Ok::<_, Error>(res)
        })?;
        self.total_constants = constants.into_inner();
        self.linear_coord = linear_coord.into_inner();
        self.row = row.into_inner();
        Ok(())
    }

    /// Check if the region is dummy
    pub fn is_dummy(&self) -> bool {
        self.region.is_none()
    }

    /// duplicate_dummy
    pub fn duplicate_dummy(&self) -> Self {
        Self {
            region: None,
            linear_coord: self.linear_coord,
            num_inner_cols: self.num_inner_cols,
            row: self.row,
            total_constants: self.total_constants,
        }
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
            var.assign_with_duplication(
                &mut region.borrow_mut(),
                self.row,
                self.linear_coord,
                values,
                check_mode,
                single_inner_col,
            )
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
    pub fn flush(&mut self) {
        // increment by the difference between the current linear coord and the next row
        let remainder = self.linear_coord % self.num_inner_cols;
        if remainder != 0 {
            let diff = self.num_inner_cols - remainder;
            self.increment(diff);
        }
        assert!(self.linear_coord % self.num_inner_cols == 0);
    }

    /// increment constants
    pub fn increment_constants(&mut self, n: usize) {
        self.total_constants += n
    }
}
