use crate::{
    circuit::BaseConfig,
    tensor::{Tensor, TensorType, ValTensor, ValType, VarTensor},
};
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

use super::{base::BaseOp, lookup::LookupOp};

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

    /// Create a new region context
    pub fn new_dummy_with_constants(offset: usize, constants: usize) -> RegionCtx<'a, F> {
        let region = None;
        RegionCtx {
            region,
            offset,
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
        let offset = AtomicUsize::new(self.offset());
        let constants = AtomicUsize::new(self.total_constants());
        *output = output.par_enum_map(|idx, _| {
            // we kick off the loop with the current offset
            let starting_offset = offset.fetch_add(0, Ordering::Relaxed);
            let starting_constants = constants.fetch_add(0, Ordering::Relaxed);
            // we need to make sure that the region is not shared between threads
            let mut local_reg = Self::new_dummy_with_constants(starting_offset, starting_constants);
            let res = inner_loop_function(idx, &mut local_reg);
            // we update the offset and constants
            offset.fetch_add(local_reg.offset() - starting_offset, Ordering::Relaxed);
            constants.fetch_add(
                local_reg.total_constants() - starting_constants,
                Ordering::Relaxed,
            );
            Ok::<_, Error>(res)
        })?;
        self.total_constants = constants.into_inner();
        self.offset = offset.into_inner();
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
            offset: self.offset,
            total_constants: self.total_constants,
        }
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
    pub fn assign_multiple_with_selector(
        &mut self,
        var: &[&VarTensor],
        values: &[ValTensor<F>],
        base_op: Option<BaseOp>,
        config: &BaseConfig<F>,
    ) -> Result<Vec<ValTensor<F>>, Error> {
        if let Some(region) = &self.region {
            // zip all values into pairs
            assert!(var.len() == values.len());
            // assert all values are of same len
            assert!(values.iter().map(|v| v.len()).collect::<HashSet<_>>().len() == 1);

            let mut results: Vec<Vec<ValType<F>>> = vec![vec![]; var.len()];
            (0..values[0].len()).for_each(|i| {
                let region = &mut region.borrow_mut();

                var.iter()
                    .zip(values.iter())
                    .enumerate()
                    .for_each(|(col, (var, value))| {
                        let val = match value {
                            ValTensor::Value { .. } => value.get_single_elem(i).unwrap(),
                            ValTensor::Instance { .. } => value.clone(),
                        };

                        results[col].push(
                            var.assign(region, self.offset + i, &val)
                                .unwrap()
                                .get_inner_tensor()
                                .unwrap()
                                .get_flat_index(0),
                        );
                    });

                // enable the selector
                if !self.is_dummy() {
                    if let Some(base_op) = &base_op {
                        let (x, y) = config.output.cartesian_coord(self.offset() + i);
                        let selector = config.selectors.get(&(base_op.clone(), x));
                        selector.unwrap().enable(region, y).unwrap();
                    }
                }
            });
            Ok(results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut t = Tensor::from(r.clone().into_iter());
                    t.reshape(values[i].dims());
                    t.into()
                })
                .collect::<Vec<_>>())
        } else {
            self.total_constants += values.iter().map(|v| v.num_constants()).sum::<usize>();
            Ok(values.to_vec())
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
            Ok(values.clone())
        }
    }

    /// Assign a valtensor to a vartensor
    pub fn assign_multiple_with_selector_and_omissions(
        &mut self,
        var: &[&VarTensor],
        values: &[ValTensor<F>],
        ommissions: &HashSet<&usize>,
        base_op: Option<BaseOp>,
        lookup_op: Option<LookupOp>,
        config: &BaseConfig<F>,
    ) -> Result<Vec<ValTensor<F>>, Error> {
        if let Some(region) = &self.region {
            // zip all values into pairs
            assert!(var.len() == values.len());
            // assert all values are of same len
            assert!(values.iter().map(|v| v.len()).collect::<HashSet<_>>().len() == 1);
            let mut results: Vec<Vec<ValType<F>>> = vec![vec![]; var.len()];
            let mut total_assigned = 0;
            (0..values[0].len()).for_each(|i| {
                let region = &mut region.borrow_mut();

                let omit_flag = if ommissions.contains(&i) {
                    HashSet::from([&0])
                } else {
                    HashSet::from([&1])
                };

                var.iter()
                    .zip(values.iter())
                    .enumerate()
                    .for_each(|(col, (var, value))| {
                        let val = match value {
                            ValTensor::Value { .. } => value.get_single_elem(i).unwrap(),
                            ValTensor::Instance { .. } => value.clone(),
                        };

                        results[col].push(
                            var.assign_with_omissions(
                                region,
                                self.offset + total_assigned,
                                &val,
                                &omit_flag,
                            )
                            .unwrap()
                            .get_inner_tensor()
                            .unwrap()
                            .get_flat_index(0),
                        );
                    });

                if !ommissions.contains(&i) {
                    total_assigned += 1;
                }

                // enable the selector
                if !self.is_dummy() {
                    let (x, y) = config
                        .output
                        .cartesian_coord(self.offset() + total_assigned);
                    if let Some(base_op) = &base_op {
                        let selector = config.selectors.get(&(base_op.clone(), x));
                        selector.unwrap().enable(region, y).unwrap();
                    }
                    if let Some(lookup_op) = &lookup_op {
                        let selector = config.lookup_selectors.get(&(lookup_op.clone(), x));
                        selector.unwrap().enable(region, y).unwrap();
                    }
                }
            });
            Ok(results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut t = Tensor::from(r.clone().into_iter());
                    t.reshape(values[i].dims());
                    t.into()
                })
                .collect::<Vec<_>>())
        } else {
            self.total_constants += values.iter().map(|v| v.num_constants()).sum::<usize>();
            let inner_tensors = values
                .iter()
                .map(|v| v.get_inner_tensor().unwrap())
                .collect::<Vec<_>>();
            for o in ommissions {
                self.total_constants -= inner_tensors
                    .iter()
                    .map(|t| t.get_flat_index(**o).is_constant() as usize)
                    .sum::<usize>();
            }
            Ok(values.to_vec())
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
            var.assign_with_omissions(&mut region.borrow_mut(), self.offset, values, ommissions)
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
    ) -> Result<(ValTensor<F>, usize), Error> {
        if let Some(region) = &self.region {
            // duplicates every nth element to adjust for column overflow
            var.assign_with_duplication(&mut region.borrow_mut(), self.offset, values, check_mode)
        } else {
            let (_, len, total_assigned_constants) =
                var.dummy_assign_with_duplication(self.offset, values)?;
            self.total_constants += total_assigned_constants;
            Ok((values.clone(), len))
        }
    }

    /// Enable a selector
    pub fn enable(&mut self, selector: Option<&Selector>, y: usize) -> Result<(), Error> {
        match &self.region {
            Some(region) => selector.unwrap().enable(&mut region.borrow_mut(), y),
            None => Ok(()),
        }
    }

    /// constrain equal
    pub fn constrain_equal(&mut self, a: &ValTensor<F>, b: &ValTensor<F>) -> Result<(), Error> {
        if let Some(region) = &self.region {
            let a = a.get_inner_tensor().unwrap();
            let b = b.get_inner_tensor().unwrap();
            a.iter().zip(b.iter()).try_for_each(|(a, b)| {
                let a = a.get_prev_assigned();
                let b = b.get_prev_assigned();
                // if they're both assigned, we can constrain them
                if let (Some(a), Some(b)) = (&a, &b) {
                    region.borrow_mut().constrain_equal(a.cell(), b.cell())
                // if one is Some and the other is None -- panic
                } else if a.is_some() || b.is_some() {
                    panic!("constrain_equal: one of the tensors is assigned and the other is not")
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
