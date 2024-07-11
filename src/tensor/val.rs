use crate::circuit::region::ConstantsMap;
use maybe_rayon::slice::Iter;

use super::{
    ops::{intercalate_values, pad, resize},
    *,
};
use halo2_proofs::{arithmetic::Field, circuit::Cell, plonk::Instance};
use maybe_rayon::iter::{FilterMap, IntoParallelIterator, ParallelIterator};

pub(crate) fn create_constant_tensor<
    F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd,
>(
    val: F,
    len: usize,
) -> ValTensor<F> {
    let mut constant = Tensor::from(vec![ValType::Constant(val); len].into_iter());
    constant.set_visibility(&crate::graph::Visibility::Fixed);
    ValTensor::from(constant)
}

pub(crate) fn create_unit_tensor<
    F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd,
>(
    len: usize,
) -> ValTensor<F> {
    let mut unit = Tensor::from(vec![ValType::Constant(F::ONE); len].into_iter());
    unit.set_visibility(&crate::graph::Visibility::Fixed);
    ValTensor::from(unit)
}

pub(crate) fn create_zero_tensor<
    F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd,
>(
    len: usize,
) -> ValTensor<F> {
    let mut zero = Tensor::from(vec![ValType::Constant(F::ZERO); len].into_iter());
    zero.set_visibility(&crate::graph::Visibility::Fixed);
    ValTensor::from(zero)
}

#[derive(Debug, Clone)]
/// A [ValType] is a wrapper around Halo2 value(s).
pub enum ValType<F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd> {
    /// value
    Value(Value<F>),
    /// assigned  value
    AssignedValue(Value<Assigned<F>>),
    /// previously assigned value
    PrevAssigned(AssignedCell<F, F>),
    /// constant
    Constant(F),
    /// assigned constant
    AssignedConstant(AssignedCell<F, F>, F),
}

impl<F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd> ValType<F> {
    /// Returns the inner cell of the [ValType].
    pub fn cell(&self) -> Option<Cell> {
        match self {
            ValType::PrevAssigned(cell) => Some(cell.cell()),
            ValType::AssignedConstant(cell, _) => Some(cell.cell()),
            _ => None,
        }
    }

    /// Returns the assigned cell of the [ValType].
    pub fn assigned_cell(&self) -> Option<AssignedCell<F, F>> {
        match self {
            ValType::PrevAssigned(cell) => Some(cell.clone()),
            ValType::AssignedConstant(cell, _) => Some(cell.clone()),
            _ => None,
        }
    }

    /// Returns true if the value is previously assigned.
    pub fn is_prev_assigned(&self) -> bool {
        matches!(
            self,
            ValType::PrevAssigned(_) | ValType::AssignedConstant(..)
        )
    }
    /// Returns true if the value is constant.
    pub fn is_constant(&self) -> bool {
        matches!(self, ValType::Constant(_) | ValType::AssignedConstant(..))
    }

    /// get felt eval
    pub fn get_felt_eval(&self) -> Option<F> {
        let mut res = None;
        match self {
            ValType::Value(v) => {
                v.map(|f| {
                    res = Some(f);
                });
            }
            ValType::AssignedValue(v) => {
                v.map(|f| {
                    res = Some(f.evaluate());
                });
            }
            ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                v.value_field().map(|f| {
                    res = Some(f.evaluate());
                });
            }
            ValType::Constant(v) => {
                res = Some(*v);
            }
        }
        res
    }

    /// get_prev_assigned
    pub fn get_prev_assigned(&self) -> Option<AssignedCell<F, F>> {
        match self {
            ValType::PrevAssigned(v) => Some(v.clone()),
            ValType::AssignedConstant(v, _) => Some(v.clone()),
            _ => None,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<ValType<F>> for i32 {
    fn from(val: ValType<F>) -> Self {
        match val {
            ValType::Value(v) => {
                let mut output = 0_i32;
                let mut i = 0;
                v.map(|y| {
                    let e = felt_to_i32(y);
                    output = e;
                    i += 1;
                });
                output
            }
            ValType::AssignedValue(v) => {
                let mut output = 0_i32;
                let mut i = 0;
                v.evaluate().map(|y| {
                    let e = felt_to_i32(y);
                    output = e;
                    i += 1;
                });
                output
            }
            ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                let mut output = 0_i32;
                let mut i = 0;
                v.value().map(|y| {
                    let e = felt_to_i32(*y);
                    output = e;
                    i += 1;
                });
                output
            }
            ValType::Constant(v) => felt_to_i32(v),
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<F> for ValType<F> {
    fn from(t: F) -> ValType<F> {
        ValType::Constant(t)
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Value<F>> for ValType<F> {
    fn from(t: Value<F>) -> ValType<F> {
        ValType::Value(t)
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Value<Assigned<F>>> for ValType<F> {
    fn from(t: Value<Assigned<F>>) -> ValType<F> {
        ValType::AssignedValue(t)
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<AssignedCell<F, F>> for ValType<F> {
    fn from(t: AssignedCell<F, F>) -> ValType<F> {
        ValType::PrevAssigned(t)
    }
}

impl<F: PrimeField + TensorType + PartialOrd> TensorType for ValType<F>
where
    F: Field,
{
    fn zero() -> Option<Self> {
        Some(ValType::Constant(<F as Field>::ZERO))
    }
    fn one() -> Option<Self> {
        Some(ValType::Constant(<F as Field>::ONE))
    }
}

/// A [ValTensor] is a wrapper around a [Tensor] of [ValType].
/// or a column of an [Instance].
/// This is the type used for all intermediate values in a circuit.
/// It is also the type used for the inputs and outputs of a circuit.
#[derive(Debug, Clone)]
pub enum ValTensor<F: PrimeField + TensorType + PartialOrd> {
    /// A tensor of [Value], each containing a field element
    Value {
        /// Underlying [Tensor].
        inner: Tensor<ValType<F>>,
        /// Vector of dimensions of the tensor.
        dims: Vec<usize>,
        ///
        scale: crate::Scale,
    },
    /// A tensor backed by an [Instance] column
    Instance {
        /// [Instance]
        inner: Column<Instance>,
        /// Vector of dimensions of the tensor.
        dims: Vec<Vec<usize>>,
        /// Current instance num
        idx: usize,
        ///
        initial_offset: usize,
        ///
        scale: crate::Scale,
    },
}

impl<F: PrimeField + TensorType + PartialOrd> TensorType for ValTensor<F> {
    fn zero() -> Option<Self> {
        Some(ValTensor::Value {
            inner: Tensor::zero()?,
            dims: vec![],
            scale: 0,
        })
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Tensor<ValType<F>>> for ValTensor<F> {
    fn from(t: Tensor<ValType<F>>) -> ValTensor<F> {
        ValTensor::Value {
            inner: t.map(|x| x),
            dims: t.dims().to_vec(),
            scale: 1,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Vec<ValType<F>>> for ValTensor<F> {
    fn from(t: Vec<ValType<F>>) -> ValTensor<F> {
        ValTensor::Value {
            inner: t.clone().into_iter().into(),
            dims: vec![t.len()],
            scale: 1,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> TryFrom<Tensor<F>> for ValTensor<F> {
    type Error = TensorError;
    fn try_from(t: Tensor<F>) -> Result<ValTensor<F>, TensorError> {
        let visibility = t.visibility.clone();
        let dims = t.dims().to_vec();
        let inner = t
            .into_iter()
            .map(|x| {
                if let Some(vis) = &visibility {
                    match vis {
                        Visibility::Fixed => Ok(ValType::Constant(x)),
                        _ => Ok(Value::known(x).into()),
                    }
                } else {
                    Err(TensorError::UnsetVisibility)
                }
            })
            .collect::<Result<Vec<_>, TensorError>>()?;

        let mut inner: Tensor<ValType<F>> = inner.into_iter().into();
        inner.reshape(&dims)?;

        Ok(ValTensor::Value {
            inner,
            dims,
            scale: 1,
        })
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Tensor<Value<F>>> for ValTensor<F> {
    fn from(t: Tensor<Value<F>>) -> ValTensor<F> {
        ValTensor::Value {
            inner: t.map(|x| x.into()),
            dims: t.dims().to_vec(),
            scale: 1,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Tensor<Value<Assigned<F>>>> for ValTensor<F> {
    fn from(t: Tensor<Value<Assigned<F>>>) -> ValTensor<F> {
        ValTensor::Value {
            inner: t.map(|x| x.into()),
            dims: t.dims().to_vec(),
            scale: 1,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<Tensor<AssignedCell<F, F>>> for ValTensor<F> {
    fn from(t: Tensor<AssignedCell<F, F>>) -> ValTensor<F> {
        ValTensor::Value {
            inner: t.map(|x| x.into()),
            dims: t.dims().to_vec(),
            scale: 1,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> ValTensor<F> {
    /// Allocate a new [ValTensor::Value] from the given [Tensor] of [i64].
    pub fn from_i64_tensor(t: Tensor<i64>) -> ValTensor<F> {
        let inner = t.map(|x| ValType::Value(Value::known(i64_to_felt(x))));
        inner.into()
    }

    /// Allocate a new [ValTensor::Instance] from the ConstraintSystem with the given tensor `dims`, optionally enabling `equality`.
    pub fn new_instance(
        cs: &mut ConstraintSystem<F>,
        dims: Vec<Vec<usize>>,
        scale: crate::Scale,
    ) -> Self {
        let col = cs.instance_column();
        cs.enable_equality(col);

        ValTensor::Instance {
            inner: col,
            dims,
            initial_offset: 0,
            idx: 0,
            scale,
        }
    }

    /// Allocate a new [ValTensor::Instance] from the ConstraintSystem with the given tensor `dims`, optionally enabling `equality`.
    pub fn new_instance_from_col(
        dims: Vec<Vec<usize>>,
        scale: crate::Scale,
        col: Column<Instance>,
    ) -> Self {
        ValTensor::Instance {
            inner: col,
            dims,
            idx: 0,
            initial_offset: 0,
            scale,
        }
    }

    ///
    pub fn get_total_instance_len(&self) -> usize {
        match self {
            ValTensor::Instance { dims, .. } => dims
                .iter()
                .map(|x| {
                    if !x.is_empty() {
                        x.iter().product::<usize>()
                    } else {
                        0
                    }
                })
                .sum(),
            _ => 0,
        }
    }

    ///
    pub fn is_instance(&self) -> bool {
        matches!(self, ValTensor::Instance { .. })
    }

    /// reverse order of elements whilst preserving the shape
    pub fn reverse(&mut self) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, .. } => {
                v.reverse();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(())
    }

    ///
    pub fn set_initial_instance_offset(&mut self, offset: usize) {
        if let ValTensor::Instance { initial_offset, .. } = self {
            *initial_offset = offset;
        }
    }

    ///
    pub fn increment_idx(&mut self) {
        if let ValTensor::Instance { idx, .. } = self {
            *idx += 1;
        }
    }

    ///
    pub fn set_idx(&mut self, val: usize) {
        if let ValTensor::Instance { idx, .. } = self {
            *idx = val;
        }
    }

    ///
    pub fn get_idx(&self) -> usize {
        match self {
            ValTensor::Instance { idx, .. } => *idx,
            _ => 0,
        }
    }

    ///
    pub fn any_unknowns(&self) -> Result<bool, TensorError> {
        match self {
            ValTensor::Instance { .. } => Ok(true),
            _ => Ok(self.get_inner()?.iter().any(|&x| {
                let mut is_empty = true;
                x.map(|_| is_empty = false);
                is_empty
            })),
        }
    }

    /// Returns true if all the [ValTensor]'s [Value]s are assigned.
    pub fn all_prev_assigned(&self) -> bool {
        match self {
            ValTensor::Value { inner, .. } => inner.iter().all(|x| x.is_prev_assigned()),
            ValTensor::Instance { .. } => false,
        }
    }

    /// Set the [ValTensor]'s scale.
    pub fn set_scale(&mut self, scale: crate::Scale) {
        match self {
            ValTensor::Value { scale: s, .. } => *s = scale,
            ValTensor::Instance { scale: s, .. } => *s = scale,
        }
    }

    /// Returns the [ValTensor]'s scale.
    pub fn scale(&self) -> crate::Scale {
        match self {
            ValTensor::Value { scale, .. } => *scale,
            ValTensor::Instance { scale, .. } => *scale,
        }
    }

    /// Returns the number of constants in the [ValTensor].
    pub fn create_constants_map_iterator(
        &self,
    ) -> FilterMap<Iter<'_, ValType<F>>, fn(&ValType<F>) -> Option<(F, ValType<F>)>> {
        match self {
            ValTensor::Value { inner, .. } => inner.par_iter().filter_map(|x| {
                if let ValType::Constant(v) = x {
                    Some((*v, x.clone()))
                } else {
                    None
                }
            }),
            ValTensor::Instance { .. } => {
                unreachable!("Instance tensors do not have constants")
            }
        }
    }

    /// Returns the number of constants in the [ValTensor].
    pub fn create_constants_map(&self) -> ConstantsMap<F> {
        match self {
            ValTensor::Value { inner, .. } => inner
                .par_iter()
                .filter_map(|x| {
                    if let ValType::Constant(v) = x {
                        Some((*v, x.clone()))
                    } else {
                        None
                    }
                })
                .collect(),
            ValTensor::Instance { .. } => ConstantsMap::new(),
        }
    }

    /// Fetch the underlying [Tensor] of field elements.
    pub fn get_felt_evals(&self) -> Result<Tensor<F>, TensorError> {
        let mut felt_evals: Vec<F> = vec![];
        match self {
            ValTensor::Value {
                inner: v, dims: _, ..
            } => {
                // we have to push to an externally created vector or else vaf.map() returns an evaluation wrapped in Value<> (which we don't want)
                let _ = v.map(|vaf| {
                    if let Some(f) = vaf.get_felt_eval() {
                        felt_evals.push(f);
                    }
                });
            }
            _ => return Err(TensorError::WrongMethod),
        };

        let mut res: Tensor<F> = felt_evals.into_iter().into();
        res.reshape(self.dims())?;
        Ok(res)
    }

    /// Calls is_singleton on the inner tensor.
    pub fn is_singleton(&self) -> bool {
        match self {
            ValTensor::Value { inner, .. } => inner.is_singleton(),
            ValTensor::Instance { .. } => false,
        }
    }

    /// Calls `int_evals` on the inner tensor.
    pub fn get_int_evals(&self) -> Result<Tensor<i64>, TensorError> {
        // finally convert to vector of integers
        let mut integer_evals: Vec<i64> = vec![];
        match self {
            ValTensor::Value {
                inner: v, dims: _, ..
            } => {
                // we have to push to an externally created vector or else vaf.map() returns an evaluation wrapped in Value<> (which we don't want)
                let _ = v.map(|vaf| match vaf {
                    ValType::Value(v) => v.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_i64(f));
                    }),
                    ValType::AssignedValue(v) => v.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_i64(f.evaluate()));
                    }),
                    ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                        v.value_field().map(|f| {
                            integer_evals.push(crate::fieldutils::felt_to_i64(f.evaluate()));
                        })
                    }
                    ValType::Constant(v) => {
                        integer_evals.push(crate::fieldutils::felt_to_i64(v));
                        Value::unknown()
                    }
                });
            }
            _ => return Err(TensorError::WrongMethod),
        };
        let mut tensor: Tensor<i64> = integer_evals.into_iter().into();
        match tensor.reshape(self.dims()) {
            _ => {}
        };

        Ok(tensor)
    }

    /// Calls `pad_to_zero_rem` on the inner tensor.
    pub fn pad_to_zero_rem(&mut self, n: usize, pad: ValType<F>) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.pad_to_zero_rem(n, pad)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(())
    }

    /// Calls `get_slice` on the inner tensor.
    pub fn last(&self) -> Result<ValTensor<F>, TensorError> {
        let slice = match self {
            ValTensor::Value {
                inner: v,
                dims: _,
                scale,
            } => {
                let inner = v.last()?;
                let dims = inner.dims().to_vec();
                ValTensor::Value {
                    inner,
                    dims,
                    scale: *scale,
                }
            }
            _ => return Err(TensorError::WrongMethod),
        };
        Ok(slice)
    }

    /// Calls `get_slice` on the inner tensor.
    pub fn get_slice(&self, indices: &[Range<usize>]) -> Result<ValTensor<F>, TensorError> {
        if indices.iter().map(|x| x.end - x.start).collect::<Vec<_>>() == self.dims() {
            return Ok(self.clone());
        }
        let slice = match self {
            ValTensor::Value {
                inner: v,
                dims: _,
                scale,
            } => {
                let inner = v.get_slice(indices)?;
                let dims = inner.dims().to_vec();
                ValTensor::Value {
                    inner,
                    dims,
                    scale: *scale,
                }
            }
            _ => return Err(TensorError::WrongMethod),
        };
        Ok(slice)
    }

    /// Calls `get_single_elem` on the inner tensor.
    pub fn get_single_elem(&self, index: usize) -> Result<ValTensor<F>, TensorError> {
        let slice = match self {
            ValTensor::Value {
                inner: v,
                dims: _,
                scale,
            } => {
                let inner = Tensor::from(vec![v.get_flat_index(index)].into_iter());
                ValTensor::Value {
                    inner,
                    dims: vec![1],
                    scale: *scale,
                }
            }
            _ => return Err(TensorError::WrongMethod),
        };
        Ok(slice)
    }

    /// Fetches the inner tensor as a `Tensor<ValType<F>`
    pub fn get_inner_tensor(&self) -> Result<&Tensor<ValType<F>>, TensorError> {
        Ok(match self {
            ValTensor::Value { inner: v, .. } => v,
            ValTensor::Instance { .. } => return Err(TensorError::WrongMethod),
        })
    }

    /// Fetches the inner tensor as a `Tensor<ValType<F>`
    pub fn get_inner_tensor_mut(&mut self) -> Result<&mut Tensor<ValType<F>>, TensorError> {
        Ok(match self {
            ValTensor::Value { inner: v, .. } => v,
            ValTensor::Instance { .. } => return Err(TensorError::WrongMethod),
        })
    }

    /// Fetches the inner tensor as a `Tensor<Value<F>>`
    pub fn get_inner(&self) -> Result<Tensor<Value<F>>, TensorError> {
        Ok(match self {
            ValTensor::Value { inner: v, .. } => v.map(|x| match x {
                ValType::Value(v) => v,
                ValType::AssignedValue(v) => v.evaluate(),
                ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                    v.value_field().evaluate()
                }
                ValType::Constant(v) => Value::known(v),
            }),
            ValTensor::Instance { .. } => return Err(TensorError::WrongMethod),
        })
    }
    /// Calls `expand` on the inner tensor.
    pub fn expand(&mut self, dims: &[usize]) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.expand(dims)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(())
    }

    /// Calls `move_axis` on the inner tensor.
    pub fn move_axis(&mut self, source: usize, destination: usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.move_axis(source, destination)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(())
    }

    /// Sets the [ValTensor]'s shape.
    pub fn reshape(&mut self, new_dims: &[usize]) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                v.reshape(new_dims)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { dims: d, idx, .. } => {
                if d[*idx].iter().product::<usize>() != new_dims.iter().product::<usize>() {
                    return Err(TensorError::DimError(format!(
                        "Cannot reshape {:?} to {:?} as they have number of elements",
                        d[*idx], new_dims
                    )));
                }
                d[*idx] = new_dims.to_vec();
            }
        };
        Ok(())
    }

    /// Sets the [ValTensor]'s shape.
    pub fn slice(&mut self, axis: &usize, start: &usize, end: &usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = crate::tensor::ops::slice(v, axis, start, end)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(())
    }

    /// Calls `flatten` on the inner [Tensor].
    pub fn flatten(&mut self) {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                v.flatten();
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { dims: d, idx, .. } => {
                d[*idx] = vec![d[*idx].iter().product()];
            }
        }
    }

    /// Calls `duplicate_every_n` on the inner [Tensor].
    pub fn duplicate_every_n(
        &mut self,
        n: usize,
        num_repeats: usize,
        initial_offset: usize,
    ) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.duplicate_every_n(n, num_repeats, initial_offset)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// remove constant zero values constants
    pub fn remove_const_zero_values(&mut self) {
        match self {
            ValTensor::Value { inner: v, dims, .. } => {
                *v = v
                    .clone()
                    .into_par_iter()
                    .filter_map(|e| {
                        if let ValType::Constant(r) = e {
                            if r == F::ZERO {
                                return None;
                            }
                        } else if let ValType::AssignedConstant(_, r) = e {
                            if r == F::ZERO {
                                return None;
                            }
                        }
                        Some(e)
                    })
                    .collect();
                *dims = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {}
        }
    }

    /// gets constants
    pub fn get_const_zero_indices(&self) -> Vec<usize> {
        match self {
            ValTensor::Value { inner: v, .. } => v
                .par_iter()
                .enumerate()
                .filter_map(|(i, e)| {
                    if let ValType::Constant(r) = e {
                        if *r == F::ZERO {
                            return Some(i);
                        }
                    } else if let ValType::AssignedConstant(_, r) = e {
                        if *r == F::ZERO {
                            return Some(i);
                        }
                    }
                    None
                })
                .collect(),
            ValTensor::Instance { .. } => vec![],
        }
    }

    /// gets constants
    pub fn get_const_indices(&self) -> Vec<usize> {
        match self {
            ValTensor::Value { inner: v, .. } => v
                .par_iter()
                .enumerate()
                .filter_map(|(i, e)| {
                    if let ValType::Constant(_) = e {
                        Some(i)
                    } else if let ValType::AssignedConstant(_, _) = e {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect(),
            ValTensor::Instance { .. } => vec![],
        }
    }

    /// calls `remove_indices` on the inner [Tensor].
    pub fn remove_indices(
        &mut self,
        indices: &mut [usize],
        is_sorted: bool,
    ) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                // this is very slow how can we speed this up ?
                *v = v.remove_indices(indices, is_sorted)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                if indices.is_empty() {
                    return Ok(());
                } else {
                    return Err(TensorError::WrongMethod);
                }
            }
        }
        Ok(())
    }

    /// Calls `duplicate_every_n` on the inner [Tensor].
    pub fn remove_every_n(
        &mut self,
        n: usize,
        num_repeats: usize,
        initial_offset: usize,
    ) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.remove_every_n(n, num_repeats, initial_offset)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `intercalate_values` on the inner [Tensor].
    pub fn intercalate_values(
        &mut self,
        value: ValType<F>,
        stride: usize,
        axis: usize,
    ) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = intercalate_values(v, value, stride, axis)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }
    /// Calls `resize` on the inner [Tensor].
    pub fn resize(&mut self, scales: &[usize]) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = resize(v, scales)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(())
    }
    /// Calls `pad_spatial_dims` on the inner [Tensor].
    pub fn pad(&mut self, padding: Vec<(usize, usize)>, offset: usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = pad(v, padding, offset)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `len` on the inner [Tensor].
    pub fn len(&self) -> usize {
        match self {
            ValTensor::Value { dims, .. } => {
                if !dims.is_empty() && (dims != &[0]) {
                    dims.iter().product::<usize>()
                } else {
                    0
                }
            }
            ValTensor::Instance { dims, idx, .. } => {
                let dims = dims[*idx].clone();
                if !dims.is_empty() && (dims != [0]) {
                    dims.iter().product::<usize>()
                } else {
                    0
                }
            }
        }
    }

    ///
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Calls `concats` on the inner [Tensor].
    pub fn concat(&self, other: Self) -> Result<Self, TensorError> {
        let res = match (self, other) {
            (ValTensor::Value { inner: v1, .. }, ValTensor::Value { inner: v2, .. }) => {
                ValTensor::from(Tensor::new(Some(&[v1.clone(), v2]), &[2])?.combine()?)
            }
            _ => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(res)
    }

    /// Calls `concats` on the inner [Tensor].
    pub fn concat_axis(&self, other: Self, axis: &usize) -> Result<Self, TensorError> {
        let res = match (self, other) {
            (ValTensor::Value { inner: v1, .. }, ValTensor::Value { inner: v2, .. }) => {
                let v = crate::tensor::ops::concat(&[v1, &v2], *axis)?;
                ValTensor::from(v)
            }
            _ => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(res)
    }

    /// Returns the `dims` attribute of the [ValTensor].
    pub fn dims(&self) -> &[usize] {
        match self {
            ValTensor::Value { dims: d, .. } => d,
            ValTensor::Instance { dims: d, idx, .. } => &d[*idx],
        }
    }
    /// A [String] representation of the [ValTensor] for display, for example in showing intermediate values in a computational graph.
    pub fn show(&self) -> String {
        match self.clone() {
            ValTensor::Value {
                inner: v, dims: _, ..
            } => {
                let r: Tensor<i32> = v.map(|x| x.into());
                if r.len() > 10 {
                    let start = r[..5].to_vec();
                    let end = r[r.len() - 5..].to_vec();
                    // print the two split by ... in the middle
                    format!(
                        "[{} ... {}]",
                        start.iter().map(|x| format!("{}", x)).join(", "),
                        end.iter().map(|x| format!("{}", x)).join(", ")
                    )
                } else {
                    format!("{:?}", r)
                }
            }
            _ => "ValTensor not PrevAssigned".into(),
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> ValTensor<F> {
    /// inverts the inner values
    pub fn inverse(&self) -> Result<ValTensor<F>, TensorError> {
        let mut cloned_self = self.clone();

        match &mut cloned_self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.map(|x| match x {
                    ValType::AssignedValue(v) => ValType::AssignedValue(v.invert()),
                    ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                        ValType::AssignedValue(v.value_field().invert())
                    }
                    ValType::Value(v) => ValType::Value(v.map(|x| x.invert().unwrap_or(F::ZERO))),
                    ValType::Constant(v) => ValType::Constant(v.invert().unwrap_or(F::ZERO)),
                });
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(cloned_self)
    }
}
