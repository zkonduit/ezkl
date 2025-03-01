use super::{
    ops::{intercalate_values, pad, resize},
    *,
};
use crate::{circuit::region::ConstantsMap, fieldutils::felt_to_integer_rep};
use halo2_proofs::{arithmetic::Field, circuit::Cell, plonk::Instance};
use maybe_rayon::iter::{FilterMap, ParallelIterator};
use maybe_rayon::slice::{Iter, ParallelSlice};

/// Creates a new ValTensor filled with a constant value
///
/// # Arguments
/// * `val` - The constant value to fill the tensor with
/// * `len` - The length of the tensor
///
/// # Returns
/// A new ValTensor containing the constant value repeated `len` times with Fixed visibility
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

/// Creates a new ValTensor filled with ones
///
/// # Arguments
/// * `len` - The length of the tensor
///
/// # Returns
/// A new ValTensor containing ones with Fixed visibility
pub(crate) fn create_unit_tensor<
    F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd,
>(
    len: usize,
) -> ValTensor<F> {
    let mut unit = Tensor::from(vec![ValType::Constant(F::ONE); len].into_iter());
    unit.set_visibility(&crate::graph::Visibility::Fixed);
    ValTensor::from(unit)
}

/// Creates a new ValTensor filled with zeros
///
/// # Arguments
/// * `len` - The length of the tensor
///
/// # Returns
/// A new ValTensor containing zeros with Fixed visibility
pub(crate) fn create_zero_tensor<
    F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd,
>(
    len: usize,
) -> ValTensor<F> {
    let mut zero = Tensor::from(vec![ValType::Constant(F::ZERO); len].into_iter());
    zero.set_visibility(&crate::graph::Visibility::Fixed);
    ValTensor::from(zero)
}

/// A wrapper type for values in a zero-knowledge circuit
///
/// ValType represents different kinds of values that can appear in a circuit:
/// - Raw values that haven't been assigned
/// - Values that have been assigned to circuit cells
/// - Constants known at circuit creation time
/// - Previously assigned values that can be referenced
#[derive(Debug, Clone)]
pub enum ValType<F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd> {
    /// An unassigned value
    Value(Value<F>),
    /// A value that has been assigned to the circuit
    AssignedValue(Value<Assigned<F>>),
    /// A reference to a previously assigned value
    PrevAssigned(AssignedCell<F, F>),
    /// A constant value known at circuit creation time
    Constant(F),
    /// A constant value that has been assigned to a circuit cell
    AssignedConstant(AssignedCell<F, F>, F),
}

impl<F: PrimeField + TensorType + PartialOrd> From<ValType<F>> for IntegerRep {
    /// Converts a ValType to its integer representation
    fn from(val: ValType<F>) -> Self {
        match val {
            ValType::Value(v) => {
                let mut output = 0;
                v.map(|y| {
                    let e = felt_to_integer_rep(y);
                    output = e;
                });
                output
            }
            ValType::AssignedValue(v) => {
                let mut output = 0;
                v.evaluate().map(|y| {
                    let e = felt_to_integer_rep(y);
                    output = e;
                });
                output
            }
            ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                let mut output = 0;
                v.value().map(|y| {
                    let e = felt_to_integer_rep(*y);
                    output = e;
                });
                output
            }
            ValType::Constant(v) => felt_to_integer_rep(v),
        }
    }
}

impl<F: PrimeField + TensorType + std::marker::Send + std::marker::Sync + PartialOrd> ValType<F> {
    /// Returns the inner circuit cell if this value has been assigned to one
    pub fn cell(&self) -> Option<Cell> {
        match self {
            ValType::PrevAssigned(cell) => Some(cell.cell()),
            ValType::AssignedConstant(cell, _) => Some(cell.cell()),
            _ => None,
        }
    }

    /// Returns the assigned cell if this value has been assigned to one
    pub fn assigned_cell(&self) -> Option<AssignedCell<F, F>> {
        match self {
            ValType::PrevAssigned(cell) => Some(cell.clone()),
            ValType::AssignedConstant(cell, _) => Some(cell.clone()),
            _ => None,
        }
    }

    /// Returns true if this value was previously assigned to a circuit cell
    pub fn is_prev_assigned(&self) -> bool {
        matches!(
            self,
            ValType::PrevAssigned(_) | ValType::AssignedConstant(..)
        )
    }

    /// Returns true if this value is a constant
    pub fn is_constant(&self) -> bool {
        matches!(self, ValType::Constant(_) | ValType::AssignedConstant(..))
    }

    /// Gets the field element value if available
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

    /// Gets the previously assigned cell if available
    pub fn get_prev_assigned(&self) -> Option<AssignedCell<F, F>> {
        match self {
            ValType::PrevAssigned(v) => Some(v.clone()),
            ValType::AssignedConstant(v, _) => Some(v.clone()),
            _ => None,
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> From<F> for ValType<F> {
    /// Creates a Constant ValType from a field element
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
/// A tensor of values used in a zero-knowledge circuit
///
/// ValTensor represents either:
/// - A tensor of ValType values that can be assigned to circuit cells
/// - A tensor backed by an Instance column for public inputs
///
/// This is the main type used for intermediate values, inputs and outputs in a circuit.
#[derive(Debug, Clone)]
pub enum ValTensor<F: PrimeField + TensorType + PartialOrd> {
    /// A tensor of circuit values
    Value {
        /// The underlying tensor of values
        inner: Tensor<ValType<F>>,
        /// The dimensions of the tensor
        dims: Vec<usize>,
        /// Scale factor applied to values
        scale: crate::Scale,
    },
    /// A tensor backed by a public input column
    Instance {
        /// The instance column
        inner: Column<Instance>,
        /// Vector of dimension vectors (one per instance)
        dims: Vec<Vec<usize>>,
        /// Current instance index
        idx: usize,
        /// Initial offset in the instance column
        initial_offset: usize,
        /// Scale factor applied to values
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

// Additional From implementations...

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> ValTensor<F> {
    /// Creates a new ValTensor from a tensor of integer representations
    ///
    /// # Arguments
    /// * `t` - Tensor of integer values to convert to field elements
    pub fn from_integer_rep_tensor(t: Tensor<IntegerRep>) -> ValTensor<F> {
        let inner = t.map(|x| ValType::Value(Value::known(integer_rep_to_felt(x))));
        inner.into()
    }

    /// Creates a new public input instance column
    ///
    /// # Arguments
    /// * `cs` - The constraint system to create the column in
    /// * `dims` - Vector of dimension vectors for the instances
    /// * `scale` - Scale factor to apply to values
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

    /// Creates a new instance ValTensor from an existing instance column
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

    /// Gets the total length across all instances
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

    /// Returns true if this is an Instance tensor
    pub fn is_instance(&self) -> bool {
        matches!(self, ValTensor::Instance { .. })
    }

    /// Reverses the elements while preserving shape
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

    /// Sets the initial offset for instance values
    pub fn set_initial_instance_offset(&mut self, offset: usize) {
        if let ValTensor::Instance { initial_offset, .. } = self {
            *initial_offset = offset;
        }
    }

    /// Increments the instance index
    pub fn increment_idx(&mut self) {
        if let ValTensor::Instance { idx, .. } = self {
            *idx += 1;
        }
    }

    /// Sets the current instance index
    pub fn set_idx(&mut self, val: usize) {
        if let ValTensor::Instance { idx, .. } = self {
            *idx = val;
        }
    }

    /// Gets the current instance index
    pub fn get_idx(&self) -> usize {
        match self {
            ValTensor::Instance { idx, .. } => *idx,
            _ => 0,
        }
    }

    /// Returns true if any values are unknown/unassigned
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

    /// Returns true if all values are previously assigned
    pub fn all_prev_assigned(&self) -> bool {
        match self {
            ValTensor::Value { inner, .. } => inner.iter().all(|x| x.is_prev_assigned()),
            ValTensor::Instance { .. } => false,
        }
    }

    /// Sets the scale factor
    pub fn set_scale(&mut self, scale: crate::Scale) {
        match self {
            ValTensor::Value { scale: s, .. } => *s = scale,
            ValTensor::Instance { scale: s, .. } => *s = scale,
        }
    }

    /// Gets the current scale factor
    pub fn scale(&self) -> crate::Scale {
        match self {
            ValTensor::Value { scale, .. } => *scale,
            ValTensor::Instance { scale, .. } => *scale,
        }
    }

    /// Returns an iterator over constant values in the tensor
    /// Uses parallel processing for tensors larger than a threshold
    ///
    /// # Returns
    /// An iterator yielding (value, ValType) pairs for constants
    #[allow(clippy::type_complexity)]
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

    /// Creates a map of all constant values in the tensor
    /// Uses parallel processing for tensors larger than 1 million elements
    ///
    /// # Returns
    /// A map from field elements to their ValType representations
    pub fn create_constants_map(&self) -> ConstantsMap<F> {
        let threshold = 1_000_000; // Tuned using the benchmarks

        if self.len() < threshold {
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
        } else {
            // Use parallel processing for larger arrays
            let num_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let chunk_size = (self.len() / num_cores).max(100_000);

            match self {
                ValTensor::Value { inner, .. } => inner
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk.par_iter().filter_map(|x| {
                            if let ValType::Constant(v) = x {
                                Some((*v, x.clone()))
                            } else {
                                None
                            }
                        })
                    })
                    .collect(),
                ValTensor::Instance { .. } => ConstantsMap::new(),
            }
        }
    }

    /// Gets the field element evaluations for all values
    ///
    /// # Returns
    /// A tensor containing the field element evaluations
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
    pub fn get_felt_evals(&self) -> Result<Tensor<F>, TensorError> {
        let mut felt_evals: Vec<F> = vec![];
        match self {
            ValTensor::Value {
                inner: v, dims: _, ..
            } => {
                v.map(|vaf| {
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

    /// Checks if this is a singleton tensor (1 element)
    pub fn is_singleton(&self) -> bool {
        match self {
            ValTensor::Value { inner, .. } => inner.is_singleton(),
            ValTensor::Instance { .. } => false,
        }
    }

    /// Computes the sign of each value in the tensor
    ///
    /// # Returns
    /// A new tensor containing -1, 0, or 1 for each value's sign
    pub fn sign(&self) -> Result<Self, TensorError> {
        let evals = self.int_evals()?;
        Ok(evals
            .par_enum_map(|_, val| {
                Ok::<_, TensorError>(ValType::Value(Value::known(integer_rep_to_felt(
                    val.signum(),
                ))))
            })?
            .into())
    }

    /// Decomposes each value into base-n digits
    ///
    /// # Arguments
    /// * `base` - The base to decompose into
    /// * `n` - Number of digits to produce
    ///
    /// # Returns
    /// A tensor containing the base-n decomposition of each value
    pub fn decompose(&self, base: usize, n: usize) -> Result<Self, TensorError> {
        let res = self
            .get_inner()?
            .par_iter()
            .map(|x| {
                let mut is_empty = true;
                x.map(|_| is_empty = false);
                if is_empty {
                    Ok::<_, TensorError>(vec![Value::<F>::unknown(); n + 1])
                } else {
                    let mut res = vec![Value::unknown(); n + 1];
                    let mut int_rep = 0;

                    x.map(|f| {
                        int_rep = crate::fieldutils::felt_to_integer_rep(f);
                    });
                    let decompe = crate::tensor::ops::get_rep(&int_rep, base, n)?;

                    for (i, x) in decompe.iter().enumerate() {
                        res[i] = Value::known(crate::fieldutils::integer_rep_to_felt(*x));
                    }
                    Ok(res)
                }
            })
            .collect::<Result<Vec<_>, _>>();

        let mut tensor = Tensor::from(res?.into_iter().flatten().collect::<Vec<_>>().into_iter());
        let mut dims = self.dims().to_vec();
        dims.push(n + 1);

        tensor.reshape(&dims)?;

        Ok(tensor.into())
    }

    /// Gets the integer representation of all values
    ///
    /// # Returns
    /// A tensor containing the integer representation of each value
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
    #[allow(unused)]
    pub fn int_evals(&self) -> Result<Tensor<IntegerRep>, TensorError> {
        let mut integer_evals: Vec<IntegerRep> = vec![];
        match self {
            ValTensor::Value {
                inner: v, dims: _, ..
            } => {
                v.map(|vaf| match vaf {
                    ValType::Value(v) => v.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_integer_rep(f));
                    }),
                    ValType::AssignedValue(v) => v.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_integer_rep(f.evaluate()));
                    }),
                    ValType::PrevAssigned(v) | ValType::AssignedConstant(v, ..) => {
                        v.value_field().map(|f| {
                            integer_evals
                                .push(crate::fieldutils::felt_to_integer_rep(f.evaluate()));
                        })
                    }
                    ValType::Constant(v) => {
                        integer_evals.push(crate::fieldutils::felt_to_integer_rep(v));
                        Value::unknown()
                    }
                });
            }
            _ => return Err(TensorError::WrongMethod),
        };
        let mut tensor: Tensor<IntegerRep> = integer_evals.into_iter().into();
        tensor.reshape(self.dims());

        Ok(tensor)
    }

    /// Pads the tensor with zeros until its size is divisible by n
    ///
    /// # Arguments
    /// * `n` - The divisor to pad to
    /// * `pad` - The value to use for padding
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

    /// Gets the last value/slice along the first dimension
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

    /// Gets the first value/slice along the first dimension
    pub fn first(&self) -> Result<ValTensor<F>, TensorError> {
        let slice = match self {
            ValTensor::Value {
                inner: v,
                dims: _,
                scale,
            } => {
                let inner = v.first()?;
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

    /// Gets a slice of the tensor
    ///
    /// # Arguments
    /// * `indices` - The ranges to slice along each dimension
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

    /// Gets a single element from the flattened tensor
    ///
    /// # Arguments
    /// * `index` - The linear index of the element
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

    /// Gets a reference to the inner tensor
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
    pub fn get_inner_tensor(&self) -> Result<&Tensor<ValType<F>>, TensorError> {
        Ok(match self {
            ValTensor::Value { inner: v, .. } => v,
            ValTensor::Instance { .. } => return Err(TensorError::WrongMethod),
        })
    }

    /// Gets a mutable reference to the inner tensor
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
    pub fn get_inner_tensor_mut(&mut self) -> Result<&mut Tensor<ValType<F>>, TensorError> {
        Ok(match self {
            ValTensor::Value { inner: v, .. } => v,
            ValTensor::Instance { .. } => return Err(TensorError::WrongMethod),
        })
    }

    /// Gets the inner values as a tensor of Value<F>
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

    /// Expands the tensor by repeating elements along new dimensions
    ///
    /// # Arguments
    /// * `dims` - The sizes of the new dimensions to add
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

    /// Moves an axis of the tensor to a new position
    ///
    /// # Arguments
    /// * `source` - The axis to move
    /// * `destination` - Where to move it to
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

    /// Reshapes the tensor to new dimensions
    /// The total number of elements must remain the same
    ///
    /// # Arguments
    /// * `new_dims` - The new dimensions to reshape to
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

    /// Takes a slice of the tensor along a given axis
    ///
    /// # Arguments
    /// * `axis` - The axis to slice along
    /// * `start` - Starting index
    /// * `end` - Ending index
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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

    /// Flattens all dimensions into a single dimension
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

    /// Duplicates elements periodically through the tensor
    ///
    /// # Arguments
    /// * `n` - Period of duplication
    /// * `num_repeats` - Number of times to duplicate each element
    /// * `initial_offset` - Offset to start duplicating from
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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

    /// Selects every nth element from the tensor
    ///
    /// # Arguments
    /// * `n` - Period of selection
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
    pub fn get_every_n(&mut self, n: usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.get_every_n(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Removes every nth element from the tensor
    ///
    /// # Arguments
    /// * `n` - Period of removal
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
    pub fn exclude_every_n(&mut self, n: usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
                *v = v.exclude_every_n(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Removes constant zero values from the tensor
    /// Uses parallel processing for tensors larger than a threshold
    pub fn remove_const_zero_values(&mut self) {
        let size_threshold = 1_000_000; // Tuned using benchmarks

        if self.len() < size_threshold {
            // Single-threaded for small tensors
            match self {
                ValTensor::Value { inner: v, dims, .. } => {
                    *v = v
                        .clone()
                        .into_iter()
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
        } else {
            // Parallel processing for large tensors
            let num_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let chunk_size = (self.len() / num_cores).max(100_000);

            match self {
                ValTensor::Value { inner: v, dims, .. } => {
                    *v = v
                        .par_chunks_mut(chunk_size)
                        .flat_map(|chunk| {
                            chunk.par_iter_mut().filter_map(|e| {
                                if let ValType::Constant(r) = e {
                                    if *r == F::ZERO {
                                        return None;
                                    }
                                } else if let ValType::AssignedConstant(_, r) = e {
                                    if *r == F::ZERO {
                                        return None;
                                    }
                                }
                                Some(e.clone())
                            })
                        })
                        .collect();
                    *dims = v.dims().to_vec();
                }
                ValTensor::Instance { .. } => {}
            }
        }
    }

    /// Gets the indices of all constant zero values
    /// Uses parallel processing for large tensors
    ///
    /// # Returns
    /// A vector of indices where constant zero values are located
    pub fn get_const_zero_indices(&self) -> Vec<usize> {
        let size_threshold = 1_000_000;

        if self.len() < size_threshold {
            // Single-threaded for small tensors
            match &self {
                ValTensor::Value { inner: v, .. } => v
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| match e {
                        ValType::Constant(r) | ValType::AssignedConstant(_, r) => {
                            (*r == F::ZERO).then_some(i)
                        }
                        _ => None,
                    })
                    .collect(),
                ValTensor::Instance { .. } => vec![],
            }
        } else {
            // Parallel processing for large tensors
            let num_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let chunk_size = (self.len() / num_cores).max(100_000);

            match &self {
                ValTensor::Value { inner: v, .. } => v
                    .par_chunks(chunk_size)
                    .enumerate()
                    .flat_map(|(chunk_idx, chunk)| {
                        chunk
                            .par_iter()
                            .enumerate()
                            .filter_map(move |(i, e)| match e {
                                ValType::Constant(r) | ValType::AssignedConstant(_, r) => {
                                    (*r == F::ZERO).then_some(chunk_idx * chunk_size + i)
                                }
                                _ => None,
                            })
                    })
                    .collect::<Vec<_>>(),
                ValTensor::Instance { .. } => vec![],
            }
        }
    }

    /// Gets the indices of all constant values
    /// Uses parallel processing for large tensors
    ///
    /// # Returns
    /// A vector of indices where constant values are located
    pub fn get_const_indices(&self) -> Vec<usize> {
        let size_threshold = 1_000_000;

        if self.len() < size_threshold {
            // Single-threaded for small tensors
            match &self {
                ValTensor::Value { inner: v, .. } => v
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| match e {
                        ValType::Constant(_) | ValType::AssignedConstant(_, _) => Some(i),
                        _ => None,
                    })
                    .collect(),
                ValTensor::Instance { .. } => vec![],
            }
        } else {
            // Parallel processing for large tensors
            let num_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let chunk_size = (self.len() / num_cores).max(100_000);

            match &self {
                ValTensor::Value { inner: v, .. } => v
                    .par_chunks(chunk_size)
                    .enumerate()
                    .flat_map(|(chunk_idx, chunk)| {
                        chunk
                            .par_iter()
                            .enumerate()
                            .filter_map(move |(i, e)| match e {
                                ValType::Constant(_) | ValType::AssignedConstant(_, _) => {
                                    Some(chunk_idx * chunk_size + i)
                                }
                                _ => None,
                            })
                    })
                    .collect::<Vec<_>>(),
                ValTensor::Instance { .. } => vec![],
            }
        }
    }

    /// Removes specified indices from the tensor
    ///
    /// # Arguments
    /// * `indices` - Indices to remove
    /// * `is_sorted` - Whether indices are pre-sorted
    ///
    /// # Errors
    /// Returns an error if called on non-empty Instance tensor
    pub fn remove_indices(
        &mut self,
        indices: &mut [usize],
        is_sorted: bool,
    ) -> Result<(), TensorError> {
        match self {
            ValTensor::Value {
                inner: v, dims: d, ..
            } => {
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

    /// Removes elements periodically from the tensor
    ///
    /// # Arguments
    /// * `n` - Period of removal
    /// * `num_repeats` - Number of elements to remove in each period
    /// * `initial_offset` - Offset to start removal from
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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

    /// Inserts values periodically into the tensor
    ///
    /// # Arguments
    /// * `value` - Value to insert
    /// * `stride` - Period between insertions
    /// * `axis` - Axis to insert along
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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

    /// Resizes the tensor by scaling dimensions
    ///
    /// # Arguments
    /// * `scales` - Scaling factors for each dimension
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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

    /// Pads the tensor with zeros along specified dimensions
    ///
    /// # Arguments
    /// * `padding` - (before, after) padding pairs for each dimension
    /// * `offset` - Offset for padding values
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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

    /// Gets the total number of elements in the tensor
    pub fn len(&self) -> usize {
        match self {
            ValTensor::Value { dims, inner, .. } => {
                if !dims.is_empty() && (dims != &[0]) {
                    dims.iter().product::<usize>()
                } else if dims.is_empty() {
                    inner.inner.len()
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

    /// Returns true if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Concatenates two tensors along the first dimension
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

    /// Concatenates two tensors along a specified axis
    ///
    /// # Arguments
    /// * `other` - Tensor to concatenate with this one
    /// * `axis` - Axis along which to concatenate
    ///
    /// # Returns
    /// A new tensor containing the concatenated values
    ///
    /// # Errors
    /// Returns an error if either tensor is an Instance tensor
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

    /// Gets the dimensions of the tensor
    ///
    /// # Returns
    /// A slice containing the size of each dimension
    pub fn dims(&self) -> &[usize] {
        match self {
            ValTensor::Value { dims: d, .. } => d,
            ValTensor::Instance { dims: d, idx, .. } => &d[*idx],
        }
    }

    /// Creates a string representation of the tensor's values
    /// Truncates long tensors to show first and last 5 elements
    ///
    /// # Returns
    /// A string showing the tensor's values
    pub fn show(&self) -> String {
        let r = match self.int_evals() {
            Ok(v) => v,
            Err(_) => return "ValTensor not PrevAssigned".into(),
        };

        if r.len() > 10 {
            let start = r[..5].to_vec();
            let end = r[r.len() - 5..].to_vec();
            format!(
                "[{} ... {}]",
                start.iter().map(|x| format!("{}", x)).join(", "),
                end.iter().map(|x| format!("{}", x)).join(", ")
            )
        } else {
            format!("{:?}", r)
        }
    }
}

impl<F: PrimeField + TensorType + PartialOrd> ValTensor<F> {
    /// Computes the multiplicative inverse of each element
    ///
    /// # Returns
    /// A new tensor containing the inverses
    /// Elements with no inverse are set to zero
    ///
    /// # Errors
    /// Returns an error if called on an Instance tensor
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
