use super::{ops::pad, *};
use halo2_proofs::plonk::Instance;
/// A wrapper around a [Tensor] where the inner type is one of Halo2's [`Value<F>`], [`Value<Assigned<F>>`], [`AssignedCell<Assigned<F>, F>`].
/// This enum is generally used to assign values to variables / advices already configured in a Halo2 circuit (usually represented as a [VarTensor]).
/// For instance can represent pre-trained neural network weights; or a known input to a network.
#[derive(Debug, Clone)]
pub enum ValTensor<F: FieldExt + TensorType> {
    /// A tensor of [Value], each containing a field element
    Value {
        /// Underlying [Tensor].
        inner: Tensor<Value<F>>,
        /// Vector of dimensions of the tensor.
        dims: Vec<usize>,
    },
    /// A tensor of [Value], each containing a ratio of field elements, which may be evaluated to produce plain field elements.
    AssignedValue {
        /// Underlying [Tensor].
        inner: Tensor<Value<Assigned<F>>>,
        /// Vector of dimensions of the [Tensor].
        dims: Vec<usize>,
    },
    /// A tensor of AssignedCells, with data both a value and the matrix cell to which it is assigned.
    PrevAssigned {
        /// Underlying [Tensor].
        inner: Tensor<AssignedCell<F, F>>,
        /// Vector of dimensions of the [Tensor].
        dims: Vec<usize>,
    },
    /// A tensor backed by an [Instance] column
    Instance {
        /// [Instance]
        inner: Column<Instance>,
        /// Vector of dimensions of the tensor.
        dims: Vec<usize>,
    },
}

impl<F: FieldExt + TensorType> From<Tensor<Value<F>>> for ValTensor<F> {
    fn from(t: Tensor<Value<F>>) -> ValTensor<F> {
        ValTensor::Value {
            inner: t.clone(),
            dims: t.dims().to_vec(),
        }
    }
}

impl<F: FieldExt + TensorType> From<Tensor<Value<Assigned<F>>>> for ValTensor<F> {
    fn from(t: Tensor<Value<Assigned<F>>>) -> ValTensor<F> {
        ValTensor::AssignedValue {
            inner: t.clone(),
            dims: t.dims().to_vec(),
        }
    }
}

impl<F: FieldExt + TensorType> From<Tensor<AssignedCell<F, F>>> for ValTensor<F> {
    fn from(t: Tensor<AssignedCell<F, F>>) -> ValTensor<F> {
        ValTensor::PrevAssigned {
            inner: t.clone(),
            dims: t.dims().to_vec(),
        }
    }
}

impl<F: FieldExt + TensorType> ValTensor<F> {
    /// Allocate a new [ValTensor::Instance] from the ConstraintSystem with the given tensor `dims`, optionally enabling `equality`.
    pub fn new_instance(cs: &mut ConstraintSystem<F>, dims: Vec<usize>, equality: bool) -> Self {
        let col = cs.instance_column();
        if equality {
            cs.enable_equality(col);
        }
        ValTensor::Instance { inner: col, dims }
    }

    /// Calls `int_evals` on the inner tensor.
    pub fn get_int_evals(&self) -> Result<Vec<i128>, Box<dyn Error>> {
        // finally convert to vector of integers
        let mut integer_evals: Vec<i128> = vec![];
        match self {
            ValTensor::Value { inner: v, dims: _ } => {
                let _ = v.map(|vaf| {
                    // we have to push to an externally created vector or else vaf.map() returns an evaluation wrapped in Value<> (which we don't want)
                    vaf.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_i128(f));
                    })
                });
            }
            ValTensor::AssignedValue { inner: v, dims: _ } => {
                let _ = v.map(|vaf| {
                    // we have to push to an externally created vector or else vaf.map() returns an evaluation wrapped in Value<> (which we don't want)
                    vaf.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_i128(f.evaluate()));
                    })
                });
            }
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                // convert assigned cells to Value<Assigned<F>> so we can extract the inner field element
                let w_vaf: Tensor<Value<Assigned<F>>> = v.map(|acaf| (acaf).value_field());

                let _ = w_vaf.map(|vaf| {
                    // we have to push to an externally created vector or else vaf.map() returns an evaluation wrapped in Value<> (which we don't want)
                    vaf.map(|f| {
                        integer_evals.push(crate::fieldutils::felt_to_i128(f.evaluate()));
                    })
                });
            }
            _ => return Err(Box::new(TensorError::WrongMethod)),
        };
        Ok(integer_evals)
    }

    /// Calls `get_slice` on the inner tensor.
    pub fn get_slice(&self, indices: &[Range<usize>]) -> Result<ValTensor<F>, Box<dyn Error>> {
        let slice = match self {
            ValTensor::Value { inner: v, dims: _ } => {
                let slice = v.get_slice(indices)?;
                ValTensor::Value {
                    inner: slice.clone(),
                    dims: slice.dims().to_vec(),
                }
            }
            ValTensor::AssignedValue { inner: v, dims: _ } => {
                let slice = v.get_slice(indices)?;
                ValTensor::AssignedValue {
                    inner: slice.clone(),
                    dims: slice.dims().to_vec(),
                }
            }
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                let slice = v.get_slice(indices)?;
                ValTensor::PrevAssigned {
                    inner: slice.clone(),
                    dims: slice.dims().to_vec(),
                }
            }
            _ => return Err(Box::new(TensorError::WrongMethod)),
        };
        Ok(slice)
    }

    /// Transposes the inner tensor
    pub fn transpose_2d(&mut self) -> Result<(), Box<dyn Error>> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                v.transpose_2d()?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                v.transpose_2d()?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                v.transpose_2d()?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { dims: d, .. } => {
                *d = vec![d.iter().product()];
            }
        }
        Ok(())
    }

    /// Transposes the inner tensor
    pub fn get_inner(&self) -> Result<Tensor<Value<F>>, TensorError> {
        Ok(match self {
            ValTensor::Value { inner: v, .. } => v.clone().into(),
            ValTensor::AssignedValue { inner: v, .. } => v.map(|x| x.evaluate()).into(),
            ValTensor::PrevAssigned { inner: v, .. } => {
                v.map(|x| x.value_field().evaluate()).into()
            }
            ValTensor::Instance { .. } => return Err(TensorError::WrongMethod),
        })
    }

    /// Sets the [ValTensor]'s shape.
    pub fn reshape(&mut self, new_dims: &[usize]) -> Result<(), Box<dyn Error>> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                v.reshape(new_dims);
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                v.reshape(new_dims);
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                v.reshape(new_dims);
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { dims: d, .. } => {
                if d.iter().product::<usize>() != new_dims.iter().product::<usize>() {
                    return Err(Box::new(TensorError::DimError));
                }
                *d = new_dims.to_vec();
            }
        };
        Ok(())
    }

    /// Calls `flatten` on the inner [Tensor].
    pub fn flatten(&mut self) {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                v.flatten();
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                v.flatten();
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                v.flatten();
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { dims: d, .. } => {
                *d = vec![d.iter().product()];
            }
        }
    }

    /// Calls `tile` on the inner [Tensor].
    pub fn tile(&mut self, n: usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                *v = v.tile(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                *v = v.tile(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                *v = v.tile(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `tile` on the inner [Tensor].
    pub fn pad(&mut self, padding: (usize, usize)) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                *v = pad(v, padding)?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                *v = pad(v, padding)?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                *v = pad(v, padding)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `repeat_rows` on the inner [Tensor].
    pub fn repeat_rows(&mut self, n: usize) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                *v = v.repeat_rows(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                *v = v.repeat_rows(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                *v = v.repeat_rows(n)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `pad_row_ones` on the inner [Tensor].
    pub fn pad_row_ones(&mut self) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                *v = v.pad_row_ones()?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                *v = v.pad_row_ones()?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                *v = v.pad_row_ones()?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `expand_new_shape` on the inner [Tensor].
    pub fn expand_new_shape(&mut self, shape: &[usize]) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                *v = v.expand_new_shape(shape)?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                *v = v.expand_new_shape(shape)?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                *v = v.expand_new_shape(shape)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Calls `expand_new_shape` on the inner [Tensor].
    pub fn doubly_blocked_toeplitz(
        &mut self,
        num_blocks: usize,
        num_cols: usize,
    ) -> Result<(), TensorError> {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                *v = v.doubly_blocked_toeplitz(num_blocks, num_cols)?;
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                *v = v.doubly_blocked_toeplitz(num_blocks, num_cols)?;
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                *v = v.doubly_blocked_toeplitz(num_blocks, num_cols)?;
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { .. } => {
                return Err(TensorError::WrongMethod);
            }
        }
        Ok(())
    }

    /// Pads each column
    pub fn append_to_row(&self, b: ValTensor<F>) -> Result<ValTensor<F>, TensorError> {
        match (self, b) {
            (ValTensor::Value { inner: v, .. }, ValTensor::Value { inner: v2, .. }) => {
                Ok(v.append_to_row(v2)?.into())
            }
            (
                ValTensor::AssignedValue { inner: v, .. },
                ValTensor::AssignedValue { inner: v2, .. },
            ) => Ok(v.append_to_row(v2)?.into()),
            (
                ValTensor::PrevAssigned { inner: v, .. },
                ValTensor::PrevAssigned { inner: v2, .. },
            ) => Ok(v.append_to_row(v2)?.into()),
            _ => {
                return Err(TensorError::WrongMethod);
            }
        }
    }

    /// Calls `tile` on the inner [Tensor].
    pub fn concat(&self, other: Self) -> Result<Self, TensorError> {
        let res = match (self, other) {
            (ValTensor::Value { inner: v1, .. }, ValTensor::Value { inner: v2, .. }) => {
                ValTensor::from(Tensor::new(Some(&[v1.clone(), v2]), &[2])?.combine()?)
            }
            (
                ValTensor::AssignedValue { inner: v1, .. },
                ValTensor::AssignedValue { inner: v2, .. },
            ) => ValTensor::from(Tensor::new(Some(&[v1.clone(), v2]), &[2])?.combine()?),
            (
                ValTensor::PrevAssigned { inner: v1, .. },
                ValTensor::PrevAssigned { inner: v2, .. },
            ) => ValTensor::from(Tensor::new(Some(&[v1.clone(), v2]), &[2])?.combine()?),
            _ => {
                return Err(TensorError::WrongMethod);
            }
        };
        Ok(res)
    }

    /// Returns the `dims` attribute of the [ValTensor].
    pub fn dims(&self) -> &[usize] {
        match self {
            ValTensor::Value { dims: d, .. }
            | ValTensor::AssignedValue { dims: d, .. }
            | ValTensor::PrevAssigned { dims: d, .. }
            | ValTensor::Instance { dims: d, .. } => d,
        }
    }
    /// A [String] representation of the [ValTensor] for display, for example in showing intermediate values in a computational graph.
    pub fn show(&self) -> String {
        match self.clone() {
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                let r: Tensor<i32> = v.into();
                format!("PrevAssigned {:?}", r)
            }
            ValTensor::Value { inner: v, dims: _ } => {
                let r: Tensor<i32> = v.into();
                format!("Value {:?}", r)
            }
            _ => "ValTensor not PrevAssigned".into(),
        }
    }
}
