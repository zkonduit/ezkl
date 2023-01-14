use super::*;
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
            _ => "ValTensor not PrevAssigned".into(),
        }
    }
}
