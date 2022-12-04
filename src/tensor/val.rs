use super::*;
use halo2_proofs::plonk::Instance;
/// A wrapper around a tensor where the inner type is one of Halo2's `Value<F>`, `Value<Assigned<F>>`, `AssignedCell<Assigned<F>, F>`.
/// This enum is generally used to assign values to variables / advices already configured in a Halo2 circuit (usually represented as a [VarTensor]).
/// For instance can represent pre-trained neural network weights; or a known input to a network.
#[derive(Debug, Clone)]
pub enum ValTensor<F: FieldExt + TensorType> {
    Value {
        inner: Tensor<Value<F>>,
        dims: Vec<usize>,
    },
    AssignedValue {
        inner: Tensor<Value<Assigned<F>>>,
        dims: Vec<usize>,
    },
    PrevAssigned {
        inner: Tensor<AssignedCell<F, F>>,
        dims: Vec<usize>,
    },
    Instance {
        inner: Column<Instance>,
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
    pub fn new_instance(cs: &mut ConstraintSystem<F>, dims: Vec<usize>, equality: bool) -> Self {
        let col = cs.instance_column();
        if equality {
            cs.enable_equality(col);
        }
        ValTensor::Instance { inner: col, dims }
    }

    /// Calls `get_slice` on the inner tensor.
    pub fn get_slice(&self, indices: &[Range<usize>]) -> ValTensor<F> {
        match self {
            ValTensor::Value { inner: v, dims: _ } => {
                let slice = v.get_slice(indices);
                ValTensor::Value {
                    inner: slice.clone(),
                    dims: slice.dims().to_vec(),
                }
            }
            ValTensor::AssignedValue { inner: v, dims: _ } => {
                let slice = v.get_slice(indices);
                ValTensor::AssignedValue {
                    inner: slice.clone(),
                    dims: slice.dims().to_vec(),
                }
            }
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                let slice = v.get_slice(indices);
                ValTensor::PrevAssigned {
                    inner: slice.clone(),
                    dims: slice.dims().to_vec(),
                }
            }
            _ => unimplemented!(),
        }
    }

    /// Sets the `ValTensor`'s shape.
    pub fn reshape(&mut self, new_dims: &[usize]) {
        match self {
            ValTensor::Value { inner: v, dims: d } => {
                assert_eq!(
                    d.iter().product::<usize>(),
                    new_dims.iter().product::<usize>()
                );
                v.reshape(new_dims);
                *d = v.dims().to_vec();
            }
            ValTensor::AssignedValue { inner: v, dims: d } => {
                assert_eq!(
                    d.iter().product::<usize>(),
                    new_dims.iter().product::<usize>()
                );
                v.reshape(new_dims);
                *d = v.dims().to_vec();
            }
            ValTensor::PrevAssigned { inner: v, dims: d } => {
                assert_eq!(
                    d.iter().product::<usize>(),
                    new_dims.iter().product::<usize>()
                );
                v.reshape(new_dims);
                *d = v.dims().to_vec();
            }
            ValTensor::Instance { dims: d, .. } => {
                assert_eq!(
                    d.iter().product::<usize>(),
                    new_dims.iter().product::<usize>()
                );
                *d = new_dims.to_vec();
            }
        }
    }

    /// Calls `flatten` on the inner tensor.
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

    /// Returns the `dims` attribute of the `ValTensor`.
    pub fn dims(&self) -> &[usize] {
        match self {
            ValTensor::Value { dims: d, .. }
            | ValTensor::AssignedValue { dims: d, .. }
            | ValTensor::PrevAssigned { dims: d, .. }
            | ValTensor::Instance { dims: d, .. } => d,
        }
    }
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
