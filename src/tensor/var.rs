use super::*;

#[derive(Clone, Debug)]
pub enum VarTensor {
    Advice {
        inner: Tensor<Column<Advice>>,
        dims: Vec<usize>,
    },
    Fixed {
        inner: Tensor<Column<Fixed>>,
        dims: Vec<usize>,
    },
}

impl From<Tensor<Column<Advice>>> for VarTensor {
    fn from(t: Tensor<Column<Advice>>) -> VarTensor {
        VarTensor::Advice {
            inner: t.clone(),
            dims: t.dims().to_vec(),
        }
    }
}

impl From<Tensor<Column<Fixed>>> for VarTensor {
    fn from(t: Tensor<Column<Fixed>>) -> VarTensor {
        VarTensor::Fixed {
            inner: t.clone(),
            dims: t.dims().to_vec(),
        }
    }
}

impl VarTensor {
    pub fn get_slice(&self, indices: &[Range<usize>], new_dims: &[usize]) -> VarTensor {
        match self {
            VarTensor::Advice { inner: v, dims: _ } => VarTensor::Advice {
                inner: v.get_slice(indices),
                dims: new_dims.to_vec(),
            },
            VarTensor::Fixed { inner: v, dims: _ } => VarTensor::Fixed {
                inner: v.get_slice(indices),
                dims: new_dims.to_vec(),
            },
        }
    }

    pub fn reshape(&mut self, new_dims: &[usize]) {
        match self {
            VarTensor::Advice { inner: _, dims: d } => {
                assert!(d.iter().product::<usize>() == new_dims.iter().product());
                *d = new_dims.to_vec();
            }
            VarTensor::Fixed { inner: _, dims: d } => {
                assert!(d.iter().product::<usize>() == new_dims.iter().product());
                *d = new_dims.to_vec();
            }
        }
    }

    pub fn enable_equality<F: FieldExt>(&self, meta: &mut ConstraintSystem<F>) {
        match self {
            VarTensor::Advice {
                inner: advices,
                dims: _,
            } => {
                for advice in advices.iter() {
                    meta.enable_equality(*advice);
                }
            }
            VarTensor::Fixed { inner: _, dims: _ } => {}
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            VarTensor::Advice { inner: _, dims: d } => d,
            VarTensor::Fixed { inner: _, dims: d } => d,
        }
    }
}
