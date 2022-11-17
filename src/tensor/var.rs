use super::*;
use crate::abort;
use log::error;
/// A wrapper around a tensor where the inner type is one of Halo2's `Column<Fixed>` or `Column<Advice>`.
/// The wrapper allows for `VarTensor`'s dimensions to differ from that of the inner (wrapped) tensor.
/// The inner tensor might, for instance, contain 3 Advice Columns. Each of those columns in turn
/// might be representing 3 elements laid out in the circuit. As such, though the inner tensor might
/// only be of dimension `[3]` we can set the VarTensor's dimension to `[3,3]` to capture information
/// about the column layout. This enum is generally used to configure and layout circuit variables / advices.
/// For instance can be used to represent neural network parameters within a circuit that we later assign to
/// using the `assign` method called on a [ValTensor].
#[derive(Clone, Debug)]
pub enum VarTensor {
    Advice {
        inner: Column<Advice>,
        dims: Vec<usize>,
    },
    Fixed {
        inner: Tensor<Column<Fixed>>,
        dims: Vec<usize>,
    },
}

impl VarTensor {
    /// Sets the `VarTensor`'s shape.
    pub fn reshape(&mut self, new_dims: &[usize]) {
        match self {
            VarTensor::Advice { inner: _, dims: d } => {
                assert_eq!(
                    d.iter().product::<usize>(),
                    new_dims.iter().product::<usize>()
                );
                *d = new_dims.to_vec();
            }
            VarTensor::Fixed { inner: _, dims: d } => {
                assert_eq!(
                    d.iter().product::<usize>(),
                    new_dims.iter().product::<usize>()
                );
                *d = new_dims.to_vec();
            }
        }
    }

    /// Enables equality on Advice type `VarTensor`.
    pub fn enable_equality<F: FieldExt>(&self, meta: &mut ConstraintSystem<F>) {
        match self {
            VarTensor::Advice {
                inner: advice,
                dims: _,
            } => {
                meta.enable_equality(*advice);
            }
            VarTensor::Fixed { inner: _, dims: _ } => {}
        }
    }

    /// Returns the `dims` attribute of the `VarTensor`.
    pub fn dims(&self) -> &[usize] {
        match self {
            VarTensor::Advice { inner: _, dims: d } => d,
            VarTensor::Fixed { inner: _, dims: d } => d,
        }
    }
}

impl VarTensor {
    /// Retrieve the values represented within the columns of the `VarTensor` (recall that `VarTensor`
    /// is a Tensor of Halo2 columns).
    pub fn query<F: FieldExt>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        offset: usize,
    ) -> Result<Tensor<Expression<F>>, TensorError> {
        match &self {
            VarTensor::Fixed { inner: _, dims: _ } => {
                todo!()
            }
            // when advice we have 1 col per row
            VarTensor::Advice { inner: a, dims: d } => {
                let mut c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..d.iter().product::<usize>())
                        .map(|i| meta.query_advice(*a, Rotation(offset as i32 + i as i32))),
                );
                c.reshape(d);
                Ok(c)
            }
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor.
    pub fn assign<F: FieldExt + TensorType>(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, TensorError> {
        match values {
            ValTensor::Value { inner: v, dims: _ } => v.enum_map(|coord, k| match &self {
                VarTensor::Fixed { inner: f, dims: _ } => {
                    match region.assign_fixed(|| "k", f[coord], offset, || k.into()) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign ValTensor to VarTensor {:?}", e);
                        }
                    }
                }
                VarTensor::Advice { inner: a, dims: _ } => {
                    match region.assign_advice(|| "k", *a, offset + coord, || k.into()) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign ValTensor to VarTensor {:?}", e);
                        }
                    }
                }
            }),
            ValTensor::PrevAssigned { inner: v, dims: _ } => v.enum_map(|coord, x| match &self {
                VarTensor::Fixed { inner: _, dims: _ } => todo!(),
                VarTensor::Advice { inner: a, dims: _ } => {
                    match x.copy_advice(|| "k", region, *a, offset + coord) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to copy ValTensor to VarTensor {:?}", e);
                        }
                    }
                }
            }),
            ValTensor::AssignedValue { inner: v, dims: _ } => v.enum_map(|coord, k| match &self {
                VarTensor::Fixed { inner: f, dims: _ } => {
                    match region.assign_fixed(|| "k", f[coord], offset, || k) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign ValTensor to VarTensor {:?}", e);
                        }
                    }
                }
                VarTensor::Advice { inner: a, dims: _ } => {
                    match region.assign_advice(|| "k", *a, offset + coord, || k) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign ValTensor to VarTensor {:?}", e);
                        }
                    }
                }
            }),
        }
    }
}
