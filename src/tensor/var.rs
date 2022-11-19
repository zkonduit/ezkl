use super::*;
use crate::abort;
use log::error;
/// A wrapper around Halo2's `Column<Fixed>` or `Column<Advice>`.
/// The wrapper allows for `VarTensor`'s dimensions to differ from that of the inner (wrapped) columns.
/// The inner vector might, for instance, contain 3 Advice Columns. Each of those columns in turn
/// might be representing 3 elements laid out in the circuit. As such, though the inner tensor might
/// only be of dimension `[3]` we can set the VarTensor's dimension to `[3,3]` to capture information
/// about the column layout. This enum is generally used to configure and layout circuit variables / advices.
/// For instance can be used to represent neural network parameters within a circuit that we later assign to
/// using the `assign` method called on a [ValTensor].
#[derive(Clone, Debug)]
pub enum VarTensor {
    Advice {
        inner: Vec<Column<Advice>>,
        col_size: usize,
        capacity: usize,
        dims: Vec<usize>,
    },
    Fixed {
        inner: Vec<Column<Fixed>>,
        col_size: usize,
        capacity: usize,
        dims: Vec<usize>,
    },
}

impl VarTensor {
    pub fn new_advice<F: FieldExt>(
        cs: &mut ConstraintSystem<F>,
        k: usize,
        capacity: usize,
        dims: Vec<usize>,
        equality: bool,
    ) -> Self {
        let base = 2u32;
        let max_rows = base.pow(k as u32) as usize - cs.blinding_factors() - 1;
        let modulo = (capacity / max_rows) + 1;
        let mut advices = vec![];
        for _ in 0..modulo {
            let col = cs.advice_column();
            if equality {
                cs.enable_equality(col);
            }
            advices.push(col);
        }

        VarTensor::Advice {
            inner: advices,
            col_size: max_rows,
            capacity,
            dims,
        }
    }

    pub fn new_fixed<F: FieldExt>(
        cs: &mut ConstraintSystem<F>,
        k: usize,
        capacity: usize,
        dims: Vec<usize>,
        equality: bool,
    ) -> Self {
        let base = 2u32;
        let max_rows = base.pow(k as u32) as usize - cs.blinding_factors() - 1;
        let modulo = (capacity / max_rows) + 1;
        let mut fixed = vec![];
        for _ in 0..modulo {
            let col = cs.fixed_column();
            if equality {
                cs.enable_equality(col);
            }
            fixed.push(col);
        }

        VarTensor::Fixed {
            inner: fixed,
            col_size: max_rows,
            capacity,
            dims,
        }
    }

    /// Gets the dims of the object the VarTensor represents
    pub fn num_cols(&self) -> usize {
        match self {
            VarTensor::Advice { inner, .. } => inner.len(),
            _ => todo!(),
        }
    }

    /// Gets the dims of the object the VarTensor represents
    pub fn dims(&self) -> Vec<usize> {
        match self {
            VarTensor::Advice { dims: d, .. } => d.to_vec(),
            _ => todo!(),
        }
    }

    /// Sets the dims of the object the VarTensor represents
    pub fn reshape(&self, new_dims: &[usize]) -> Self {
        match self {
            VarTensor::Advice {
                inner,
                col_size,
                capacity,
                ..
            } => VarTensor::Advice {
                inner: inner.clone(),
                col_size: *col_size,
                capacity: *capacity,
                dims: new_dims.to_vec(),
            },
            _ => todo!(),
        }
    }

    pub fn cartesian_coord(&self, linear_coord: usize) -> (usize, usize) {
        match self {
            VarTensor::Advice { col_size, .. } => {
                let x = linear_coord / col_size;
                let y = linear_coord % col_size;
                (x, y)
            }
            VarTensor::Fixed { col_size, .. } => {
                let x = linear_coord / col_size;
                let y = linear_coord % col_size;
                (x, y)
            }
        }
    }

    /// Enables equality on Advice type `VarTensor`.
    pub fn enable_equality<F: FieldExt>(&self, cs: &mut ConstraintSystem<F>) {
        match self {
            VarTensor::Advice { inner: advices, .. } => {
                let _ = advices
                    .iter()
                    .map(|a| {
                        cs.enable_equality(*a);
                    })
                    .collect_vec();
            }
            VarTensor::Fixed { .. } => {}
        }
    }

    /// Returns the `capacity` attribute of the `VarTensor`.
    pub fn capacity(&self) -> usize {
        match self {
            VarTensor::Advice { capacity, .. } => *capacity,
            VarTensor::Fixed { capacity, .. } => *capacity,
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
            VarTensor::Fixed { .. } => {
                todo!()
            }
            // when advice we have 1 col per row
            VarTensor::Advice {
                inner: advices,
                dims,
                ..
            } => {
                let mut c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..dims.iter().product::<usize>()).map(|i| {
                        let (x, y) = self.cartesian_coord(i);
                        meta.query_advice(advices[x], Rotation(offset as i32 + y as i32))
                    }),
                );
                c.reshape(dims);
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
                VarTensor::Fixed { .. } => {
                    todo!()
                }
                VarTensor::Advice { inner: advices, .. } => {
                    let (x, y) = self.cartesian_coord(offset + coord);
                    match region.assign_advice(|| "k", advices[x], y, || k.into()) {
                        Ok(a) => a,
                        Err(e) => {
                            abort!("failed to assign ValTensor to VarTensor {:?}", e);
                        }
                    }
                }
            }),
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                v.enum_map(|coord, xcell| match &self {
                    VarTensor::Fixed { .. } => todo!(),
                    VarTensor::Advice { inner: advices, .. } => {
                        let (x, y) = self.cartesian_coord(offset + coord);
                        match xcell.copy_advice(|| "k", region, advices[x], y) {
                            Ok(a) => a,
                            Err(e) => {
                                abort!("failed to copy ValTensor to VarTensor {:?}", e);
                            }
                        }
                    }
                })
            }
            ValTensor::AssignedValue { inner: v, dims: _ } => v.enum_map(|coord, k| match &self {
                VarTensor::Fixed { .. } => {
                    todo!()
                }
                VarTensor::Advice { inner: advices, .. } => {
                    let (x, y) = self.cartesian_coord(offset + coord);
                    match region.assign_advice(|| "k", advices[x], y, || k) {
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
