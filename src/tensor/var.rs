use super::*;

/// A wrapper around a tensor where the inner type is one of Halo2's Column<Fixed> or Column<Advice>.
/// The wrapper allows for VarTensor's dimensions to differ from that of the inner (wrapped) tensor.
/// The inner tensor might, for instance, contain 3 Advice Columns. Each of those columns in turn
/// might be representing 3 elements laid out in the circuit. As such, though the inner tensor might
/// only be of dimension `[3]` we can set the VarTensor's dimension to `[3,3]` to capture information
/// about the column layout. This enum is generally used to configure and layout circuit variables / advices.
/// For instance can be used to represent neural network parameters within a circuit that we later assign to
/// using a the `assign` method called on a `ValTensor`.
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
    /// Calls `get_slice` on the inner tensor.
    pub fn get_slice(&self, indices: &[Range<usize>], new_dims: &[usize]) -> VarTensor {
        match self {
            VarTensor::Advice { inner: v, dims: _ } => {
                let mut new_inner = v.get_slice(indices);
                if new_dims.len() > 1 {
                    new_inner.reshape(&new_dims[0..new_dims.len() - 1]);
                }
                VarTensor::Advice {
                    inner: new_inner,
                    dims: new_dims.to_vec(),
                }
            }
	    VarTensor::Fixed { inner: v, dims: _ } => {
                let mut new_inner = v.get_slice(indices);
                if new_dims.len() > 1 {
                    new_inner.reshape(&new_dims[0..new_dims.len() - 1]);
                }
                VarTensor::Fixed {
                    inner: new_inner,
                    dims: new_dims.to_vec(),
                }
            }

            // VarTensor::Fixed { inner: v, dims: _ } => VarTensor::Fixed {
            //     inner: v.get_slice(indices),
            //     dims: new_dims.to_vec(),
            // },
        }
    }

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
    ) -> Tensor<Expression<F>> {
        let mut t = match &self {
            // when fixed we have 1 col per param
            VarTensor::Fixed { inner: f, dims: _ } => {
                f.map(|c| meta.query_fixed(c, Rotation(offset as i32)))
            }
            // when advice we have 1 col per row
            VarTensor::Advice { inner: a, dims: d } => a
                .map(|column| {
                    Tensor::from(
                        (0..*d.last().unwrap())
                            .map(|i| meta.query_advice(column, Rotation(offset as i32 + i as i32))),
                    )
                })
                .combine(),
        };
        t.reshape(self.dims());
        t
    }

    /// Retrieve the value represented at a specific index within the columns of the inner tensor.
    pub fn query_idx<F: FieldExt>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        idx: usize,
        offset: usize,
    ) -> Expression<F> {
        match &self {
            VarTensor::Fixed { inner: f, dims: _ } => {
                meta.query_fixed(f[idx], Rotation(offset as i32))
            }
            VarTensor::Advice { inner: a, dims: _ } => {
                meta.query_advice(a[idx], Rotation(offset as i32))
            }
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor.
    pub fn assign<F: FieldExt + TensorType>(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Tensor<AssignedCell<Assigned<F>, F>> {
        match values {
            ValTensor::Value { inner: v, dims: _ } => v.mc_enum_map(|coord, k| match &self {
                VarTensor::Fixed { inner: f, dims: _ } => region
                    .assign_fixed(|| "k", f.get(&coord), offset, || k.into())
                    .unwrap(),
                VarTensor::Advice { inner: a, dims: _ } => {
                    let coord = format_advice_coord(coord);
                    let last = coord.len() - 1;
                    region
                        .assign_advice(
                            || "k",
                            a.get(&coord[0..last]),
                            offset + coord[last],
                            || k.into(),
                        )
                        .unwrap()
                }
            }),
            ValTensor::PrevAssigned { inner: v, dims: _ } => {
                v.mc_enum_map(|coord, x| match &self {
                    VarTensor::Fixed { inner: _, dims: _ } => todo!(),
                    VarTensor::Advice { inner: a, dims: _ } => {
                        let coord = format_advice_coord(coord);
                        let last = coord.len() - 1;
                        x.copy_advice(|| "k", region, a.get(&coord[0..last]), offset + coord[last])
                            .unwrap()
                    }
                })
            }
            ValTensor::AssignedValue { inner: v, dims: _ } => {
                v.mc_enum_map(|coord, k| match &self {
                    VarTensor::Fixed { inner: f, dims: _ } => region
                        .assign_fixed(|| "k", f.get(&coord), offset, || k)
                        .unwrap(),
                    VarTensor::Advice { inner: a, dims: _ } => {
                        let coord = format_advice_coord(coord);
                        let last = coord.len() - 1;
                        region
                            .assign_advice(
                                || "k",
                                a.get(&coord[0..last]),
                                offset + coord[last],
                                || k.into(),
                            )
                            .unwrap()
                    }
                })
            }
        }
    }
}

fn format_advice_coord(coord: &[usize]) -> Vec<usize> {
    let last = coord.len() - 1;
    let mut v = coord.to_vec();
    if last == 0 {
        v.insert(0, 0);
    }
    v
}
