use super::*;
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
    /// A VarTensor for holding Advice values, which are assigned at proving time.
    Advice {
        /// Vec of Advice columns
        inner: Vec<Column<Advice>>,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
        /// Total capacity (number of advice cells), usually inner.len()*col_size
        capacity: usize,
        /// Vector of dimensions of the tensor we are representing using this storage. Note that the shape of the storage and this shape can differ.
        dims: Vec<usize>,
    },
    /// A VarTensor for holding Fixed values, which are assigned at circuit definition time.
    Fixed {
        /// Vec of Fixed columns
        inner: Vec<Column<Fixed>>,
        /// Number of rows available to be used in each column of the storage
        col_size: usize,
        /// Total capacity (number of advice cells), usually inner.len()*col_size
        capacity: usize,
        /// Vector of dimensions of the tensor we are representing using this storage. Note that the shape of the storage and this shape can differ.
        dims: Vec<usize>,
    },
}

impl VarTensor {
    /// Create a new VarTensor::Advice
    /// Arguments
    ///
    /// * `cs` - `ConstraintSystem` from which the columns will be allocated.
    /// * `k` - log2 number of rows in the matrix, including any system and blinding rows.
    /// * `capacity` - number of advice cells for this tensor
    /// * `dims` - `Vec` of dimensions of the tensor we are representing. Note that the shape of the storage and this shape can differ.
    /// * `equality` - true if we want to enable equality constraints for the columns involved.
    /// * `max_rot` - maximum number of rotations that we allow for this VarTensor. Rotations affect performance.
    pub fn new_advice<F: FieldExt>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        capacity: usize,
        dims: Vec<usize>,
        equality: bool,
    ) -> Self {
        let base = 2u32;
        let max_rows = base.pow(logrows as u32) as usize - cs.blinding_factors() - 1;

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

    /// Create a new VarTensor::Fixed
    /// `cs` is the `ConstraintSystem` from which the columns will be allocated.
    /// `k` is the log2 number of rows in the matrix, including any system and blinding rows.
    /// `capacity` is the number of fixed cells for this tensor
    /// `dims` is the `Vec` of dimensions of the tensor we are representing. Note that the shape of the storage and this shape can differ.
    /// `equality` should be true if we want to enable equality constraints for the columns involved.
    /// `max_rot` is the maximum number of rotations that we allow for this VarTensor. Rotations affect performance.
    pub fn new_fixed<F: FieldExt>(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        capacity: usize,
        dims: Vec<usize>,
        equality: bool,
    ) -> Self {
        let base = 2u32;
        let max_rows = base.pow(logrows as u32) as usize - cs.blinding_factors() - 1;
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
            VarTensor::Fixed { inner, .. } => inner.len(),
        }
    }

    /// Gets the size of each column
    pub fn col_size(&self) -> usize {
        match self {
            VarTensor::Advice { col_size, .. } | VarTensor::Fixed { col_size, .. } => *col_size,
        }
    }

    /// Gets the dims of the object the VarTensor represents
    pub fn dims(&self) -> Vec<usize> {
        match self {
            VarTensor::Advice { dims: d, .. } | VarTensor::Fixed { dims: d, .. } => d.to_vec(),
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
            VarTensor::Fixed {
                inner,
                col_size,
                capacity,
                ..
            } => VarTensor::Fixed {
                inner: inner.clone(),
                col_size: *col_size,
                capacity: *capacity,
                dims: new_dims.to_vec(),
            },
        }
    }

    /// Take a linear coordinate and output the (column, row) position in the storage block.
    pub fn cartesian_coord(&self, linear_coord: usize) -> (usize, usize) {
        match self {
            VarTensor::Advice { col_size, .. } | VarTensor::Fixed { col_size, .. } => {
                let x = linear_coord / col_size;
                let y = linear_coord % col_size;
                (x, y)
            }
        }
    }

    /// Returns the `capacity` attribute of the `VarTensor`.
    pub fn capacity(&self) -> usize {
        match self {
            VarTensor::Advice { capacity, .. } | VarTensor::Fixed { capacity, .. } => *capacity,
        }
    }
}

impl VarTensor {
    /// Retrieve the values represented within the columns of the `VarTensor` (recall that `VarTensor`
    /// is a Tensor of Halo2 columns).
    pub fn query_rng<F: FieldExt>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        rotation_offset: i32,
        idx_offset: usize,
        rng: usize,
    ) -> Result<Tensor<Expression<F>>, halo2_proofs::plonk::Error> {
        match &self {
            VarTensor::Fixed { inner: fixed, .. } => {
                let c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..rng).map(|i| {
                        let (x, y) = self.cartesian_coord(idx_offset + i);
                        meta.query_fixed(fixed[x], Rotation(rotation_offset + y as i32))
                    }),
                );
                Ok(c)
            }
            // when advice we have 1 col per row
            VarTensor::Advice { inner: advices, .. } => {
                let c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..rng).map(|i| {
                        let (x, y) = self.cartesian_coord(idx_offset + i);
                        meta.query_advice(advices[x], Rotation(rotation_offset + y as i32))
                    }),
                );
                Ok(c)
            }
        }
    }

    /// Retrieve the values represented within the columns of the `VarTensor` (recall that `VarTensor`
    /// is a Tensor of Halo2 columns).
    pub fn query<F: FieldExt>(
        &self,
        meta: &mut VirtualCells<'_, F>,
        offset: usize,
    ) -> Result<Tensor<Expression<F>>, halo2_proofs::plonk::Error> {
        match &self {
            VarTensor::Fixed {
                inner: fixed, dims, ..
            } => {
                let mut c = Tensor::from(
                    // this should fail if dims is empty, should be impossible
                    (0..dims.iter().product::<usize>()).map(|i| {
                        let (x, y) = self.cartesian_coord(i);
                        meta.query_fixed(fixed[x], Rotation(offset as i32 + y as i32))
                    }),
                );
                c.reshape(dims);
                Ok(c)
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
    ) -> Result<Tensor<AssignedCell<F, F>>, halo2_proofs::plonk::Error> {
        match values {
            ValTensor::Instance {
                inner: instance,
                dims,
            } => match &self {
                VarTensor::Advice { inner: v, .. } => {
                    // this should never ever fail
                    let t: Tensor<i32> = Tensor::new(None, dims).unwrap();
                    t.enum_map(|coord, _| {
                        let (x, y) = self.cartesian_coord(offset + coord);
                        if x > 0 {
                            unimplemented!("multi-column instance variables will arrive in a future release but for now, increase K.")
                        }
                        region.assign_advice_from_instance(
                            || "pub input anchor",
                            *instance,
                            coord,
                            v[x],
                            y,
                        )
                    })
                }
                _ => Err(halo2_proofs::plonk::Error::Synthesis),
            },
            ValTensor::Value { inner: v, .. } => v.enum_map(|coord, k| {
                let (x, y) = self.cartesian_coord(offset + coord);
                
                match k {
                    ValType::Value(v) => match &self {
                        VarTensor::Fixed { inner: fixed, .. } => {
                            region.assign_fixed(|| "k", fixed[x], y, || v)
                        }
                        VarTensor::Advice { inner: advices, .. } => {
                            region.assign_advice(|| "k", advices[x], y, || v)
                        }
                    },
                    ValType::PrevAssigned(v) => match &self {
                        VarTensor::Advice { inner: advices, .. } => {
                            v.copy_advice(|| "k", region, advices[x], y)
                        }
                        _ => Err(halo2_proofs::plonk::Error::Synthesis),
                    },
                    ValType::AssignedValue(v) => match &self {
                        VarTensor::Fixed { inner: fixed, .. } => region
                            .assign_fixed(|| "k", fixed[x], y, || v)
                            .map(|a| a.evaluate()),
                        VarTensor::Advice { inner: advices, .. } => region
                            .assign_advice(|| "k", advices[x], y, || v)
                            .map(|a| a.evaluate()),
                    },
                }
            }),
        }
    }

    /// Assigns specific values (`ValTensor`) to the columns of the inner tensor.
    pub fn assign_with_duplication<F: FieldExt + TensorType>(
        &self,
        region: &mut Region<'_, F>,
        offset: usize,
        values: &ValTensor<F>,
    ) -> Result<(Tensor<AssignedCell<F, F>>, usize), halo2_proofs::plonk::Error> {
       
        match values {
        
            ValTensor::Instance { .. } => unimplemented!("duplication is not supported on instance columns. increase K if you require more rows."),
            ValTensor::Value { inner: v, dims } => {
                // duplicates every nth element to adjust for column overflow               
                let v = v.duplicate_every_n(self.col_size()).unwrap();
                let res = v.enum_map(|coord, k| {
                    let (x, y) = self.cartesian_coord(offset + coord);
                    match k {
                        ValType::Value(v) => match &self {
                            VarTensor::Fixed { inner: fixed, .. } => {
                                region.assign_fixed(|| "k", fixed[x], y, || v)
                            }
                            VarTensor::Advice { inner: advices, .. } => {
                                region.assign_advice(|| "k", advices[x], y, || v)
                            }
                        },
                        ValType::PrevAssigned(v) => match &self {
                            VarTensor::Advice { inner: advices, .. } => {
                                v.copy_advice(|| "k", region, advices[x], y)
                            }
                            _ => Err(halo2_proofs::plonk::Error::Synthesis),
                        },
                        ValType::AssignedValue(v) => match &self {
                            VarTensor::Fixed { inner: fixed, .. } => region
                                .assign_fixed(|| "k", fixed[x], y, || v)
                                .map(|a| a.evaluate()),
                            VarTensor::Advice { inner: advices, .. } => region
                                .assign_advice(|| "k", advices[x], y, || v)
                                .map(|a| a.evaluate()),
                        },
                    }
                })?;
                let mut non_duplicated_res = res.remove_every_n(self.col_size()).unwrap();
                
                non_duplicated_res.reshape(dims);
                Ok((non_duplicated_res, res.len()))
            }
        }
    }
}
