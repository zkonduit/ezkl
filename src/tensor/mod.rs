/// Tensor related errors.
pub mod errors;
/// Implementations of common operations on tensors.
pub mod ops;
/// A wrapper around a tensor of circuit variables / advices.
pub mod val;
/// A wrapper around a tensor of Halo2 Value types.
pub mod var;

pub use errors::TensorError;

use core::hash::Hash;
use halo2curves::ff::PrimeField;
use maybe_rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::io::Write;
use std::path::PathBuf;
pub use val::*;
pub use var::*;

use crate::{
    circuit::utils,
    fieldutils::{integer_rep_to_felt, IntegerRep},
    graph::Visibility,
};

use halo2_proofs::{
    arithmetic::Field,
    circuit::{AssignedCell, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Fixed, VirtualCells},
    poly::Rotation,
};
use itertools::Itertools;
use std::error::Error;
use std::fmt::Debug;
use std::io::Read;
use std::iter::Iterator;
use std::ops::{Add, Deref, Mul, Neg, Range, Sub};
use std::sync::Arc;

/// A view into a tensor with specified ranges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorView {
    /// the ranges to iterate over
    pub ranges: Vec<Range<usize>>,
    /// the original dimensions of the tensor
    pub original_dims: Vec<usize>,
}

impl PartialOrd for TensorView {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.ranges.len() != other.ranges.len() {
            return None;
        }
        for (a, b) in self.ranges.iter().zip(other.ranges.iter()) {
            if a.start != b.start || a.end != b.end {
                return None;
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

impl Ord for TensorView {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.ranges.len() != other.ranges.len() {
            return std::cmp::Ordering::Less;
        }
        for (a, b) in self.ranges.iter().zip(other.ranges.iter()) {
            if a.start != b.start || a.end != b.end {
                return std::cmp::Ordering::Less;
            }
        }
        std::cmp::Ordering::Equal
    }
}

/// The (inner) type of tensor elements.
pub trait TensorType: Clone + Debug {
    /// Returns the zero value.
    fn zero() -> Option<Self> {
        None
    }
    /// Returns the unit value.
    fn one() -> Option<Self> {
        None
    }
}

macro_rules! tensor_type {
    ($rust_type:ty, $tensor_type:ident, $zero:expr_2021, $one:expr_2021) => {
        impl TensorType for $rust_type {
            fn zero() -> Option<Self> {
                Some($zero)
            }
            fn one() -> Option<Self> {
                Some($one)
            }
        }
    };
}

impl TensorType for f32 {
    fn zero() -> Option<Self> {
        Some(0.0)
    }
}

impl TensorType for f64 {
    fn zero() -> Option<Self> {
        Some(0.0)
    }
}

tensor_type!(bool, Bool, false, true);
tensor_type!(IntegerRep, IntegerRep, 0, 1);
tensor_type!(usize, USize, 0, 1);
tensor_type!((), Empty, (), ());
tensor_type!(utils::F32, F32, utils::F32(0.0), utils::F32(1.0));

impl<T: TensorType> TensorType for Tensor<T> {
    fn zero() -> Option<Self> {
        Some(Tensor::new(Some(&[T::zero().unwrap()]), &[1]).unwrap())
    }
    fn one() -> Option<Self> {
        Some(Tensor::new(Some(&[T::one().unwrap()]), &[1]).unwrap())
    }
}

impl<T: TensorType> TensorType for Value<T> {
    fn zero() -> Option<Self> {
        Some(Value::known(T::zero().unwrap()))
    }

    fn one() -> Option<Self> {
        Some(Value::known(T::one().unwrap()))
    }
}

impl<F: PrimeField + PartialOrd> TensorType for Assigned<F>
where
    F: Field,
{
    fn zero() -> Option<Self> {
        Some(F::ZERO.into())
    }

    fn one() -> Option<Self> {
        Some(F::ONE.into())
    }
}

impl<F: PrimeField> TensorType for Expression<F>
where
    F: Field,
{
    fn zero() -> Option<Self> {
        Some(Expression::Constant(F::ZERO))
    }

    fn one() -> Option<Self> {
        Some(Expression::Constant(F::ONE))
    }
}

impl TensorType for Column<Advice> {}
impl TensorType for Column<Fixed> {}

impl<F: PrimeField + PartialOrd> TensorType for AssignedCell<Assigned<F>, F> {}

impl<F: PrimeField + PartialOrd> TensorType for AssignedCell<F, F> {}

impl TensorType for halo2curves::bn256::Fr {
    fn zero() -> Option<Self> {
        Some(halo2curves::bn256::Fr::zero())
    }
    fn one() -> Option<Self> {
        Some(halo2curves::bn256::Fr::one())
    }
}

impl<F: TensorType> TensorType for &F {
    fn zero() -> Option<Self> {
        None
    }

    fn one() -> Option<Self> {
        None
    }
}

/// A generic multi-dimensional array representation of a Tensor.
/// The `inner` attribute contains a vector of values whereas `dims` corresponds to the dimensionality of the array
/// and as such determines how we index, query for values, or slice a Tensor.
#[derive(Clone, Debug, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Tensor<T: TensorType> {
    inner: Arc<Vec<T>>,
    dims: Vec<usize>,
    scale: Option<crate::Scale>,
    visibility: Option<Visibility>,
    view: Option<TensorView>,
}

impl<T: PartialEq + TensorType> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.dims == other.dims && self.view == other.view && self.inner == other.inner
    }
}

impl<I: Iterator> From<I> for Tensor<I::Item>
where
    I::Item: TensorType + Clone,
    Vec<I::Item>: FromIterator<I::Item>,
{
    fn from(value: I) -> Tensor<I::Item> {
        let data: Vec<I::Item> = value.collect::<Vec<I::Item>>();
        Tensor::new(Some(&data), &[data.len()]).unwrap()
    }
}

impl<T> FromIterator<T> for Tensor<T>
where
    T: TensorType + Clone,
    Vec<T>: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(value: I) -> Tensor<T> {
        let data: Vec<I::Item> = value.into_iter().collect::<Vec<I::Item>>();
        Tensor::new(Some(&data), &[data.len()]).unwrap()
    }
}

impl<F: PrimeField + Clone + TensorType + PartialOrd> From<Tensor<AssignedCell<Assigned<F>, F>>>
    for Tensor<Value<F>>
{
    fn from(value: Tensor<AssignedCell<Assigned<F>, F>>) -> Tensor<Value<F>> {
        let mut output = Vec::new();
        for x in value.iter() {
            output.push(x.value_field().evaluate());
        }
        Tensor::new(Some(&output), value.dims()).unwrap()
    }
}

impl<F: PrimeField + TensorType + Clone + PartialOrd> From<Tensor<Value<F>>>
    for Tensor<Value<Assigned<F>>>
{
    fn from(t: Tensor<Value<F>>) -> Tensor<Value<Assigned<F>>> {
        let mut ta: Tensor<Value<Assigned<F>>> =
            Tensor::from((0..t.len()).map(|i| t.get_flat(i).into()));
        // safe to unwrap as we know the dims are correct
        ta.reshape(t.dims()).unwrap();
        ta
    }
}

impl<F: PrimeField + TensorType + Clone> From<Tensor<IntegerRep>> for Tensor<Value<F>> {
    fn from(t: Tensor<IntegerRep>) -> Tensor<Value<F>> {
        let mut ta: Tensor<Value<F>> = Tensor::from(
            (0..t.len()).map(|i| Value::known(integer_rep_to_felt::<F>(t.get_flat(i)))),
        );
        // safe to unwrap as we know the dims are correct
        ta.reshape(t.dims()).unwrap();
        ta
    }
}

impl<T: Clone + TensorType + std::marker::Send + std::marker::Sync>
    maybe_rayon::iter::FromParallelIterator<T> for Tensor<T>
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: maybe_rayon::iter::IntoParallelIterator<Item = T>,
    {
        let inner: Vec<T> = par_iter.into_par_iter().collect();
        Tensor::new(Some(&inner), &[inner.len()]).unwrap()
    }
}

impl<'data, T: Clone + TensorType + std::marker::Send + std::marker::Sync + 'data>
    maybe_rayon::iter::IntoParallelRefIterator<'data> for Tensor<T>
{
    type Item = &'data T;
    type Iter = maybe_rayon::iter::IterBridge<std::vec::IntoIter<&'data T>>;

    fn par_iter(&'data self) -> Self::Iter {
        let ranges = if let Some(view) = &self.view {
            view.ranges.clone()
        } else {
            self.dims.iter().map(|&d| 0..d).collect_vec()
        };

        // First collect into a Vec to resolve the lifetime issues
        let elements: Vec<&'data T> = ranges
            .iter()
            .cloned()
            .multi_cartesian_product()
            .map(|coord| {
                let index = self.get_index_raw(&coord);
                &self.inner[index]
            })
            .collect();

        // Then convert to parallel iterator
        elements.into_iter().par_bridge()
    }
}

impl<T: Clone + TensorType + Ord + PartialOrd> Tensor<T> {
    /// call sort unstable on the inner vector
    pub fn sort_unstable(&mut self) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.sort_unstable();
    }
}

impl<T: Clone + TensorType + PrimeField> Tensor<T> {
    /// save to a file
    pub fn save(&self, path: &PathBuf) -> Result<(), TensorError> {
        let writer =
            std::fs::File::create(path).map_err(|e| TensorError::FileSaveError(e.to_string()))?;
        let mut buf_writer = std::io::BufWriter::new(writer);

        self.iter().copied().for_each(|x| {
            let x = x.to_repr();
            buf_writer.write_all(x.as_ref()).unwrap();
        });

        Ok(())
    }

    /// load from a file
    pub fn load(path: &PathBuf) -> Result<Self, TensorError> {
        let reader =
            std::fs::File::open(path).map_err(|e| TensorError::FileLoadError(e.to_string()))?;
        let mut buf_reader = std::io::BufReader::new(reader);

        let mut inner = Vec::new();
        while let Ok(true) = buf_reader.has_data_left() {
            let mut repr = T::Repr::default();
            match buf_reader.read_exact(repr.as_mut()) {
                Ok(_) => {
                    inner.push(T::from_repr(repr).unwrap());
                }
                Err(_) => {
                    return Err(TensorError::FileLoadError(
                        "Failed to read tensor".to_string(),
                    ));
                }
            }
        }
        Ok(Tensor::new(Some(&inner), &[inner.len()]).unwrap())
    }
}

impl<T: Clone + TensorType> Tensor<T> {
    /// Sets (copies) the tensor values to the provided ones.
    pub fn new(values: Option<&[T]>, dims: &[usize]) -> Result<Self, TensorError> {
        let total_dims: usize = if !dims.is_empty() {
            dims.iter().product()
        } else if values.is_some() {
            1
        } else {
            0
        };
        match values {
            Some(v) => {
                if total_dims != v.len() {
                    return Err(TensorError::DimError(format!(
                        "Cannot create tensor of length {} with dims {:?}",
                        v.len(),
                        dims
                    )));
                }
                Ok(Tensor {
                    inner: Arc::new(Vec::from(v)),
                    dims: Vec::from(dims),
                    scale: None,
                    visibility: None,
                    view: None,
                })
            }
            None => Ok(Tensor {
                inner: Arc::new(vec![T::zero().unwrap(); total_dims]),
                dims: Vec::from(dims),
                scale: None,
                visibility: None,
                view: None,
            }),
        }
    }

    /// set the tensor's (optional) scale parameter
    pub fn set_scale(&mut self, scale: crate::Scale) {
        self.scale = Some(scale)
    }

    /// to vec
    pub fn to_vec(&self) -> Vec<T> {
        self.iter().cloned().collect()
    }

    /// set the tensor's (optional) visibility parameter
    pub fn set_visibility(&mut self, visibility: &Visibility) {
        self.visibility = Some(visibility.clone())
    }

    /// getter for scale
    pub fn scale(&self) -> Option<crate::Scale> {
        self.scale
    }

    /// getter for visibility
    pub fn visibility(&self) -> Option<Visibility> {
        self.visibility.clone()
    }

    /// Returns the number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.dims().iter().product::<usize>()
    }
    /// Checks if the number of elements in tensor is 0.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Checks if the number of elements in tensor is 1 but with an empty dimension (this is for onnx compatibility).
    pub fn is_singleton(&self) -> bool {
        self.dims().is_empty() && self.len() == 1
    }

    /// Set one single value on the tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// a.set(&[0, 0, 1], 10);
    /// assert_eq!(a[0 + 0 + 1], 10);
    ///
    /// a.set(&[2, 2, 0], 9);
    /// assert_eq!(a[2*9 + 2*3 + 0], 9);
    /// ```
    fn set(&mut self, indices: &[usize], value: T) {
        let index = self.get_index(indices);
        let inner = Arc::make_mut(&mut self.inner);
        inner[index] = value;
    }

    /// reverse the values of the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(&vec![0,1,2], &[3, 3, 3]).unwrap();
    ///
    /// a.reverse();
    /// assert_eq!(a[0], 2);
    /// assert_eq!(a[1], 1);
    /// assert_eq!(a[2], 0);
    /// ```
    pub fn reverse(&mut self) {
        let inner = Arc::make_mut(&mut self.inner);
        inner.reverse();
    }

    /// Get a single value from the Tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get(&self, indices: &[usize]) -> T {
        let index = self.get_index(indices);
        self.inner[index].clone()
    }

    /// Get a single value from the Tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get_flat(0), 0);
    /// ```
    pub fn get_flat(&self, index: usize) -> T {
        self.get_flat_ref(index).clone()
    }

    /// Get a single value from the Tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get_flat_ref(0), 0);
    /// ```
    pub fn get_flat_ref(&self, index: usize) -> &T {
        let dims = self.dims.clone();

        let cartesian_coord: Vec<Vec<usize>> = dims
            .iter()
            .cloned()
            .map(|d| 0..d)
            .multi_cartesian_product()
            .collect();

        let real_index = self.get_index(&cartesian_coord[index]);

        &self.inner[real_index]
    }

    /// Get a flat slice from the Tensor.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get_flat(0), 0);
    /// ```
    pub fn get_flat_slice(&self, range: Range<usize>) -> Vec<T> {
        self.inner[range].iter().cloned().collect()
    }

    /// Create an iterator over values of the tensor.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ + Clone {
        let ranges = if let Some(view) = &self.view {
            view.ranges.clone()
        } else {
            self.dims.iter().map(|&d| 0..d).collect_vec()
        };

        // Create an iterator over the cartesian product of ranges
        // and map each coordinate to the corresponding element in the tensor
        ranges
            .iter()
            .cloned()
            .multi_cartesian_product()
            .map(move |coord| {
                let index = self.get_index_raw(&coord);
                &self.inner[index]
            })
    }

    /// Create an iterator over values of the tensor.
    pub fn rev(&self) -> impl Iterator<Item = &T> + '_ + Clone {
        let ranges = if let Some(view) = &self.view {
            view.ranges.clone()
        } else {
            self.dims.iter().map(|&d| 0..d).collect_vec()
        };

        // Create an iterator over the cartesian product of ranges
        // and map each coordinate to the corresponding element in the tensor
        ranges
            .iter()
            .cloned()
            .multi_cartesian_product()
            .map(move |coord| {
                let index = self.get_index_raw(&coord);
                &self.inner[index]
            })
    }

    /// Concretize the values of the tensor iterating over the elements of the view
    /// and returning a new tensor with the values.
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// assert_eq!(a.concretize().unwrap(), expected);
    ///
    /// let b = Tensor { inner: Arc::new(vec![1, 2, 3, 4, 5, 6]), dims: vec![2, 3], scale: None, visibility: None, view: Some(TensorView { ranges: vec![0..1, 0..2] }) };
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// assert_eq!(b.concretize().unwrap(), expected);
    /// ```
    fn concretize(&mut self) {
        if self.view().is_none() {
            return;
        }
        let mut inner = vec![];

        for val in self.iter() {
            inner.push(val.clone())
        }

        self.inner = Arc::new(inner);
        self.view = None;
    }

    /// Pad to a length that is divisible by n
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<IntegerRep>::new(Some(&[1,2,3,4,5,6]), &[2, 3]).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 0, 0]), &[8]).unwrap();
    /// assert_eq!(a.pad_to_zero_rem(4, 0).unwrap(), expected);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 0, 0, 0]), &[9]).unwrap();
    /// assert_eq!(a.pad_to_zero_rem(9, 0).unwrap(), expected);
    /// ```
    pub fn pad_to_zero_rem(&self, n: usize, pad: T) -> Result<Tensor<T>, TensorError> {
        let mut copy = self.clone();
        copy.concretize();

        let mut inner = copy.inner.deref().clone();
        let remainder = self.len() % n;
        if remainder != 0 {
            inner.resize(self.len() + n - remainder, pad);
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Display a tensor
    pub fn show(&self) -> String {
        if self.len() > 12 {
            let start = self.get_flat_slice(0..12);
            // print the two split by ... in the middle
            format!(
                "[{} ...]",
                start.iter().map(|x| format!("{:?}", x)).join(", "),
            )
        } else {
            format!("[{:?}]", self.iter().map(|x| format!("{:?}", x)).join(", "))
        }
    }

    /// Get the slice ranges for the tensor.
    fn get_slice_ranges(&self, indices: &[Range<usize>]) -> Result<Vec<Range<usize>>, TensorError> {
        if indices.is_empty() {
            // If no indices are provided, use the full tensor dimensions
            if let Some(view) = &self.view {
                // If we have a view, return the view's ranges
                return Ok(view.ranges.clone());
            } else {
                // Otherwise return the full range for each dimension
                return Ok(self.dims.iter().map(|&d| 0..d).collect());
            }
        }

        // Check that the requested slice dimensionality doesn't exceed the tensor's
        if self.dims.len() < indices.len() {
            return Err(TensorError::DimError(format!(
                "The dimensionality of the slice {:?} is greater than the tensor's {:?}",
                indices, self.dims
            )));
        }

        // Handle the case where the tensor already has a view
        if let Some(existing_view) = &self.view {
            // Map the requested ranges through the existing view's ranges
            let mut adjusted_ranges = Vec::with_capacity(self.dims.len());

            // Process the explicitly provided indices
            for (i, new_range) in indices.iter().enumerate() {
                // Get the existing view's range for this dimension
                let existing_range = &existing_view.ranges[i];
                let current_dim_size = existing_range.end - existing_range.start;

                // Validate the requested range against the current view's dimension
                if new_range.start >= current_dim_size || new_range.end > current_dim_size {
                    return Err(TensorError::IndexOutOfBounds(i, current_dim_size));
                }

                // Map the range through the existing view
                let absolute_start = existing_range.start + new_range.start;
                let absolute_end = existing_range.start + new_range.end;

                adjusted_ranges.push(absolute_start..absolute_end);
            }

            // Fill in the remaining dimensions with the full ranges from the existing view
            adjusted_ranges
                .extend((indices.len()..self.dims.len()).map(|i| existing_view.ranges[i].clone()));

            return Ok(adjusted_ranges);
        }

        // No existing view, handle as before
        // Pre-allocate the full indices vector with capacity
        let mut full_indices = Vec::with_capacity(self.dims.len());

        // Process the explicitly provided indices
        for (i, range) in indices.iter().enumerate() {
            // Validate the range against the tensor dimension
            if range.start >= self.dims[i] || range.end > self.dims[i] {
                return Err(TensorError::IndexOutOfBounds(range.start, self.dims[i]));
            }
            full_indices.push(range.clone());
        }

        // Fill remaining dimensions with full ranges
        full_indices.extend((indices.len()..self.dims.len()).map(|i| 0..self.dims[i]));

        // Validate all ranges
        for (i, range) in full_indices.iter().enumerate() {
            if range.end > self.dims[i] {
                return Err(TensorError::IndexOutOfBounds(range.end, self.dims[i]));
            }
        }

        Ok(full_indices)
    }

    /// Get a slice from the Tensor.
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut b = Tensor::<IntegerRep>::new(Some(&[1, 2]), &[2]).unwrap();
    ///
    /// assert_eq!(a.get_slice(&[0..2]).unwrap(), b);
    /// ```
    pub fn get_slice(&self, indices: &[Range<usize>]) -> Result<Tensor<T>, TensorError>
    where
        T: Send + Sync,
    {
        // Get the adjusted ranges for the slice
        let full_indices = self.get_slice_ranges(indices)?;

        // Calculate new dimensions once
        let dims: Vec<usize> = full_indices.iter().map(|e| e.end - e.start).collect();

        // Get the original dimensions - if we already have a view, use its original_dims
        // otherwise, the current dims are the original dims
        let original_dims = if let Some(view) = &self.view {
            view.original_dims.clone()
        } else {
            self.dims.clone()
        };

        Ok(Tensor {
            inner: self.inner.clone(),
            dims,
            scale: self.scale.clone(),
            visibility: self.visibility.clone(),
            view: Some(TensorView {
                ranges: full_indices,
                original_dims,
            }),
        })
    }

    /// Set a slice of the Tensor.
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// let b = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 1, 2, 3]), &[2, 3]).unwrap();
    /// a.set_slice(&[1..2], &Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[1, 3]).unwrap()).unwrap();
    /// assert_eq!(a, b);
    /// ```
    pub fn set_slice(
        &mut self,
        indices: &[Range<usize>],
        value: &Tensor<T>,
    ) -> Result<(), TensorError>
    where
        T: Send + Sync,
    {
        if indices.is_empty() {
            return Ok(());
        }
        if self.dims.len() < indices.len() {
            return Err(TensorError::DimError(format!(
                "The dimensionality of the slice {:?} is greater than the tensor's {:?}",
                indices, self.dims
            )));
        }

        // if indices weren't specified we fill them in as required
        let mut full_indices = indices.to_vec();

        let omitted_dims = (indices.len()..self.dims.len())
            .map(|i| self.dims[i])
            .collect::<Vec<_>>();

        for dim in &omitted_dims {
            full_indices.push(0..*dim);
        }

        let full_dims = full_indices
            .iter()
            .map(|x| x.end - x.start)
            .collect::<Vec<_>>();

        // now broadcast the value to the full dims
        let value = value.expand(&full_dims)?;

        let cartesian_coord: Vec<Vec<usize>> = full_indices
            .iter()
            .cloned()
            .multi_cartesian_product()
            .collect();

        let _ = cartesian_coord
            .iter()
            .enumerate()
            .map(|(i, e)| {
                self.set(e, value.get_flat(i).clone());
            })
            .collect::<Vec<_>>();

        Ok(())
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    fn get_index(&self, indices: &[usize]) -> usize {
        assert_eq!(
            self.dims.len(),
            indices.len(),
            "Index dimensionality mismatch"
        );

        if let Some(view) = &self.view {
            // When we have a view, we need to:
            // 1. Map the provided indices to the original tensor space
            // 2. Calculate the flat index using original dimensions

            assert_eq!(
                view.ranges.len(),
                indices.len(),
                "View range dimensionality mismatch"
            );

            // Map each index through the corresponding view range
            let original_indices: Vec<usize> = indices
                .iter()
                .zip(view.ranges.iter())
                .map(|(&idx, range)| {
                    // Verify the index is within the view's range
                    let view_size = range.end - range.start;
                    assert!(
                        idx < view_size,
                        "Index {} out of bounds for view dimension with size {}",
                        idx,
                        view_size
                    );

                    // Map to original tensor space
                    range.start + idx
                })
                .collect();

            self.get_index_raw(&original_indices)
        } else {
            self.get_index_raw(indices)
        }
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    fn get_index_raw(&self, indices: &[usize]) -> usize {
        let dims = if let Some(view) = &self.view {
            view.original_dims.clone()
        } else {
            self.dims.clone()
        };

        assert_eq!(dims.len(), indices.len(), "Index dimensionality mismatch");

        // Without a view, use the standard algorithm
        let mut index = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            assert!(
                dims[i] > indices[i],
                "Index {} out of bounds for dimension {} of size {}",
                indices[i],
                i,
                dims[i]
            );

            index += indices[i] * stride;
            stride *= dims[i];
        }

        index
    }

    /// Fetches every nth element
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5]), &[3]).unwrap();
    /// assert_eq!(a.get_every_n(2).unwrap(), expected);
    /// assert_eq!(a.get_every_n(1).unwrap(), a);
    ///
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 6]), &[2]).unwrap();
    /// assert_eq!(a.get_every_n(5).unwrap(), expected);
    ///
    /// ```
    pub fn get_every_n(&self, n: usize) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = vec![];
        for (i, elem) in self.inner.deref().clone().into_iter().enumerate() {
            if i % n == 0 {
                inner.push(elem.clone());
            }
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Duplicates every nth element
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 3, 4, 5, 5, 6]), &[8]).unwrap();
    /// assert_eq!(a.duplicate_every_n(3, 1, 0).unwrap(), expected);
    /// assert_eq!(a.duplicate_every_n(7, 1, 0).unwrap(), a);
    ///
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 2, 3, 3, 4, 5, 5, 6]), &[9]).unwrap();
    /// assert_eq!(a.duplicate_every_n(3, 1, 2).unwrap(), expected);
    ///
    /// ```
    pub fn duplicate_every_n(
        &self,
        n: usize,
        num_repeats: usize,
        initial_offset: usize,
    ) -> Result<Tensor<T>, TensorError> {
        if n == 0 {
            return Err(TensorError::InvalidArgument(
                "Cannot duplicate every 0th element".to_string(),
            ));
        }

        let mut inner: Vec<T> = Vec::with_capacity(self.inner.len());
        let mut offset = initial_offset;
        for (i, elem) in self.iter().enumerate() {
            if (i + offset + 1) % n == 0 {
                inner.extend(vec![elem.clone(); 1 + num_repeats]);
                offset += num_repeats;
            } else {
                inner.push(elem.clone());
            }
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Removes every nth element
    ///
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 3, 4, 5, 6, 6]), &[8]).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 3, 5, 6, 6]), &[7]).unwrap();
    /// assert_eq!(a.remove_every_n(4, 1, 0).unwrap(), expected);
    ///
    ///
    pub fn remove_every_n(
        &self,
        n: usize,
        num_repeats: usize,
        initial_offset: usize,
    ) -> Result<Tensor<T>, TensorError> {
        if n == 0 {
            return Err(TensorError::InvalidArgument(
                "Cannot remove every 0th element".to_string(),
            ));
        }

        // Pre-calculate capacity to avoid reallocations
        let estimated_size = self.inner.len() - (self.inner.len() / n) * num_repeats;
        let mut inner = Vec::with_capacity(estimated_size);

        // Use iterator directly instead of creating intermediate collectionsif
        let mut i = 0;

        while i < self.len() {
            // Add the current element
            inner.push(self.get_flat(i).clone());

            // If this is an nth position (accounting for offset)
            if (i + initial_offset + 1) % n == 0 {
                // Skip the next num_repeats elements
                i += num_repeats + 1;
            } else {
                i += 1;
            }
        }

        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Remove indices
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 6]), &[4]).unwrap();
    /// let mut indices = vec![3, 4];
    /// assert_eq!(a.remove_indices(&mut indices, true).unwrap(), expected);
    ///
    ///
    /// let a = Tensor::<IntegerRep>::new(Some(&[52, -245, 153, 13, -4, -56, -163, 249, -128, -172, 396, 143, 2, -96, 504, -44, -158, -393, 61, 95, 191, 74, 64, -219, 553, 104, 235, 222, 44, -216, 63, -251, 40, -140, 112, -355, 60, 123, 26, -116, -89, -200, -109, 168, 135, -34, -99, -54, 5, -81, 322, 87, 4, -139, 420, 92, -295, -12, 262, -1, 26, -48, 231, 1, -335, 244, 188, -4, 5, -362, 57, -198, -184, -117, 40, 305, 49, 30, -59, -26, -37, 96]), &[82]).unwrap();
    /// let b = Tensor::<IntegerRep>::new(Some(&[52, -245, 153, 13, -4, -56, -163, 249, -128, -172, 396, 143, 2, -96, 504, -44, -158, -393, 61, 95, 191, 74, 64, -219, 553, 104, 235, 222, 44, -216, 63, -251, 40, -140, 112, -355, 60, 123, 26, -116, -89, -200, -109, 168, 135, -34, -99, -54, 5, -81, 322, 87, 4, -139, 420, 92, -295, -12, 262, -1, 26, -48, 231, -335, 244, 188, 5, -362, 57, -198, -184, -117, 40, 305, 49, 30, -59, -26, -37, 96]), &[80]).unwrap();
    /// let mut indices = vec![63, 67];
    /// assert_eq!(a.remove_indices(&mut indices, true).unwrap(), b);
    /// ```
    pub fn remove_indices(
        &self,
        indices: &mut [usize],
        is_sorted: bool,
    ) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = self.inner.deref().clone();
        // time it
        if !is_sorted {
            indices.par_sort_unstable();
        }
        // remove indices
        for elem in indices.iter().rev() {
            if *elem < self.len() {
                inner.remove(*elem);
            } else {
                return Err(TensorError::IndexOutOfBounds(*elem, self.len()));
            }
        }

        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Returns the tensor's dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the tensor's view.
    pub fn view(&self) -> Option<&TensorView> {
        self.view.as_ref()
    }

    ///Reshape the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.reshape(&[9, 3]);
    /// assert_eq!(a.dims(), &[9, 3]);
    /// ```
    pub fn reshape(&mut self, new_dims: &[usize]) -> Result<(), TensorError> {
        // in onnx parlance this corresponds to converting a tensor to a single element
        if new_dims.is_empty() {
            if !(self.len() == 1 || self.is_empty()) {
                return Err(TensorError::DimError(
                    "Cannot reshape to empty tensor".to_string(),
                ));
            }
            self.dims = vec![];
        }
        if self.dims() == [0] && new_dims.iter().product::<usize>() == 1 {
            self.dims = Vec::from(new_dims);
        } else {
            let product = if new_dims != [0] {
                new_dims.iter().product::<usize>()
            } else {
                0
            };
            if self.len() != product {
                return Err(TensorError::DimError(format!(
                    "Cannot reshape tensor of length {} to {:?}",
                    self.len(),
                    new_dims
                )));
            }
            self.dims = Vec::from(new_dims);
        }
        Ok(())
    }

    /// Move axis of the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// let b = a.move_axis(0, 2).unwrap();
    /// assert_eq!(b.dims(), &[3, 3, 3]);
    ///
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 1, 2]).unwrap();
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5, 2, 4, 6]), &[1, 2, 3]).unwrap();
    /// let b = a.move_axis(0, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();
    /// let b = a.move_axis(1, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();
    /// let b = a.move_axis(2, 1).unwrap();
    /// assert_eq!(b, expected);
    /// ```
    pub fn move_axis(&mut self, source: usize, destination: usize) -> Result<Self, TensorError> {
        assert!(source < self.dims.len());
        assert!(destination < self.dims.len());

        let mut new_dims = self.dims.clone();
        new_dims.remove(source);
        new_dims.insert(destination, self.dims[source]);

        // now reconfigure the elements appropriately in the new array
        //  eg. if we have a 3x3x3 array and we want to move the 0th axis to the 2nd position
        //  we need to move the elements at 0, 1, 2, 3, 4, 5, 6, 7, 8 to 0, 3, 6, 1, 4, 7, 2, 5, 8
        //  so we need to move the elements at 0, 1, 2 to 0, 3, 6
        //  and the elements at 3, 4, 5 to 1, 4, 7
        //  and the elements at 6, 7, 8 to 2, 5, 8
        let cartesian_coords = new_dims
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product()
            .collect::<Vec<Vec<usize>>>();

        let mut output = Tensor::new(None, &new_dims)?;

        for coord in cartesian_coords {
            let mut old_coord = vec![0; self.dims.len()];

            // now fetch the old index
            for (i, c) in coord.iter().enumerate() {
                if i == destination {
                    old_coord[source] = *c;
                } else if i == source && source < destination {
                    old_coord[source + 1] = *c;
                } else if i == source && source > destination {
                    old_coord[source - 1] = *c;
                } else if (i < source && source < destination)
                    || (i < destination && source > destination)
                    || (i > source && source > destination)
                    || (i > destination && source < destination)
                {
                    old_coord[i] = *c;
                } else if i > source && source < destination {
                    old_coord[i + 1] = *c;
                } else if i > destination && source > destination {
                    old_coord[i - 1] = *c;
                } else {
                    return Err(TensorError::DimError(
                        "Unknown condition for moving the axis".to_string(),
                    ));
                }
            }

            let value = self.get(&old_coord);

            output.set(&coord, value);
        }

        Ok(output)
    }

    /// Swap axes of the tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// let b = a.swap_axes(0, 2).unwrap();
    /// assert_eq!(b.dims(), &[3, 3, 3]);
    ///
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 1, 2]).unwrap();
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5, 2, 4, 6]), &[2, 1, 3]).unwrap();
    /// let b = a.swap_axes(0, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();
    /// let b = a.swap_axes(1, 2).unwrap();
    /// assert_eq!(b, expected);
    ///
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 3, 2]).unwrap();
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12]), &[2, 2, 3]).unwrap();
    /// let b = a.swap_axes(2, 1).unwrap();
    /// assert_eq!(b, expected);
    /// ```
    pub fn swap_axes(&mut self, source: usize, destination: usize) -> Result<Self, TensorError> {
        assert!(source < self.dims.len());
        assert!(destination < self.dims.len());
        let mut new_dims = self.dims.clone();
        new_dims[source] = self.dims[destination];
        new_dims[destination] = self.dims[source];

        // now reconfigure the elements appropriately in the new array
        //  eg. if we have a 3x3x3 array and we want to move the 0th axis to the 2nd position
        //  we need to move the elements at 0, 1, 2, 3, 4, 5, 6, 7, 8 to 0, 3, 6, 1, 4, 7, 2, 5, 8
        //  so we need to move the elements at 0, 1, 2 to 0, 3, 6
        //  and the elements at 3, 4, 5 to 1, 4, 7
        //  and the elements at 6, 7, 8 to 2, 5, 8
        let cartesian_coords = new_dims
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product()
            .collect::<Vec<Vec<usize>>>();

        let mut output = Tensor::new(None, &new_dims)?;

        for coord in cartesian_coords {
            let mut old_coord = vec![0; self.dims.len()];

            // now fetch the old index
            for (i, c) in coord.iter().enumerate() {
                if i == destination {
                    old_coord[source] = *c;
                } else if i == source {
                    old_coord[destination] = *c;
                } else {
                    old_coord[i] = *c;
                }
            }
            output.set(&coord, self.get(&old_coord));
        }

        Ok(output)
    }

    /// Broadcasts the tensor to a given shape
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3, 1]).unwrap();
    ///
    /// let mut expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 2, 2, 2, 3, 3, 3]), &[3, 3]).unwrap();
    /// assert_eq!(a.expand(&[3, 3]).unwrap(), expected);
    ///
    /// ```
    pub fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        // if both have length 1 then we can just return the tensor
        if self.dims().iter().product::<usize>() == 1 && shape.iter().product::<usize>() == 1 {
            let mut output = self.clone();
            output.reshape(shape)?;
            return Ok(output);
        } else if self.dims() == &[0] && shape.iter().product::<usize>() == 1 {
            let mut output = self.clone();
            output.reshape(shape)?;
            return Ok(output);
        }

        if self.dims().len() > shape.len() {
            return Err(TensorError::DimError(format!(
                "Cannot expand {:?} to the smaller shape {:?}",
                self.dims(),
                shape
            )));
        }

        if shape == self.dims() {
            return Ok(self.clone());
        }

        for d in self.dims() {
            if !(shape.contains(d) || *d == 1) {
                return Err(TensorError::DimError(format!(
                    "The current dimension {} must be contained in the new shape {:?} or be 1",
                    d, shape
                )));
            }
        }

        let cartesian_coords = shape
            .iter()
            .map(|d| 0..*d)
            .multi_cartesian_product()
            .collect::<Vec<Vec<usize>>>();

        let mut output = Tensor::new(None, shape)?;

        for coord in cartesian_coords {
            let mut new_coord = Vec::with_capacity(self.dims().len());
            for (i, c) in coord.iter().enumerate() {
                if i < self.dims().len() && self.dims()[i] == 1 {
                    new_coord.push(0);
                } else if i >= self.dims().len() {
                    // do nothing at this point does not exist in the original tensor
                } else {
                    new_coord.push(*c);
                }
            }
            output.set(&coord, self.get(&new_coord));
        }

        Ok(output)
    }

    ///Flatten the tensor shape
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.flatten();
    /// assert_eq!(a.dims(), &[27]);
    /// ```
    pub fn flatten(&mut self) {
        if !self.dims().is_empty() && (self.dims() != [0]) {
            self.dims = Vec::from([self.dims.iter().product::<usize>()]);
        }
    }

    /// Maps a function to tensors
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.map(|x| IntegerRep::pow(x,2));
    /// assert_eq!(c, Tensor::from([1, 16].into_iter()))
    /// ```
    pub fn map<F: FnMut(T) -> G, G: TensorType>(&self, mut f: F) -> Tensor<G> {
        let mut t = Tensor::from(self.iter().map(|e| f(e.clone())));
        // safe to unwrap as we know the dims are correct
        t.reshape(self.dims()).unwrap();
        t
    }

    /// Maps a function to tensors and enumerates
    /// ```
    /// use ezkl::tensor::{Tensor, TensorError};
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.enum_map::<_,_,TensorError>(|i, x| Ok(IntegerRep::pow(x + i as IntegerRep, 2))).unwrap();
    /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn enum_map<F: FnMut(usize, &T) -> Result<G, E>, G: TensorType, E: Error>(
        &self,
        mut f: F,
    ) -> Result<Tensor<G>, E> {
        let vec: Result<Vec<G>, E> = self
            .inner
            .iter()
            .enumerate()
            .map(|(i, e)| f(i, e))
            .collect();
        let mut t: Tensor<G> = Tensor::from(vec?.iter().cloned());
        // safe to unwrap as we know the dims are correct
        t.reshape(self.dims()).unwrap();
        Ok(t)
    }

    /// Maps a function to tensors and enumerates in parallel
    /// ```
    /// use ezkl::tensor::{Tensor, TensorError};
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.par_map::<_,_,TensorError>(|x| Ok(IntegerRep::pow(x + i as IntegerRep, 2))).unwrap();
    /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn par_map<
        F: Fn(&T) -> Result<G, E> + std::marker::Send + std::marker::Sync,
        G: TensorType + std::marker::Send + std::marker::Sync,
        E: Error + std::marker::Send + std::marker::Sync,
    >(
        &self,
        f: F,
    ) -> Result<Tensor<G>, E>
    where
        T: std::marker::Send + std::marker::Sync,
    {
        let vec: Result<Vec<G>, E> = self.par_iter().map(move |e| f(e)).collect();
        let mut t: Tensor<G> = Tensor::from(vec?.iter().cloned());
        // safe to unwrap as we know the dims are correct
        t.reshape(self.dims()).unwrap();
        Ok(t)
    }

    /// Get last elem from Tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut b = Tensor::<IntegerRep>::new(Some(&[3]), &[1]).unwrap();
    ///
    /// assert_eq!(a.last().unwrap(), b);
    /// ```
    pub fn last(&self) -> Result<Tensor<T>, TensorError>
    where
        T: Send + Sync,
    {
        let slice = self.dims().iter().map(|x| x - 1..*x).collect::<Vec<_>>();
        let indices = self.get_slice_ranges(&slice)?;

        let original_dims = if let Some(view) = &self.view {
            view.original_dims.clone()
        } else {
            self.dims.clone()
        };

        // the view for the last element is the slice of the size of the first dimension
        let view = Some(TensorView {
            ranges: indices,
            original_dims,
        });

        let dims = self.dims().iter().map(|_x| 1).collect::<Vec<_>>();

        Ok(Tensor {
            inner: self.inner.clone(),
            dims,
            view,
            scale: self.scale,
            visibility: self.visibility.clone(),
        })
    }

    /// Get first elem from Tensor
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut b = Tensor::<IntegerRep>::new(Some(&[1]), &[1]).unwrap();
    ///
    /// assert_eq!(a.first().unwrap(), b);
    /// ```
    pub fn first(&self) -> Result<Tensor<T>, TensorError>
    where
        T: Send + Sync,
    {
        let slice = self.dims().iter().map(|_| 0..1).collect::<Vec<_>>();
        let indices = self.get_slice_ranges(&slice)?;

        let original_dims = if let Some(view) = &self.view {
            view.original_dims.clone()
        } else {
            self.dims.clone()
        };

        // the view for the last element is the slice of the size of the first dimension
        let view = Some(TensorView {
            ranges: indices,
            original_dims,
        });

        let dims = self.dims().iter().map(|_x| 1).collect::<Vec<_>>();

        Ok(Tensor {
            inner: self.inner.clone(),
            dims,
            view,
            scale: self.scale,
            visibility: self.visibility.clone(),
        })
    }
}

impl<T: Clone + TensorType> Tensor<Tensor<T>> {
    /// Flattens a tensor of tensors
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// let mut a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// let mut b = Tensor::<IntegerRep>::new(Some(&[1, 4]), &[2, 1]).unwrap();
    /// let mut c = Tensor::new(Some(&[a,b]), &[2]).unwrap();
    /// let mut d = c.combine().unwrap();
    /// assert_eq!(d.dims(), &[8]);
    /// ```
    pub fn combine(&self) -> Result<Tensor<T>, TensorError> {
        let mut dims = 0;
        let mut inner = Vec::new();
        for t in self.inner.deref().iter() {
            dims += t.len();
            inner.extend(t.iter().cloned());
        }
        Tensor::new(Some(&inner), &[dims])
    }
}

impl<T: TensorType + Add<Output = T> + std::marker::Send + std::marker::Sync> Add for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Adds tensors.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use std::ops::Add;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 4, 4, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 1D casting
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2]),
    ///     &[1]).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 3, 4, 3, 3, 3]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// // Now test 2D casting
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 3]),
    ///     &[2]).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 3, 4, 4, 4, 4]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        let res = {
            let mut res: Tensor<T> = lhs
                .iter()
                .zip(rhs.iter())
                .par_bridge()
                .map(|(o, r)| o.clone() + r.clone())
                .collect();
            res.reshape(&broadcasted_shape).unwrap();
            res
        };

        Ok(res)
    }
}

impl<T: TensorType + Neg<Output = T> + std::marker::Send + std::marker::Sync> Neg for Tensor<T> {
    type Output = Tensor<T>;
    /// Negates a tensor.
    /// # Arguments
    /// * `self` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use std::ops::Neg;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 1, 2, 1, 1, 1]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = x.neg();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[-2, -1, -2, -1, -1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn neg(self) -> Self {
        self.par_iter().map(|x| x.clone().neg()).collect()
    }
}

impl<T: TensorType + Sub<Output = T> + std::marker::Send + std::marker::Sync> Sub for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Subtracts tensors.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use std::ops::Sub;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, -2, 0, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 1D sub
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2]),
    ///     &[1],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, -1, 0, -1, -1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 2D sub
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 3]),
    ///     &[2],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, -1, 0, -2, -2, -2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        let res = {
            let mut res: Tensor<T> = lhs
                .iter()
                .zip(rhs.iter())
                .par_bridge()
                .map(|(o, r)| o.clone() - r.clone())
                .collect();
            res.reshape(&broadcasted_shape).unwrap();
            res
        };

        Ok(res)
    }
}

impl<T: TensorType + Mul<Output = T> + std::marker::Send + std::marker::Sync> Mul for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Elementwise multiplies tensors.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use std::ops::Mul;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 3, 4, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 1D mult
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2]),
    ///     &[1]).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 2D mult
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 2]),
    ///     &[2]).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        let broadcasted_shape = get_broadcasted_shape(self.dims(), rhs.dims()).unwrap();
        let lhs = self.expand(&broadcasted_shape).unwrap();
        let rhs = rhs.expand(&broadcasted_shape).unwrap();

        let res = {
            let mut res: Tensor<T> = lhs
                .iter()
                .zip(rhs.iter())
                .par_bridge()
                .map(|(o, r)| o.clone() * r.clone())
                .collect();
            res.reshape(&broadcasted_shape).unwrap();
            res
        };

        Ok(res)
    }
}

/// Returns the broadcasted shape of two tensors
/// ```
/// use ezkl::tensor::get_broadcasted_shape;
/// let a = vec![2, 3];
/// let b = vec![2, 3];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![3];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![2, 1];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![1, 3];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// let a = vec![2, 3];
/// let b = vec![1, 1];
/// let c = get_broadcasted_shape(&a, &b).unwrap();
/// assert_eq!(c, vec![2, 3]);
///
/// ```
pub fn get_broadcasted_shape(
    shape_a: &[usize],
    shape_b: &[usize],
) -> Result<Vec<usize>, TensorError> {
    let num_dims_a = shape_a.len();
    let num_dims_b = shape_b.len();

    if num_dims_a == num_dims_b {
        let mut broadcasted_shape = Vec::with_capacity(num_dims_a);
        for (dim_a, dim_b) in shape_a.iter().zip(shape_b.iter()) {
            let max_dim = dim_a.max(dim_b);
            broadcasted_shape.push(*max_dim);
        }
        Ok(broadcasted_shape)
    } else if num_dims_a < num_dims_b {
        Ok(shape_b.to_vec())
    } else if num_dims_a > num_dims_b {
        Ok(shape_a.to_vec())
    } else {
        Err(TensorError::DimError(
            "Unknown condition for broadcasting".to_string(),
        ))
    }
}
////////////////////////
///

/// The shape of data for some operations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default, Copy)]
pub enum DataFormat {
    /// NCHW
    #[default]
    NCHW,
    /// NHWC
    NHWC,
    /// CHW
    CHW,
    /// HWC
    HWC,
}

// as str
impl core::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFormat::NCHW => write!(f, "NCHW"),
            DataFormat::NHWC => write!(f, "NHWC"),
            DataFormat::CHW => write!(f, "CHW"),
            DataFormat::HWC => write!(f, "HWC"),
        }
    }
}

impl DataFormat {
    /// Get the format's canonical form
    pub fn canonical(&self) -> DataFormat {
        match self {
            DataFormat::NHWC => DataFormat::NCHW,
            DataFormat::HWC => DataFormat::CHW,
            _ => *self,
        }
    }

    /// no batch dim
    pub fn has_no_batch(&self) -> bool {
        match self {
            DataFormat::CHW | DataFormat::HWC => true,
            _ => false,
        }
    }

    /// Convert tensor to canonical format (NCHW or CHW)
    pub fn to_canonical<F: PrimeField + TensorType + PartialOrd + Hash>(
        &self,
        tensor: &mut ValTensor<F>,
    ) -> Result<(), TensorError> {
        match self {
            DataFormat::NHWC => {
                // For ND: Move channels from last axis to position after batch
                let ndims = tensor.dims().len();
                if ndims > 2 {
                    tensor.move_axis(ndims - 1, 1)?;
                }
            }
            DataFormat::HWC => {
                // For ND: Move channels from last axis to first position
                let ndims = tensor.dims().len();
                if ndims > 1 {
                    tensor.move_axis(ndims - 1, 0)?;
                }
            }
            _ => {} // NCHW/CHW are already in canonical format
        }
        Ok(())
    }

    /// Convert tensor from canonical format to target format
    pub fn from_canonical<F: PrimeField + TensorType + PartialOrd + Hash>(
        &self,
        tensor: &mut ValTensor<F>,
    ) -> Result<(), TensorError> {
        match self {
            DataFormat::NHWC => {
                // Move channels from position 1 to end
                let ndims = tensor.dims().len();
                if ndims > 2 {
                    tensor.move_axis(1, ndims - 1)?;
                }
            }
            DataFormat::HWC => {
                // Move channels from position 0 to end
                let ndims = tensor.dims().len();
                if ndims > 1 {
                    tensor.move_axis(0, ndims - 1)?;
                }
            }
            _ => {} // NCHW/CHW don't need conversion
        }
        Ok(())
    }

    /// Get the position of the channel dimension
    pub fn get_channel_dim(&self, ndims: usize) -> usize {
        match self {
            DataFormat::NCHW => 1,
            DataFormat::NHWC => ndims - 1,
            DataFormat::CHW => 0,
            DataFormat::HWC => ndims - 1,
        }
    }
}
/// The shape of the kernel for some operations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default, Copy)]
pub enum KernelFormat {
    /// HWIO
    HWIO,
    /// OIHW
    #[default]
    OIHW,
    /// OHWI
    OHWI,
}

impl core::fmt::Display for KernelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelFormat::HWIO => write!(f, "HWIO"),
            KernelFormat::OIHW => write!(f, "OIHW"),
            KernelFormat::OHWI => write!(f, "OHWI"),
        }
    }
}

impl KernelFormat {
    /// Get the format's canonical form
    pub fn canonical(&self) -> KernelFormat {
        match self {
            KernelFormat::HWIO => KernelFormat::OIHW,
            KernelFormat::OHWI => KernelFormat::OIHW,
            _ => *self,
        }
    }

    /// Convert kernel to canonical format (OIHW)
    pub fn to_canonical<F: PrimeField + TensorType + PartialOrd + Hash>(
        &self,
        kernel: &mut ValTensor<F>,
    ) -> Result<(), TensorError> {
        match self {
            KernelFormat::HWIO => {
                let kdims = kernel.dims().len();
                // Move output channels from last to first
                kernel.move_axis(kdims - 1, 0)?;
                // Move input channels from new last to second position
                kernel.move_axis(kdims - 1, 1)?;
            }
            KernelFormat::OHWI => {
                let kdims = kernel.dims().len();
                // Move input channels from last to second position
                kernel.move_axis(kdims - 1, 1)?;
            }
            _ => {} // OIHW is already canonical
        }
        Ok(())
    }

    /// Convert kernel from canonical format to target format
    pub fn from_canonical<F: PrimeField + TensorType + PartialOrd + Hash>(
        &self,
        kernel: &mut ValTensor<F>,
    ) -> Result<(), TensorError> {
        match self {
            KernelFormat::HWIO => {
                let kdims = kernel.dims().len();
                // Move input channels from second position to last
                kernel.move_axis(1, kdims - 1)?;
                // Move output channels from first to last
                kernel.move_axis(0, kdims - 1)?;
            }
            KernelFormat::OHWI => {
                let kdims = kernel.dims().len();
                // Move input channels from second position to last
                kernel.move_axis(1, kdims - 1)?;
            }
            _ => {} // OIHW doesn't need conversion
        }
        Ok(())
    }

    /// Get the position of input and output channel dimensions
    pub fn get_channel_dims(&self, ndims: usize) -> (usize, usize) {
        // (input_ch, output_ch)
        match self {
            KernelFormat::OIHW => (1, 0),
            KernelFormat::HWIO => (ndims - 2, ndims - 1),
            KernelFormat::OHWI => (ndims - 1, 0),
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl From<tract_onnx::tract_hir::ops::nn::DataFormat> for DataFormat {
    fn from(fmt: tract_onnx::tract_hir::ops::nn::DataFormat) -> Self {
        match fmt {
            tract_onnx::tract_hir::ops::nn::DataFormat::NCHW => DataFormat::NCHW,
            tract_onnx::tract_hir::ops::nn::DataFormat::NHWC => DataFormat::NHWC,
            tract_onnx::tract_hir::ops::nn::DataFormat::CHW => DataFormat::CHW,
            tract_onnx::tract_hir::ops::nn::DataFormat::HWC => DataFormat::HWC,
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl From<tract_onnx::tract_hir::tract_core::ops::cnn::conv::KernelFormat> for KernelFormat {
    fn from(fmt: tract_onnx::tract_hir::tract_core::ops::cnn::conv::KernelFormat) -> Self {
        match fmt {
            tract_onnx::tract_hir::tract_core::ops::cnn::conv::KernelFormat::HWIO => {
                KernelFormat::HWIO
            }
            tract_onnx::tract_hir::tract_core::ops::cnn::conv::KernelFormat::OIHW => {
                KernelFormat::OIHW
            }
            tract_onnx::tract_hir::tract_core::ops::cnn::conv::KernelFormat::OHWI => {
                KernelFormat::OHWI
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let data: Vec<f32> = vec![-1.0f32, 0.0, 1.0, 2.5];
        let tensor = Tensor::<f32>::new(Some(&data), &[2, 2]).unwrap();
        assert_eq!(tensor.to_vec(), data[..]);
    }

    #[test]
    fn tensor_clone() {
        let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        assert_eq!(x, x.clone());
    }

    #[test]
    fn tensor_eq() {
        let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        let mut b = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3, 1]).unwrap();
        b.reshape(&[3]).unwrap();
        let c = Tensor::<IntegerRep>::new(Some(&[1, 2, 4]), &[3]).unwrap();
        let d = Tensor::<IntegerRep>::new(Some(&[1, 2, 4]), &[3, 1]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
    #[test]
    fn tensor_slice() {
        let a = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
        let b = Tensor::<IntegerRep>::new(Some(&[1, 4]), &[2, 1]).unwrap();
        assert_eq!(
            a.get_slice(&[0..2, 0..1]).unwrap(),
            b.get_slice(&[0..2, 0..1]).unwrap()
        );
    }
}
