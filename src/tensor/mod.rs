/// Implementations of common operations on tensors.
pub mod ops;
/// A wrapper around a tensor of circuit variables / advices.
pub mod val;
/// A wrapper around a tensor of Halo2 Value types.
pub mod var;

use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
pub use val::*;
pub use var::*;

use crate::{
    circuit::utils,
    fieldutils::{felt_to_i32, i128_to_felt, i32_to_felt},
};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Expression, Fixed, VirtualCells},
    poly::Rotation,
};
use itertools::Itertools;
use std::ops::DerefMut;
use std::ops::Range;
use std::{cmp::max, ops::Add};
use std::{cmp::min, ops::Deref};
use std::{error::Error, ops::Mul};
use std::{fmt::Debug, ops::Sub};
use std::{iter::Iterator, ops::Div};
use thiserror::Error;
/// A wrapper for tensor related errors.
#[derive(Debug, Error)]
pub enum TensorError {
    /// Shape mismatch in a operation
    #[error("dimension mismatch in tensor op: {0}")]
    DimMismatch(String),
    /// Shape when instantiating
    #[error("dimensionality error when manipulating a tensor")]
    DimError,
    /// wrong method was called on a tensor-like struct
    #[error("wrong method called")]
    WrongMethod,
}

/// The (inner) type of tensor elements.
pub trait TensorType: Clone + Debug + 'static {
    /// Returns the zero value.
    fn zero() -> Option<Self> {
        None
    }
    /// Returns the unit value.
    fn one() -> Option<Self> {
        None
    }
    /// Max operator for ordering values.
    fn tmax(&self, _: &Self) -> Option<Self> {
        None
    }
}

macro_rules! tensor_type {
    ($rust_type:ty, $tensor_type:ident, $zero:expr, $one:expr) => {
        impl TensorType for $rust_type {
            fn zero() -> Option<Self> {
                Some($zero)
            }
            fn one() -> Option<Self> {
                Some($one)
            }

            fn tmax(&self, other: &Self) -> Option<Self> {
                Some(max(*self, *other))
            }
        }
    };
}

impl TensorType for f32 {
    fn zero() -> Option<Self> {
        Some(0.0)
    }

    // f32 doesnt impl Ord so we cant just use max like we can for i32, usize.
    // A comparison between f32s needs to handle NAN values.
    fn tmax(&self, other: &Self) -> Option<Self> {
        match (self.is_nan(), other.is_nan()) {
            (true, true) => Some(f32::NAN),
            (true, false) => Some(*other),
            (false, true) => Some(*self),
            (false, false) => {
                if self >= other {
                    Some(*self)
                } else {
                    Some(*other)
                }
            }
        }
    }
}

tensor_type!(i128, Int128, 0, 1);
tensor_type!(i32, Int32, 0, 1);
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

    fn tmax(&self, other: &Self) -> Option<Self> {
        Some(
            (self.clone())
                .zip(other.clone())
                .map(|(a, b)| a.tmax(&b).unwrap()),
        )
    }
}

impl<F: FieldExt> TensorType for Assigned<F> {
    fn zero() -> Option<Self> {
        Some(F::zero().into())
    }

    fn one() -> Option<Self> {
        Some(F::one().into())
    }

    fn tmax(&self, other: &Self) -> Option<Self> {
        if self.evaluate() >= other.evaluate() {
            Some(*self)
        } else {
            Some(*other)
        }
    }
}

impl<F: FieldExt> TensorType for Expression<F> {
    fn zero() -> Option<Self> {
        Some(Expression::Constant(F::zero()))
    }

    fn one() -> Option<Self> {
        Some(Expression::Constant(F::one()))
    }

    fn tmax(&self, _: &Self) -> Option<Self> {
        todo!()
    }
}

impl TensorType for Column<Advice> {}
impl TensorType for Column<Fixed> {}

impl<F: FieldExt> TensorType for AssignedCell<Assigned<F>, F> {
    fn tmax(&self, other: &Self) -> Option<Self> {
        let mut output: Option<Self> = None;
        self.value_field().zip(other.value_field()).map(|(a, b)| {
            if a.evaluate() >= b.evaluate() {
                output = Some(self.clone());
            } else {
                output = Some(other.clone());
            }
        });
        output
    }
}

impl<F: FieldExt> TensorType for AssignedCell<F, F> {
    fn tmax(&self, other: &Self) -> Option<Self> {
        let mut output: Option<Self> = None;
        self.value().zip(other.value()).map(|(a, b)| {
            if a >= b {
                output = Some(self.clone());
            } else {
                output = Some(other.clone());
            }
        });
        output
    }
}

// specific types
impl TensorType for halo2curves::pasta::Fp {
    fn zero() -> Option<Self> {
        Some(halo2curves::pasta::Fp::zero())
    }

    fn one() -> Option<Self> {
        Some(halo2curves::pasta::Fp::one())
    }

    fn tmax(&self, other: &Self) -> Option<Self> {
        Some((*self).max(*other))
    }
}

impl TensorType for halo2curves::bn256::Fr {
    fn zero() -> Option<Self> {
        Some(halo2curves::bn256::Fr::zero())
    }

    fn one() -> Option<Self> {
        Some(halo2curves::bn256::Fr::one())
    }

    fn tmax(&self, other: &Self) -> Option<Self> {
        Some((*self).max(*other))
    }
}

/// A generic multi-dimensional array representation of a Tensor.
/// The `inner` attribute contains a vector of values whereas `dims` corresponds to the dimensionality of the array
/// and as such determines how we index, query for values, or slice a Tensor.
#[derive(Clone, Debug, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Tensor<T: TensorType> {
    inner: Vec<T>,
    dims: Vec<usize>,
}

impl<T: TensorType> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = ::std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<T: TensorType> Deref for Tensor<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.inner.deref()
    }
}

impl<T: TensorType> DerefMut for Tensor<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.inner.deref_mut()
    }
}

impl<T: PartialEq + TensorType> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.dims == other.dims && self.deref() == other.deref()
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

impl<F: FieldExt + Clone + TensorType> From<Tensor<AssignedCell<Assigned<F>, F>>> for Tensor<i32> {
    fn from(value: Tensor<AssignedCell<Assigned<F>, F>>) -> Tensor<i32> {
        let mut output = Vec::new();
        value.map(|x| {
            x.evaluate().value().map(|y| {
                let e = felt_to_i32(*y);
                output.push(e);
                e
            })
        });
        Tensor::new(Some(&output), value.dims()).unwrap()
    }
}

impl<F: FieldExt + Clone + TensorType> From<Tensor<AssignedCell<F, F>>> for Tensor<i32> {
    fn from(value: Tensor<AssignedCell<F, F>>) -> Tensor<i32> {
        let mut output = Vec::new();
        value.map(|x| {
            let mut i = 0;
            x.value().map(|y| {
                let e = felt_to_i32(*y);
                output.push(e);
                i += 1;
            });
            if i == 0 {
                output.push(0);
            }
        });
        Tensor::new(Some(&output), value.dims()).unwrap()
    }
}

impl<F: FieldExt + Clone + TensorType> From<Tensor<AssignedCell<Assigned<F>, F>>>
    for Tensor<Value<F>>
{
    fn from(value: Tensor<AssignedCell<Assigned<F>, F>>) -> Tensor<Value<F>> {
        let mut output = Vec::new();
        for (_, x) in value.iter().enumerate() {
            output.push(x.value_field().evaluate());
        }
        Tensor::new(Some(&output), value.dims()).unwrap()
    }
}

impl<F: FieldExt + TensorType + Clone> From<Tensor<Value<F>>> for Tensor<i32> {
    fn from(t: Tensor<Value<F>>) -> Tensor<i32> {
        let mut output = Vec::new();
        t.map(|x| {
            let mut i = 0;
            x.map(|y| {
                let e = felt_to_i32(y);
                output.push(e);
                i += 1;
            });
            if i == 0 {
                output.push(0);
            }
        });
        Tensor::new(Some(&output), t.dims()).unwrap()
    }
}

impl<F: FieldExt + TensorType + Clone> From<Tensor<Value<F>>> for Tensor<Value<Assigned<F>>> {
    fn from(t: Tensor<Value<F>>) -> Tensor<Value<Assigned<F>>> {
        let mut ta: Tensor<Value<Assigned<F>>> = Tensor::from((0..t.len()).map(|i| t[i].into()));
        ta.reshape(t.dims());
        ta
    }
}

impl<F: FieldExt + TensorType + Clone> From<Tensor<i32>> for Tensor<Value<F>> {
    fn from(t: Tensor<i32>) -> Tensor<Value<F>> {
        let mut ta: Tensor<Value<F>> =
            Tensor::from((0..t.len()).map(|i| Value::known(i32_to_felt::<F>(t[i]))));
        ta.reshape(t.dims());
        ta
    }
}

impl<F: FieldExt + TensorType + Clone> From<Tensor<i128>> for Tensor<Value<F>> {
    fn from(t: Tensor<i128>) -> Tensor<Value<F>> {
        let mut ta: Tensor<Value<F>> =
            Tensor::from((0..t.len()).map(|i| Value::known(i128_to_felt::<F>(t[i]))));
        ta.reshape(t.dims());
        ta
    }
}

impl<T: Clone + TensorType + std::marker::Send + std::marker::Sync>
    rayon::iter::IntoParallelIterator for Tensor<T>
{
    type Iter = rayon::vec::IntoIter<T>;
    type Item = T;
    fn into_par_iter(self) -> Self::Iter {
        self.inner.into_par_iter()
    }
}

impl<'data, T: Clone + TensorType + std::marker::Send + std::marker::Sync>
    rayon::iter::IntoParallelRefMutIterator<'data> for Tensor<T>
{
    type Iter = rayon::slice::IterMut<'data, T>;
    type Item = &'data mut T;
    fn par_iter_mut(&'data mut self) -> Self::Iter {
        self.inner.par_iter_mut()
    }
}

impl<T: Clone + TensorType> Tensor<T> {
    /// Sets (copies) the tensor values to the provided ones.
    pub fn new(values: Option<&[T]>, dims: &[usize]) -> Result<Self, TensorError> {
        let total_dims: usize = dims.iter().product();
        match values {
            Some(v) => {
                if total_dims != v.len() {
                    return Err(TensorError::DimError);
                }
                Ok(Tensor {
                    inner: Vec::from(v),
                    dims: Vec::from(dims),
                })
            }
            None => Ok(Tensor {
                inner: vec![T::zero().unwrap(); total_dims],
                dims: Vec::from(dims),
            }),
        }
    }

    /// Returns the number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.dims().iter().product::<usize>()
    }
    /// Checks if the number of elements in tensor is 0.
    pub fn is_empty(&self) -> bool {
        self.dims().iter().product::<usize>() == 0
    }

    /// Set one single value on the tensor.
    ///
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// a.set(&[0, 0, 1], 10);
    /// assert_eq!(a[0 + 0 + 1], 10);
    ///
    /// a.set(&[2, 2, 0], 9);
    /// assert_eq!(a[2*9 + 2*3 + 0], 9);
    /// ```
    pub fn set(&mut self, indices: &[usize], value: T) {
        let index = self.get_index(indices);
        self[index] = value;
    }

    /// Get a single value from the Tensor.
    ///
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get(&self, indices: &[usize]) -> T {
        let index = self.get_index(indices);
        self[index].clone()
    }

    /// Get a slice from the Tensor.
    ///
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[1, 2]), &[2]).unwrap();
    ///
    /// assert_eq!(a.get_slice(&[0..2]).unwrap(), b);
    /// ```
    pub fn get_slice(&self, indices: &[Range<usize>]) -> Result<Tensor<T>, TensorError> {
        if self.dims.len() < indices.len() {
            return Err(TensorError::DimError);
        }
        let mut res = Vec::new();
        // if indices weren't specified we fill them in as required
        let mut full_indices = indices.to_vec();

        for i in 0..(self.dims.len() - indices.len()) {
            full_indices.push(0..self.dims()[indices.len() + i])
        }
        for e in full_indices.iter().cloned().multi_cartesian_product() {
            let index = self.get_index(&e);
            res.push(self[index].clone())
        }
        let dims: Vec<usize> = full_indices.iter().map(|e| e.end - e.start).collect();
        // for i in (0..indices.len()).rev() {
        //     if (dims[i] == 1) && (dims.len() > 1) {
        //         dims.remove(i);
        //     }
        // }

        Tensor::new(Some(&res), &dims)
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    pub fn get_index(&self, indices: &[usize]) -> usize {
        assert_eq!(self.dims.len(), indices.len());
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        index
    }

    /// Duplicates every nth element
    ///
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 3, 4, 5, 5, 6]), &[8]).unwrap();
    /// assert_eq!(a.duplicate_every_n(3, 0).unwrap(), expected);
    /// assert_eq!(a.duplicate_every_n(7, 0).unwrap(), a);
    ///
    /// let expected = Tensor::<i32>::new(Some(&[1, 1, 2, 3, 3, 4, 5, 5, 6]), &[9]).unwrap();
    /// assert_eq!(a.duplicate_every_n(3, 2).unwrap(), expected);
    ///
    /// ```
    pub fn duplicate_every_n(
        &self,
        n: usize,
        initial_offset: usize,
    ) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = vec![];
        let mut offset = initial_offset;
        for (i, elem) in self.inner.clone().into_iter().enumerate() {
            if (i + offset + 1) % n == 0 {
                inner.extend(vec![elem; 2].into_iter());
                offset += 1;
            } else {
                inner.push(elem.clone());
            }
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Removes every nth element
    ///
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let a = Tensor::<i32>::new(Some(&[1, 2, 3, 3, 4, 5, 6, 6]), &[8]).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    /// assert_eq!(a.remove_every_n(4, 0).unwrap(), expected);
    ///
    /// let a = Tensor::<i32>::new(Some(&[1, 2, 3, 3, 4, 5, 6]), &[7]).unwrap();
    /// assert_eq!(a.remove_every_n(4, 0).unwrap(), expected);
    /// assert_eq!(a.remove_every_n(9, 0).unwrap(), a);
    ///
    /// let a = Tensor::<i32>::new(Some(&[1, 1, 2, 3, 3, 4, 5, 5, 6]), &[9]).unwrap();
    /// assert_eq!(a.remove_every_n(3, 2).unwrap(), expected);
    ///
    pub fn remove_every_n(
        &self,
        n: usize,
        initial_offset: usize,
    ) -> Result<Tensor<T>, TensorError> {
        let mut inner: Vec<T> = vec![];
        for (i, elem) in self.inner.clone().into_iter().enumerate() {
            if (i + initial_offset + 1) % n == 0 {
            } else {
                inner.push(elem.clone());
            }
        }
        Tensor::new(Some(&inner), &[inner.len()])
    }

    /// Returns the tensor's dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    ///Reshape the tensor
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.reshape(&[9, 3]);
    /// assert_eq!(a.dims(), &[9, 3]);
    /// ```
    pub fn reshape(&mut self, new_dims: &[usize]) {
        assert!(self.len() == new_dims.iter().product::<usize>());
        self.dims = Vec::from(new_dims);
    }

    ///Flatten the tensor shape
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.flatten();
    /// assert_eq!(a.dims(), &[27]);
    /// ```
    pub fn flatten(&mut self) {
        self.dims = Vec::from([self.dims.iter().product::<usize>()]);
    }

    /// Maps a function to tensors
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.map(|x| i32::pow(x,2));
    /// assert_eq!(c, Tensor::from([1, 16].into_iter()))
    /// ```
    pub fn map<F: FnMut(T) -> G, G: TensorType>(&self, mut f: F) -> Tensor<G> {
        let mut t = Tensor::from(self.inner.iter().map(|e| f(e.clone())));
        t.reshape(self.dims());
        t
    }

    /// Maps a function to tensors and enumerates
    /// ```
    /// use ezkl_lib::tensor::{Tensor, TensorError};
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.enum_map::<_,_,TensorError>(|i, x| Ok(i32::pow(x + i as i32, 2))).unwrap();
    /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn enum_map<F: FnMut(usize, T) -> Result<G, E>, G: TensorType, E: Error>(
        &self,
        mut f: F,
    ) -> Result<Tensor<G>, E> {
        let vec: Result<Vec<G>, E> = self
            .inner
            .iter()
            .enumerate()
            .map(|(i, e)| f(i, e.clone()))
            .collect();
        let mut t: Tensor<G> = Tensor::from(vec?.iter().cloned());
        t.reshape(self.dims());
        Ok(t)
    }

    /// Maps a function to tensors and enumerates using multi cartesian coordinates
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.mc_enum_map(|i, x| i32::pow(x + i[0] as i32, 2)).unwrap();
    /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn mc_enum_map<F: FnMut(&[usize], T) -> G, G: TensorType>(
        &self,
        mut f: F,
    ) -> Result<Tensor<G>, TensorError> {
        let mut indices = Vec::new();
        for i in self.dims.clone() {
            indices.push(0..i);
        }
        let mut res = Vec::new();
        for coord in indices.iter().cloned().multi_cartesian_product() {
            res.push(f(&coord, self.get(&coord)));
        }

        Tensor::new(Some(&res), self.dims())
    }

    /// Transposes a 2D tensor
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 3, 2, 4]), &[2, 2]).unwrap();
    /// a.transpose_2d();
    /// assert_eq!(a, expected);
    /// ```
    pub fn transpose_2d(&mut self) -> Result<(), TensorError> {
        if self.dims().len() != 2 {
            return Err(TensorError::DimError);
        }
        let mut indices = Vec::new();
        for i in self.dims.clone() {
            indices.push(0..i);
        }
        let mut v_transpose = Tensor::new(None, &[self.dims()[1], self.dims()[0]])?;
        for coord in indices.iter().cloned().multi_cartesian_product() {
            v_transpose.set(&[coord[1], coord[0]], self.get(&coord));
        }
        *self = v_transpose;
        Ok(())
    }

    /// Adds a row of ones
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4, 1, 4]), &[2, 2]).unwrap();
    /// let mut c = a.pad_row_ones().unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 4, 1, 4, 1, 1]), &[3, 2]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn pad_row_ones(&self) -> Result<Tensor<T>, TensorError> {
        let mut result = self.inner.clone();
        for _ in 0..self.dims[1] {
            result.push(T::one().unwrap());
        }
        let mut dims = self.dims().to_vec();
        dims[0] += 1;
        Tensor::new(Some(&result), &dims)
    }

    /// Tiles a tensor n times
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.tile(2).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 4, 1, 4]), &[2, 2]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn tile(&self, n: usize) -> Result<Tensor<T>, TensorError> {
        let mut res = vec![];
        for _ in 0..n {
            res.push(self.clone());
        }
        let mut tiled = Tensor::new(Some(&res), &[n])?.combine()?;
        tiled.reshape(&[&[n], self.dims()].concat());
        Ok(tiled)
    }

    /// Extends another tensor to rows
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4, 1, 4, 1, 4]), &[3, 2]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[2, 3, 2, 3, 2, 3]), &[3, 2]).unwrap();
    /// let mut c = a.append_to_row(&[&b]).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2, 3]), &[3, 4]).unwrap();
    /// assert_eq!(c, expected);
    ///
    /// let mut a =Tensor::<i32>::new(Some(&[10, 0, 0, 20, 10, 0, 0, 20, 10, 0, 0, 20]), &[4, 3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[30, 0, 0, 40, 30, 0, 0, 40, 30, 0, 0, 40]), &[4, 3]).unwrap();
    /// let mut c = a.append_to_row(&[&b]).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[10, 0, 0, 30, 0, 0, 20, 10, 0, 40, 30, 0, 0, 20, 10, 0, 40, 30, 0, 0, 20, 0, 0, 40]), &[4, 6]).unwrap();
    /// assert_eq!(c, expected);
    ///
    /// let mut a =Tensor::<i32>::new(Some(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), &[4, 3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[10, 0, 0, 20, 10, 0, 0, 20, 10, 0, 0, 20]), &[4, 3]).unwrap();
    /// let mut c = a.append_to_row(&[&b]).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[0, 0, 0, 10, 0, 0, 0, 0, 0, 20, 10, 0, 0, 0, 0, 0, 20, 10, 0, 0, 0, 0, 0, 20]), &[4, 6]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn append_to_row(&self, vecs: &[&Tensor<T>]) -> Result<Tensor<T>, TensorError> {
        for b in vecs {
            if self.dims()[0] != b.dims()[0] {
                return Err(TensorError::DimMismatch("append to row".to_string()));
            }
        }
        let mut rows = Vec::new();
        for i in 0..self.dims[0] {
            let mut row = vec![self.get_slice(&[i..i + 1])?];
            for b in vecs {
                row.push(b.get_slice(&[i..i + 1])?);
            }
            rows.push(Tensor::new(Some(&row), &[vecs.len() + 1])?.combine()?);
        }
        let mut res = Tensor::new(Some(&rows), &[self.dims[0]])?.combine()?;
        let mut dims = self.dims().to_vec();
        let len = dims.len();
        for b in vecs {
            dims[len - 1] += b.dims()[1];
        }
        res.reshape(&dims);
        Ok(res)
    }

    /// Repeats the rows of a tensor n times
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
    /// let mut c = a.repeat_rows(2).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]), &[3, 2, 2]).unwrap();
    /// assert_eq!(c, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut c = a.repeat_rows(2).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 1, 2, 2, 3, 3]), &[3, 2]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn repeat_rows(&self, n: usize) -> Result<Tensor<T>, TensorError> {
        let mut rows = vec![];
        for i in 0..self.dims[0] {
            let mut row = self.get_slice(&[i..i + 1])?;
            row.flatten();
            rows.push(row);
        }
        let mut res = vec![];
        for row in rows.iter().take(self.dims[0]) {
            for _ in 0..n {
                res.push(row.clone());
            }
        }

        let mut tiled = Tensor::new(Some(&res), &[n * self.dims()[0]])?.combine()?;
        tiled.reshape(&[self.dims(), &[n]].concat());
        Ok(tiled)
    }

    /// Multi-ch doubly blocked toeplitz matrix
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 0, 0, 0, 0]), &[1, 1, 2, 4]).unwrap();
    /// let mut c = a.multi_ch_doubly_blocked_toeplitz(2, 2, 3, 4, 1, 1).unwrap();
    /// let mut expected = Tensor::<i32>::new(
    /// Some(&[1, 2, 3, 4, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0,
    ///     1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0,
    ///      0, 0, 1, 2]), &[6, 8]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn multi_ch_doubly_blocked_toeplitz(
        &self,
        h_blocks: usize,
        w_blocks: usize,
        num_rows: usize,
        num_cols: usize,
        h_stride: usize,
        w_stride: usize,
    ) -> Result<Tensor<T>, TensorError>
    where
        T: std::marker::Send + std::marker::Sync,
    {
        assert!(self.dims().len() > 2);
        let first_channels = self.dims()[0];
        let second_channels = self.dims()[1];

        let mut toeplitz_tower = vec![Tensor::new(None, &[0])?; first_channels];

        toeplitz_tower
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, tower_elem)| {
                println!("outer i: {}", i);
                let zero = Tensor::new(None, &[0]).unwrap();
                let mut row = vec![zero; second_channels];
                for (j, row_block) in row.iter_mut().enumerate().take(second_channels) {
                    let mut r = self.get_slice(&[i..i + 1, j..j + 1]).unwrap();
                    let dims = r.dims()[2..].to_vec();
                    r.reshape(&dims);
                    *row_block = r
                        .doubly_blocked_toeplitz(
                            h_blocks, w_blocks, num_rows, num_cols, h_stride, w_stride,
                        )
                        .unwrap();
                }
                let mut concatenated_tensor = row[0].clone();
                concatenated_tensor = concatenated_tensor
                    .append_to_row(&row[1..].iter().map(|e| e).collect::<Vec<_>>())
                    .unwrap();
                *tower_elem = concatenated_tensor;
            });

        let mut toeplitz_tower =
            Tensor::new(Some(&toeplitz_tower[..]), &[toeplitz_tower.len()])?.combine()?;

        toeplitz_tower.reshape(&[
            first_channels * h_blocks * num_rows,
            second_channels * w_blocks * num_cols,
        ]);

        Ok(toeplitz_tower)
    }

    /// Doubly blocked toeplitz matrix
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 0, 0, 0, 0]), &[2, 4]).unwrap();
    /// let mut c = a.doubly_blocked_toeplitz(2, 2, 3, 4, 1, 1).unwrap();
    /// let mut expected = Tensor::<i32>::new(
    /// Some(&[1, 2, 3, 4, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0,
    ///     1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0,
    ///      0, 0, 1, 2]), &[6, 8]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn doubly_blocked_toeplitz(
        &self,
        h_blocks: usize,
        w_blocks: usize,
        num_rows: usize,
        num_cols: usize,
        h_stride: usize,
        w_stride: usize,
    ) -> Result<Tensor<T>, TensorError>
    where
        T: std::marker::Send + std::marker::Sync,
    {
        let mut t_matrices = vec![Tensor::new(None, &[0])?; self.dims[0]];
        t_matrices.par_iter_mut().enumerate().for_each(|(i, t)| {
            *t = self.toeplitz(i, num_rows, num_cols).unwrap();
        });

        let mut doubly_blocked_toeplitz: Vec<Tensor<T>> = vec![Tensor::new(None, &[0])?; h_blocks];

        doubly_blocked_toeplitz
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, block)| {
                println!("block j: {}", i);
                let j = i * h_stride;
                let zero_matrix = Tensor::new(None, t_matrices[0].dims()).unwrap();
                let mut row = vec![&zero_matrix; w_blocks];

                for i in 0..t_matrices.len() {
                    if i + j < w_blocks {
                        row[i + j] = &t_matrices[i];
                    }
                }

                let mut concatenated_tensor = row[0].clone();
                concatenated_tensor = concatenated_tensor.append_to_row(&row[1..]).unwrap();
                *block = concatenated_tensor;
            });

        let mut doubly_blocked_toeplitz = Tensor::new(
            Some(&doubly_blocked_toeplitz[..]),
            &[doubly_blocked_toeplitz.len()],
        )?
        .combine()?;

        doubly_blocked_toeplitz.reshape(&[h_blocks * num_rows, w_blocks * num_cols]);

        if w_stride > 1 {
            let mut shifted_rows = vec![Tensor::new(None, &[0])?; h_blocks * num_rows];

            shifted_rows
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, row)| {
                    let offset = r % num_rows;
                    *row = doubly_blocked_toeplitz.get_slice(&[r..r + 1]).unwrap();
                    if offset > 0 {
                        let mut shifted_row = Tensor::new(None, &[row.len()]).unwrap();
                        let local_offset = offset * (w_stride - 1);
                        for i in 0..shifted_row.len() - local_offset {
                            shifted_row.set(&[local_offset + i], row.get(&[0, i]).clone());
                        }
                        *row = shifted_row;
                    }
                });

            let mut doubly_blocked_toeplitz =
                Tensor::new(Some(&shifted_rows[..]), &[shifted_rows.len()])?.combine()?;

            doubly_blocked_toeplitz.reshape(&[h_blocks * num_rows, w_blocks * num_cols]);

            Ok(doubly_blocked_toeplitz)
        } else {
            Ok(doubly_blocked_toeplitz)
        }
    }

    /// Toeplitz matrix of a given row.
    /// ```
    /// // these tests were all verified against scipy.linalg.toeplitz
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[0, 0, 0, 0, 1, 2, 0, 0]), &[2, 4]).unwrap();
    /// let mut c = a.toeplitz(1, 4, 3).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 2, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0]), &[4, 3]).unwrap();
    /// assert_eq!(c, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[0, 0, 0, 0, 1, 2, 0, 0]), &[2, 4]).unwrap();
    /// let mut c = a.toeplitz(1, 2, 3).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 2, 0, 0, 1, 2]), &[2, 3]).unwrap();
    /// assert_eq!(c, expected);
    ///
    /// let mut a = Tensor::<i32>::new(Some(&[0, 0, 0, 0, 1, 2, 0, 0]), &[2, 4]).unwrap();
    /// let mut c = a.toeplitz(1, 5, 3).unwrap();
    /// let mut expected = Tensor::<i32>::new(Some(&[1, 2, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0]), &[5, 3]).unwrap();
    /// assert_eq!(c, expected);
    /// ```
    pub fn toeplitz(
        &self,
        row: usize,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Tensor<T>, TensorError>
    where
        T: std::marker::Send + std::marker::Sync,
    {
        // let n = self.dims()[1..].iter().product::<usize>();
        let mut row = self.get_slice(&[row..row + 1])?;
        row.flatten();
        let mut toeplitz = Tensor::new(None, &[num_rows, num_cols])?;
        // initialize the first row
        for j in 0..min(num_cols, row.len()) {
            toeplitz.set(&[0, j], row[j].clone());
        }
        for i in 1..num_rows {
            if (num_cols as i32 - i as i32) > 0 {
                for j in 0..(num_cols - i) {
                    toeplitz.set(&[i, i + j], toeplitz.get(&[0, j]).clone());
                }
            }
        }
        Ok(toeplitz)
    }
}

impl<T: Clone + TensorType> Tensor<Tensor<T>> {
    /// Flattens a tensor of tensors
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[1, 4]), &[2, 1]).unwrap();
    /// let mut c = Tensor::new(Some(&[a,b]), &[2]).unwrap();
    /// let mut d = c.combine().unwrap();
    /// assert_eq!(d.dims(), &[8]);
    /// ```
    pub fn combine(&self) -> Result<Tensor<T>, TensorError> {
        let mut dims = 0;
        let mut inner = Vec::new();
        for t in self.inner.clone().into_iter() {
            dims += t.len();
            inner.extend(t.inner);
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
    /// use ezkl_lib::tensor::Tensor;
    /// use std::ops::Add;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 4, 4, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 1D casting
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1]).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 3, 3, 3]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// // Now test 2D casting
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3]),
    ///     &[2]).unwrap();
    /// let result = x.add(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 4, 4, 4]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        // calculate value of output
        let mut output: Tensor<T> = self.clone();

        if self.len() != rhs.len() {
            if self.dims().iter().map(|x| (x > &1) as usize).sum::<usize>() == 1
                && rhs.dims().iter().product::<usize>() > 1
                && self.dims() != rhs.dims()
            {
                assert_eq!(rhs.dims()[0], self.dims()[0]);
                output = rhs.clone();
                let lhs = self.clone();
                let full_indices = rhs
                    .dims()
                    .iter()
                    .map(|d| 0..*d)
                    .multi_cartesian_product()
                    .collect::<Vec<Vec<usize>>>();
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    let coord = &full_indices[i];
                    *x = x.clone() + lhs[coord[0]].clone();
                });
            } else if rhs.dims().iter().map(|x| (x > &1) as usize).sum::<usize>() == 1
                && self.dims().iter().product::<usize>() > 1
                && self.dims() != rhs.dims()
            {
                assert_eq!(self.dims()[0], rhs.dims()[0]);
                let full_indices = self
                    .dims()
                    .iter()
                    .map(|d| 0..*d)
                    .multi_cartesian_product()
                    .collect::<Vec<Vec<usize>>>();
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    let coord = &full_indices[i];
                    *x = x.clone() + rhs[coord[0]].clone();
                });
            }
            // casts a 1D addition
            else if rhs.dims().iter().product::<usize>() == 1 {
                output.par_iter_mut().for_each(|o| {
                    *o = o.clone() + rhs[0].clone();
                });
            }
            // make 1D casting commutative
            else if self.dims().iter().product::<usize>() == 1 {
                output = rhs.clone();
                output.par_iter_mut().for_each(|o| {
                    *o = o.clone() + self[0].clone();
                });
            } else {
                return Err(TensorError::DimMismatch("add".to_string()));
            }
        } else {
            output.par_iter_mut().zip(rhs).for_each(|(o, r)| {
                *o = o.clone() + r.clone();
            });
        }
        Ok(output)
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
    /// use ezkl_lib::tensor::Tensor;
    /// use std::ops::Sub;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[0, -2, 0, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 1D sub
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[0, -1, 0, -1, -1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 2D sub
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3]),
    ///     &[2],
    /// ).unwrap();
    /// let result = x.sub(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[0, -1, 0, -2, -2, -2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        // calculate value of output
        let mut output: Tensor<T> = self.clone();

        if self.len() != rhs.len() {
            if self.dims().iter().map(|x| (x > &1) as usize).sum::<usize>() == 1
                && rhs.dims().iter().product::<usize>() > 1
                && self.dims() != rhs.dims()
            {
                assert_eq!(rhs.dims()[0], self.dims()[0]);
                output = rhs.clone();
                let lhs = self.clone();
                let full_indices = rhs
                    .dims()
                    .iter()
                    .map(|d| 0..*d)
                    .multi_cartesian_product()
                    .collect::<Vec<Vec<usize>>>();
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    let coord = &full_indices[i];
                    *x = x.clone() - lhs[coord[0]].clone();
                });
            } else if rhs.dims().iter().map(|x| (x > &1) as usize).sum::<usize>() == 1
                && self.dims().iter().product::<usize>() > 1
                && self.dims() != rhs.dims()
            {
                assert_eq!(self.dims()[0], rhs.dims()[0]);
                let full_indices = self
                    .dims()
                    .iter()
                    .map(|d| 0..*d)
                    .multi_cartesian_product()
                    .collect::<Vec<Vec<usize>>>();
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    let coord = &full_indices[i];
                    *x = x.clone() - rhs[coord[0]].clone();
                });
            }
            // casts a 1D addition
            else if rhs.dims().iter().product::<usize>() == 1 {
                output.par_iter_mut().for_each(|o| {
                    *o = o.clone() - rhs[0].clone();
                });
            }
            // make 1D casting commutative
            else if self.dims().iter().product::<usize>() == 1 {
                output = rhs.clone();
                output.par_iter_mut().for_each(|o| {
                    *o = self[0].clone() - o.clone();
                });
            } else {
                return Err(TensorError::DimMismatch("sub".to_string()));
            }
        } else {
            output.par_iter_mut().zip(rhs).for_each(|(o, r)| {
                *o = o.clone() - r.clone();
            });
        }
        Ok(output)
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
    /// use ezkl_lib::tensor::Tensor;
    /// use std::ops::Mul;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 3, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 1D mult
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1]).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // Now test 2D mult
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = Tensor::<i32>::new(
    ///     Some(&[2, 2]),
    ///     &[2]).unwrap();
    /// let result = x.mul(k).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        // calculate value of output
        let mut output: Tensor<T> = self.clone();

        if self.len() != rhs.len() {
            if self.dims().iter().map(|x| (x > &1) as usize).sum::<usize>() == 1
                && rhs.dims().iter().product::<usize>() > 1
                && self.dims() != rhs.dims()
            {
                assert_eq!(rhs.dims()[0], self.dims()[0]);
                output = rhs.clone();
                let lhs = self.clone();
                let full_indices = rhs
                    .dims()
                    .iter()
                    .map(|d| 0..*d)
                    .multi_cartesian_product()
                    .collect::<Vec<Vec<usize>>>();
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    let coord = &full_indices[i];
                    *x = x.clone() * lhs[coord[0]].clone();
                });
            } else if rhs.dims().iter().map(|x| (x > &1) as usize).sum::<usize>() == 1
                && self.dims().iter().product::<usize>() > 1
                && self.dims() != rhs.dims()
            {
                assert_eq!(self.dims()[0], rhs.dims()[0]);
                let full_indices = self
                    .dims()
                    .iter()
                    .map(|d| 0..*d)
                    .multi_cartesian_product()
                    .collect::<Vec<Vec<usize>>>();
                output.par_iter_mut().enumerate().for_each(|(i, x)| {
                    let coord = &full_indices[i];
                    *x = rhs[coord[0]].clone() * x.clone();
                });
            }
            // casts a 1D addition
            else if rhs.dims().iter().product::<usize>() == 1 {
                output.par_iter_mut().for_each(|o| {
                    *o = o.clone() * rhs[0].clone();
                });
            }
            // make 1D casting commutative
            else if self.dims().iter().product::<usize>() == 1 {
                output = rhs.clone();
                output.par_iter_mut().for_each(|o| {
                    *o = self[0].clone() * o.clone();
                });
            } else {
                return Err(TensorError::DimMismatch("sub".to_string()));
            }
        } else {
            output.par_iter_mut().zip(rhs).for_each(|(o, r)| {
                *o = o.clone() * r.clone();
            });
        }
        Ok(output)
    }
}

impl<T: TensorType + Mul<Output = T> + std::marker::Send + std::marker::Sync> Tensor<T> {
    /// Elementwise raise a tensor to the nth power.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// use std::ops::Mul;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.pow(3).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[8, 3375, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn pow(&self, mut exp: u32) -> Result<Self, TensorError> {
        // calculate value of output
        let mut base = self.clone();
        let mut acc = base.map(|_| T::one().unwrap());

        while exp > 1 {
            if (exp & 1) == 1 {
                acc = acc.mul(base.clone())?;
            }
            exp /= 2;
            base = base.clone().mul(base)?;
        }

        // since exp!=0, finally the exp must be 1.
        // Deal with the final bit of the exponent separately, since
        // squaring the base afterwards is not necessary and may cause a
        // needless overflow.
        acc.mul(base)
    }
}

impl<T: TensorType + Div<Output = T>> Div for Tensor<T> {
    type Output = Result<Tensor<T>, TensorError>;
    /// Elementwise divide a tensor with another tensor.
    /// # Arguments
    ///
    /// * `self` - Tensor
    /// * `rhs` - Tensor
    /// # Examples
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// use std::ops::Div;
    /// let x = Tensor::<i32>::new(
    ///     Some(&[4, 1, 4, 1, 1, 4]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let y = Tensor::<i32>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = x.div(y).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[2, 1, 2, 1, 1, 4]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// // test 1D casting
    /// let x = Tensor::<i32>::new(
    ///     Some(&[4, 1, 4, 1, 1, 4]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let y = Tensor::<i32>::new(
    ///     Some(&[2]),
    ///     &[1],
    /// ).unwrap();
    /// let result = x.div(y).unwrap();
    /// let expected = Tensor::<i32>::new(Some(&[2, 0, 2, 0, 0, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    fn div(self, rhs: Self) -> Self::Output {
        // calculate value of output
        let mut output: Tensor<T> = self.clone();

        // casts a 1D multiplication
        if rhs.dims().len() == 1 && rhs.dims()[0] == 1 {
            for i in 0..output.len() {
                output[i] = output[i].clone() / rhs[0].clone();
            }
        } else if self.dims().len() == 1 && self.dims()[0] == 1 {
            output = rhs.clone();
            for i in 0..rhs.len() {
                output[i] = self[0].clone() / output[i].clone();
            }
        } else {
            if self.dims() != rhs.dims() {
                return Err(TensorError::DimMismatch("div".to_string()));
            }

            for (i, e_i) in rhs.iter().enumerate() {
                output[i] = output[i].clone() / e_i.clone()
            }
        }
        Ok(output)
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor() {
        let data: Vec<f32> = vec![-1.0f32, 0.0, 1.0, 2.5];
        let tensor = Tensor::<f32>::new(Some(&data), &[2, 2]).unwrap();
        assert_eq!(&tensor[..], &data[..]);
    }

    #[test]
    fn tensor_clone() {
        let x = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        let clone = x.clone();
        assert_eq!(x, clone);
    }

    #[test]
    fn tensor_eq() {
        let a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
        let mut b = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3, 1]).unwrap();
        b.reshape(&[3]);
        let c = Tensor::<i32>::new(Some(&[1, 2, 4]), &[3]).unwrap();
        let d = Tensor::<i32>::new(Some(&[1, 2, 4]), &[3, 1]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
    #[test]
    fn tensor_slice() {
        let a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
        let b = Tensor::<i32>::new(Some(&[1, 4]), &[2, 1]).unwrap();
        assert_eq!(a.get_slice(&[0..2, 0..1]).unwrap(), b);
    }
}
