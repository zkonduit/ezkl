use crate::fieldutils::i32tofelt;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Constraints, Expression, Selector},
    poly::Rotation,
};

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Error;
use std::ops::Deref;
use std::ops::DerefMut;

pub trait TensorType: Clone + Debug + 'static {
    /// Returns the zero value.
    fn zero() -> Option<Self> {
        None
    }
}

macro_rules! tensor_type {
    ($rust_type:ty, $tensor_type:ident, $zero:expr) => {
        impl TensorType for $rust_type {
            fn zero() -> Option<Self> {
                Some($zero)
            }
        }
    };
}

tensor_type!(f32, Float, 0.0);
tensor_type!(i32, Int32, 0);

impl<F: FieldExt> TensorType for Value<Assigned<F>> {
    /// Returns the zero value.
    fn zero() -> Option<Self> {
        Some(Value::known(F::zero().into()))
    }
}

impl<F: FieldExt> TensorType for AssignedCell<Assigned<F>, F> {}

impl TensorType for halo2curves::pasta::Fp {}

#[derive(Debug)]
pub struct TensorError(String);

#[derive(Clone, Debug, Eq)]
pub struct Tensor<T> {
    inner: Vec<T>,
    dims: Vec<usize>,
}

impl<T> Deref for Tensor<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.inner.deref()
    }
}

impl<T> DerefMut for Tensor<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.inner.deref_mut()
    }
}

impl<T: PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        self.dims == other.dims && self.deref() == other.deref()
    }
}

impl<'a, T: Clone + TensorType> From<&'a [T]> for Tensor<T> {
    fn from(value: &'a [T]) -> Tensor<T> {
        Tensor::new(Some(&value), &[value.len() as usize]).unwrap()
    }
}

impl<'a, T: Clone + TensorType> From<Vec<T>> for Tensor<T> {
    fn from(value: Vec<T>) -> Tensor<T> {
        Tensor::new(Some(&value), &[value.len() as usize]).unwrap()
    }
}

impl<F: Clone + FieldExt + TensorType> From<Tensor<i32>> for Tensor<Value<Assigned<F>>> {
    fn from(mut t: Tensor<i32>) -> Tensor<Value<Assigned<F>>> {
        let data: Vec<Value<Assigned<F>>> = (0..t.len())
            .map(|i| Value::known(i32tofelt::<F>(t[i]).into()))
            .collect();
        Tensor::new(Some(&data), t.dims()).unwrap()
    }
}

impl<F: Clone + FieldExt + TensorType> Tensor<Value<Assigned<F>>> {
    pub fn assign_cell(
        self,
        region: &mut Region<'_, F>,
        name: &str,
        columns: &[Column<Advice>],
        offset: usize,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        assert!(self.dims.len() <= 2);
        let mut eq = Vec::new();
        if self.dims.len() == 1 {
            for i in 0..self.dims()[0] {
                let v = region.assign_advice(
                    || format!("{}", name),
                    columns[0],
                    offset + i,
                    || self.get(&[i]),
                )?;
                eq.push(v);
            }
            Ok(Tensor::new(Some(&eq), self.dims()).unwrap())
        } else if self.dims.len() == 2 {
            for i in 0..self.dims()[0] {
                for j in 0..self.dims()[1] {
                    let weight = region.assign_advice(
                        || format!("{}", name),
                        columns[i],
                        offset + j,
                        || self.get(&[i, j]),
                    )?;
                    eq.push(weight);
                }
            }
            Ok(Tensor::new(Some(&eq), self.dims()).unwrap())
        } else {
            panic!("should never reach here")
        }
    }

    pub fn assign_cell_const_offst(
        self,
        region: &mut Region<'_, F>,
        name: &str,
        columns: &[Column<Advice>],
        offset: usize,
    ) -> Result<Tensor<AssignedCell<Assigned<F>, F>>, halo2_proofs::plonk::Error> {
        assert!(self.dims.len() <= 2);
        let mut eq = Vec::new();
        if self.dims.len() == 1 {
            for i in 0..self.dims()[0] {
                let v = region.assign_advice(
                    || format!("{}", name),
                    columns[i],
                    offset,
                    || self.get(&[i]),
                )?;
                eq.push(v);
            }
            Ok(Tensor::new(Some(&eq), self.dims()).unwrap())
        } else if self.dims.len() == 2 {
            for i in 0..self.dims()[0] {
                for j in 0..self.dims()[1] {
                    let weight = region.assign_advice(
                        || format!("{}", name),
                        columns[i],
                        offset,
                        || self.get(&[i, j]),
                    )?;
                    eq.push(weight);
                }
            }
            Ok(Tensor::new(Some(&eq), self.dims()).unwrap())
        } else {
            panic!("should never reach here")
        }
    }
}

impl<T: Clone + TensorType> Tensor<T> {
    /// Sets (copies) the tensor values to the provided ones.
    /// ```
    pub fn new(values: Option<&[T]>, dims: &[usize]) -> Result<Self, TensorError> {
        let total_dims: usize = dims.iter().product();
        match values {
            Some(v) => {
                if total_dims != v.len().try_into().unwrap() {
                    return Err(TensorError(
                        "length of values array is not equal to tensor total elements".to_string(),
                    ));
                }
                Ok(Tensor {
                    inner: Vec::from(v),
                    dims: Vec::from(dims),
                })
            }
            None => Ok(Tensor {
                inner: vec![T::zero().unwrap(); total_dims as usize],
                dims: Vec::from(dims),
            }),
        }
    }

    pub fn len(&mut self) -> usize {
        self.dims().iter().product::<usize>()
    }

    /// Set one single value on the tensor.
    ///
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
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

    /// Get one single value from the Tensor.
    ///
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(None, &[2, 3, 5]).unwrap();
    ///
    /// a[1*15 + 1*5 + 1] = 5;
    /// assert_eq!(a.get(&[1, 1, 1]), 5);
    /// ```
    pub fn get(&self, indices: &[usize]) -> T {
        let index = self.get_index(indices);
        self[index].clone()
    }

    /// Get the array index from rows / columns indices.
    ///
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    ///
    /// assert_eq!(a.get_index(&[2, 2, 2]), 26);
    /// assert_eq!(a.get_index(&[1, 2, 2]), 17);
    /// assert_eq!(a.get_index(&[1, 2, 0]), 15);
    /// assert_eq!(a.get_index(&[1, 0, 1]), 10);
    /// ```
    pub fn get_index(&self, indices: &[usize]) -> usize {
        assert!(self.dims.len() == indices.len());
        let mut index = 0;
        let mut d = 1;
        for i in (0..indices.len()).rev() {
            assert!(self.dims[i] > indices[i]);
            index += indices[i] * d;
            d *= self.dims[i];
        }
        index as usize
    }

    /// Returns the tensor's dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the tensor's dimensions.
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.reshape(&[9, 3])
    /// assert_eq!(a.dims(), &[9, 3]);
    /// ```
    pub fn reshape(&mut self, new_dims: &[usize]) {
        assert!(self.len() == new_dims.iter().product::<usize>());
        self.dims = Vec::from(new_dims);
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
        let b = Tensor::<i32>::from(&[1, 2, 3][..]);
        let c = Tensor::<i32>::new(Some(&[1, 2, 4]), &[3]).unwrap();
        let d = Tensor::<i32>::new(Some(&[1, 2, 4]), &[3, 1]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
}
