use crate::fieldutils::{felt_to_i32, i32tofelt};

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Value},
    plonk::{Advice, Assigned, Column, Expression, Fixed},
};
use itertools::Itertools;
use std::fmt::Debug;
use std::iter::Iterator;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Range;

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
tensor_type!(usize, USize, 0);
tensor_type!((), Empty, ());

impl<T: TensorType> TensorType for Tensor<T> {
    fn zero() -> Option<Self> {
        Some(Tensor::new(Some(&[T::zero().unwrap()]), &[1]).unwrap())
    }
}

impl<T: TensorType> TensorType for Value<T> {
    fn zero() -> Option<Self> {
        Some(Value::known(T::zero().unwrap()))
    }
}

impl<F: FieldExt> TensorType for Assigned<F> {
    fn zero() -> Option<Self> {
        Some(F::zero().into())
    }
}

impl<F: FieldExt> TensorType for Expression<F> {
    fn zero() -> Option<Self> {
        Some(Expression::Constant(F::zero()))
    }
}

impl TensorType for Column<Advice> {}
impl TensorType for Column<Fixed> {}

impl<F: FieldExt> TensorType for AssignedCell<Assigned<F>, F> {}

// specific types
impl TensorType for halo2curves::pasta::Fp {
    fn zero() -> Option<Self> {
        Some(halo2curves::pasta::Fp::zero())
    }
}

#[derive(Debug)]
pub struct TensorError(String);

#[derive(Clone, Debug, Eq)]
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

impl<I: Iterator, T: Clone + TensorType + From<I::Item>> From<I> for Tensor<T>
where
    I::Item: Clone + TensorType,
    Vec<T>: FromIterator<I::Item>,
{
    fn from(value: I) -> Tensor<T> {
        let data: Vec<T> = value.collect::<Vec<T>>();
        Tensor::new(Some(&data), &[data.len() as usize]).unwrap()
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

impl<F: FieldExt + TensorType + Clone> From<Tensor<Value<F>>> for Tensor<Value<Assigned<F>>> {
    fn from(mut t: Tensor<Value<F>>) -> Tensor<Value<Assigned<F>>> {
        let mut ta: Tensor<Value<Assigned<F>>> = Tensor::from((0..t.len()).map(|i| t[i].into()));
        ta.reshape(t.dims());
        ta
    }
}

impl<F: FieldExt + TensorType + Clone> From<Tensor<i32>> for Tensor<Value<F>> {
    fn from(mut t: Tensor<i32>) -> Tensor<Value<F>> {
        let mut ta: Tensor<Value<F>> =
            Tensor::from((0..t.len()).map(|i| Value::known(i32tofelt::<F>(t[i]))));
        ta.reshape(t.dims());
        ta
    }
}

impl<T: Clone + TensorType> Tensor<T> {
    /// Sets (copies) the tensor values to the provided ones.
    /// ```
    pub fn new(values: Option<&[T]>, dims: &[usize]) -> Result<Self, TensorError> {
        let total_dims: usize = dims.iter().product();
        match values {
            Some(v) => {
                if total_dims != v.len() {
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

    pub fn is_empty(&mut self) -> bool {
        self.dims().iter().product::<usize>() == 0
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

    /// Get a slice from the Tensor.
    ///
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[1, 2]), &[2]).unwrap();
    ///
    /// assert_eq!(a.get_slice(&[0..2]), b);
    /// ```
    pub fn get_slice(&self, indices: &[Range<usize>]) -> Tensor<T> {
        assert!(self.dims.len() >= indices.len());
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
        let mut dims: Vec<usize> = full_indices.iter().map(|e| e.end - e.start).collect();
        for i in (0..indices.len()).rev() {
            if (dims[i] == 1) && (dims.len() > 1) {
                dims.remove(i);
            }
        }

        Tensor::new(Some(&res), &dims).unwrap()
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

    ///Reshape the tensor
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let mut a = Tensor::<f32>::new(None, &[3, 3, 3]).unwrap();
    /// a.reshape(&[9, 3]);
    /// assert_eq!(a.dims(), &[9, 3]);
    /// ```
    pub fn reshape(&mut self, new_dims: &[usize]) {
        assert!(self.len() == new_dims.iter().product::<usize>());
        self.dims = Vec::from(new_dims);
    }

    /// Maps a function to tensors
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.map(|x| i32::pow(x,2));
    /// assert_eq!(c, Tensor::from([1, 16].into_iter()))
    /// ```
    pub fn map<F: FnMut(T) -> G, G: TensorType>(&self, mut f: F) -> Tensor<G> {
        Tensor::from(self.inner.iter().map(|e| f(e.clone())))
    }

    /// Maps a function to tensors and enumerates
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
    /// let mut c = a.enum_map(|i, x| i32::pow(x + i as i32, 2));
    /// assert_eq!(c, Tensor::from([1, 25].into_iter()));
    /// ```
    pub fn enum_map<F: FnMut(usize, T) -> G, G: TensorType>(&self, mut f: F) -> Tensor<G> {
        let mut t = Tensor::from(self.inner.iter().enumerate().map(|(i, e)| f(i, e.clone())));
        t.reshape(self.dims());
        t
    }
}

impl<T: Clone + TensorType> Tensor<Tensor<T>> {
    /// Flattens a tensor of tensors
    /// ```
    /// use halo2deeplearning::tensor::Tensor;
    /// let mut a = Tensor::<i32>::new(Some(&[1, 2, 3, 4, 5, 6]), &[2, 3]).unwrap();
    /// let mut b = Tensor::<i32>::new(Some(&[1, 4]), &[2, 1]).unwrap();
    /// let mut c = Tensor::new(Some(&[a,b]), &[2]).unwrap();
    /// let mut d = c.flatten();
    /// assert_eq!(d.dims(), &[8]);
    /// ```
    pub fn flatten(&self) -> Tensor<T> {
        let mut dims = 0;
        let mut inner = Vec::new();
        for mut t in self.inner.clone().into_iter() {
            dims += t.len();
            inner.extend(t.inner);
        }
        Tensor::new(Some(&inner), &[dims]).unwrap()
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
        let b = Tensor::<i32>::new(Some(&[1, 4]), &[2]).unwrap();
        assert_eq!(a.get_slice(&[0..2, 0..1]), b);
    }
}
