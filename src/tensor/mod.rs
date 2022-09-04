use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Error;
use std::ops::Deref;
use std::ops::DerefMut;

#[derive(Debug)]
pub struct TensorError(String);

////////////////////////
// Consider removing this when integrating, if a pain
pub trait TensorType: Default + Clone + Display + Debug + 'static {
    /// Returns the zero value.
    fn zero() -> Self;
}

macro_rules! tensor_type {
    ($rust_type:ty, $tensor_type:ident, $zero:expr, $one:expr) => {
        impl TensorType for $rust_type {
            fn zero() -> Self {
                $zero
            }

            fn one() -> Self {
                $one
            }
        }
    };
}

tensor_type!(f32, Float, 0.0, 1.0);
tensor_type!(f64, Double, 0.0, 1.0);
tensor_type!(i32, Int32, 0, 1);
tensor_type!(u8, UInt8, 0, 1);
tensor_type!(u16, UInt16, 0, 1);
tensor_type!(u32, UInt32, 0, 1);
tensor_type!(u64, UInt64, 0, 1);
tensor_type!(i16, Int16, 0, 1);
tensor_type!(i8, Int8, 0, 1);
tensor_type!(i64, Int64, 0, 1);
tensor_type!(bool, Bool, false, true);

#[derive(Clone, Debug, Eq)]
pub struct Tensor<T> {
    inner: Vec<T>,
    dims: Vec<u64>,
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
        Tensor::new(Some(&value), &[value.len() as u64]).unwrap()
    }
}

impl<T: Clone + TensorType> Tensor<T> {
    /// Sets (copies) the tensor values to the provided ones.
    /// ```
    pub fn new(values: Option<&[T]>, dims: &[u64]) -> Result<Self, TensorError> {
        let total_dims: u64 = dims.iter().product();
        match values {
            Some(val) => {
                if total_dims != val.len().try_into().unwrap() {
                    return Err(TensorError(
                        "length of values array is not equal to tensor total elements".to_string(),
                    ));
                }
                Ok(Tensor {
                    inner: Vec::from(val),
                    dims: Vec::from(dims),
                })
            }
            None => Ok(Tensor {
                inner: vec![T::zero(); total_dims.try_into().unwrap()],
                dims: Vec::from(dims),
            }),
        }
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
    pub fn set(&mut self, indices: &[u64], value: T) {
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
    pub fn get(&self, indices: &[u64]) -> T {
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
    pub fn get_index(&self, indices: &[u64]) -> usize {
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
    pub fn dims(&self) -> &[u64] {
        &self.dims
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
        let d = Tensor::<i32>::new(Some(&[1, 2, 3]), &[3, 1]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
}
