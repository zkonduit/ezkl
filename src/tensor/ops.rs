use super::TensorError;
use crate::tensor::{Tensor, TensorType};
use itertools::Itertools;
use maybe_rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    prelude::IntoParallelRefIterator,
};
use std::collections::{HashMap, HashSet};
pub use std::ops::{Add, Div, Mul, Neg, Sub};

/// IFF operation.
/// # Arguments
/// * `mask` - Tensor of 0s and 1s
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::iff;
/// let mask = Tensor::<i128>::new(
///    Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let a = Tensor::<i128>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///   Some(&[7, 8, 9, 10, 11, 12]),
/// &[2, 3],
/// ).unwrap();
/// let result = iff(&mask, &a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 8, 3, 10, 5, 12]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn iff<
    T: TensorType
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialEq,
>(
    mask: &Tensor<T>,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // assert is boolean
    if !mask
        .par_iter()
        .all(|x| *x == T::one().unwrap() || *x == T::zero().unwrap())
    {
        return Err(TensorError::WrongMethod);
    }

    let masked_a = (mask.clone() * a.clone())?;

    let masked_b = ((Tensor::from(vec![T::one().ok_or(TensorError::Unsupported)?].into_iter())
        - mask.clone())?
        * b.clone())?;

    masked_a + masked_b
}

/// Elementwise applies not to a tensor of integers.
/// # Arguments
/// * `a` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::not;
/// let x = Tensor::<i128>::new(
///    Some(&[1, 1, 1, 1, 1, 0]),
///   &[2, 3],
/// ).unwrap();
/// let result = not(&x).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 0, 0, 0, 0, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn not<
    T: TensorType
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialEq,
>(
    a: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    iff(
        a,
        &Tensor::from(vec![T::zero().unwrap()].into_iter()),
        &Tensor::from(vec![T::one().unwrap()].into_iter()),
    )
}

/// Elementwise applies or to two tensors of integers.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::or;
/// let a = Tensor::<i128>::new(
///   Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///  Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let result = or(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 1, 1, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn or<
    T: TensorType
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialEq,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    if !b
        .par_iter()
        .all(|x| *x == T::one().unwrap() || *x == T::zero().unwrap())
    {
        return Err(TensorError::WrongMethod);
    }

    iff(a, a, b)
}

/// Elementwise applies xor to two tensors
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::xor;
/// let a = Tensor::<i128>::new(
///  Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
/// Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let result = xor(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 1, 0, 1, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
///
pub fn xor<
    T: TensorType
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialEq,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    let a_not_b = (a.clone() * not(b)?)?;
    let b_not_a = (b.clone() * not(a)?)?;
    a_not_b + b_not_a
}

/// Elementwise applies and to two tensors
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::and;
/// let a = Tensor::<i128>::new(
///  Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
/// Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let result = and(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 0, 1, 0, 1, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn and<
    T: TensorType
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialEq,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // assert is boolean
    if !b
        .par_iter()
        .all(|x| *x == T::one().unwrap() || *x == T::zero().unwrap())
    {
        return Err(TensorError::WrongMethod);
    }

    // assert is boolean
    if !a
        .par_iter()
        .all(|x| *x == T::one().unwrap() || *x == T::zero().unwrap())
    {
        return Err(TensorError::WrongMethod);
    }

    a.clone() * b.clone()
}

/// Elementwise applies equals to two tensors of integers.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::equals;
/// let a = Tensor::<i128>::new(
/// Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
/// Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let result = equals(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 0, 1, 0, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn equals<
    T: TensorType
        + std::marker::Send
        + std::marker::Sync
        + Sub<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
        + std::cmp::PartialEq
        + std::cmp::PartialOrd
        + std::convert::From<u64>,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    let a = a.clone();
    let b = b.clone();

    let diff = (a - b)?;

    let result = nonlinearities::kronecker_delta(&diff);

    Ok(result)
}

/// Greater than operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::greater;
/// let a = Tensor::<i128>::new(
///   Some(&[1, 12, 6, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///  Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = greater(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 1, 1, 0, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn greater<
    T: TensorType
        + Sub<Output = T>
        + Mul<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialOrd
        + std::convert::TryFrom<u64>,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    let mask_inter = (a.clone() - b.clone())?;
    let mask = mask_inter.map(|x| {
        if x > T::zero().ok_or(TensorError::Unsupported).unwrap() {
            T::one().ok_or(TensorError::Unsupported).unwrap()
        } else {
            T::zero().ok_or(TensorError::Unsupported).unwrap()
        }
    });
    Ok(mask)
}

/// Greater equals than operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::greater_equal;
/// let a = Tensor::<i128>::new(
///   Some(&[1, 12, 6, 4, 3, 2]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///  Some(&[1, 2, 3, 4, 5, 4]),
/// &[2, 3],
/// ).unwrap();
/// let result = greater_equal(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 1, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn greater_equal<
    T: TensorType
        + Sub<Output = T>
        + Mul<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialOrd
        + std::convert::TryFrom<u64>,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    let mask_inter = (a.clone() - b.clone())?;
    let mask = mask_inter.map(|x| {
        if x >= T::zero().ok_or(TensorError::Unsupported).unwrap() {
            T::one().ok_or(TensorError::Unsupported).unwrap()
        } else {
            T::zero().ok_or(TensorError::Unsupported).unwrap()
        }
    });
    Ok(mask)
}

/// Less than to operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::less;
/// let a = Tensor::<i128>::new(
///  Some(&[1, 0, 5, 4, 5, 1]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
/// Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = less(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 1, 0, 0, 0, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
///
pub fn less<
    T: TensorType
        + Sub<Output = T>
        + Mul<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialOrd
        + std::convert::TryFrom<u64>,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // a < b <=> b > a
    greater(b, a)
}

/// Less equals than operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::less_equal;
/// let a = Tensor::<i128>::new(
///  Some(&[1, 0, 5, 4, 5, 1]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
/// Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = less_equal(&a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 0, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
///
pub fn less_equal<
    T: TensorType
        + Sub<Output = T>
        + Mul<Output = T>
        + std::marker::Send
        + std::marker::Sync
        + std::cmp::PartialOrd
        + std::convert::TryFrom<u64>,
>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // a < b <=> b > a
    greater_equal(b, a)
}

/// Resize using nearest neighbour interpolation.
/// # Arguments
/// * `a` - Tensor
/// * `scales` - Vector of scales
/// # Examples
/// ```
///
///
/// let a = Tensor::<i128>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = resize(&a, &[1, 2]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]), &[2, 6]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let a = Tensor::<i128>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = resize(&a, &[2, 2]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6]), &[4, 6]).unwrap();
/// assert_eq!(result, expected);
///
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::resize;
/// let a = Tensor::<i128>::new(
///   Some(&[1, 2, 3, 4]),
/// &[2, 2],
/// ).unwrap();
/// let result = resize(&a, &[2, 2]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]), &[4, 4]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let a = Tensor::<i128>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[3, 2],
/// ).unwrap();
/// let result = resize(&a, &[2, 3]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6]), &[6, 6]).unwrap();
/// assert_eq!(result, expected);
///
///
/// ```
pub fn resize<T: TensorType + Send + Sync>(
    a: &Tensor<T>,
    scales: &[usize],
) -> Result<Tensor<T>, TensorError> {
    let mut new_shape = vec![];
    for (s, d) in scales.iter().zip(a.dims()) {
        new_shape.push(s * d);
    }

    let mut output = Tensor::new(None, &new_shape)?;

    let cartesian_coord: Vec<Vec<usize>> = new_shape
        .iter()
        .map(|d| (0..*d))
        .multi_cartesian_product()
        .collect();

    // resize using nearest neighbour interpolation
    // (i.e. just copy the value of the nearest neighbour to pad the tensor)
    output = output.par_enum_map(|i, _| {
        let mut coord = vec![];
        for (j, (c, _d)) in cartesian_coord[i].iter().zip(new_shape.iter()).enumerate() {
            let scale = scales[j];
            let fragment = c / scale;
            coord.push(fragment);
        }

        Ok::<_, TensorError>(a.get(&coord))
    })?;

    Ok(output)
}

/// Computes the einstein sum of a set of tensors.
/// # Arguments
/// * `equation` - Einstein summation equation
/// * `inputs` - Vector of tensors
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::einsum;
///
/// // matmul case
/// let x = Tensor::<i128>::new(
///    Some(&[2, 1, 2, 1, 1, 1]),
///  &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///   Some(&[2, 3, 2, 1, 1, 1]),
/// &[3, 2],
/// ).unwrap();
/// let result = einsum("ij,jk->ik", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[8, 9, 5, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// // element wise multiplication
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
///  &[3, 3],
/// ).unwrap();
/// let result = einsum("ij,ij->ij", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 4, 9, 2, 6, 12, 3, 8, 15]), &[3, 3]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // dot product of A with the transpose of B.
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
///  &[3, 3],
/// ).unwrap();
/// let result = einsum("ik,jk->ij", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[14, 14, 14, 20, 20, 20, 26, 26, 26]), &[3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// // dot product
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
///  &[3, 3],
/// ).unwrap();
/// let result = einsum("ik,ik->i", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[14, 20, 26]), &[3]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // dot product
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3]),
///  &[3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[1, 2, 3]),
///  &[3],
/// ).unwrap();
/// let result = einsum("i,i->", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[14]), &[1]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // wut ?
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap();
/// let result = einsum("anm,bm->ba", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[68, 80, 95, 113, 134, 158]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// // wutttttt ?
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap();
/// let z = Tensor::<i128>::new(
///    Some(&[4, 5, 7, 8, 9, 9]),
///  &[2, 3],
/// ).unwrap();
///
/// let result = einsum("bn,anm,bm->ba", &[z, x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[390, 414, 534, 994, 1153, 1384]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // contraction with a single common axis
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap();
/// let result = einsum("abc,cd->", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[648]), &[1]).unwrap();
/// assert_eq!(result, expected);
///
/// // contraction with no common axes (outer product)
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap();
/// let result = einsum("abc,ed->", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1296]), &[1]).unwrap();
/// assert_eq!(result, expected);
///
/// // trivial axes mapping
/// let x = Tensor::<i128>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[4, 5]),
///  &[2],
/// ).unwrap();
///
/// let result = einsum("mk,k->m", &[x.clone(), k.clone()]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[41, 68]), &[2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = einsum("mk,k->mn", &[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[41, 68]), &[2, 1]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(
///    Some(&[0, 0, 0, 3]),
///  &[1, 4],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///    Some(&[213, 227, 74, 77]),
///  &[4],
/// ).unwrap();
///
/// let result = einsum("mk,k->ma", &[x.clone(), k.clone()]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[231]), &[1, 1]).unwrap();
/// assert_eq!(result, expected);
/// // subtle difference
/// let result = einsum("mk,n->ma", &[x.clone(), k.clone()]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1773]), &[1, 1]).unwrap();
/// assert_eq!(result, expected);
///
////// ```
pub fn einsum<
    T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Send + std::marker::Sync,
>(
    equation: &str,
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    // Parse equation into an operation

    let mut equation = equation.split("->");
    let inputs_eq = equation.next().unwrap();
    let output_eq = equation.next().unwrap();
    let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();

    // Check that the number of inputs matches the number of inputs in the equation
    if inputs.len() != inputs_eq.len() {
        return Err(TensorError::DimMismatch("einsum".to_string()));
    }

    let mut indices_to_size = HashMap::new();
    for (i, input) in inputs.iter().enumerate() {
        for j in 0..inputs_eq[i].len() {
            let c = inputs_eq[i].chars().nth(j).unwrap();
            if let std::collections::hash_map::Entry::Vacant(e) = indices_to_size.entry(c) {
                e.insert(input.dims()[j]);
            } else if indices_to_size[&c] != input.dims()[j] {
                return Err(TensorError::DimMismatch("einsum".to_string()));
            }
        }
    }

    // maps unrepresented indices in the output to a trivial 1
    for c in output_eq.chars() {
        indices_to_size.entry(c).or_insert(1);
    }

    // Compute the output tensor shape
    let mut output_shape: Vec<usize> = output_eq
        .chars()
        .map(|c| *indices_to_size.get(&c).unwrap())
        .collect();

    if output_shape.is_empty() {
        output_shape.push(1);
    }

    let mut seen = HashSet::new();
    let mut common_indices_to_inputs = vec![];
    for input in &inputs_eq {
        for c in input.chars() {
            if !seen.contains(&c) {
                seen.insert(c);
            } else {
                common_indices_to_inputs.push(c);
            }
        }
    }

    let cartesian_coord = output_shape
        .iter()
        .map(|d| 0..*d)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    // Compute the cartesian product of all indices
    let output: Vec<T> = cartesian_coord
        .par_iter()
        .map(|coord| {
            // Compute the slice of each input tensor given the current coordinate of the output tensor
            let inputs = (0..inputs.len())
                .map(|idx| {
                    let mut slice = vec![];
                    for (i, c) in inputs_eq[idx].chars().enumerate() {
                        // If the current index is in the output equation, then the slice should be the current coordinate
                        if let Some(idx) = output_eq.find(c) {
                            slice.push(coord[idx]..coord[idx] + 1);
                        // Otherwise, the slice should be the entire dimension of the input tensor
                        } else {
                            slice.push(0..inputs[idx].dims()[i]);
                        }
                    }

                    // Get the slice of the input tensor
                    inputs[idx].get_slice(&slice).unwrap()
                })
                .collect::<Vec<_>>();

            // Get the indices common across input tensors
            let mut common_coord = common_indices_to_inputs
                .iter()
                .map(|d| {
                    // If the current index is in the output equation, then the slice should be the current coordinate
                    if output_eq.contains(*d) {
                        0..1
                    // Otherwise, the slice should be the entire dimension of the input tensor
                    } else {
                        0..*indices_to_size.get(d).unwrap()
                    }
                })
                .multi_cartesian_product()
                .collect::<Vec<_>>();

            // If there are no common indices, then we need to add an empty slice to force one iteration of the loop
            if common_coord.is_empty() {
                common_coord.push(vec![]);
            }

            let mut prod = T::zero().unwrap();

            // Compute the cartesian product of all common indices
            for common_dim in common_coord {
                let inputs = (0..inputs.len())
                    .map(|idx| {
                        let mut slice = vec![];
                        // Iterate over all indices in the input equation
                        for (i, c) in inputs_eq[idx].chars().enumerate() {
                            // If the current index is common to multiple inputs, then the slice should be the current coordinate
                            if let Some(j) = common_indices_to_inputs.iter().position(|&r| r == c) {
                                slice.push(common_dim[j]..common_dim[j] + 1);
                            } else {
                                slice.push(0..inputs[idx].dims()[i]);
                            }
                        }
                        // Get the slice of the input tensor
                        inputs[idx].get_slice(&slice).unwrap()
                    })
                    .collect::<Vec<_>>();

                let input_pairs = inputs
                    .iter()
                    .map(|d| d.iter())
                    .multi_cartesian_product()
                    .collect::<Vec<_>>();

                // Compute the product of all input tensors
                for pair in input_pairs {
                    prod = prod
                        + pair
                            .into_iter()
                            .fold(T::one().unwrap(), |acc, x| acc * x.clone());
                }
            }
            prod
        })
        .collect();

    let mut output: Tensor<T> = output.into_iter().into();
    output.reshape(&output_shape)?;

    Ok(output)
}

/// Adds multiple tensors.
/// # Arguments
///
/// * `t` - Vector of tensors
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::add;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = add(&[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 4, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test 1D casting
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2]),
///     &[1]).unwrap();
/// let result = add(&[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 3, 4, 3, 3, 3]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn add<T: TensorType + Add<Output = T> + std::marker::Send + std::marker::Sync>(
    t: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        output = output.add(e.clone())?;
    }

    Ok(output)
}

/// Subtracts multiple tensors.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::sub;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = sub(&[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, -2, 0, 0, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test 1D sub
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2]),
///     &[1],
/// ).unwrap();
/// let result = sub(&[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, -1, 0, -1, -1, -1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn sub<T: TensorType + Sub<Output = T> + std::marker::Send + std::marker::Sync>(
    t: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        output = (output - e.clone())?;
    }

    Ok(output)
}

/// Negates a tensor.
/// # Arguments
///
/// * `a` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::neg;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = neg(&x).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[-2, -1, -2, -1, -1, -1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn neg<T: TensorType + Neg<Output = T> + std::marker::Send + std::marker::Sync>(
    t: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    Ok(-t.clone())
}

/// Elementwise multiplies multiple tensors.
/// # Arguments
///
/// * `t` - Tensors
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::mult;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = mult(&[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 3, 4, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test 1D mult
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2]),
///     &[1]).unwrap();
/// let result = mult(&[x, k]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn mult<T: TensorType + Mul<Output = T> + std::marker::Send + std::marker::Sync>(
    t: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        output = (output * e.clone())?;
    }

    Ok(output)
}

/// Rescale a tensor with a const integer (similar to const_mult).
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::rescale;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = rescale(&x, k).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn rescale<T: TensorType + Add<Output = T> + std::marker::Send + std::marker::Sync>(
    a: &Tensor<T>,
    mult: u128,
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();
    output.par_iter_mut().enumerate().for_each(|(i, a_i)| {
        for _ in 1..mult {
            *a_i = a_i.clone() + a[i].clone();
        }
    });
    Ok(output)
}

/// Sums a tensor.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::sum;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = sum(&x).unwrap();
/// let expected = 21;
/// assert_eq!(result[0], expected);
/// ```
pub fn sum<T: TensorType + Add<Output = T>>(a: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut res = T::zero().unwrap();

    let _ = a.map(|a_i| res = res.clone() + a_i);
    Tensor::new(Some(&[res]), &[1])
}

/// Takes prod of tensor's elements.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::prod;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = prod(&x).unwrap();
/// let expected = 0;
/// assert_eq!(result[0], expected);
/// ```
pub fn prod<T: TensorType + Mul<Output = T>>(a: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut res = T::one().unwrap();

    let _ = a.map(|a_i| res = res.clone() * a_i);
    Tensor::new(Some(&[res]), &[1])
}

/// Downsamples a tensor along a dimension.
/// # Arguments
/// * `input` - Tensor
/// * `dim` - Dimension to downsample along
/// * `stride` - Stride to downsample by
/// *  `modulo` - Modulo to downsample by
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::downsample;
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 4, 5, 6]),
///  &[2, 3],
/// ).unwrap();
/// let result = downsample(&x, 0, 1, 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 5, 6]), &[1, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = downsample(&x, 1, 2, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 3, 4, 6]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = downsample(&x, 1, 2, 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 5]), &[2, 1]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = downsample(&x, 1, 2, 2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[3, 6]), &[2, 1]).unwrap();
/// assert_eq!(result, expected);
pub fn downsample<T: TensorType + Send + Sync>(
    input: &Tensor<T>,
    dim: usize,
    stride: usize,
    modulo: usize,
) -> Result<Tensor<T>, TensorError> {
    let mut output_shape = input.dims().to_vec();
    // now downsample along axis dim offset by modulo, rounding up (+1 if remaidner is non-zero)
    let remainder = (input.dims()[dim] - modulo) % stride;
    let div = (input.dims()[dim] - modulo) / stride;
    output_shape[dim] = div + (remainder > 0) as usize;
    let mut output = Tensor::<T>::new(None, &output_shape)?;

    if modulo > input.dims()[dim] {
        return Err(TensorError::DimMismatch("downsample".to_string()));
    }

    // now downsample along axis dim offset by modulo
    let indices = (0..output_shape.len())
        .map(|i| {
            if i == dim {
                let mut index = vec![0; output_shape[i]];
                for (i, idx) in index.iter_mut().enumerate() {
                    *idx = i * stride + modulo;
                }
                index
            } else {
                (0..output_shape[i]).collect_vec()
            }
        })
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output = output.par_enum_map(|i, _: T| {
        let coord = indices[i].clone();
        Ok(input.get(&coord))
    })?;

    Ok(output)
}

/// Gathers a tensor along a dimension.
/// # Arguments
/// * `input` - Tensor
/// * `dim` - Dimension to gather along
/// * `index` - Tensor of indices to gather
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::gather;
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 4, 5, 6]),
///   &[2, 3],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///   Some(&[0, 1]),
///  &[2],
/// ).unwrap();
/// let result = gather(&x, &index, 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn gather<T: TensorType + Send + Sync>(
    input: &Tensor<T>,
    index: &Tensor<usize>,
    dim: usize,
) -> Result<Tensor<T>, TensorError> {
    let mut index_clone = index.clone();
    index_clone.flatten();
    if index_clone.is_singleton() {
        index_clone.reshape(&[1])?;
    }

    // Calculate the output tensor size
    let mut output_size = input.dims().to_vec();
    output_size[dim] = index_clone.dims()[0];

    // Allocate memory for the output tensor
    let mut output = Tensor::new(None, &output_size)?;
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output = output.par_enum_map(|i, _: T| {
        let coord = cartesian_coord[i].clone();
        let index_val = index_clone.get(&[coord[dim]]);
        let new_coord = coord
            .iter()
            .enumerate()
            .map(|(i, x)| if i == dim { index_val } else { *x })
            .collect::<Vec<_>>();

        Ok(input.get(&new_coord))
    })?;

    // Reshape the output tensor
    if index.is_singleton() {
        output_size.remove(dim);
    }

    output.reshape(&output_size)?;

    Ok(output)
}

/// Scatters a tensor along a dimension.
/// # Arguments
/// * `input` - Tensor
/// * `dim` - Dimension to scatter along
/// * `index` - Tensor of indices to scatter
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::scatter;
/// let x = Tensor::<f64>::new(
///   Some(&[1.0, 2.0, 3.0, 4.0]),
/// &[2, 2],
/// ).unwrap();
/// let src = Tensor::<f64>::new(
///  Some(&[5.0, 6.0, 7.0, 8.0]),
/// &[2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///  Some(&[0, 0, 1, 0]),
/// &[2, 2],
/// ).unwrap();
/// let result = scatter(&x, &index, &src, 0).unwrap();
/// let expected = Tensor::<f64>::new(Some(&[5.0, 8.0, 7.0, 4.0]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn scatter<T: TensorType + Send + Sync>(
    input: &Tensor<T>,
    index: &Tensor<usize>,
    src: &Tensor<T>,
    dim: usize,
) -> Result<Tensor<T>, TensorError> {
    // Writes all values from the tensor src into self at the indices specified in the index tensor.
    // For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
    assert_eq!(index.dims(), src.dims());
    // Calculate the output tensor size
    let src_size = src.dims().to_vec();

    // For a 3-D tensor, self is updated as:
    // self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
    // self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
    // self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    let mut output = input.clone();

    let cartesian_coord = src_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    cartesian_coord.iter().for_each(|coord| {
        let mut new_coord = coord.clone();
        let index_val = index.get(coord);
        new_coord[dim] = index_val;
        let val = src.get(coord);
        output.set(&new_coord, val);
    });

    Ok(output)
}

/// Gathers a tensor along a dimension.
/// # Arguments
/// * `input` - Tensor
/// * `dim` - Dimension to gather along
/// * `index` - Tensor of indices to gather
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::gather_elements;
/// let x = Tensor::<i128>::new(
///    Some(&[1, 2, 3, 4]),
///   &[2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///   Some(&[0, 0, 1, 0]),
///  &[2, 2],
/// ).unwrap();
/// let result = gather_elements(&x, &index, 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 1, 4, 3]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn gather_elements<T: TensorType + Send + Sync>(
    input: &Tensor<T>,
    index: &Tensor<usize>,
    dim: usize,
) -> Result<Tensor<T>, TensorError> {
    // Calculate the output tensor size
    let output_size = index.dims().to_vec();
    // same rank
    assert_eq!(input.dims().len(), index.dims().len());

    // Allocate memory for the output tensor
    let mut output = Tensor::new(None, &output_size)?;
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output = output.par_enum_map(|i, _: T| {
        let coord = cartesian_coord[i].clone();
        let index_val = index.get(&coord);

        let mut new_coord = coord.clone();
        new_coord[dim] = index_val;

        let val = input.get(&new_coord);

        Ok(val)
    })?;

    // Reshape the output tensor
    output.reshape(&output_size)?;

    Ok(output)
}

/// Gather ND.
/// # Arguments
/// * `input` - Tensor
/// * `index` - Tensor of indices to gather
/// * `batch_dims` - Number of batch dimensions
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::gather_nd;
/// let x = Tensor::<i128>::new(
///   Some(&[0, 1, 2, 3]),
/// &[2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
/// Some(&[0, 0, 1, 1]),
/// &[2, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 3]), &[2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
/// Some(&[1, 0]),
/// &[2, 1],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 3, 0, 1]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(
///  Some(&[0, 1, 2, 3, 4, 5, 6, 7]),
/// &[2, 2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 1, 0]),
/// &[2, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 3, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 1, 0]),
/// &[2, 1, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 3, 4, 5]), &[2, 1, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
/// Some(&[1, 0]),
/// &[2, 1],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 3, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]),
/// &[2, 2, 3],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 3, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
pub fn gather_nd<T: TensorType + Send + Sync>(
    input: &Tensor<T>,
    index: &Tensor<usize>,
    batch_dims: usize,
) -> Result<Tensor<T>, TensorError> {
    // Calculate the output tensor size
    let index_dims = index.dims().to_vec();
    let input_dims = input.dims().to_vec();

    println!("----------- {:?} {:?}", input.dims(), index.dims());

    let output_size = 
    // If indices_shape[-1] == r-b, since the rank of indices is q,
    // indices can be thought of as N (q-b-1)-dimensional tensors containing 1-D tensors of dimension r-b,
    // where N is an integer equals to the product of 1 and all the elements in the batch dimensions of the indices_shape.
    // Let us think of each such r-b ranked tensor as indices_slice.
    // Each scalar value corresponding to data[0:b-1,indices_slice] is filled into
    // the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
    if index_dims.last() == Some(&(input_dims.len() - batch_dims)) {
        input_dims[..input_dims.len() - 1].to_vec()
    // if indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor containing 1-D tensors of dimension < r-b.
    // Let us think of each such tensors as indices_slice.
    // Each tensor slice corresponding to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
    } else if index_dims.last() < Some(&(input_dims.len() - batch_dims)) {
        let last_value = index_dims.last().unwrap();
        let output_rank = input_dims.len() + index_dims.len() - 1 - batch_dims - last_value;

        let mut dims = index_dims[..index_dims.len() - 1].to_vec();
        let input_offset = batch_dims + last_value;
        dims.extend(input_dims[input_offset..input_dims.len()].to_vec());

        println!("{:?} {:?}", dims, output_rank);

        assert_eq!(output_rank, dims.len());
        dims

    } else {
        return Err(TensorError::DimMismatch("gather_nd".to_string()));
    };

    // cartesian coord over batch dims
    let mut batch_cartesian_coord = input_dims[0..batch_dims]
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    if batch_cartesian_coord.is_empty() {
        batch_cartesian_coord.push(vec![]);
    }

    let outputs = batch_cartesian_coord
        .par_iter()
        .map(|batch_coord| {
            let batch_slice = batch_coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let mut index_slice = index.get_slice(&batch_slice)?;
            index_slice.reshape(&index.dims()[batch_dims..])?;
            let mut input_slice = input.get_slice(&batch_slice)?;
            input_slice.reshape(&input.dims()[batch_dims..])?;

            println!("{:?}", index_slice.dims());
            println!("{:?}", input_slice.dims());

            let mut inner_cartesian_coord = index_slice.dims()[0..index_slice.dims().len() - 1]
                .iter()
                .map(|x| 0..*x)
                .multi_cartesian_product()
                .collect::<Vec<_>>();

            if inner_cartesian_coord.is_empty() {
                inner_cartesian_coord.push(vec![]);
            }

            let output = inner_cartesian_coord
                .iter()
                .map(|coord| {
                    println!("inner coord {:?}", coord);
                    let slice = coord
                        .iter()
                        .map(|x| *x..*x + 1)
                        .chain(batch_coord.iter().map(|x| *x..*x + 1))
                        .collect::<Vec<_>>();

                    println!("input slice {:?}", input_slice.dims());

                    let index_slice = index_slice
                        .get_slice(&slice)
                        .unwrap()
                        .iter()
                        .map(|x| *x..*x + 1)
                        .collect::<Vec<_>>();

                    println!("index slice {:?}", index_slice);

                    input_slice.get_slice(&index_slice).unwrap()
                })
                .collect::<Tensor<_>>();

            println!("output {:?}", output.dims());

            Ok(output.combine()?)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut outputs = outputs.into_iter().flatten().collect::<Tensor<_>>();

    println!("outputs {}", outputs.show());

    outputs.reshape(&output_size)?;

    Ok(outputs)
}

fn axes_op<T: TensorType + Send + Sync>(
    a: &Tensor<T>,
    axes: &[usize],
    op: impl Fn(&Tensor<T>) -> Result<Tensor<T>, TensorError> + Send + Sync,
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output

    if axes.is_empty() {
        return Ok(a.clone());
    }

    let mut new_dims = vec![];
    for i in 0..a.dims().len() {
        if !axes.contains(&i) {
            new_dims.push(a.dims()[i]);
        } else {
            new_dims.push(1);
        }
    }

    let res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let res = res.par_enum_map(|i, _: T| {
        let coord = cartesian_coord[i].clone();
        let mut prod_dims = vec![];
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                prod_dims.push(0..a.dims()[i]);
            } else {
                prod_dims.push(*c..*c + 1);
            }
        }

        Ok(op(&a.get_slice(&prod_dims)?)?[0].clone())
    })?;

    Ok(res)
}

/// Takes product of a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::prod_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = prod_axes(&x, &[1]).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[60, 0]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn prod_axes<T: TensorType + Mul<Output = T> + Send + Sync>(
    a: &Tensor<T>,
    axes: &[usize],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    axes_op(a, axes, prod)
}

/// Returns top K values.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::topk;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[6],
/// ).unwrap();
/// let result = topk(&x, 3, true).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[15, 2, 2]),
///     &[3],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn topk<T: TensorType + PartialOrd>(
    a: &Tensor<T>,
    k: usize,
    largest: bool,
) -> Result<Tensor<T>, TensorError> {
    let mut indexed_a = a.clone();
    indexed_a.flatten();

    let mut indexed_a = a
        .iter()
        .enumerate()
        .map(|(i, x)| (i, x))
        .collect::<Vec<_>>();

    if largest {
        indexed_a.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
    } else {
        indexed_a.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    }

    let indexed_a = indexed_a
        .into_iter()
        .take(k)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();

    let mut output = Tensor::new(None, &[k])?;

    for (i, x) in indexed_a.iter().enumerate() {
        output.set(&[i], a[*x].clone());
    }

    Ok(output)
}

/// Returns top K values.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// * `dim` - Dimension to topk along
/// * `largest` - Whether to return the largest or largest values
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::topk_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2,3],
/// ).unwrap();
/// let result = topk_axes(&x, 2, 1, true).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[15, 2, 1, 1]),
///     &[2,2],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn topk_axes<T: TensorType + PartialOrd + Send + Sync>(
    a: &Tensor<T>,
    k: usize,
    dim: usize,
    largest: bool,
) -> Result<Tensor<T>, TensorError> {
    let mut new_dims = a.dims().to_vec();
    new_dims[dim] = k;

    let res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let res = res.par_enum_map(|i, _: T| {
        let coord = cartesian_coord[i].clone();
        let mut slice = vec![];
        for (i, c) in coord.iter().enumerate() {
            if i == dim {
                slice.push(0..a.dims()[i]);
            } else {
                slice.push(*c..*c + 1);
            }
        }
        let sliced_value = a.get_slice(&slice)?;
        let topk = topk(&sliced_value, k, largest)?;
        Ok(topk[coord[dim]].clone())
    })?;

    Ok(res)
}

/// Sums a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::sum_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = sum_axes(&x, &[1]).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[19, 2]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn sum_axes<T: TensorType + Add<Output = T> + Send + Sync>(
    a: &Tensor<T>,
    axes: &[usize],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    axes_op(a, axes, sum)
}

/// Mins a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::min_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = min_axes(&x, &[1]).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[2, 0]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn min_axes<T: TensorType + Add<Output = T> + std::cmp::Ord + Send + Sync>(
    a: &Tensor<T>,
    axes: &[usize],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output

    let min_fn = |a: &Tensor<T>| -> Result<Tensor<T>, TensorError> {
        Ok(vec![a.par_iter().min().unwrap().clone()].into_iter().into())
    };

    axes_op(a, axes, min_fn)
}

/// Abs a tensor.
/// # Arguments
/// * `a` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::abs;
/// let x = Tensor::<i128>::new(
///    Some(&[-2, 15, 2, -1, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let result = abs(&x).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 15, 2, 1, 1, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn abs<T: TensorType + Add<Output = T> + std::cmp::Ord + Neg<Output = T>>(
    a: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();
    output.iter_mut().for_each(|a_i| {
        if *a_i < T::zero().unwrap() {
            *a_i = -a_i.clone();
        }
    });
    Ok(output)
}

/// Max of a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::max_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = max_axes(&x, &[1]).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[15, 1]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn max_axes<T: TensorType + Add<Output = T> + std::cmp::Ord + Send + Sync>(
    a: &Tensor<T>,
    axes: &[usize],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output

    let max_fn = |a: &Tensor<T>| -> Result<Tensor<T>, TensorError> {
        Ok(vec![a.par_iter().max().unwrap().clone()].into_iter().into())
    };

    axes_op(a, axes, max_fn)
}

/// Argmax of a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::argmax_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = argmax_axes(&x, 1).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[1, 0]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn argmax_axes<T: TensorType + Add<Output = T> + std::cmp::Ord + From<u64> + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
) -> Result<Tensor<T>, TensorError> {
    let argmax_fn = |a: &Tensor<T>| -> Result<Tensor<T>, TensorError> {
        Ok(vec![a
            .clone()
            .into_iter()
            .enumerate()
            // we value the first index in the case of a tie
            .max_by_key(|(idx, value)| (value.clone(), -(*idx as i64)))
            .map(|(idx, _)| T::from(idx as u64))
            .unwrap()]
        .into_iter()
        .into())
    };

    // calculate value of output
    axes_op(a, &[dim], argmax_fn)
}

/// Argmin of a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::argmin_axes;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 15, 0, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = argmin_axes(&x, 0).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[1, 1, 0]),
///     &[1, 3],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn argmin_axes<T: TensorType + Add<Output = T> + std::cmp::Ord + From<u64> + Send + Sync>(
    a: &Tensor<T>,
    dim: usize,
) -> Result<Tensor<T>, TensorError> {
    let argmax_fn = |a: &Tensor<T>| -> Result<Tensor<T>, TensorError> {
        Ok(vec![a
            .clone()
            .into_iter()
            .enumerate()
            // we value the first index in the case of a tie
            .min_by_key(|(idx, value)| (value.clone(), (*idx as i64)))
            .map(|(idx, _)| T::from(idx as u64))
            .unwrap()]
        .into_iter()
        .into())
    };

    // calculate value of output
    axes_op(a, &[dim], argmax_fn)
}

/// Applies convolution over a 3D tensor of shape C x H x W (and adds a bias).
/// # Arguments
///
/// * `inputs` - A vector of tensors holding in order: input image, convolution kernel, convolution bias.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// # Examples
/// ```
/// // expected ouputs are taken from pytorch torch.nn.functional.conv2d
///
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::conv;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[5, 1, 1, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[0]),
///     &[1],
/// ).unwrap();
/// let result = conv::<i128>(&[x, k, b], [(0, 0); 2], (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[31, 16, 8, 26]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test single channel
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 2, 3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[5, 1, 1, 1, 5, 2, 1, 1]),
///     &[2, 1, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[1, 1]),
///     &[2],
/// ).unwrap();
///
/// let result = conv::<i128>(&[x, k, b], [(0, 0); 2], (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[32, 17, 9, 27, 34, 20, 13, 26]), &[1, 2, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test multi channel
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 2, 3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[5, 1, 1, 1, 5, 2, 1, 1, 5, 3, 1, 1, 5, 4, 1, 1, 5, 1, 1, 1, 5, 2, 1, 1, 5, 3, 1, 1, 5, 4, 1, 1]),
///     &[4, 2, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[1, 1, 1, 1]),
///     &[4],
/// ).unwrap();
///
/// let result = conv::<i128>(&[x, k, b], [(0, 0); 2], (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[65, 36, 21, 52, 73, 48, 37, 48, 65, 36, 21, 52, 73, 48, 37, 48]), &[1, 4, 2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn conv<
    T: TensorType
        + Mul<Output = T>
        + Add<Output = T>
        + std::marker::Sync
        + std::marker::Send
        + std::iter::Sum,
>(
    inputs: &[Tensor<T>],
    padding: [(usize, usize); 2],
    stride: (usize, usize),
) -> Result<Tensor<T>, TensorError> {
    let has_bias = inputs.len() == 3;
    let (image, kernel) = (&mut inputs[0].clone(), &mut inputs[1].clone());

    let og_image_dims = image.dims().to_vec();
    let og_kernel_dims = kernel.dims().to_vec();
    // ensure inputs are 4D tensors
    if og_image_dims.len() == 3 {
        // adds a dummy image_channels dimension
        let mut new_dims = image.dims().to_vec();
        // insert 1 at the input_channels pos
        if og_kernel_dims.len() == 3 {
            new_dims.insert(1, 1);
        } else {
            new_dims.insert(0, 1);
        }
        image.reshape(&new_dims)?;
    }

    // ensure kernel is 4D tensor
    if og_kernel_dims.len() == 3 && og_image_dims.len() == 3 {
        // adds a dummy image_channels dimension
        let mut new_dims = kernel.dims().to_vec();
        // insert 1 at the input_channels pos
        new_dims.insert(1, 1);
        kernel.reshape(&new_dims)?;
    }

    if (image.dims().len() != 4)
        || (kernel.dims().len() != 4)
        // ensure number of groups makes sense
        || (image.dims()[1] % kernel.dims()[1] != 0)
    {
        return Err(TensorError::DimMismatch("conv".to_string()));
    }

    let image_dims = image.dims();
    let kernel_dims = kernel.dims();

    if has_bias {
        let bias = &mut inputs[2].clone();

        if bias.dims().is_empty() {
            bias.reshape(&[1])?;
        }

        if (bias.dims().len() != 1) && (bias.dims()[0] != kernel.dims()[0]) {
            return Err(TensorError::DimMismatch("conv bias".to_string()));
        }
    }

    let (batch_size, output_channels, input_channels, kernel_height, kernel_width) = (
        image_dims[0],
        kernel_dims[0],
        image_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[2], image_dims[3]);

    let padded_image = pad::<T>(image, padding)?;

    let vert_slides = (image_height + padding[0].0 + padding[1].0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + padding[0].1 + padding[1].1 - kernel_width) / stride.1 + 1;

    let num_groups = input_channels / kernel_dims[1];
    let input_channels_per_group = input_channels / num_groups;
    let output_channels_per_group = output_channels / num_groups;

    if output_channels_per_group == 0 {
        return Err(TensorError::DimMismatch(format!(
            "Given groups={}, expected kernel to be at least {} at dimension 0 but got {} instead",
            num_groups, num_groups, output_channels_per_group
        )));
    }

    let num_outputs =
        batch_size * num_groups * output_channels_per_group * vert_slides * horz_slides;

    let mut output = Tensor::new(None, &[num_outputs])?;

    let cartesian_coord = [
        (0..batch_size),
        (0..num_groups),
        (0..output_channels_per_group),
        (0..vert_slides),
        (0..horz_slides),
    ]
    .iter()
    .cloned()
    .multi_cartesian_product()
    .collect::<Vec<_>>();

    output.par_iter_mut().enumerate().for_each(|(i, o)| {
        let cartesian_coord_per_group = &cartesian_coord[i];
        let (batch, group, i, j, k) = (
            cartesian_coord_per_group[0],
            cartesian_coord_per_group[1],
            cartesian_coord_per_group[2],
            cartesian_coord_per_group[3],
            cartesian_coord_per_group[4],
        );
        let rs = j * stride.0;
        let cs = k * stride.1;

        let start_channel = group * input_channels_per_group;
        let end_channel = start_channel + input_channels_per_group;

        let local_image = padded_image
            .get_slice(&[
                batch..batch + 1,
                start_channel..end_channel,
                rs..(rs + kernel_height),
                cs..(cs + kernel_width),
            ])
            .unwrap();

        let start_kernel_index = group * output_channels_per_group + i;
        let end_kernel_index = start_kernel_index + 1;
        let local_kernel = kernel
            .get_slice(&[start_kernel_index..end_kernel_index])
            .unwrap();

        let res = dot(&[local_image, local_kernel]).unwrap()[0].clone();
        if has_bias {
            let bias_index = if inputs[2].len() > 1 {
                start_kernel_index
            } else {
                0
            };
            *o = res + inputs[2][bias_index].clone();
        } else {
            *o = res;
        }
    });

    // remove dummy batch dimension if we added one
    if og_image_dims.len() == 3 && vert_slides == 1 {
        output.reshape(&[batch_size, output_channels, horz_slides])?;
    } else if og_image_dims.len() == 3 {
        output.reshape(&[output_channels, vert_slides, horz_slides])?;
    } else {
        output.reshape(&[batch_size, output_channels, vert_slides, horz_slides])?;
    }

    Ok(output)
}

/// Intercalates values into a tensor along a given axis.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::intercalate_values;
///
/// let tensor = Tensor::<i128>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
/// let result = intercalate_values(&tensor, 0, 2, 1).unwrap();
///
/// let expected = Tensor::<i128>::new(Some(&[1, 0, 2, 3, 0, 4]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = intercalate_values(&expected, 0, 2, 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 0, 2, 0, 0, 0, 3, 0, 4]), &[3, 3]).unwrap();
///
/// assert_eq!(result, expected);
///
/// ```
pub fn intercalate_values<T: TensorType>(
    tensor: &Tensor<T>,
    value: T,
    stride: usize,
    axis: usize,
) -> Result<Tensor<T>, TensorError> {
    if stride == 1 {
        return Ok(tensor.clone());
    }

    let mut output_dims = tensor.dims().to_vec();
    output_dims[axis] = output_dims[axis] * stride - 1;

    let mut output: Tensor<T> = Tensor::new(None, &output_dims)?;

    let cartesian_coord = output
        .dims()
        .iter()
        .map(|d| (0..*d))
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut tensor_slice_iter = tensor.iter();

    output.iter_mut().enumerate().for_each(|(i, o)| {
        let coord = &cartesian_coord[i];

        if coord[axis] % stride == 0 {
            *o = tensor_slice_iter.next().unwrap().clone();
        } else {
            *o = value.clone();
        }
    });

    Ok(output)
}

/// One hot encodes a tensor along a given axis.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::one_hot;
/// let tensor = Tensor::<i128>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
/// let result = one_hot(&tensor, 5, 2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 1, 0, 0, 0,
///                                           0, 0, 1, 0, 0,
///                                           0, 0, 0, 1, 0,
///                                           0, 0, 0, 0, 1]), &[2, 2, 5]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn one_hot(
    tensor: &Tensor<i128>,
    num_classes: usize,
    axis: usize,
) -> Result<Tensor<i128>, TensorError> {
    let mut output_dims = tensor.dims().to_vec();
    output_dims.insert(axis, num_classes);

    let mut output: Tensor<i128> = Tensor::new(None, &output_dims)?;

    let cartesian_coord = output
        .dims()
        .iter()
        .map(|d| (0..*d))
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output
        .iter_mut()
        .enumerate()
        .map(|(i, o)| {
            let coord = &cartesian_coord[i];
            let coord_axis = coord[axis];

            let mut coord_without_axis = coord.clone();
            coord_without_axis.remove(axis);

            let elem = tensor.get(&coord_without_axis) as usize;
            if elem > num_classes {
                return Err(TensorError::DimMismatch(format!(
                    "Expected element to be less than num_classes, but got {}",
                    elem
                )));
            };

            if coord_axis == elem {
                *o = 1;
            } else {
                *o = 0;
            }
            Ok(())
        })
        .collect::<Result<Vec<()>, TensorError>>()?;

    Ok(output)
}

/// Performs a 2D deconvolution on the given input tensor.
/// # Examples
/// ```
// // expected ouputs are taken from pytorch torch.nn.functional.conv_transpose2d
///
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::deconv;
///
/// let c = Tensor::<i128>::new(Some(&[6, 0, 12, 4, 0, 8, 0, 0, 3, 0, 0, 2]), &[1, 2, 2, 3]).unwrap();
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
///
/// let result = deconv::<i128>(&[x, c], [(1, 1); 2], (1, 1), (2, 2)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 32, 0, 32, 0, 6, 0, 12, 0, 4, 0, 8, 0, 4, 0, 8, 0, 0, 0, 3, 0, 0, 0, 2]), &[1, 2, 3, 4]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k], [(0, 0); 2], (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[6, 14, 4, 2, 17, 21, 0, 1, 5]), &[1, 1, 3, 3]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k], [(1, 1); 2], (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[17]), &[1, 1, 1, 1]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k], [(1, 1); 2], (0, 0), (2, 2)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[10, 4, 0, 3]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k], [(0, 0); 2], (0, 0), (2, 2)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[6, 2, 12, 4, 2, 10, 4, 20, 0, 0, 3, 1, 0, 0, 1, 5]), &[1, 1, 4, 4]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[3, 2]),
///     &[1, 1, 2, 1],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k], [(1, 1); 2], (0, 0), (2, 2)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 0]), &[1, 1, 2, 1]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[3, 2]),
///     &[1, 1, 2, 1],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k], [(0, 0); 2], (0, 0), (2, 2)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[6, 0, 12, 4, 0, 8, 0, 0, 3, 0, 0, 2]), &[1, 1, 4, 3]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let c = Tensor::<i128>::new(Some(&[6, 0, 12, 4, 0, 8, 0, 0, 3, 0, 0, 2]), &[1, 2, 2, 3]).unwrap();
/// let x = Tensor::<i128>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
///
/// let result = deconv::<i128>(&[x, c], [(1, 1); 2], (0, 0), (2, 2)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[0, 32, 0, 0, 6, 0, 0, 4, 0, 0, 0, 0]), &[1, 2, 2, 3]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let x = Tensor::<i128>::new(
///     Some(&[3, 8, 0, 8, 4, 9, 8, 1, 8]),
///     &[1, 1, 3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[1, 0, 4, 6]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[1]),
///     &[1],
/// ).unwrap();
/// let result = deconv::<i128>(&[x, k, b], [(1, 1); 2], (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[55, 58, 66, 69]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// ```
pub fn deconv<
    T: TensorType
        + Mul<Output = T>
        + Add<Output = T>
        + std::marker::Sync
        + std::marker::Send
        + std::iter::Sum,
>(
    inputs: &[Tensor<T>],
    padding: [(usize, usize); 2],
    output_padding: (usize, usize),
    stride: (usize, usize),
) -> Result<Tensor<T>, TensorError> {
    let has_bias = inputs.len() == 3;
    let (image, kernel) = (&inputs[0], &inputs[1]);

    if (image.dims().len() != 4) || (kernel.dims().len() != 4) {
        return Err(TensorError::DimMismatch("deconv".to_string()));
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(TensorError::DimMismatch(
            "nil stride is not supported for deconv".to_string(),
        ));
    }

    if has_bias {
        let bias = &mut inputs[2].clone();

        if bias.dims().is_empty() {
            bias.reshape(&[1])?;
        }

        if (bias.dims().len() != 1) && (bias.dims()[0] != kernel.dims()[0]) {
            return Err(TensorError::DimMismatch("deconv bias".to_string()));
        }
    }

    let (kernel_height, kernel_width) = (kernel.dims()[2], kernel.dims()[3]);

    let mut expanded_image = intercalate_values(image, T::zero().unwrap(), stride.0, 2)?;
    expanded_image = intercalate_values(&expanded_image, T::zero().unwrap(), stride.1, 3)?;
    expanded_image = pad(&expanded_image, [(kernel_height - 1, kernel_width - 1); 2])?;

    // flip order
    let channel_coord = (0..kernel.dims()[0])
        .cartesian_product(0..kernel.dims()[1])
        .collect::<Vec<_>>();

    let slice_coord = expanded_image
        .dims()
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if i == 2 {
                padding[0].0..d - padding[1].0 + output_padding.0
            } else if i == 3 {
                padding[0].1..d - padding[1].1 + output_padding.1
            } else {
                0..*d
            }
        })
        .collect::<Vec<_>>();

    let sliced_expanded_image = expanded_image.get_slice(&slice_coord)?;

    let mut inverted_kernels = vec![];

    for (i, j) in channel_coord {
        let mut channel = kernel.get_slice(&[i..i + 1, j..j + 1])?;
        channel = Tensor::from(channel.clone().into_iter().rev());
        channel.reshape(&[kernel.dims()[2], kernel.dims()[3]])?;
        inverted_kernels.push(channel);
    }

    let mut deconv_kernel =
        Tensor::new(Some(&inverted_kernels), &[inverted_kernels.len()])?.combine()?;
    deconv_kernel.reshape(kernel.dims())?;

    // tensorflow formatting patch
    if kernel.dims()[0] == sliced_expanded_image.dims()[1] {
        deconv_kernel.reshape(&[
            kernel.dims()[1],
            kernel.dims()[0],
            kernel.dims()[2],
            kernel.dims()[3],
        ])?;
    }

    let input = if has_bias {
        vec![
            sliced_expanded_image,
            deconv_kernel.clone(),
            inputs[2].clone(),
        ]
    } else {
        vec![sliced_expanded_image, deconv_kernel.clone()]
    };

    let output = conv(&input, [(0, 0); 2], (1, 1))?;

    Ok(output)
}

/// Applies 2D sum pooling over a 4D tensor of shape B x C x H x W.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// * `pool_dims` - Tuple of pooling window size in x and y directions.
/// * `normalize` - Flag to normalize the output by the number of elements in the pooling window.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::sumpool;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap();
/// let pooled = sumpool(&x, [(0, 0); 2], (1, 1), (2, 2), false).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[11, 8, 8, 10]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
///
/// // This time with normalization
/// let pooled = sumpool(&x, [(0, 0); 2], (1, 1), (2, 2), true).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[3, 2, 2, 3]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
/// ```
pub fn sumpool(
    image: &Tensor<i128>,
    padding: [(usize, usize); 2],
    stride: (usize, usize),
    kernel_shape: (usize, usize),
    normalize: bool,
) -> Result<Tensor<i128>, TensorError> {
    let image_dims = image.dims();
    let batch_size = image_dims[0];
    let image_channels = image_dims[1];

    let unit = 1_i128;

    let mut kernel = Tensor::from(0..kernel_shape.0 * kernel_shape.1).map(|_| unit);
    kernel.reshape(&[1, 1, kernel_shape.0, kernel_shape.1])?;

    let cartesian_coord = [(0..batch_size), (0..image_channels)]
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let res = cartesian_coord
        .iter()
        .map(|coord| {
            let (b, i) = (coord[0], coord[1]);
            let input = image.get_slice(&[b..b + 1, i..i + 1])?;
            let output = conv(&[input, kernel.clone()], padding, stride)?;
            Ok(output)
        })
        .collect::<Result<Tensor<_>, TensorError>>()?;

    let shape = &res[0].dims()[2..];
    let mut combined = res.combine()?;
    combined.reshape(&[&[batch_size, image_channels], shape].concat())?;

    if normalize {
        let norm = kernel.len();
        combined = nonlinearities::const_div(&combined, norm as f64);
    }

    Ok(combined)
}

/// Applies 2D max pooling over a 4D tensor of shape B x C x H x W.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// * `pool_dims` - Tuple of pooling window size in x and y directions.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::max_pool2d;
/// use ezkl::circuit::utils::F32;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap();
/// let pooled = max_pool2d::<i128>(&x, &[(0, 0); 2], &(1, 1), &(2, 2)).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[5, 4, 4, 6]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
///
/// let x = Tensor::<f32>::new(Some(&[-0.9180, -0.4702, -0.0882, -0.0885, 0.3940,
///                                  -0.4884, 0.1395,  1.7860, -0.9729,  1.5160, -0.3346,
///                                 -0.0601, -0.1140,  0.2522, -0.2938, -0.0355]), &[1,1,4,4]).unwrap();
/// let x = x.map(|x| F32(x));
/// let pooled = max_pool2d::<F32>(&x, &[(0, 0); 2], &(2, 2), &(2, 2)).unwrap();
/// let expected = Tensor::<f32>::new(Some(&[0.3940,  1.7860, 1.5160, -0.0355]), &[1, 1, 2, 2]).unwrap();
/// let expected = expected.map(|x| F32(x));
/// assert_eq!(pooled, expected);
/// ```
pub fn max_pool2d<T: TensorType + std::marker::Sync + std::marker::Send + std::cmp::Ord>(
    image: &Tensor<T>,
    padding: &[(usize, usize); 2],
    stride: &(usize, usize),
    pool_dims: &(usize, usize),
) -> Result<Tensor<T>, TensorError> {
    if image.dims().len() != 4 {
        return Err(TensorError::DimMismatch("max_pool2d".to_string()));
    }
    let image_dims = image.dims();

    let (batch, input_channels, image_height, image_width) =
        (image_dims[0], image_dims[1], image_dims[2], image_dims[3]);

    let padded_image = pad::<T>(image, *padding)?;

    let vert_slides = (image_height + padding[0].0 + padding[1].0 - pool_dims.0) / stride.0 + 1;
    let horz_slides = (image_width + padding[0].1 + padding[1].1 - pool_dims.1) / stride.1 + 1;

    let mut output: Tensor<T> =
        Tensor::new(None, &[batch, input_channels, horz_slides, vert_slides]).unwrap();

    let cartesian_coord = [
        (0..batch),
        (0..input_channels),
        (0..vert_slides),
        (0..horz_slides),
    ]
    .iter()
    .cloned()
    .multi_cartesian_product()
    .collect::<Vec<_>>();

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, o)| {
            let coord = &cartesian_coord[flat_index];
            let (b, i, j, k) = (coord[0], coord[1], coord[2], coord[3]);
            let rs = j * stride.0;
            let cs = k * stride.1;
            let themax = padded_image
                .get_slice(&[
                    b..(b + 1),
                    i..(i + 1),
                    rs..(rs + pool_dims.0),
                    cs..(cs + pool_dims.1),
                ])
                .unwrap()
                .into_iter()
                .max()
                .unwrap();
            *o = themax;
        });

    Ok(output)
}

/// Dot product of two tensors.
/// # Arguments
///
/// * `inputs` - Vector of tensors of length 2.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::dot;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let y = Tensor::<i128>::new(
///     Some(&[5, 5, 10, -4, 2, -1, 2, 0, 1]),
///     &[1, 3, 3],
/// ).unwrap();
/// assert_eq!(dot(&[x, y]).unwrap()[0], 86);
/// ```
pub fn dot<T: TensorType + Mul<Output = T> + Add<Output = T> + Send + Sync + std::iter::Sum>(
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    if (inputs.len() != 2) || (inputs[0].clone().len() != inputs[1].clone().len()) {
        return Err(TensorError::DimMismatch("dot".to_string()));
    }

    let (a, b): (Tensor<T>, Tensor<T>) = (inputs[0].clone(), inputs[1].clone());
    let res: Vec<T> = a
        .par_iter()
        .zip(b.par_iter())
        .fold(
            || T::zero().unwrap(),
            |acc, (k, i)| acc + k.clone() * i.clone(),
        )
        .collect();

    let res = res.into_iter().sum();

    Tensor::new(Some(&[res]), &[1])
}

/// Pads a 4D tensor of shape `B x C x H x W` to a tensor of shape `B x C x (H + 2xPADDING) x (W + 2xPADDING)` using 0 values.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::pad;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap();
/// let result = pad::<i128>(&x, [(1, 1); 2]).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[0, 0, 0, 0, 0, 0, 5, 2, 3, 0, 0, 0, 4, -1, 0, 0, 3, 1, 6, 0, 0, 0, 0, 0, 0]),
///     &[1, 1, 5, 5],
/// ).unwrap();
/// assert_eq!(result, expected);
///
///
///
/// ```
pub fn pad<T: TensorType>(
    image: &Tensor<T>,
    padding: [(usize, usize); 2],
) -> Result<Tensor<T>, TensorError> {
    if image.dims().len() != 4 {
        return Err(TensorError::DimMismatch("pad".to_string()));
    }
    let (batch_size, channels, height, width) = (
        image.dims()[0],
        image.dims()[1],
        image.dims()[2],
        image.dims()[3],
    );

    let (padding_before, padding_after) = padding.into();

    let padded_height = height + padding_before.0 + padding_after.0;
    let padded_width = width + padding_before.1 + padding_after.1;

    let mut output =
        Tensor::<T>::new(None, &[batch_size, channels, padded_height, padded_width]).unwrap();

    for b in 0..batch_size {
        for channel in 0..channels {
            for row in 0..height {
                for col in 0..width {
                    output.set(
                        &[b, channel, row + padding_before.0, col + padding_before.1],
                        image.get(&[b, channel, row, col]).clone(),
                    );
                }
            }
        }
    }

    output.reshape(&[batch_size, channels, padded_height, padded_width])?;
    Ok(output)
}

/// Packs a multi-dim tensor into a single elem tensor
/// # Arguments
///
/// * `a` - Tensor.
/// * `base` - Base to use when packing
/// * `scale` - fixed point representation scale
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::pack;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 1]),
///     &[1, 3],
/// ).unwrap();
/// let result = pack::<i128>(&x, 2, 2).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[90]),
///     &[1],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn pack<T: TensorType + std::marker::Send + std::marker::Sync>(
    a: &Tensor<T>,
    base: T,
    scale: u32,
) -> Result<Tensor<T>, TensorError>
where
    T: Add<Output = T>,
    T: Mul<Output = T>,
{
    // base ^ (scale + tensor)
    let mut output = T::zero().unwrap();
    let base_tensor = Tensor::new(Some(&[base]), &[1])?;
    for (i, a_i) in a.iter().enumerate() {
        let pow_value = &base_tensor.pow((i as u32) * (scale + 1))?[0];
        output = output + pow_value.clone() * a_i.clone();
    }
    Tensor::new(Some(&[output]), &[1])
}

/// Concatenates a list of tensors along a specified axis.
/// # Arguments
/// * `inputs` - A slice of tensors to concatenate.
/// * `axis` - The axis along which to concatenate the tensors.
///
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::concat;
/// // tested against pytorch outputs for reference :)
///
/// // 1D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3]), &[3]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[4, 5, 6]), &[3]).unwrap();
/// let result = concat(&[&x, &y], 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
/// assert_eq!(result, expected);
///
/// // 2D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[7, 8, 9]), &[3, 1]).unwrap();
/// let result = concat(&[&x, &y], 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 7, 3, 4, 8, 5, 6, 9]), &[3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// /// 4D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), &[2, 2, 2, 2]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[17, 18, 19, 20, 21, 22, 23, 14]), &[2, 2, 1, 2]).unwrap();
/// let result = concat(&[&x, &y], 2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 17, 18, 5, 6, 7, 8, 19, 20, 9, 10, 11, 12, 21, 22, 13, 14, 15, 16, 23, 14]), &[2, 2, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // 5D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), &[8, 1, 1, 1, 2]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[17, 18, 19, 20, 21, 22, 23, 14]), &[4, 1, 1, 1, 2]).unwrap();
/// let result = concat(&[&x, &y], 0).unwrap();
///
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 14]), &[12, 1, 1, 1, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// ```
///
/// # Errors
/// Returns a TensorError if the tensors in `inputs` have incompatible dimensions for concatenation along the specified `axis`.

pub fn concat<T: TensorType + Send + Sync>(
    inputs: &[&Tensor<T>],
    axis: usize,
) -> Result<Tensor<T>, TensorError> {
    if inputs.len() == 1 {
        return Ok(inputs[0].clone());
    }

    // Calculate the output tensor size
    let mut output_size = inputs[0].dims().to_vec();
    output_size[axis] = inputs.iter().map(|x| x.dims()[axis]).sum();

    // Allocate memory for the output tensor
    let mut output = Tensor::new(None, &output_size)?;
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let get_input_index = |index_along_axis: usize| -> (usize, usize) {
        let mut current_idx = 0;
        let mut input_idx = 0;
        let mut input_coord_at_idx = 0;
        for (i, elem) in inputs.iter().enumerate() {
            current_idx += elem.dims()[axis];
            if index_along_axis < current_idx {
                input_idx = i;
                // subtract the current
                input_coord_at_idx = index_along_axis - (current_idx - elem.dims()[axis]);
                break;
            }
        }
        (input_idx, input_coord_at_idx)
    };

    output = output.par_enum_map(|i, _: T| {
        let coord = cartesian_coord[i].clone();
        let mut index = 0;
        let mut input_index = 0;
        let mut input_coord = coord.clone();
        for (j, x) in coord.iter().enumerate() {
            if j == axis {
                (input_index, input_coord[j]) = get_input_index(*x);
                break;
            }
            index += x;
        }

        Ok(inputs[input_index].get(&input_coord))
    })?;

    // Reshape the output tensor
    output.reshape(&output_size)?;

    Ok(output)
}

/// Slices a tensor from start to end along a given axis
///
/// /// # Examples
/// ```
/// // tested against pytorch output
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::slice;
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let result = slice(&x, &0, &1, &2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[3, 4]), &[1, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let result = slice(&x, &1, &1, &2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 4, 6]), &[3, 1]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 2, 3]).unwrap();
/// let result = slice(&x, &2, &1, &2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 5, 8, 11]), &[2, 2, 1]).unwrap();
/// assert_eq!(result, expected);
/// ```
///
pub fn slice<T: TensorType + Send + Sync>(
    t: &Tensor<T>,
    axis: &usize,
    start: &usize,
    end: &usize,
) -> Result<Tensor<T>, TensorError> {
    let mut slice = vec![];
    for (i, d) in t.dims().iter().enumerate() {
        if i != *axis {
            slice.push(0..*d)
        } else {
            slice.push(*start..*end)
        }
    }

    t.get_slice(&slice)
}

// ---------------------------------------------------------------------------------------------------------
// -- nonlinear Functions ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------

/// Activation functions
pub mod nonlinearities {
    use super::*;

    /// Ceiling operator.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    ///
    /// use ezkl::tensor::ops::nonlinearities::ceil;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[1, 2, 3, 4, 5, 6]),
    ///  &[3, 2],
    /// ).unwrap();
    /// let result = ceil(&x, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[2, 2, 4, 4, 6, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn ceil(a: &Tensor<i128>, scale: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.ceil() * scale;
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Floor operator.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::floor;
    /// let x = Tensor::<i128>::new(
    ///   Some(&[1, 2, 3, 4, 5, 6]),
    ///  &[3, 2],
    /// ).unwrap();
    /// let result = floor(&x, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[0, 2, 2, 4, 4, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn floor(a: &Tensor<i128>, scale: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.floor() * scale;
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Round operator.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::round;
    /// let x = Tensor::<i128>::new(
    ///   Some(&[1, 2, 3, 4, 5, 6]),
    /// &[3, 2],
    /// ).unwrap();
    /// let result = round(&x, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[2, 2, 4, 4, 6, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn round(a: &Tensor<i128>, scale: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.round() * scale;
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Round half to even operator.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::round_half_to_even;
    /// let x = Tensor::<i128>::new(
    ///   Some(&[1, 2, 3, 4, 5, 6]),
    /// &[3, 2],
    /// ).unwrap();
    /// let result = round_half_to_even(&x, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[0, 2, 4, 4, 4, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn round_half_to_even(a: &Tensor<i128>, scale: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.round_ties_even() * scale;
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Raises to a floating point power.
    /// # Arguments
    /// * `a` - Tensor
    /// * `power` - Floating point power
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::pow;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[2, 15, 2, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = pow(&x, 1.0, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 225, 4, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn pow(a: &Tensor<i128>, scale_input: f64, power: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let kix = scale_input * (kix).powf(power);
            let rounded = kix.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Applies Kronecker delta to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::kronecker_delta;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[2, 15, 2, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = kronecker_delta(&x);
    /// let expected = Tensor::<i128>::new(Some(&[0, 0, 0, 0, 0, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn kronecker_delta<T: TensorType + std::cmp::PartialEq + Send + Sync>(
        a: &Tensor<T>,
    ) -> Tensor<T> {
        a.par_enum_map(|_, a_i| {
            if a_i == T::zero().unwrap() {
                Ok::<_, TensorError>(T::one().unwrap())
            } else {
                Ok::<_, TensorError>(T::zero().unwrap())
            }
        })
        .unwrap()
    }

    /// Elementwise applies sigmoid to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::sigmoid;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sigmoid(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 1, 1, 1]), &[2, 3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// let x = Tensor::<i128>::new(
    ///    Some(&[65536]),
    ///   &[1],
    /// ).unwrap();
    /// let result = sigmoid(&x, 65536.0);
    /// let expected = Tensor::<i128>::new(Some(&[47911]), &[1]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// /// assert_eq!(result, expected);
    /// let x = Tensor::<i128>::new(
    ///    Some(&[256]),
    ///   &[1],
    /// ).unwrap();
    /// let result = sigmoid(&x, 256.0);
    /// let expected = Tensor::<i128>::new(Some(&[187]), &[1]).unwrap();
    ///
    /// ```
    pub fn sigmoid(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input / (1.0 + (-kix).exp());
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies exponential to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::exp;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = exp(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[7, 3269017, 7, 3, 3, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// let x = Tensor::<i128>::new(
    ///    Some(&[37, 12, 41]),
    ///   &[3],
    /// ).unwrap();
    /// let result = exp(&x, 512.0);
    ///
    /// let expected = Tensor::<i128>::new(Some(&[550, 524, 555]), &[3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn exp(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.exp();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies exponential to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::ln;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 3000]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = ln(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[1, 3, 1, 0, 0, 8]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// let x = Tensor::<i128>::new(
    ///    Some(&[37, 12, 41]),
    ///   &[3],
    /// ).unwrap();
    /// let result = ln(&x, 512.0);
    ///
    /// let expected = Tensor::<i128>::new(Some(&[-1345, -1922, -1293]), &[3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn ln(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.ln();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies sign to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::sign;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[-2, 15, 2, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = sign(&x);
    /// let expected = Tensor::<i128>::new(Some(&[-1, 1, 1, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sign(a: &Tensor<i128>) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(a_i.signum()))
            .unwrap()
    }

    /// softmax layout
    pub fn softmax_axes(a: &Tensor<i128>, scale: f64, axes: &[usize]) -> Tensor<i128> {
        // we want this to be as small as possible so we set the output scale to 1
        let dims = a.dims();

        if dims.len() == 1 {
            return softmax(a, scale);
        }

        let cartesian_coord = dims[..dims.len() - 1]
            .iter()
            .map(|x| 0..*x)
            .multi_cartesian_product()
            .collect::<Vec<_>>();

        let mut outputs = vec![];

        for coord in cartesian_coord {
            let mut sum_dims = vec![];
            for (i, c) in coord.iter().enumerate() {
                if axes.contains(&i) {
                    sum_dims.push(0..a.dims()[i]);
                } else {
                    sum_dims.push(*c..*c + 1);
                }
            }

            let softmax_input = a.get_slice(&sum_dims).unwrap();

            let res = softmax(&softmax_input, scale);

            outputs.push(res);
        }

        let mut res = Tensor::new(Some(&outputs), &[outputs.len()])
            .unwrap()
            .combine()
            .unwrap();
        res.reshape(dims).unwrap();
        res
    }

    /// Applies softmax
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::softmax;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 2, 3, 2, 2, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = softmax(&x, 128.0);
    /// // doubles the scale of the input
    /// let expected = Tensor::<i128>::new(Some(&[2730, 2730, 2751, 2730, 2730, 2688]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn softmax(a: &Tensor<i128>, scale: f64) -> Tensor<i128> {
        // the more accurate calculation is commented out and we implement as below so it matches the steps in layout

        let exp = exp(a, scale);

        let sum = sum(&exp).unwrap();
        let inv_denom = recip(&sum, scale, scale);

        (exp * inv_denom).unwrap()
    }

    /// Applies range_check_percent
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `b` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::range_check_percent;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[100, 200, 300, 400, 500, 600]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let y = Tensor::<i128>::new(
    ///    Some(&[103, 204, 303, 404, 505, 607]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = range_check_percent(&[x, y], 1024, 1024, 1.0); // 1% tolerance
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 0, 0, 0, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn range_check_percent(
        t: &[Tensor<i128>],
        input_scale: usize,
        output_scale: usize,
        tol: f32,
    ) -> Tensor<i128> {
        // the more accurate calculation is commented out and we implement as below so it matches the steps in layout
        let scale = input_scale * output_scale;
        let diff: Tensor<i128> = sub(t).unwrap();
        let recip = recip(&t[0], input_scale as f64, output_scale as f64);
        let product = mult(&[diff, recip]).unwrap();
        let _tol = ((tol / 100.0) * scale as f32).round() as f64;
        let upper_bound = greater_than(&product, _tol);
        let neg_product =
            mult(&[product, Tensor::<i128>::new(Some(&[-1]), &[1]).unwrap()]).unwrap();
        let lower_bound = greater_than(&neg_product, _tol);

        add(&[upper_bound, lower_bound]).unwrap()
    }

    /// Elementwise applies square root to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::sqrt;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sqrt(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[2, 5, 3, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sqrt(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.sqrt();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies reciprocal square root to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::rsqrt;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[4, 25, 8, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = rsqrt(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[1, 0, 0, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn rsqrt(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input / (kix.sqrt() + f64::EPSILON);
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies cosine to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::cos;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = cos(&x, 2.0);
    /// let expected = Tensor::<i128>::new(Some(& [-1, 2, -1, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn cos(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.cos();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies arccosine to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::acos;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = acos(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[0, 0, 0, 0, 0, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn acos(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.acos();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies cosh to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::cosh;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = cosh(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[27, 36002449669, 1490, 2, 2, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn cosh(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.cosh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies arccosineh to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::acosh;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = acosh(&x, 1.0);
    /// let expected = Tensor::<i128>::new(Some(& [2, 4, 3, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn acosh(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.acosh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies sine to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::sin;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = sin(&x, 128.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sin(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.sin();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies arcsine to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::asin;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = asin(&x, 128.0);
    /// let expected = Tensor::<i128>::new(Some(& [4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn asin(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.asin();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies sineh to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::sinh;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = sinh(&x, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[7, 268337, 55, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sinh(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.sinh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies arcsineh to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::asinh;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = asinh(&x, 128.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn asinh(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.asinh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies tan activation to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::tan;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = tan(&x, 64.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 26, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn tan(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.tan();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies arctan activation to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::atan;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = atan(&x, 128.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn atan(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.atan();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies tanh activation to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::tanh;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = tanh(&x, 128.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn tanh(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.tanh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies arctanh activation to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::atanh;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[4, 25, 8, 2, 2, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = atanh(&x, 32.0);
    /// let expected = Tensor::<i128>::new(Some(&[4, 34, 8, 2, 2, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn atanh(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.atanh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Applies error function (erf) on a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::erffunc;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[5, 28, 9, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = erffunc(&x, 128.0);
    /// let expected = Tensor::<i128>::new(Some(&[6, 31, 10, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn erffunc(a: &Tensor<i128>, scale_input: f64) -> Tensor<i128> {
        const NCOEF: usize = 28;
        const COF: [f64; 28] = [
            -1.3026537197817094,
            6.419_697_923_564_902e-1,
            1.9476473204185836e-2,
            -9.561_514_786_808_63e-3,
            -9.46595344482036e-4,
            3.66839497852761e-4,
            4.2523324806907e-5,
            -2.0278578112534e-5,
            -1.624290004647e-6,
            1.303655835580e-6,
            1.5626441722e-8,
            -8.5238095915e-8,
            6.529054439e-9,
            5.059343495e-9,
            -9.91364156e-10,
            -2.27365122e-10,
            9.6467911e-11,
            2.394038e-12,
            -6.886027e-12,
            8.94487e-13,
            3.13092e-13,
            -1.12708e-13,
            3.81e-16,
            7.106e-15,
            -1.523e-15,
            -9.4e-17,
            1.21e-16,
            -2.8e-17,
        ];

        /// Chebyshev coefficients
        fn erfccheb(z: f64) -> f64 {
            let mut d = 0f64;
            let mut dd = 0f64;

            assert!(z >= 0f64, "erfccheb requires nonnegative argument");
            let t = 2f64 / (2f64 + z);
            let ty = 4f64 * t - 2f64;
            for j in (1..NCOEF - 1).rev() {
                let tmp = d;
                d = ty * d - dd + COF[j];
                dd = tmp;
            }
            t * (-z.powi(2) + 0.5 * (COF[0] + ty * d) - dd).exp()
        }

        pub fn erf(x: f64) -> f64 {
            if x >= 0f64 {
                1.0 - erfccheb(x)
            } else {
                erfccheb(-x) - 1f64
            }
        }

        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * erf(kix);
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as i128)
        })
        .unwrap()
    }

    /// Elementwise applies leaky relu to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale` - Single value
    /// * `slope` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::leakyrelu;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, -5]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = leakyrelu(&x, 0.1);
    /// let expected = Tensor::<i128>::new(Some(&[2, 15, 2, 1, 1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn leakyrelu(a: &Tensor<i128>, slope: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let rounded = if a_i < 0 {
                let d_inv_x = (slope) * (a_i as f64);
                d_inv_x.round() as i128
            } else {
                let d_inv_x = a_i as f64;
                d_inv_x.round() as i128
            };
            Ok::<_, TensorError>(rounded)
        })
        .unwrap()
    }

    /// Elementwise applies max to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `b` - scalar
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::max;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[2, 15, 2, 1, 1, -5]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = max(&x, 1.0, 1.0);
    /// let expected = Tensor::<i128>::new(Some(&[2, 15, 2, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn max(a: &Tensor<i128>, scale_input: f64, threshold: f64) -> Tensor<i128> {
        // calculate value of output
        a.par_enum_map(|_, a_i| {
            let d_inv_x = (a_i as f64) / scale_input;
            let rounded = if d_inv_x <= threshold {
                (threshold * scale_input).round() as i128
            } else {
                (d_inv_x * scale_input).round() as i128
            };
            Ok::<_, TensorError>(rounded)
        })
        .unwrap()
    }

    /// Elementwise applies min to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// * `b` - scalar
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::min;
    /// let x = Tensor::<i128>::new(
    ///    Some(&[2, 15, 2, 1, 1, -5]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = min(&x, 1.0, 2.0);
    /// let expected = Tensor::<i128>::new(Some(&[2, 2, 2, 1, 1, -5]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn min(a: &Tensor<i128>, scale_input: f64, threshold: f64) -> Tensor<i128> {
        // calculate value of output
        a.par_enum_map(|_, a_i| {
            let d_inv_x = (a_i as f64) / scale_input;
            let rounded = if d_inv_x >= threshold {
                (threshold * scale_input).round() as i128
            } else {
                (d_inv_x * scale_input).round() as i128
            };
            Ok::<_, TensorError>(rounded)
        })
        .unwrap()
    }

    /// Elementwise divides a tensor with a const integer element.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::const_div;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = const_div(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 4, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn const_div(a: &Tensor<i128>, denom: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let d_inv_x = (a_i as f64) / (denom);
            Ok::<_, TensorError>(d_inv_x.round() as i128)
        })
        .unwrap()
    }

    /// Elementwise inverse.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::recip;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2_f64;
    /// let result = recip(&x, 1.0, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 2, 1, 0, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn recip(a: &Tensor<i128>, input_scale: f64, out_scale: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| {
            let rescaled = (a_i as f64) / input_scale;
            let denom = (1_f64) / (rescaled + f64::EPSILON);
            let d_inv_x = out_scale * denom;
            Ok::<_, TensorError>(d_inv_x.round() as i128)
        })
        .unwrap()
    }

    /// Elementwise inverse.
    /// # Arguments
    /// * `out_scale` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::zero_recip;
    /// let k = 2_f64;
    /// let result = zero_recip(1.0);
    /// let expected = Tensor::<i128>::new(Some(&[4503599627370496]), &[1]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn zero_recip(out_scale: f64) -> Tensor<i128> {
        let a = Tensor::<i128>::new(Some(&[0]), &[1]).unwrap();

        a.par_enum_map(|_, a_i| {
            let rescaled = a_i as f64;
            let denom = (1_f64) / (rescaled + f64::EPSILON);
            let d_inv_x = out_scale * denom;
            Ok::<_, TensorError>(d_inv_x.round() as i128)
        })
        .unwrap()
    }

    /// Elementwise greater than
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::greater_than;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = greater_than(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[0, 0, 0, 1, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn greater_than(a: &Tensor<i128>, b: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(i128::from((a_i as f64 - b) > 0_f64)))
            .unwrap()
    }

    /// Elementwise greater than
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::greater_than_equal;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = greater_than_equal(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 0, 1, 1, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn greater_than_equal(a: &Tensor<i128>, b: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(i128::from((a_i as f64 - b) >= 0_f64)))
            .unwrap()
    }

    /// Elementwise less than
    /// # Arguments
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::less_than;
    ///
    /// let x = Tensor::<i128>::new(
    ///    Some(&[2, 1, 2, 7, 1, 1]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    ///
    /// let result = less_than(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[0, 1, 0, 0, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn less_than(a: &Tensor<i128>, b: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(i128::from((a_i as f64 - b) < 0_f64)))
            .unwrap()
    }

    /// Elementwise less than
    /// # Arguments
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::less_than_equal;
    ///
    /// let x = Tensor::<i128>::new(
    ///    Some(&[2, 1, 2, 7, 1, 1]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    ///
    /// let result = less_than_equal(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 0, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn less_than_equal(a: &Tensor<i128>, b: f64) -> Tensor<i128> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(i128::from((a_i as f64 - b) <= 0_f64)))
            .unwrap()
    }

    /// Takes the mean of a tensor
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::mean;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = mean(&x, 1);
    /// let expected = Tensor::<i128>::new(Some(&[2]), &[1]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn mean(a: &Tensor<i128>, scale: usize) -> Tensor<i128> {
        let sum = sum(a).unwrap();
        const_div(&sum, (scale * a.len()) as f64)
    }
}

/// Ops that return the transcript i.e intermediate calcs of an op
pub mod accumulated {
    use super::*;

    /// Dot product of two tensors.
    /// # Arguments
    ///
    /// * `inputs` - Vector of tensors of length 2.
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::accumulated::dot;
    ///
    /// let x = Tensor::<i128>::new(
    ///     Some(&[5, 2]),
    ///     &[2],
    /// ).unwrap();
    /// let y = Tensor::<i128>::new(
    ///     Some(&[5, 5]),
    ///     &[2],
    /// ).unwrap();
    /// let expected = Tensor::<i128>::new(
    ///     Some(&[25, 35]),
    ///     &[2],
    /// ).unwrap();
    /// assert_eq!(dot(&[x, y], 1).unwrap(), expected);
    /// ```
    pub fn dot<T: TensorType + Mul<Output = T> + Add<Output = T>>(
        inputs: &[Tensor<T>; 2],
        chunk_size: usize,
    ) -> Result<Tensor<T>, TensorError> {
        if inputs[0].clone().len() != inputs[1].clone().len() {
            return Err(TensorError::DimMismatch("dot".to_string()));
        }
        let (a, b): (Tensor<T>, Tensor<T>) = (inputs[0].clone(), inputs[1].clone());

        let transcript: Tensor<T> = a
            .iter()
            .zip(b)
            .chunks(chunk_size)
            .into_iter()
            .scan(T::zero().unwrap(), |acc, chunk| {
                let k = chunk.fold(T::zero().unwrap(), |acc, (a_i, b_i)| {
                    acc.clone() + a_i.clone() * b_i.clone()
                });
                *acc = acc.clone() + k.clone();
                Some(acc.clone())
            })
            .collect();

        Ok(transcript)
    }

    /// Sums a tensor.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::accumulated::sum;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sum(&x, 1).unwrap();
    /// let expected = Tensor::<i128>::new(
    ///     Some(&[2, 17, 19, 20, 21, 21]),
    ///     &[6],
    /// ).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sum<T: TensorType + Mul<Output = T> + Add<Output = T>>(
        a: &Tensor<T>,
        chunk_size: usize,
    ) -> Result<Tensor<T>, TensorError> {
        let transcript: Tensor<T> = a
            .iter()
            .chunks(chunk_size)
            .into_iter()
            .scan(T::zero().unwrap(), |acc, chunk| {
                let k = chunk.fold(T::zero().unwrap(), |acc, a_i| acc.clone() + a_i.clone());
                *acc = acc.clone() + k.clone();
                Some(acc.clone())
            })
            .collect();

        Ok(transcript)
    }

    /// Prod of a tensor.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::accumulated::prod;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = prod(&x, 1).unwrap();
    /// let expected = Tensor::<i128>::new(
    ///     Some(&[2, 30, 60, 60, 60, 0]),
    ///     &[6],
    /// ).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn prod<T: TensorType + Mul<Output = T> + Add<Output = T>>(
        a: &Tensor<T>,
        chunk_size: usize,
    ) -> Result<Tensor<T>, TensorError> {
        let transcript: Tensor<T> = a
            .iter()
            .chunks(chunk_size)
            .into_iter()
            .scan(T::one().unwrap(), |acc, chunk| {
                let k = chunk.fold(T::one().unwrap(), |acc, a_i| acc.clone() * a_i.clone());
                *acc = acc.clone() * k.clone();
                Some(acc.clone())
            })
            .collect();

        Ok(transcript)
    }
}
