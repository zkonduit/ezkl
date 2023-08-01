use super::TensorError;
use crate::tensor::{Tensor, TensorType};
use itertools::Itertools;
use rayon::{
    iter::IndexedParallelIterator, iter::IntoParallelRefMutIterator, iter::ParallelIterator,
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
///   Some(&[2, 1, 2, 1, 1, 1]),
/// &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///   Some(&[2, 3, 2, 1, 1, 1]),
/// &[2, 3],
/// ).unwrap();
/// let result = iff(&mask, &a, &b).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[2, 1, 2, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn iff<
    T: TensorType
        + Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + std::marker::Send
        + std::marker::Sync,
>(
    mask: &Tensor<T>,
    b: &Tensor<T>,
    a: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    let masked_a = (mask.clone() * a.clone())?;
    let masked_b = ((Tensor::from(vec![T::one().ok_or(TensorError::DimError)?].into_iter())
        - mask.clone())?
        * b.clone())?;

    masked_a + masked_b
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
pub fn resize<T: TensorType>(a: &Tensor<T>, scales: &[usize]) -> Result<Tensor<T>, TensorError> {
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
    output.iter_mut().enumerate().for_each(|(i, o)| {
        let mut coord = vec![];
        for (j, (c, _d)) in cartesian_coord[i].iter().zip(new_shape.iter()).enumerate() {
            let scale = scales[j];
            let fragment = c / scale;
            coord.push(fragment);
        }

        *o = a.get(&coord);
    });

    Ok(output)
}

/// Matrix multiplies two 2D tensors.
/// # Arguments
///
/// * `inputs` - Vector of tensors of length 2
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::matmul;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 2, 1, 1]),
///     &[3, 4],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = matmul(&vec![k, x]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn matmul<
    T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Send + std::marker::Sync,
>(
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    let (mut a, mut b) = (inputs[0].clone(), inputs[1].clone());

    if a.dims().len() == 1 {
        a.reshape(&[1, a.dims()[0]]);
    }
    if b.dims().len() == 1 {
        b.reshape(&[b.dims()[0], 1]);
    }

    if (inputs.len() != 2)
        || (a.dims()[a.dims().len() - 1] != b.dims()[a.dims().len() - 2])
        || (a.dims()[0..a.dims().len() - 2] != b.dims()[0..a.dims().len() - 2])
    {
        return Err(TensorError::DimMismatch("matmul".to_string()));
    }

    let mut dims = Vec::from(&a.dims()[0..a.dims().len() - 2]);
    dims.push(a.dims()[a.dims().len() - 2]);
    dims.push(b.dims()[a.dims().len() - 1]);
    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &dims).unwrap();

    let cartesian_coord = dims
        .iter()
        .map(|d| 0..*d)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, o)| {
            let coord = &cartesian_coord[flat_index];
            let row = coord[0..coord.len() - 1]
                .iter()
                .map(|&d| d..(d + 1))
                .collect::<Vec<_>>();
            let mut col = coord[0..coord.len()]
                .iter()
                .map(|&d| d..(d + 1))
                .collect::<Vec<_>>();
            col[coord.len() - 2] = 0..b.dims()[coord.len() - 2];
            let prod = dot(&[
                a.get_slice(&row[0..]).unwrap(),
                b.get_slice(&col[0..]).unwrap(),
            ])
            .unwrap();

            *o = prod[0].clone();
        });

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
/// ```
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

            // Get the indices common accross input tensors
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
    output.reshape(&output_shape);

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
/// * `a` - Tensor
/// * `b` - Tensor
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

pub fn downsample<T: TensorType>(
    input: &Tensor<T>,
    dim: usize,
    stride: usize,
    modulo: usize,
) -> Result<Tensor<T>, TensorError> {
    let mut output_shape = input.dims().to_vec();
    output_shape[dim] = (input.dims()[dim] - modulo).div_ceil(stride);
    let mut output = Tensor::<T>::new(None, &output_shape)?;

    assert!(modulo <= input.dims()[dim]);
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

    output
        .iter_mut()
        .zip(indices.iter())
        .for_each(|(o, i)| *o = input.get(i.as_slice()));

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
/// let result = gather(&x, 1, &index).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn gather<T: TensorType>(
    input: &Tensor<T>,
    dim: usize,
    index: &Tensor<usize>,
) -> Result<Tensor<T>, TensorError> {
    // Calculate the output tensor size
    let mut output_size = input.dims().to_vec();
    output_size[dim] = index.dims()[0];

    assert!(index.dims().len() == 1, "Index must be 1D for now");

    // Allocate memory for the output tensor
    let mut output = Tensor::new(None, &output_size)?;
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output = output.enum_map(|i, _: T| {
        let coord = cartesian_coord[i].clone();
        let index_val = index.get(&[coord[dim]]);
        let new_coord = coord
            .iter()
            .enumerate()
            .map(|(i, x)| if i == dim { index_val } else { *x })
            .collect::<Vec<_>>();

        Ok(input.get(&new_coord))
    })?;

    // Reshape the output tensor
    output.reshape(&output_size);

    Ok(output)
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
pub fn sum_axes<T: TensorType + Add<Output = T>>(
    a: &Tensor<T>,
    axes: &[usize],
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

    let mut res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    for coord in cartesian_coord.iter() {
        let mut sum_dims = vec![];
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(*c..*c + 1);
            }
        }

        res.set(coord, sum(&a.get_slice(&sum_dims)?)?[0].clone());
    }

    Ok(res)
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
pub fn min_axes<T: TensorType + Add<Output = T> + std::cmp::Ord>(
    a: &Tensor<T>,
    axes: &[usize],
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

    let mut res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    for coord in cartesian_coord.iter() {
        let mut sum_dims = vec![];
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(*c..*c + 1);
            }
        }

        res.set(coord, a.get_slice(&sum_dims)?.iter().min().unwrap().clone());
    }

    Ok(res)
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

/// Mins a tensor along specific axes.
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
pub fn max_axes<T: TensorType + Add<Output = T> + std::cmp::Ord>(
    a: &Tensor<T>,
    axes: &[usize],
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output

    let mut new_dims = vec![];
    for i in 0..a.dims().len() {
        if !axes.contains(&i) {
            new_dims.push(a.dims()[i]);
        } else {
            new_dims.push(1);
        }
    }

    let mut res = Tensor::new(None, &new_dims)?;

    let cartesian_coord = new_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    for coord in cartesian_coord.iter() {
        let mut sum_dims = vec![];
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(*c..*c + 1);
            }
        }

        res.set(coord, a.get_slice(&sum_dims)?.iter().max().unwrap().clone());
    }

    Ok(res)
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
/// let result = conv::<i128>(&[x, k, b], (0, 0), (1, 1)).unwrap();
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
/// let result = conv::<i128>(&[x, k, b], (0, 0), (1, 1)).unwrap();
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
/// let result = conv::<i128>(&[x, k, b], (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[65, 36, 21, 52, 73, 48, 37, 48, 65, 36, 21, 52, 73, 48, 37, 48]), &[1, 4, 2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn conv<
    T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Sync + std::marker::Send,
>(
    inputs: &[Tensor<T>],
    padding: (usize, usize),
    stride: (usize, usize),
) -> Result<Tensor<T>, TensorError> {
    let has_bias = inputs.len() == 3;
    let (image, kernel) = (&mut inputs[0].clone(), &inputs[1]);
    let og_dims = image.dims().to_vec();

    // ensure inputs are 4D tensors
    if og_dims.len() == 3 {
        // adds a dummy batch dimension
        let mut new_dims = vec![1];
        new_dims.extend_from_slice(image.dims());
        image.reshape(&new_dims);
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
        let bias = &inputs[2];
        if (bias.dims().len() != 1) || (bias.dims()[0] != kernel.dims()[0]) {
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

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

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

    let cartesian_coord = vec![
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
            *o = res + inputs[2][start_kernel_index].clone();
        } else {
            *o = res;
        }
    });

    // remove dummy batch dimension if we added one
    if og_dims.len() == 3 {
        output.reshape(&[output_channels, vert_slides, horz_slides]);
    } else {
        output.reshape(&[batch_size, output_channels, vert_slides, horz_slides]);
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
/// let result = deconv::<i128>(&[x, c], (1, 1), (1, 1), (2, 2)).unwrap();
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
/// let result = deconv::<i128>(&[x, k], (0, 0), (0, 0), (1, 1)).unwrap();
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
/// let result = deconv::<i128>(&[x, k], (1, 1), (0, 0), (1, 1)).unwrap();
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
/// let result = deconv::<i128>(&[x, k], (1, 1), (0, 0), (2, 2)).unwrap();
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
/// let result = deconv::<i128>(&[x, k], (0, 0), (0, 0), (2, 2)).unwrap();
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
/// let result = deconv::<i128>(&[x, k], (1, 1), (0, 0), (2, 2)).unwrap();
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
/// let result = deconv::<i128>(&[x, k], (0, 0), (0, 0), (2, 2)).unwrap();
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
/// let result = deconv::<i128>(&[x, c], (1, 1), (0, 0), (2, 2)).unwrap();
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
/// let result = deconv::<i128>(&[x, k, b], (1, 1), (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[55, 58, 66, 69]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// ```
pub fn deconv<
    T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Sync + std::marker::Send,
>(
    inputs: &[Tensor<T>],
    padding: (usize, usize),
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
            "non-positive stride is not supported for deconv".to_string(),
        ));
    }

    if has_bias {
        let bias = &inputs[2];
        if (bias.dims().len() != 1) || (bias.dims()[0] != kernel.dims()[0]) {
            return Err(TensorError::DimMismatch("deconv bias".to_string()));
        }
    }

    let (kernel_height, kernel_width) = (kernel.dims()[2], kernel.dims()[3]);

    let mut expanded_image = intercalate_values(image, T::zero().unwrap(), stride.0, 2)?;
    expanded_image = intercalate_values(&expanded_image, T::zero().unwrap(), stride.1, 3)?;
    expanded_image = pad(&expanded_image, (kernel_height - 1, kernel_width - 1))?;

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
                padding.0..d - padding.0 + output_padding.0
            } else if i == 3 {
                padding.1..d - padding.1 + output_padding.1
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
        channel.reshape(&[kernel.dims()[2], kernel.dims()[3]]);
        inverted_kernels.push(channel);
    }

    let mut deconv_kernel =
        Tensor::new(Some(&inverted_kernels), &[inverted_kernels.len()])?.combine()?;
    deconv_kernel.reshape(kernel.dims());

    // tensorflow formatting patch
    if kernel.dims()[0] == sliced_expanded_image.dims()[1] {
        deconv_kernel.reshape(&[
            kernel.dims()[1],
            kernel.dims()[0],
            kernel.dims()[2],
            kernel.dims()[3],
        ]);
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

    let output = conv(&input, (0, 0), (1, 1))?;

    Ok(output)
}

/// Applies 2D sum pooling over a 4D tensor of shape B x C x H x W.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// * `pool_dims` - Tuple of pooling window size in x and y directions.
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
/// let pooled = sumpool::<i128>(&x, (0, 0), (1, 1), (2, 2)).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[11, 8, 8, 10]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
/// ```
pub fn sumpool<
    T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Sync + std::marker::Send,
>(
    image: &Tensor<T>,
    padding: (usize, usize),
    stride: (usize, usize),
    kernel_shape: (usize, usize),
) -> Result<Tensor<T>, TensorError> {
    if image.dims().len() != 4 {
        return Err(TensorError::DimMismatch("sumpool".to_string()));
    }
    let image_dims = image.dims();

    let (batch, image_channels, image_height, image_width) =
        (image_dims[0], image_dims[1], image_dims[2], image_dims[3]);

    let (output_channels, kernel_height, kernel_width) =
        (image_channels, kernel_shape.0, kernel_shape.1);

    let padded_image = pad::<T>(image, padding)?;

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    // calculate value of output
    let mut output: Tensor<T> =
        Tensor::new(None, &[batch, output_channels, vert_slides, horz_slides]).unwrap();

    let cartesian_coord = vec![
        (0..batch),
        (0..output_channels),
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
            let thesum = sum(&padded_image
                .get_slice(&[
                    b..b + 1,
                    i..i + 1,
                    rs..(rs + kernel_height),
                    cs..(cs + kernel_width),
                ])
                .unwrap())
            .unwrap();
            *o = thesum[0].clone();
        });

    Ok(output)
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
/// let pooled = max_pool2d::<i128>(&x, &(0, 0), &(1, 1), &(2, 2)).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[5, 4, 4, 6]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
///
/// let x = Tensor::<f32>::new(Some(&[-0.9180, -0.4702, -0.0882, -0.0885, 0.3940,
///                                  -0.4884, 0.1395,  1.7860, -0.9729,  1.5160, -0.3346,
///                                 -0.0601, -0.1140,  0.2522, -0.2938, -0.0355]), &[1,1,4,4]).unwrap();
/// let x = x.map(|x| F32(x));
/// let pooled = max_pool2d::<F32>(&x, &(0, 0), &(2, 2), &(2, 2)).unwrap();
/// let expected = Tensor::<f32>::new(Some(&[0.3940,  1.7860, 1.5160, -0.0355]), &[1, 1, 2, 2]).unwrap();
/// let expected = expected.map(|x| F32(x));
/// assert_eq!(pooled, expected);
/// ```
pub fn max_pool2d<T: TensorType + std::marker::Sync + std::marker::Send + std::cmp::Ord>(
    image: &Tensor<T>,
    padding: &(usize, usize),
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

    let vert_slides = (image_height + 2 * padding.0 - pool_dims.0) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - pool_dims.1) / stride.1 + 1;

    let mut output: Tensor<T> =
        Tensor::new(None, &[batch, input_channels, horz_slides, vert_slides]).unwrap();

    let cartesian_coord = vec![
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
pub fn dot<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    if (inputs.len() != 2) || (inputs[0].clone().len() != inputs[1].clone().len()) {
        return Err(TensorError::DimMismatch("dot".to_string()));
    }

    let (a, b): (Tensor<T>, Tensor<T>) = (inputs[0].clone(), inputs[1].clone());
    let res = a
        .iter()
        .zip(b)
        .fold(T::zero().unwrap(), |acc, (k, i)| acc + k.clone() * i);
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
/// let result = pad::<i128>(&x, (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[0, 0, 0, 0, 0, 0, 5, 2, 3, 0, 0, 0, 4, -1, 0, 0, 3, 1, 6, 0, 0, 0, 0, 0, 0]),
///     &[1, 1, 5, 5],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn pad<T: TensorType>(
    image: &Tensor<T>,
    padding: (usize, usize),
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
    let padded_height = height + 2 * padding.0;
    let padded_width = width + 2 * padding.1;

    let mut output =
        Tensor::<T>::new(None, &[batch_size, channels, padded_height, padded_width]).unwrap();

    for b in 0..batch_size {
        for channel in 0..channels {
            for row in 0..height {
                for col in 0..width {
                    output.set(
                        &[b, channel, row + padding.0, col + padding.1],
                        image.get(&[b, channel, row, col]).clone(),
                    );
                }
            }
        }
    }

    output.reshape(&[batch_size, channels, padded_height, padded_width]);
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
/// let result = concat(&[x, y], 0).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
/// assert_eq!(result, expected);
///
/// // 2D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[7, 8, 9]), &[3, 1]).unwrap();
/// let result = concat(&[x, y], 1).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 7, 3, 4, 8, 5, 6, 9]), &[3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// /// 4D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), &[2, 2, 2, 2]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[17, 18, 19, 20, 21, 22, 23, 14]), &[2, 2, 1, 2]).unwrap();
/// let result = concat(&[x, y], 2).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 17, 18, 5, 6, 7, 8, 19, 20, 9, 10, 11, 12, 21, 22, 13, 14, 15, 16, 23, 14]), &[2, 2, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // 5D example
/// let x = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), &[8, 1, 1, 1, 2]).unwrap();
/// let y = Tensor::<i128>::new(Some(&[17, 18, 19, 20, 21, 22, 23, 14]), &[4, 1, 1, 1, 2]).unwrap();
/// let result = concat(&[x, y], 0).unwrap();
///
/// let expected = Tensor::<i128>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 14]), &[12, 1, 1, 1, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// ```
///
/// # Errors
/// Returns a TensorError if the tensors in `inputs` have incompatible dimensions for concatenation along the specified `axis`.

pub fn concat<T: TensorType>(inputs: &[Tensor<T>], axis: usize) -> Result<Tensor<T>, TensorError> {
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

    output = output.enum_map(|i, _: T| {
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
    output.reshape(&output_size);

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
pub fn slice<T: TensorType>(
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
    /// let result = sigmoid(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 1, 1, 1]), &[2, 3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// let x = Tensor::<i128>::new(
    ///    Some(&[65536]),
    ///   &[1],
    /// ).unwrap();
    /// let result = sigmoid(&x, 65536, 256);
    /// let expected = Tensor::<i128>::new(Some(&[187]), &[1]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// /// assert_eq!(result, expected);
    /// let x = Tensor::<i128>::new(
    ///    Some(&[256]),
    ///   &[1],
    /// ).unwrap();
    /// let result = sigmoid(&x, 256, 256);
    /// let expected = Tensor::<i128>::new(Some(&[187]), &[1]).unwrap();
    ///
    /// ```
    pub fn sigmoid(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let kix = (*a_i as f64) / (scale_input as f64);
            let fout = (scale_output as f64) / (1.0 + (-kix).exp());
            let rounded = fout.round();
            output[i] = rounded as i128;
        }
        output
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
    /// let result = exp(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[7, 3269017, 7, 3, 3, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// let x = Tensor::<i128>::new(
    ///    Some(&[37, 12, 41]),
    ///   &[3],
    /// ).unwrap();
    /// let result = exp(&x, 512, 512);
    ///
    /// let expected = Tensor::<i128>::new(Some(&[550, 524, 555]), &[3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn exp(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let kix = (*a_i as f64) / (scale_input as f64);
            let fout = (scale_output as f64) * kix.exp();
            let rounded = fout.round();
            output[i] = rounded as i128;
        }
        output
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
    /// let result = ln(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[1, 3, 1, 0, 0, 8]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// let x = Tensor::<i128>::new(
    ///    Some(&[37, 12, 41]),
    ///   &[3],
    /// ).unwrap();
    /// let result = ln(&x, 512, 512);
    ///
    /// let expected = Tensor::<i128>::new(Some(&[-1345, -1922, -1293]), &[3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn ln(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let kix = (*a_i as f64) / (scale_input as f64);
            let fout = (scale_output as f64) * kix.ln();
            let rounded = fout.round();
            output[i] = rounded as i128;
        }
        output
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
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            if *a_i > 0 {
                output[i] = 1;
            } else if *a_i < 0 {
                output[i] = -1;
            } else {
                output[i] = 0;
            }
        }
        output
    }

    /// softmax layout
    pub fn multi_dim_softmax(
        a: &Tensor<i128>,
        scale_input: usize,
        scale_output: usize,
    ) -> (Tensor<i128>, Vec<Tensor<i128>>) {
        // we want this to be as small as possible so we set the output scale to 1
        let dims = a.dims();

        if dims.len() == 1 {
            return softmax(a, scale_input, scale_output);
        }

        let mut intermediate_values = vec![];

        let cartesian_coord = dims[..dims.len() - 1]
            .iter()
            .map(|x| 0..*x)
            .multi_cartesian_product()
            .collect::<Vec<_>>();

        let mut outputs = vec![];

        for coord in cartesian_coord {
            let mut sum_dims = vec![];
            for c in coord {
                sum_dims.push(c..c + 1);
            }
            sum_dims.push(0..dims[dims.len() - 1]);

            let softmax_input = a.get_slice(&sum_dims).unwrap();

            let res = softmax(&softmax_input, scale_input, scale_output);

            outputs.push(res.0);
            intermediate_values.extend(res.1);
        }

        let mut res = Tensor::new(Some(&outputs), &[outputs.len()])
            .unwrap()
            .combine()
            .unwrap();
        res.reshape(dims);

        (res, intermediate_values)
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
    ///     Some(&[2, 4, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = softmax(&x, 128, 128).0;
    /// // doubles the scale of the input
    /// let expected = Tensor::<i128>::new(Some(&[2730, 2772, 2730, 2709, 2709, 2688]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn softmax(
        a: &Tensor<i128>,
        scale_input: usize,
        scale_output: usize,
    ) -> (Tensor<i128>, Vec<Tensor<i128>>) {
        // the more accurate calculation is commented out and we implement as below so it matches the steps in layout
        let mut intermediate_values = vec![];

        let exp = exp(a, scale_input, scale_output);

        let sum = sum(&exp).unwrap();
        intermediate_values.push(sum.clone());
        let inv_denom = recip(&sum, scale_output.pow(2) as u32);

        ((exp * inv_denom).unwrap(), intermediate_values)
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
        let recip = recip(&t[0], scale as u32);
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
    /// let result = sqrt(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[2, 5, 3, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sqrt(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let kix = (*a_i as f64) / (scale_input as f64);
            let fout = (scale_output as f64) * kix.sqrt();
            let rounded = fout.round();
            output[i] = rounded as i128;
        }
        output
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
    /// let result = rsqrt(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[1, 0, 0, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn rsqrt(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let kix = (*a_i as f64) / (scale_input as f64);
            let fout = (scale_output as f64) * (1.0 / kix.sqrt());
            let rounded = fout.round();
            output[i] = rounded as i128;
        }
        output
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
    /// let result = cos(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[-83, 126, -18, 69, 69, 128]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn cos(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let cosz = (scale_output as f64) * z.cos();
            output[i] = cosz as i128;
        }

        output
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
    /// let result = acos(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[0, 0, 0, 0, 0, 201]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn acos(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let acos = (scale_output as f64) * z.acos();
            output[i] = acos as i128;
        }

        output
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
    /// let result = cosh(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[3495, 4608313557592, 190781, 197, 197, 128]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn cosh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let coshz = (scale_output as f64) * z.cosh();
            output[i] = coshz as i128;
        }

        output
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
    /// let result = acosh(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[264, 500, 354, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn acosh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let acoshz = (scale_output as f64) * z.acosh();
            output[i] = acoshz as i128;
        }

        output
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
    /// let result = sin(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[-96, -16, 126, 107, 107, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sin(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let sinz = (scale_output as f64) * z.sin();
            output[i] = sinz as i128;
        }

        output
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
    /// let result = asin(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(& [0, 0, 0, 201, 201, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn asin(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let asinz = (scale_output as f64) * z.asin();
            output[i] = asinz as i128;
        }

        output
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
    /// let result = sinh(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(& [3493, 4608313557592, 190781, 150, 150, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sinh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let asinz = (scale_output as f64) * z.sinh();
            output[i] = asinz as i128;
        }

        output
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
    /// let result = asinh(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[268, 500, 355, 112, 112, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn asinh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let asinhz = (scale_output as f64) * z.asinh();
            output[i] = asinhz as i128;
        }

        output
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
    /// let result = tan(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[148, -17, -870, 199, 199, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn tan(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let tanz = (scale_output as f64) * z.tan();
            output[i] = tanz as i128;
        }

        output
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
    /// let result = atan(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[169, 195, 185, 100, 100, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn atan(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let atanz = (scale_output as f64) * z.atan();
            output[i] = atanz as i128;
        }

        output
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
    /// let result = tanh(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[127, 128, 127, 97, 97, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn tanh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let tanhz: f64 = (scale_output as f64) * z.tanh();
            output[i] = tanhz as i128;
        }

        output
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
    /// let result = atanh(&x, 1, 128);
    /// let expected = Tensor::<i128>::new(Some(&[0, 0, 0, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn atanh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f64 / (scale_input as f64);
            let atanhz = (scale_output as f64) * z.atanh();
            output[i] = atanhz as i128;
        }

        output
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
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = erffunc(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[0, 1, 1, 0, 0, 0]), &[2, 3]).unwrap(); // TODO
    /// assert_eq!(result, expected);
    /// ```
    pub fn erffunc(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

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

        for i in 0..a.len() {
            let mut z = a[i] as f64 / (scale_input as f64);
            z = (scale_output as f64) * (erf(z));
            output[i] = z as i128;
        }
        output
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
    /// let result = leakyrelu(&x, 1, 0.1);
    /// let expected = Tensor::<i128>::new(Some(&[2, 15, 2, 1, 1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn leakyrelu(a: &Tensor<i128>, scale: usize, slope: f64) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            output[i] = if a_i < &0 {
                let d_inv_x = (slope) * (*a_i as f64) / (scale as f64);
                d_inv_x.round() as i128
            } else {
                let d_inv_x = (*a_i as f64) / (scale as f64);
                d_inv_x.round() as i128
            };
        }
        output
    }

    /// Elementwise applies prelu to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale` - Single value
    /// * `slopes` - Array of values
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::nonlinearities::prelu;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[-10, 15, 2, 1, 1, -5]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = prelu(&x, 1, &[0.1, 25.0]);
    /// let expected = Tensor::<i128>::new(Some(&[-1, 15, 2, 1, 1, -125]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn prelu(a: &Tensor<i128>, scale: usize, slopes: &[f64]) -> Tensor<i128> {
        if slopes.len() == 1 {
            return leakyrelu(a, scale, slopes[0]);
        } else {
            // assert number of slopes is equal to number of channels
            assert_eq!(slopes.len(), a.dims()[0])
        }
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            output[i] = if a_i < &0 {
                let slope_i: f64 = slopes[i / (a.dims()[1..].iter().product::<usize>())];
                let d_inv_x = (slope_i) * (*a_i as f64) / (scale as f64);
                d_inv_x.round() as i128
            } else {
                let d_inv_x = (*a_i as f64) / (scale as f64);
                d_inv_x.round() as i128
            };
        }
        output
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
        // calculate value of output
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let d_inv_x = (*a_i as f64) / denom;
            output[i] = d_inv_x.round() as i128;
        }
        output
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
    /// let k = 2_u32;
    /// let result = recip(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 2, 1, 0, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn recip(a: &Tensor<i128>, scale: u32) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let d_inv_x = (scale as f64) * (1_f64) / (*a_i as f64);
            output[i] = d_inv_x.round() as i128;
        }
        output
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
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            output[i] = i128::from((*a_i as f64 - b) > 0_f64);
        }
        output
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
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 0, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn less_than(a: &Tensor<i128>, b: f64) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            output[i] = i128::from((*a_i as f64 - b) < 0_f64);
        }
        output
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
    /// assert_eq!(dot(&[x, y]).unwrap(), expected);
    /// ```
    pub fn dot<T: TensorType + Mul<Output = T> + Add<Output = T>>(
        inputs: &[Tensor<T>; 2],
    ) -> Result<Tensor<T>, TensorError> {
        if inputs[0].clone().len() != inputs[1].clone().len() {
            return Err(TensorError::DimMismatch("dot".to_string()));
        }
        let (a, b): (Tensor<T>, Tensor<T>) = (inputs[0].clone(), inputs[1].clone());

        let transcript: Tensor<T> = a
            .iter()
            .zip(b)
            .scan(T::zero().unwrap(), |acc, (k, i)| {
                *acc = acc.clone() + k.clone() * i;
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
    /// let result = sum(&x).unwrap();
    /// let expected = Tensor::<i128>::new(
    ///     Some(&[2, 17, 19, 20, 21, 21]),
    ///     &[6],
    /// ).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sum<T: TensorType + Mul<Output = T> + Add<Output = T>>(
        a: &Tensor<T>,
    ) -> Result<Tensor<T>, TensorError> {
        let transcript: Tensor<T> = a
            .iter()
            .scan(T::zero().unwrap(), |acc, k| {
                *acc = acc.clone() + k.clone();
                Some(acc.clone())
            })
            .collect();

        Ok(transcript)
    }

    /// Matrix multiplies two 2D tensors.
    /// # Arguments
    ///
    /// * `inputs` - Vector of tensors of length 2
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::tensor::ops::accumulated::matmul;
    ///
    /// let x = Tensor::<i128>::new(
    ///     Some(&[5, 2, 3]),
    ///     &[3, 1],
    /// ).unwrap();
    /// let k = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = matmul(&[k, x]).unwrap();
    /// let expected = Tensor::<i128>::new(Some(&[10, 12, 18, 5, 7, 10]), &[2, 1, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn matmul<
        T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Send + std::marker::Sync,
    >(
        inputs: &[Tensor<T>; 2],
    ) -> Result<Tensor<T>, TensorError> {
        let (mut a, mut b) = (inputs[0].clone(), inputs[1].clone());

        if a.dims().len() == 1 {
            a.reshape(&[1, a.dims()[0]]);
        }
        if b.dims().len() == 1 {
            b.reshape(&[b.dims()[0], 1]);
        }

        if (a.dims()[a.dims().len() - 1] != b.dims()[a.dims().len() - 2])
            || (a.dims()[0..a.dims().len() - 2] != b.dims()[0..a.dims().len() - 2])
        {
            return Err(TensorError::DimMismatch("matmul".to_string()));
        }

        let mut dims = Vec::from(&a.dims()[0..a.dims().len() - 2]);
        dims.push(a.dims()[a.dims().len() - 2]);
        dims.push(b.dims()[a.dims().len() - 1]);
        // calculate value of output

        let indices = dims.iter().map(|d| 0..*d).collect::<Vec<_>>();

        let cartesian_product = indices
            .iter()
            .cloned()
            .multi_cartesian_product()
            .collect::<Vec<_>>();
        let mut transcripts = vec![Tensor::<T>::new(None, &[0])?; cartesian_product.len()];

        transcripts.par_iter_mut().enumerate().for_each(|(i, t)| {
            let row = cartesian_product[i][0..cartesian_product[i].len() - 1]
                .iter()
                .map(|&d| d..(d + 1))
                .collect::<Vec<_>>();
            let mut col = cartesian_product[i][0..cartesian_product[i].len()]
                .iter()
                .map(|&d| d..(d + 1))
                .collect::<Vec<_>>();
            col[cartesian_product[i].len() - 2] = 0..b.dims()[cartesian_product[i].len() - 2];
            let dot_transcript = dot(&[
                a.get_slice(&row[0..]).unwrap(),
                b.get_slice(&col[0..]).unwrap(),
            ])
            .unwrap();
            *t = dot_transcript;
        });

        let mut output = Tensor::new(Some(&transcripts), &[transcripts.len()])?.combine()?;
        output.reshape(&[dims.as_slice(), &[transcripts[0].len()]].concat());

        Ok(output)
    }
}
