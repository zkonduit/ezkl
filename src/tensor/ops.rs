use super::TensorError;
use crate::{
    fieldutils::IntegerRep,
    tensor::{Tensor, TensorType},
};
use itertools::Itertools;
use maybe_rayon::{iter::ParallelIterator, prelude::IntoParallelRefIterator};
pub use std::ops::{Add, Mul, Neg, Sub};

/// Trilu operation.
/// # Arguments
/// * `a` - Tensor
/// * `k` - i32
/// * `upper` - Boolean
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::trilu;
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[1, 3, 2],
/// ).unwrap();
/// let result = trilu(&a, 1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 2, 0, 0, 0, 0]), &[1, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 1, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[1, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 0, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 0, 4, 0, 0]), &[1, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 0, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 3, 4, 5, 6]), &[1, 3, 2]).unwrap();
/// assert_eq!(result, expected);  
///
/// let result = trilu(&a, -1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 0, 6]), &[1, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, -1, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 3, 0, 5, 6]), &[1, 3, 2]).unwrap();
/// assert_eq!(result, expected);  
///
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[1, 2, 3],
/// ).unwrap();
/// let result = trilu(&a, 1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 2, 3, 0, 0, 6]), &[1, 2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 1, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 0, 4, 5, 6]), &[1, 2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 0, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 0, 5, 6]), &[1, 2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 0, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 0, 4, 5, 0]), &[1, 2, 3]).unwrap();
/// assert_eq!(result, expected);  
///
/// let result = trilu(&a, -1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[1, 2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, -1, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 4, 0, 0]), &[1, 2, 3]).unwrap();
/// assert_eq!(result, expected);  
///
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9]),
/// &[1, 3, 3],
/// ).unwrap();
/// let result = trilu(&a, 1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 2, 3, 0, 0, 6, 0, 0, 0]), &[1, 3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 1, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 0, 4, 5, 6, 7, 8, 9]), &[1, 3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 0, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 0, 5, 6, 0, 0, 9]), &[1, 3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, 0, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 0, 4, 5, 0, 7, 8, 9]), &[1, 3, 3]).unwrap();
/// assert_eq!(result, expected);  
///
/// let result = trilu(&a, -1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 0, 8, 9]), &[1, 3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = trilu(&a, -1, false).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 4, 0, 0, 7, 8, 0]), &[1, 3, 3]).unwrap();
/// assert_eq!(result, expected);  
/// ```
pub fn trilu<T: TensorType + std::marker::Send + std::marker::Sync>(
    a: &Tensor<T>,
    k: i32,
    upper: bool,
) -> Result<Tensor<T>, TensorError> {
    let mut output = a.clone();

    // Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
    // The attribute “upper” determines whether the upper or lower part is retained.
    // If set to true, the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
    // Default value for the “upper” attribute is true. Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions.
    // The upper triangular part consists of the elements on and above the given diagonal (k).
    // The lower triangular part consists of elements on and below the diagonal. All other elements in the matrix are set to zero.

    let batch_dims = &a.dims()[0..a.dims().len() - 2];
    let batch_cartiesian = batch_dims.iter().map(|d| 0..*d).multi_cartesian_product();

    for b in batch_cartiesian {
        for i in 0..a.dims()[1] {
            for j in 0..a.dims()[2] {
                let mut coord = b.clone();
                coord.push(i);
                coord.push(j);
                // If k = 0, the triangular part on and above/below the main diagonal is retained.

                if upper {
                    // If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
                    if (j as i32) < (i as i32) + k {
                        output.set(&coord, T::zero().ok_or(TensorError::Unsupported)?);
                    }
                } else {
                    // If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
                    if (j as i32) > (i as i32) + k {
                        output.set(&coord, T::zero().ok_or(TensorError::Unsupported)?);
                    }
                }
            }
        }
    }

    Ok(output)
}

/// Resize using nearest neighbour interpolation.
/// # Arguments
/// * `a` - Tensor
/// * `scales` - Vector of scales
/// # Examples
/// ```
///
///
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = resize(&a, &[1, 2]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]), &[2, 6]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap();
/// let result = resize(&a, &[2, 2]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4, 4, 5, 5, 6, 6]), &[4, 6]).unwrap();
/// assert_eq!(result, expected);
///
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::resize;
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4]),
/// &[2, 2],
/// ).unwrap();
/// let result = resize(&a, &[2, 2]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]), &[4, 4]).unwrap();
/// assert_eq!(result, expected);
///
///
/// let a = Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[3, 2],
/// ).unwrap();
/// let result = resize(&a, &[2, 3]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 6, 6]), &[6, 6]).unwrap();
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

/// Adds multiple tensors.
/// # Arguments
///
/// * `t` - Vector of tensors
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::add;
/// let x = Tensor::<IntegerRep>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<IntegerRep>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = add(&[x, k]).unwrap();
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
/// let result = add(&[x, k]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, 3, 4, 3, 3, 3]), &[2, 3]).unwrap();
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
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::sub;
/// let x = Tensor::<IntegerRep>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<IntegerRep>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = sub(&[x, k]).unwrap();
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
/// let result = sub(&[x, k]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, -1, 0, -1, -1, -1]), &[2, 3]).unwrap();
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

/// Elementwise multiplies multiple tensors.
/// # Arguments
///
/// * `t` - Tensors
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::mult;
/// let x = Tensor::<IntegerRep>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<IntegerRep>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = mult(&[x, k]).unwrap();
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
/// let result = mult(&[x, k]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
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

/// Downsamples a tensor along a dimension.
/// # Arguments
/// * `input` - Tensor
/// * `dim` - Dimension to downsample along
/// * `stride` - Stride to downsample by
/// *  `modulo` - Modulo to downsample by
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::downsample;
/// let x = Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 4, 5, 6]),
///  &[2, 3],
/// ).unwrap();
/// let result = downsample(&x, 0, 1, 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, 5, 6]), &[1, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = downsample(&x, 1, 2, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 4, 6]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = downsample(&x, 1, 2, 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 5]), &[2, 1]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = downsample(&x, 1, 2, 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[3, 6]), &[2, 1]).unwrap();
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
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::gather;
/// let x = Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 4, 5, 6]),
///   &[2, 3],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///   Some(&[0, 1]),
///  &[2],
/// ).unwrap();
/// let result = gather(&x, &index, 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 4, 5]), &[2, 2]).unwrap();
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
/// use ezkl::fieldutils::IntegerRep;
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
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::gather_elements;
/// let x = Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 4]),
///   &[2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///   Some(&[0, 0, 1, 0]),
///  &[2, 2],
/// ).unwrap();
/// let result = gather_elements(&x, &index, 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 4, 3]), &[2, 2]).unwrap();
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
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::gather_nd;
/// let x = Tensor::<IntegerRep>::new(
///   Some(&[0, 1, 2, 3]),
/// &[2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
/// Some(&[0, 0, 1, 1]),
/// &[2, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 3]), &[2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
/// Some(&[1, 0]),
/// &[2, 1],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 3, 0, 1]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<IntegerRep>::new(
///  Some(&[0, 1, 2, 3, 4, 5, 6, 7]),
/// &[2, 2, 2],
/// ).unwrap();
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 1, 0]),
/// &[2, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 3, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 1, 0]),
/// &[2, 1, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 3, 4, 5]), &[2, 1, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
/// Some(&[1, 0]),
/// &[2, 1],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 3, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]),
/// &[2, 2, 3],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 3, 4, 5]), &[2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 0, 0, 1, 1, 1, 0]),
/// &[2, 2, 2],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 3, 0, 1, 6, 7, 4, 5]), &[2, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let index = Tensor::<usize>::new(
///  Some(&[0, 1, 0, 1, 1, 1]),
/// &[2, 3],
/// ).unwrap();
/// let result = gather_nd(&x, &index, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 7]), &[2]).unwrap();
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
    let last_value = index_dims
        .last()
        .ok_or(TensorError::DimMismatch("gather_nd".to_string()))?;
    if last_value > &(input_dims.len() - batch_dims) {
        return Err(TensorError::DimMismatch("gather_nd".to_string()));
    }

    let output_size =
    // If indices_shape[-1] == r-b, since the rank of indices is q,
    // indices can be thought of as N (q-b-1)-dimensional tensors containing 1-D tensors of dimension r-b,
    // where N is an integer equals to the product of 1 and all the elements in the batch dimensions of the indices_shape.
    // Let us think of each such r-b ranked tensor as indices_slice.
    // Each scalar value corresponding to data[0:b-1,indices_slice] is filled into
    // the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
     // if indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor containing 1-D tensors of dimension < r-b.
    // Let us think of each such tensors as indices_slice.
    // Each tensor slice corresponding to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
    {
        let output_rank = input_dims.len() + index_dims.len() - 1 - batch_dims - last_value;

        let mut dims = index_dims[..index_dims.len() - 1].to_vec();
        let input_offset = batch_dims + last_value;
        dims.extend(input_dims[input_offset..input_dims.len()].to_vec());

        assert_eq!(output_rank, dims.len());
        dims

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
                    let slice = coord
                        .iter()
                        .map(|x| *x..*x + 1)
                        .chain(batch_coord.iter().map(|x| *x..*x + 1))
                        .collect::<Vec<_>>();

                    let index_slice = index_slice
                        .get_slice(&slice)
                        .unwrap()
                        .iter()
                        .map(|x| *x..*x + 1)
                        .collect::<Vec<_>>();

                    input_slice.get_slice(&index_slice).unwrap()
                })
                .collect::<Tensor<_>>();

            output.combine()
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut outputs = outputs.into_iter().flatten().collect::<Tensor<_>>();

    outputs.reshape(&output_size)?;

    Ok(outputs)
}

/// Scatter ND.
/// This operator is the inverse of GatherND.
/// # Arguments
/// * `input` - Tensor
/// * `index` - Tensor of indices to scatter
/// * `src` - Tensor of src
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::scatter_nd;
/// let x = Tensor::<IntegerRep>::new(
///  Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
/// &[8],
/// ).unwrap();
///
/// let index = Tensor::<usize>::new(
/// Some(&[4, 3, 1, 7]),
/// &[4, 1],
/// ).unwrap();
/// let src = Tensor::<IntegerRep>::new(
/// Some(&[9, 10, 11, 12]),
/// &[4],
/// ).unwrap();
/// let result = scatter_nd(&x, &index, &src).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 11, 3, 10, 9, 6, 7, 12]), &[8]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<IntegerRep>::new(
///  Some(&[1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
///         1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
///         8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8,
///         8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
/// &[4, 4, 4],
/// ).unwrap();
///
/// let index = Tensor::<usize>::new(
///   Some(&[0, 2]),
/// &[2, 1],
/// ).unwrap();
///
/// let src = Tensor::<IntegerRep>::new(
///  Some(&[5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
///         1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
///   ]),
/// &[2, 4, 4],
/// ).unwrap();
///
/// let result = scatter_nd(&x, &index, &src).unwrap();
///
/// let expected = Tensor::<IntegerRep>::new(
///  Some(&[5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,
///         1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1,
///         1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
///         8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
/// &[4, 4, 4],
/// ).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<IntegerRep>::new(
///  Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
/// &[2, 4],
/// ).unwrap();
///
/// let index = Tensor::<usize>::new(
/// Some(&[0, 1]),
/// &[2, 1],
/// ).unwrap();
/// let src = Tensor::<IntegerRep>::new(
/// Some(&[9, 10]),
/// &[2],
/// ).unwrap();
/// let result = scatter_nd(&x, &index, &src).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[9, 9, 9, 9, 10, 10, 10, 10]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<IntegerRep>::new(
///  Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
/// &[2, 4],
/// ).unwrap();
///
/// let index = Tensor::<usize>::new(
/// Some(&[0, 1]),
/// &[1, 1, 2],
/// ).unwrap();
/// let src = Tensor::<IntegerRep>::new(
/// Some(&[9]),
/// &[1, 1],
/// ).unwrap();
/// let result = scatter_nd(&x, &index, &src).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 9, 3, 4, 5, 6, 7, 8]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ````
///
pub fn scatter_nd<T: TensorType + Send + Sync>(
    input: &Tensor<T>,
    index: &Tensor<usize>,
    src: &Tensor<T>,
) -> Result<Tensor<T>, TensorError> {
    // Calculate the output tensor size
    let index_dims = index.dims().to_vec();
    let input_dims = input.dims().to_vec();
    let last_value = index_dims
        .last()
        .ok_or(TensorError::DimMismatch("scatter_nd".to_string()))?;
    if last_value > &input_dims.len() {
        return Err(TensorError::DimMismatch("scatter_nd".to_string()));
    }

    let mut output = input.clone();

    let cartesian_coord = index_dims[0..index_dims.len() - 1]
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    cartesian_coord
        .iter()
        .map(|coord| {
            let slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let index_val = index.get_slice(&slice)?;
            let index_slice = index_val.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let src_val = src.get_slice(&slice)?;
            output.set_slice(&index_slice, &src_val)?;
            Ok(())
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(output)
}

/// Abs a tensor.
/// # Arguments
/// * `a` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::abs;
/// let x = Tensor::<IntegerRep>::new(
///    Some(&[-2, 15, 2, -1, 1, 0]),
/// &[2, 3],
/// ).unwrap();
/// let result = abs(&x).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 15, 2, 1, 1, 0]), &[2, 3]).unwrap();
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

/// Intercalates values into a tensor along a given axis.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::intercalate_values;
///
/// let tensor = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
/// let result = intercalate_values(&tensor, 0, 2, 1).unwrap();
///
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 2, 3, 0, 4]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// let result = intercalate_values(&expected, 0, 2, 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 2, 0, 0, 0, 3, 0, 4]), &[3, 3]).unwrap();
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
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::one_hot;
/// let tensor = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4]), &[2, 2]).unwrap();
/// let result = one_hot(&tensor, 5, 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 1, 0, 0, 0,
///                                           0, 0, 1, 0, 0,
///                                           0, 0, 0, 1, 0,
///                                           0, 0, 0, 0, 1]), &[2, 2, 5]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn one_hot(
    tensor: &Tensor<IntegerRep>,
    num_classes: usize,
    axis: usize,
) -> Result<Tensor<IntegerRep>, TensorError> {
    let mut output_dims = tensor.dims().to_vec();
    output_dims.insert(axis, num_classes);

    let mut output: Tensor<IntegerRep> = Tensor::new(None, &output_dims)?;

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

/// Pads a ND tensor of shape `B x C x H x D1 x D2 x ...` along all dimensions.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::pad;
///
/// let x = Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap();
/// let result = pad::<IntegerRep>(&x, vec![(0, 0), (0, 0), (1, 1), (1, 1)], 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[0, 0, 0, 0, 0, 0, 5, 2, 3, 0, 0, 0, 4, -1, 0, 0, 3, 1, 6, 0, 0, 0, 0, 0, 0]),
///     &[1, 1, 5, 5],
/// ).unwrap();
/// assert_eq!(result, expected);
///
/// let result = pad::<IntegerRep>(&x, vec![(1, 1), (1, 1)], 2).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn pad<T: TensorType>(
    image: &Tensor<T>,
    padding: Vec<(usize, usize)>,
    offset: usize,
) -> Result<Tensor<T>, TensorError> {
    let padded_dims = image.dims()[offset..]
        .iter()
        .enumerate()
        .map(|(i, d)| d + padding[i].0 + padding[i].1)
        .collect::<Vec<_>>();

    let mut output_dims = image.dims()[..offset].to_vec();
    output_dims.extend(padded_dims);

    let mut output = Tensor::<T>::new(None, &output_dims).unwrap();

    let cartesian_coord = image
        .dims()
        .iter()
        .map(|d| (0..*d))
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    for coord in cartesian_coord {
        let rest = &coord[offset..];
        let mut padded_res = coord[..offset].to_vec();
        padded_res.extend(rest.iter().zip(padding.iter()).map(|(c, p)| c + p.0));
        let image_val = image.get(&coord);
        output.set(&padded_res, image_val);
    }

    output.reshape(&output_dims)?;
    Ok(output)
}

/// Concatenates a list of tensors along a specified axis.
/// # Arguments
/// * `inputs` - A slice of tensors to concatenate.
/// * `axis` - The axis along which to concatenate the tensors.
///
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::concat;
/// // tested against pytorch outputs for reference :)
///
/// // 1D example
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3]), &[3]).unwrap();
/// let y = Tensor::<IntegerRep>::new(Some(&[4, 5, 6]), &[3]).unwrap();
/// let result = concat(&[&x, &y], 0).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
/// assert_eq!(result, expected);
///
/// // 2D example
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let y = Tensor::<IntegerRep>::new(Some(&[7, 8, 9]), &[3, 1]).unwrap();
/// let result = concat(&[&x, &y], 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 7, 3, 4, 8, 5, 6, 9]), &[3, 3]).unwrap();
/// assert_eq!(result, expected);
///
/// /// 4D example
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), &[2, 2, 2, 2]).unwrap();
/// let y = Tensor::<IntegerRep>::new(Some(&[17, 18, 19, 20, 21, 22, 23, 14]), &[2, 2, 1, 2]).unwrap();
/// let result = concat(&[&x, &y], 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 17, 18, 5, 6, 7, 8, 19, 20, 9, 10, 11, 12, 21, 22, 13, 14, 15, 16, 23, 14]), &[2, 2, 3, 2]).unwrap();
/// assert_eq!(result, expected);
///
///
/// // 5D example
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), &[8, 1, 1, 1, 2]).unwrap();
/// let y = Tensor::<IntegerRep>::new(Some(&[17, 18, 19, 20, 21, 22, 23, 14]), &[4, 1, 1, 1, 2]).unwrap();
/// let result = concat(&[&x, &y], 0).unwrap();
///
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 14]), &[12, 1, 1, 1, 2]).unwrap();
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
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::tensor::ops::slice;
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let result = slice(&x, &0, &1, &2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[3, 4]), &[1, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[3, 2]).unwrap();
/// let result = slice(&x, &1, &1, &2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 4, 6]), &[3, 1]).unwrap();
/// assert_eq!(result, expected);
///
/// let x = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), &[2, 2, 3]).unwrap();
/// let result = slice(&x, &2, &1, &2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, 5, 8, 11]), &[2, 2, 1]).unwrap();
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
    /// use ezkl::fieldutils::IntegerRep;
    ///
    /// use ezkl::tensor::ops::nonlinearities::ceil;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[1, 2, 3, 4, 5, 6]),
    ///  &[3, 2],
    /// ).unwrap();
    /// let result = ceil(&x, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[2, 2, 4, 4, 6, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn ceil(a: &Tensor<IntegerRep>, scale: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.ceil() * scale;
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::floor;
    /// let x = Tensor::<IntegerRep>::new(
    ///   Some(&[1, 2, 3, 4, 5, 6]),
    ///  &[3, 2],
    /// ).unwrap();
    /// let result = floor(&x, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 2, 2, 4, 4, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn floor(a: &Tensor<IntegerRep>, scale: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.floor() * scale;
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::round;
    /// let x = Tensor::<IntegerRep>::new(
    ///   Some(&[1, 2, 3, 4, 5, 6]),
    /// &[3, 2],
    /// ).unwrap();
    /// let result = round(&x, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[2, 2, 4, 4, 6, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn round(a: &Tensor<IntegerRep>, scale: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.round() * scale;
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::round_half_to_even;
    /// let x = Tensor::<IntegerRep>::new(
    ///   Some(&[1, 2, 3, 4, 5, 6]),
    /// &[3, 2],
    /// ).unwrap();
    /// let result = round_half_to_even(&x, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 2, 4, 4, 4, 6]), &[3, 2]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn round_half_to_even(a: &Tensor<IntegerRep>, scale: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale;
            let rounded = kix.round_ties_even() * scale;
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::pow;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 15, 2, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = pow(&x, 1.0, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 225, 4, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn pow(a: &Tensor<IntegerRep>, scale_input: f64, power: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let kix = scale_input * (kix).powf(power);
            let rounded = kix.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
        })
        .unwrap()
    }

    /// Applies Kronecker delta to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::kronecker_delta;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 15, 2, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = kronecker_delta(&x);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 0, 0, 1]), &[2, 3]).unwrap();
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::sigmoid;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sigmoid(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 1, 1, 1]), &[2, 3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[65536]),
    ///   &[1],
    /// ).unwrap();
    /// let result = sigmoid(&x, 65536.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[47911]), &[1]).unwrap();
    /// assert_eq!(result, expected);
    ///
    /// /// assert_eq!(result, expected);
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[256]),
    ///   &[1],
    /// ).unwrap();
    /// let result = sigmoid(&x, 256.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[187]), &[1]).unwrap();
    ///
    /// ```
    pub fn sigmoid(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input / (1.0 + (-kix).exp());
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
        })
        .unwrap()
    }

    /// Elementwise applies hardswish to a tensor of integers.
    /// Hardswish is defined as:
    // Hardswish(x)={0if x≤−3,xif x≥+3,x⋅(x+3)/6otherwise
    // Hardswish(x)=⎩
    // ⎨
    // ⎧​0xx⋅(x+3)/6​if x≤−3,if x≥+3,otherwise​
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::hardswish;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[-12, -3, 2, 1, 1, 15]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = hardswish(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 2, 1, 1, 15]), &[2, 3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    ///
    /// ```
    pub fn hardswish(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let res = if kix <= -3.0 {
                0.0
            } else if kix >= 3.0 {
                kix
            } else {
                kix * (kix + 3.0) / 6.0
            };
            let rounded = (res * scale_input).round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::exp;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = exp(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[7, 3269017, 7, 3, 3, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[37, 12, 41]),
    ///   &[3],
    /// ).unwrap();
    /// let result = exp(&x, 512.0);
    ///
    /// let expected = Tensor::<IntegerRep>::new(Some(&[550, 524, 555]), &[3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn exp(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.exp();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::ln;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 15, 2, 1, 1, 3000]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = ln(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 3, 1, 0, 0, 8]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    ///
    ///
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[37, 12, 41]),
    ///   &[3],
    /// ).unwrap();
    /// let result = ln(&x, 512.0);
    ///
    /// let expected = Tensor::<IntegerRep>::new(Some(&[-1345, -1922, -1293]), &[3]).unwrap();
    ///
    /// assert_eq!(result, expected);
    /// ```
    pub fn ln(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.ln();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
        })
        .unwrap()
    }

    /// Elementwise applies sign to a tensor of integers.
    /// # Arguments
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::sign;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[-2, 15, 2, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = sign(&x);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[-1, 1, 1, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sign(a: &Tensor<IntegerRep>) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(a_i.signum()))
            .unwrap()
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::sqrt;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sqrt(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[2, 5, 3, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sqrt(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.sqrt();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::rsqrt;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[4, 25, 8, 1, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = rsqrt(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 0, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn rsqrt(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input / (kix.sqrt() + f64::EPSILON);
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::cos;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = cos(&x, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(& [-1, 2, -1, 2, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn cos(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.cos();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::acos;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = acos(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 0, 0, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn acos(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.acos();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::cosh;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = cosh(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[27, 36002449669, 1490, 2, 2, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn cosh(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.cosh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::acosh;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = acosh(&x, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(& [2, 4, 3, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn acosh(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.acosh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::sin;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = sin(&x, 128.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sin(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.sin();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::asin;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = asin(&x, 128.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(& [4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn asin(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.asin();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::sinh;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = sinh(&x, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[7, 268337, 55, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sinh(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.sinh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::asinh;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = asinh(&x, 128.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn asinh(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.asinh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::tan;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = tan(&x, 64.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 26, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn tan(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.tan();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::atan;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[4, 25, 8, 1, 1, 0]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let result = atan(&x, 128.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn atan(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.atan();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::tanh;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = tanh(&x, 128.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 25, 8, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn tanh(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.tanh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::atanh;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[4, 25, 8, 2, 2, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = atanh(&x, 32.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4, 34, 8, 2, 2, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn atanh(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let kix = (a_i as f64) / scale_input;
            let fout = scale_input * kix.atanh();
            let rounded = fout.round();
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::erffunc;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[5, 28, 9, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = erffunc(&x, 128.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[6, 31, 10, 1, 1, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn erffunc(a: &Tensor<IntegerRep>, scale_input: f64) -> Tensor<IntegerRep> {
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
            Ok::<_, TensorError>(rounded as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::leakyrelu;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 15, 2, 1, 1, -5]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = leakyrelu(&x, 0.1);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[2, 15, 2, 1, 1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn leakyrelu(a: &Tensor<IntegerRep>, slope: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let rounded = if a_i < 0 {
                let d_inv_x = (slope) * (a_i as f64);
                d_inv_x.round() as IntegerRep
            } else {
                let d_inv_x = a_i as f64;
                d_inv_x.round() as IntegerRep
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::max;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 15, 2, 1, 1, -5]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = max(&x, 1.0, 1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[2, 15, 2, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn max(a: &Tensor<IntegerRep>, scale_input: f64, threshold: f64) -> Tensor<IntegerRep> {
        // calculate value of output
        a.par_enum_map(|_, a_i| {
            let d_inv_x = (a_i as f64) / scale_input;
            let rounded = if d_inv_x <= threshold {
                (threshold * scale_input).round() as IntegerRep
            } else {
                (d_inv_x * scale_input).round() as IntegerRep
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::min;
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 15, 2, 1, 1, -5]),
    ///   &[2, 3],
    /// ).unwrap();
    /// let result = min(&x, 1.0, 2.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[2, 2, 2, 1, 1, -5]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn min(a: &Tensor<IntegerRep>, scale_input: f64, threshold: f64) -> Tensor<IntegerRep> {
        // calculate value of output
        a.par_enum_map(|_, a_i| {
            let d_inv_x = (a_i as f64) / scale_input;
            let rounded = if d_inv_x >= threshold {
                (threshold * scale_input).round() as IntegerRep
            } else {
                (d_inv_x * scale_input).round() as IntegerRep
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::const_div;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = const_div(&x, k);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 4, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn const_div(a: &Tensor<IntegerRep>, denom: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let d_inv_x = (a_i as f64) / (denom);
            Ok::<_, TensorError>(d_inv_x.round() as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::recip;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2_f64;
    /// let result = recip(&x, 1.0, k);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 2, 1, 0, 2, 2]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn recip(a: &Tensor<IntegerRep>, input_scale: f64, out_scale: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| {
            let rescaled = (a_i as f64) / input_scale;
            let denom = (1_f64) / (rescaled + f64::EPSILON);
            let d_inv_x = out_scale * denom;
            Ok::<_, TensorError>(d_inv_x.round() as IntegerRep)
        })
        .unwrap()
    }

    /// Elementwise inverse.
    /// # Arguments
    /// * `out_scale` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::zero_recip;
    /// let k = 2_f64;
    /// let result = zero_recip(1.0);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[4503599627370496]), &[1]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn zero_recip(out_scale: f64) -> Tensor<IntegerRep> {
        let a = Tensor::<IntegerRep>::new(Some(&[0]), &[1]).unwrap();

        a.par_enum_map(|_, a_i| {
            let rescaled = a_i as f64;
            let denom = (1_f64) / (rescaled + f64::EPSILON);
            let d_inv_x = out_scale * denom;
            Ok::<_, TensorError>(d_inv_x.round() as IntegerRep)
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::greater_than;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = greater_than(&x, k);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 1, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn greater_than(a: &Tensor<IntegerRep>, b: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(IntegerRep::from((a_i as f64 - b) > 0_f64)))
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::greater_than_equal;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = greater_than_equal(&x, k);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 1, 1, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn greater_than_equal(a: &Tensor<IntegerRep>, b: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(IntegerRep::from((a_i as f64 - b) >= 0_f64)))
            .unwrap()
    }

    /// Elementwise less than
    /// # Arguments
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::less_than;
    ///
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 1, 2, 7, 1, 1]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    ///
    /// let result = less_than(&x, k);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[0, 1, 0, 0, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn less_than(a: &Tensor<IntegerRep>, b: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(IntegerRep::from((a_i as f64 - b) < 0_f64)))
            .unwrap()
    }

    /// Elementwise less than
    /// # Arguments
    /// * `a` - Tensor
    /// * `b` - Single value
    /// # Examples
    /// ```
    /// use ezkl::tensor::Tensor;
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::nonlinearities::less_than_equal;
    ///
    /// let x = Tensor::<IntegerRep>::new(
    ///    Some(&[2, 1, 2, 7, 1, 1]),
    ///  &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    ///
    /// let result = less_than_equal(&x, k);
    /// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 0, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn less_than_equal(a: &Tensor<IntegerRep>, b: f64) -> Tensor<IntegerRep> {
        a.par_enum_map(|_, a_i| Ok::<_, TensorError>(IntegerRep::from((a_i as f64 - b) <= 0_f64)))
            .unwrap()
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::accumulated::dot;
    ///
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[5, 2]),
    ///     &[2],
    /// ).unwrap();
    /// let y = Tensor::<IntegerRep>::new(
    ///     Some(&[5, 5]),
    ///     &[2],
    /// ).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::accumulated::sum;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sum(&x, 1).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(
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
    /// use ezkl::fieldutils::IntegerRep;
    /// use ezkl::tensor::ops::accumulated::prod;
    /// let x = Tensor::<IntegerRep>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = prod(&x, 1).unwrap();
    /// let expected = Tensor::<IntegerRep>::new(
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
