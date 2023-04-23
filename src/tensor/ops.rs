use super::TensorError;
use crate::tensor::{Tensor, TensorType};
use itertools::Itertools;
use puruspe::erf;
use rayon::{
    iter::IndexedParallelIterator, iter::IntoParallelRefMutIterator, iter::ParallelIterator,
};
pub use std::ops::{Add, Div, Mul, Sub};

/// Matrix multiplies two 2D tensors (and adds an offset).
/// # Arguments
///
/// * `inputs` - A vector of tensors holding in order: input data, affine kernel, convolution bias.
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::affine;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 2, 1, 1]),
///     &[3, 4],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[0, 0]),
///     &[2],
/// ).unwrap();
/// let result = affine(&[x, k, b]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn affine<
    T: TensorType + Mul<Output = T> + Add<Output = T> + std::marker::Send + std::marker::Sync,
>(
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    let (mut input, kernel, bias) = (inputs[0].clone(), inputs[1].clone(), inputs[2].clone());
    if (inputs.len() != 3)
        || (bias.dims()[0] != kernel.dims()[0])
        || (input.dims()[0] != kernel.dims()[1])
    {
        return Err(TensorError::DimMismatch("affine".to_string()));
    }

    // does matrix to vector multiplication
    if input.dims().len() == 1 {
        input.reshape(&[input.dims()[0], 1])
    }

    // calculate value of output
    let mut output: Tensor<T> = matmul(&[kernel.clone(), input.clone()])?;

    for i in 0..kernel.dims()[0] {
        for j in 0..input.dims()[1] {
            output.set(&[i, j], output.get(&[i, j]) + bias[i].clone());
        }
    }
    // does matrix to vector multiplication
    if output.dims()[1] == 1 {
        output.flatten();
    }
    Ok(output)
}

/// Scales and shifts a tensor.
/// Given inputs (x,k,b) computes k*x + b elementwise
/// # Arguments
///
/// * `inputs` - Vector of tensors of length 2
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::scale_and_shift;
///
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = scale_and_shift(&[x, k, b]).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[6, 2, 6, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn scale_and_shift<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &[Tensor<T>],
) -> Result<Tensor<T>, TensorError> {
    if (inputs.len() != 3)
        || (inputs[1].dims() != inputs[2].dims())
        || (inputs[0].dims() != inputs[1].dims())
    {
        return Err(TensorError::DimMismatch("scale and shift".to_string()));
    }
    let (input, kernel, bias) = (inputs[0].clone(), inputs[1].clone(), inputs[2].clone());
    let mut output: Tensor<T> = input;
    for (i, bias_i) in bias.iter().enumerate() {
        output[i] = kernel[i].clone() * output[i].clone() + bias_i.clone()
    }
    Ok(output)
}

/// Matrix multiplies two 2D tensors.
/// # Arguments
///
/// * `inputs` - Vector of tensors of length 2
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::matmul;
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
            let prod = dot(&vec![
                &a.get_slice(&row[0..]).unwrap(),
                &b.get_slice(&col[0..]).unwrap(),
            ])
            .unwrap();

            *o = prod[0].clone();
        });

    Ok(output)
}

/// Adds multiple tensors.
/// # Arguments
///
/// * `t` - Vector of tensors
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::add;
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
pub fn add<T: TensorType + Add<Output = T>>(t: &[Tensor<T>]) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        output = (output + e.clone())?;
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
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::sub;
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
pub fn sub<T: TensorType + Sub<Output = T>>(t: &[Tensor<T>]) -> Result<Tensor<T>, TensorError> {
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
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::mult;
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
pub fn mult<T: TensorType + Mul<Output = T>>(t: &[Tensor<T>]) -> Result<Tensor<T>, TensorError> {
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
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::rescale;
/// let x = Tensor::<i128>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = rescale(&x, k).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn rescale<T: TensorType + Add<Output = T>>(
    a: &Tensor<T>,
    mult: usize,
) -> Result<Tensor<T>, TensorError> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();
    for (i, a_i) in a.iter().enumerate() {
        for _ in 1..mult {
            output[i] = output[i].clone() + a_i.clone();
        }
    }
    Ok(output)
}

/// Sums a tensor.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::sum;
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

/// Sums a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::sum_axes;
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

    if axes.len() == 0 {
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
        for i in 0..a.dims().len() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(coord[i]..coord[i] + 1);
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
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::min_axes;
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

    if axes.len() == 0 {
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
        for i in 0..a.dims().len() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(coord[i]..coord[i] + 1);
            }
        }

        res.set(coord, a.get_slice(&sum_dims)?.iter().min().unwrap().clone());
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
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::max_axes;
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

    if axes.len() == 0 {
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
        for i in 0..a.dims().len() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(coord[i]..coord[i] + 1);
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
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::conv;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
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
/// let expected = Tensor::<i128>::new(Some(&[31, 16, 8, 26]), &[1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test single channel
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[2, 3, 3],
/// ).unwrap();
/// let k = Tensor::<i128>::new(
///     Some(&[5, 1, 1, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[0]),
///     &[1],
/// ).unwrap();
///
/// let result = conv::<i128>(&[x, k, b], (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[62, 32, 16, 52]), &[1, 2, 2]).unwrap();
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
    let (image, mut kernel) = (inputs[0].clone(), inputs[1].clone());

    if (image.dims().len() != 3)
        || (kernel.dims().len() != 4)
        || ((image.dims()[0] != kernel.dims()[1]) && (kernel.dims()[1] != 1))
    {
        return Err(TensorError::DimMismatch("conv".to_string()));
    }

    if kernel.dims()[1] == 1 && kernel.dims()[1] != image.dims()[0] {
        kernel = kernel.repeat_rows(image.dims()[0])?;
        kernel.reshape(&[
            kernel.dims()[0],
            image.dims()[0],
            kernel.dims()[2],
            kernel.dims()[3],
        ]);
    }

    let image_dims = image.dims();
    let kernel_dims = kernel.dims();

    if has_bias {
        let bias = inputs[2].clone();
        if (bias.dims().len() != 1) || (bias.dims()[0] != kernel.dims()[0]) {
            return Err(TensorError::DimMismatch("conv bias".to_string()));
        }
    }

    let (output_channels, input_channels, kernel_height, kernel_width) = (
        kernel_dims[0],
        kernel_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T>(&image, padding)?;

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    // calculate value of output
    let mut output: Tensor<T> =
        Tensor::new(None, &[output_channels, vert_slides, horz_slides]).unwrap();

    let cartesian_coord = vec![(0..output_channels), (0..vert_slides), (0..horz_slides)]
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, o)| {
            let coord = &cartesian_coord[flat_index];
            let (i, j, k) = (coord[0], coord[1], coord[2]);
            let rs = j * stride.0;
            let cs = k * stride.1;

            let mut res = dot(&vec![
                &kernel.get_slice(&[i..i + 1]).unwrap().clone(),
                &padded_image
                    .get_slice(&[
                        0..input_channels,
                        rs..(rs + kernel_height),
                        cs..(cs + kernel_width),
                    ])
                    .unwrap(),
            ])
            .unwrap();

            if has_bias {
                // increment result by the bias
                res[0] = res[0].clone() + inputs[2][i].clone();
            }

            *o = res[0].clone();
        });

    Ok(output)
}

/// Applies 2D sum pooling over a 3D tensor of shape C x H x W.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// * `pool_dims` - Tuple of pooling window size in x and y directions.
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::sumpool;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let pooled = sumpool::<i128>(&x, (0, 0), (1, 1), (2, 2)).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[11, 8, 8, 10]), &[1, 2, 2]).unwrap();
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
    if image.dims().len() != 3 {
        return Err(TensorError::DimMismatch("sumpool".to_string()));
    }
    let image_dims = image.dims();

    let (image_channels, image_height, image_width) = (image_dims[0], image_dims[1], image_dims[2]);

    let (output_channels, kernel_height, kernel_width) =
        (image_channels, kernel_shape.0, kernel_shape.1);

    let padded_image = pad::<T>(image, padding)?;

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    // calculate value of output
    let mut output: Tensor<T> =
        Tensor::new(None, &[output_channels, vert_slides, horz_slides]).unwrap();

    let cartesian_coord = vec![(0..output_channels), (0..vert_slides), (0..horz_slides)]
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, o)| {
            let coord = &cartesian_coord[flat_index];
            let (i, j, k) = (coord[0], coord[1], coord[2]);
            let rs = j * stride.0;
            let cs = k * stride.1;
            let thesum = sum(&padded_image
                .get_slice(&[i..i + 1, rs..(rs + kernel_height), cs..(cs + kernel_width)])
                .unwrap())
            .unwrap();
            *o = thesum[0].clone();
        });

    Ok(output)
}

/// Applies 2D max pooling over a 3D tensor of shape C x H x W.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// * `pool_dims` - Tuple of pooling window size in x and y directions.
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::max_pool2d;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let pooled = max_pool2d::<i128>(&x, &(0, 0), &(1, 1), &(2, 2)).unwrap();
/// let expected: Tensor<i128> = Tensor::<i128>::new(Some(&[5, 4, 4, 6]), &[1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
/// ```
pub fn max_pool2d<T: TensorType + std::marker::Sync + std::marker::Send>(
    image: &Tensor<T>,
    padding: &(usize, usize),
    stride: &(usize, usize),
    pool_dims: &(usize, usize),
) -> Result<Tensor<T>, TensorError> {
    if image.dims().len() != 3 {
        return Err(TensorError::DimMismatch("max_pool2d".to_string()));
    }
    let image_dims = image.dims();

    let input_channels = image_dims[0];
    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T>(image, *padding)?;

    let horz_slides = (image_height + 2 * padding.0 - pool_dims.0) / stride.0 + 1;
    let vert_slides = (image_width + 2 * padding.1 - pool_dims.1) / stride.1 + 1;

    let mut output: Tensor<T> =
        Tensor::new(None, &[input_channels, horz_slides, vert_slides]).unwrap();

    let fmax = |acc: Option<T>, x: T| -> Option<T> {
        match (acc, x) {
            (None, x) => Some(x),
            (Some(a), x) => a.tmax(&x),
        }
    };

    let cartesian_coord = vec![(0..input_channels), (0..vert_slides), (0..horz_slides)]
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    output
        .par_iter_mut()
        .enumerate()
        .for_each(|(flat_index, o)| {
            let coord = &cartesian_coord[flat_index];
            let (i, j, k) = (coord[0], coord[1], coord[2]);
            let rs = j * stride.0;
            let cs = k * stride.1;
            let themax = padded_image
                .get_slice(&[i..(i + 1), rs..(rs + pool_dims.0), cs..(cs + pool_dims.1)])
                .unwrap()
                .into_iter()
                .fold(None, fmax)
                .unwrap();
            *o = themax.clone();
        });

    Ok(output)
}

/// Dot product of two tensors.
/// # Arguments
///
/// * `inputs` - Vector of tensors of length 2.
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::dot;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let y = Tensor::<i128>::new(
///     Some(&[5, 5, 10, -4, 2, -1, 2, 0, 1]),
///     &[1, 3, 3],
/// ).unwrap();
/// assert_eq!(dot(&vec![&x, &y]).unwrap()[0], 86);
/// ```
pub fn dot<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<&Tensor<T>>,
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

/// Pads a 3D tensor of shape `C x H x W` to a tensor of shape `C x (H + 2xPADDING) x (W + 2xPADDING)` using 0 values.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::pad;
///
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let result = pad::<i128>(&x, (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(
///     Some(&[0, 0, 0, 0, 0, 0, 5, 2, 3, 0, 0, 0, 4, -1, 0, 0, 3, 1, 6, 0, 0, 0, 0, 0, 0]),
///     &[1, 5, 5],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn pad<T: TensorType>(
    image: &Tensor<T>,
    padding: (usize, usize),
) -> Result<Tensor<T>, TensorError> {
    if image.dims().len() != 3 {
        return Err(TensorError::DimMismatch("pad".to_string()));
    }
    let (channels, height, width) = (image.dims()[0], image.dims()[1], image.dims()[2]);
    let padded_height = height + 2 * padding.0;
    let padded_width = width + 2 * padding.1;

    let mut output = Tensor::<T>::new(None, &[channels, padded_height, padded_width]).unwrap();

    for channel in 0..channels {
        for row in 0..height {
            for col in 0..width {
                output.set(
                    &[channel, row + padding.0, col + padding.1],
                    image.get(&[channel, row, col]).clone(),
                );
            }
        }
    }

    output.reshape(&[channels, padded_height, padded_width]);
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
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::pack;
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
pub fn pack<T: TensorType>(a: &Tensor<T>, base: T, scale: u32) -> Result<Tensor<T>, TensorError>
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::sigmoid;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = sigmoid(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 1, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn sigmoid(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let kix = (*a_i as f32) / (scale_input as f32);
            let fout = (scale_output as f32) / (1.0 + (-kix).exp());
            let rounded = fout.round();
            output[i] = rounded as i128;
        }
        output
    }

    /// Elementwise applies square root to a tensor of integers.
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// * `scale_input` - Single value
    /// * `scale_output` - Single value
    /// # Examples
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::sqrt;
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
            let kix = (*a_i as f32) / (scale_input as f32);
            let fout = (scale_output as f32) * kix.sqrt();
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::rsqrt;
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
            let kix = (*a_i as f32) / (scale_input as f32);
            let fout = (scale_output as f32) * (1.0 / kix.sqrt());
            let rounded = fout.round();
            output[i] = rounded as i128;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::tanh;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = tanh(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[0, 1, 0, 0, 0, 0]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```

    pub fn tanh(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let z = a[i] as f32 / (scale_input as f32);
            let numerator = z.exp() - (1.0 / z.exp());
            let denominator = z.exp() + (1.0 / z.exp());
            let tanhz = (scale_output as f32) * (numerator / denominator);
            output[i] = tanhz as i128;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::erffunc;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[4, 25, 8, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = erffunc(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 0, 0, 0]), &[2, 3]).unwrap(); // TODO
    /// assert_eq!(result, expected);
    /// ```

    pub fn erffunc(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

        for i in 0..a.len() {
            let mut z = a[i] as f32 / (scale_input as f32);
            z = (scale_output as f32) * (erf(z as f64) as f32);
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::leakyrelu;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, -5]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = leakyrelu(&x, 1, 0.1);
    /// let expected = Tensor::<i128>::new(Some(&[2, 15, 2, 1, 1, -1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn leakyrelu(a: &Tensor<i128>, scale: usize, slope: f32) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            output[i] = if a_i < &0 {
                let d_inv_x = (slope) * (*a_i as f32) / (scale as f32);
                d_inv_x.round() as i128
            } else {
                let d_inv_x = (*a_i as f32) / (scale as f32);
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::prelu;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[-10, 15, 2, 1, 1, -5]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = prelu(&x, 1, &[0.1, 25.0]);
    /// let expected = Tensor::<i128>::new(Some(&[-1, 15, 2, 1, 1, -125]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn prelu(a: &Tensor<i128>, scale: usize, slopes: &[f32]) -> Tensor<i128> {
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
                let slope_i: f32 = slopes[i / (a.dims()[1..].iter().product::<usize>())];
                let d_inv_x = (slope_i) * (*a_i as f32) / (scale as f32);
                d_inv_x.round() as i128
            } else {
                let d_inv_x = (*a_i as f32) / (scale as f32);
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::const_div;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = const_div(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 4, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn const_div(a: &Tensor<i128>, denom: f32) -> Tensor<i128> {
        // calculate value of output
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let d_inv_x = (*a_i as f32) / denom;
            output[i] = d_inv_x.round() as i128;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::recip;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = recip(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 4, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn recip(a: &Tensor<i128>, scale: u32) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            let d_inv_x = (scale as f32) * (1_f32) / (*a_i as f32);
            output[i] = d_inv_x.round() as i128;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::recip;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 1, 2, 7, 1, 1]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let k = 2.0;
    /// let result = recip(&x, k);
    /// let expected = Tensor::<i128>::new(Some(&[1, 1, 1, 4, 1, 1]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn greater_than(a: &Tensor<i128>, b: f32) -> Tensor<i128> {
        // calculate value of output
        let mut output: Tensor<i128> = a.clone();

        for (i, a_i) in a.iter().enumerate() {
            output[i] = if (*a_i as f32 - b) < 0_f32 {
                0_i128
            } else {
                1_i128
            };
        }
        output
    }

    /// Takes the mean of a tensor
    /// # Arguments
    ///
    /// * `a` - Tensor
    /// # Examples
    /// ```
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::mean;
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
        const_div(&sum, (scale * a.len()) as f32)
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::accumulated::dot;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::accumulated::sum;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::accumulated::matmul;
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
        let (a, b) = (inputs[0].clone(), inputs[1].clone());
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
