use super::TensorError;
use crate::tensor::{Tensor, TensorType};
use itertools::Itertools;
use puruspe::erf;
use rayon::{
    iter::IndexedParallelIterator, iter::IntoParallelRefMutIterator, iter::ParallelIterator,
};
pub use std::ops::{Add, Div, Mul, Sub};

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

/// Gathers a tensor along a dimension.
/// # Arguments
/// * `input` - Tensor
/// * `dim` - Dimension to gather along
/// * `index` - Tensor of indices to gather
/// # Examples
/// ```
/// use ezkl_lib::tensor::Tensor;
/// use ezkl_lib::tensor::ops::gather;
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
/// // expected ouputs are taken from pytorch torch.nn.functional.conv2d
///
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
///     Some(&[5, 1, 1, 1, 5, 2, 1, 1]),
///     &[2, 1, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i128>::new(
///     Some(&[1, 1]),
///     &[2],
/// ).unwrap();
///
/// let result = conv::<i128>(&[x, k, b], (0, 0), (1, 1)).unwrap();
/// let expected = Tensor::<i128>::new(Some(&[32, 17, 9, 27, 34, 20, 13, 26]), &[2, 2, 2]).unwrap();
/// assert_eq!(result, expected);
///
/// // Now test multi channel
/// let x = Tensor::<i128>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[2, 3, 3],
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
/// let expected = Tensor::<i128>::new(Some(&[65, 36, 21, 52, 73, 48, 37, 48, 65, 36, 21, 52, 73, 48, 37, 48]), &[4, 2, 2]).unwrap();
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
    let (image, kernel) = (&inputs[0], &inputs[1]);

    if (image.dims().len() != 3)
        || (kernel.dims().len() != 4)
        // ensure number of groups makes sense
        || (image.dims()[0] % kernel.dims()[1] != 0)
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

    let (output_channels, input_channels, kernel_height, kernel_width) = (
        kernel_dims[0],
        image_dims[0],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T>(&image, padding)?;

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    let num_groups = image_dims[0] / kernel_dims[1];
    let input_channels_per_group = input_channels / num_groups;
    let output_channels_per_group = output_channels / num_groups;

    if output_channels_per_group == 0 {
        return Err(TensorError::DimMismatch(format!(
            "Given groups={}, expected kernel to be at least {} at dimension 0 but got {} instead",
            num_groups, num_groups, output_channels_per_group
        )));
    }

    let mut outputs_per_group = vec![Tensor::new(None, &[0])?; num_groups];

    outputs_per_group
        .par_iter_mut()
        .enumerate()
        .for_each(|(group, o)| {
            let start_channel = group * input_channels_per_group;
            let end_channel = start_channel + input_channels_per_group;
            let padded_image_per_group = &padded_image
                .get_slice(&[start_channel..end_channel])
                .unwrap();

            let kernel_per_group = &kernel
                .get_slice(&[
                    group * output_channels_per_group..(group + 1) * output_channels_per_group
                ])
                .unwrap();
            let mut output_per_group =
                Tensor::new(None, &[output_channels_per_group, vert_slides, horz_slides]).unwrap();

            let cartesian_coord_per_group = vec![
                (0..output_channels_per_group),
                (0..vert_slides),
                (0..horz_slides),
            ]
            .iter()
            .cloned()
            .multi_cartesian_product()
            .collect::<Vec<_>>();

            output_per_group
                .par_iter_mut()
                .enumerate()
                .for_each(|(flat_index, o)| {
                    let coord = &cartesian_coord_per_group[flat_index];
                    let (i, j, k) = (coord[0], coord[1], coord[2]);
                    let rs = j * stride.0;
                    let cs = k * stride.1;

                    let res = dot(&vec![
                        &kernel_per_group.get_slice(&[i..i + 1]).unwrap().clone(),
                        &padded_image_per_group
                            .get_slice(&[
                                0..input_channels_per_group,
                                rs..(rs + kernel_height),
                                cs..(cs + kernel_width),
                            ])
                            .unwrap(),
                    ])
                    .unwrap();

                    *o = res[0].clone();
                });

            *o = output_per_group;
        });

    let mut output = Tensor::new(Some(&outputs_per_group), &[num_groups])?.combine()?;

    output.reshape(&[output_channels, vert_slides, horz_slides]);

    if has_bias {
        // increment result by the bias
        output = (output + inputs[2].clone())?;
    }

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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::exp;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 15, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = exp(&x, 1, 1);
    /// let expected = Tensor::<i128>::new(Some(&[7, 3269017, 7, 3, 3, 1]), &[2, 3]).unwrap();
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

    /// softmax layout
    pub fn multi_dim_softmax(
        a: &Tensor<i128>,
        scale_input: usize,
        scale_output: usize,
    ) -> Tensor<i128> {
        // we want this to be as small as possible so we set the output scale to 1
        let dims = a.dims();

        if dims.len() == 1 {
            return softmax(a, scale_input, scale_output);
        }

        let cartesian_coord = dims[..dims.len() - 1]
            .iter()
            .map(|x| 0..*x)
            .multi_cartesian_product()
            .collect::<Vec<_>>();

        let mut outputs = vec![];

        for coord in cartesian_coord {
            let mut sum_dims = vec![];
            for i in 0..coord.len() {
                sum_dims.push(coord[i]..coord[i] + 1);
            }
            sum_dims.push(0..dims[dims.len() - 1]);

            let softmax_input = a.get_slice(&sum_dims).unwrap();

            outputs.push(softmax(&softmax_input, scale_input, scale_output));
        }

        let mut res = Tensor::new(Some(&outputs), &[outputs.len()])
            .unwrap()
            .combine()
            .unwrap();
        res.reshape(dims);

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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::softmax;
    /// let x = Tensor::<i128>::new(
    ///     Some(&[2, 4, 2, 1, 1, 0]),
    ///     &[2, 3],
    /// ).unwrap();
    /// let result = softmax(&x, 128, 128);
    /// // doubles the scale of the input
    /// let expected = Tensor::<i128>::new(Some(&[2730, 2772, 2730, 2709, 2709, 2688]), &[2, 3]).unwrap();
    /// assert_eq!(result, expected);
    /// ```
    pub fn softmax(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        // the more accurate calculation is commented out and we implement as below so it matches the steps in layout
        let exp = exp(a, scale_input, scale_output);
        let sum = sum(&exp).unwrap();
        let inv_denom = recip(&sum, scale_output.pow(2) as u32);
        let output = (exp * inv_denom).unwrap();

        // let mut output = a.clone();

        // let denominator = a.iter().fold(0.0, |acc, x| {
        //     let kix = (*x as f64) / (scale_input as f64);
        //     acc + kix.exp()
        // });

        // let mut res = vec![];
        // for (i, a_i) in a.iter().enumerate() {
        //     let kix = (*a_i as f64) / (scale_input as f64);
        //     let numerator = kix.exp();
        //     let fout = numerator / denominator;
        //     res.push(fout);
        //     let rounded = ((scale_output as f64) * fout).round();
        //     output[i] = rounded as i128;
        // }

        // assert_eq!(res.iter().fold(0.0, |acc, x| acc + x), 1.0);

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
            let kix = (*a_i as f64) / (scale_input as f64);
            let fout = (scale_output as f64) * (1.0 / kix.sqrt());
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
            let z = a[i] as f64 / (scale_input as f64);
            let numerator = z.exp() - (1.0 / z.exp());
            let denominator = z.exp() + (1.0 / z.exp());
            let tanhz = (scale_output as f64) * (numerator / denominator);
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
    /// let expected = Tensor::<i128>::new(Some(&[0, 1, 1, 0, 0, 0]), &[2, 3]).unwrap(); // TODO
    /// assert_eq!(result, expected);
    /// ```

    pub fn erffunc(a: &Tensor<i128>, scale_input: usize, scale_output: usize) -> Tensor<i128> {
        let mut output = a.clone();

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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::recip;
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
    /// use ezkl_lib::tensor::Tensor;
    /// use ezkl_lib::tensor::ops::nonlinearities::greater_than;
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
            output[i] = if (*a_i as f64 - b) <= 0_f64 {
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
