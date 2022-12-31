use crate::tensor::{Tensor, TensorType};
use itertools::Itertools;
pub use std::ops::{Add, Div, Mul, Sub};

/// Matrix multiplies two 2D tensors (and adds an offset).
/// # Arguments
///
/// * `inputs` - A vector of tensors holding in order: input data, affine kernel, convolution bias.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::affine;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 2, 1, 1]),
///     &[3, 4],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let b = Tensor::<i32>::new(
///     Some(&[0, 0]),
///     &[2],
/// ).unwrap();
/// let result = affine(&vec![x, k, b]);
/// let expected = Tensor::<i32>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn affine<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<Tensor<T>>,
) -> Tensor<T> {
    assert_eq!(inputs.len(), 3);
    let (mut input, kernel, bias) = (inputs[0].clone(), inputs[1].clone(), inputs[2].clone());
    assert_eq!(bias.dims()[0], kernel.dims()[0]);
    assert_eq!(input.dims()[0], kernel.dims()[1]);

    // does matrix to vector multiplication
    if input.dims().len() == 1 {
        input.reshape(&[input.dims()[0], 1])
    }

    let input_dims = input.dims();
    let kernel_dims = kernel.dims();

    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &[kernel_dims[0], input_dims[1]]).unwrap();

    for i in 0..kernel_dims[0] {
        for j in 0..input_dims[1] {
            let prod = dot(&vec![
                &kernel.get_slice(&[i..i + 1]),
                &input.get_slice(&[0..input_dims[0], j..j + 1]),
            ]);
            output.set(&[i, j], prod[0].clone() + bias[i].clone());
        }
    }
    // does matrix to vector multiplication
    if output.dims()[1] == 1 {
        output.flatten();
    }
    output
}

/// Scales and shifts a tensor.
/// Given inputs (x,k,b) computes k*x + b elementwise
pub fn scale_and_shift<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<Tensor<T>>,
) -> Tensor<T> {
    assert_eq!(inputs.len(), 3);
    let (input, kernel, bias) = (inputs[0].clone(), inputs[1].clone(), inputs[2].clone());
    assert_eq!(input.dims(), kernel.dims());
    assert_eq!(bias.dims(), kernel.dims());
    let mut output: Tensor<T> = input;
    for (i, bias_i) in bias.iter().enumerate() {
        output[i] = kernel[i].clone() * output[i].clone() + bias_i.clone()
    }
    output
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
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 2, 1, 1]),
///     &[3, 4],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = matmul(&vec![k, x]);
/// let expected = Tensor::<i32>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn matmul<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<Tensor<T>>,
) -> Tensor<T> {
    assert_eq!(inputs.len(), 2);
    let (a, b) = (inputs[0].clone(), inputs[1].clone());
    assert_eq!(a.dims()[a.dims().len() - 1], b.dims()[a.dims().len() - 2]);
    assert_eq!(
        a.dims()[0..a.dims().len() - 2],
        b.dims()[0..a.dims().len() - 2]
    );

    let mut dims = Vec::from(&a.dims()[0..a.dims().len() - 2]);
    dims.push(a.dims()[a.dims().len() - 2]);
    dims.push(b.dims()[a.dims().len() - 1]);
    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &dims).unwrap();

    let indices = dims.iter().map(|d| 0..*d).collect::<Vec<_>>();

    for coord in indices.iter().cloned().multi_cartesian_product() {
        let row = coord[0..coord.len() - 1]
            .iter()
            .map(|&d| d..(d + 1))
            .collect::<Vec<_>>();
        let mut col = coord[0..coord.len()]
            .iter()
            .map(|&d| d..(d + 1))
            .collect::<Vec<_>>();
        col[coord.len() - 2] = 0..b.dims()[coord.len() - 2];
        let prod = dot(&vec![&a.get_slice(&row[0..]), &b.get_slice(&col[0..])]);
        output.set(&coord, prod[0].clone());
    }

    output
}

/// Adds multiple tensors.
/// # Arguments
///
/// * `t` - Vector of tensors
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::add;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = add(&vec![x, k]);
/// let expected = Tensor::<i32>::new(Some(&[4, 4, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn add<T: TensorType + Add<Output = T>>(t: &Vec<Tensor<T>>) -> Tensor<T> {
    // determines if we're multiplying by a 1D const
    if t.len() == 2 && t[1].dims().len() == 1 && t[1].dims()[0] == 1 {
        return const_add(&t[0], t[1][0].clone());
    }
    for e in t.iter() {
        assert_eq!(t[0].dims(), e.dims());
    }
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        for (i, e_i) in e.iter().enumerate() {
            output[i] = output[i].clone() + e_i.clone()
        }
    }

    output
}

/// Elementwise adds a tensor with a const element.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::const_add;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = const_add(&x, k);
/// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 3, 3, 3]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn const_add<T: TensorType + Add<Output = T>>(a: &Tensor<T>, b: T) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();

    for i in 0..output.len() {
        output[i] = output[i].clone() + b.clone();
    }

    output
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
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = sub(&vec![x, k]);
/// let expected = Tensor::<i32>::new(Some(&[0, -2, 0, 0, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn sub<T: TensorType + Sub<Output = T>>(t: &Vec<Tensor<T>>) -> Tensor<T> {
    // determines if we're multiplying by a 1D const
    if t.len() == 2 && t[1].dims().len() == 1 && t[1].dims()[0] == 1 {
        return const_sub(&t[0], t[1][0].clone());
    }

    for e in t.iter() {
        assert_eq!(t[0].dims(), e.dims());
    }
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        for (i, e_i) in e.iter().enumerate() {
            output[i] = output[i].clone() - e_i.clone()
        }
    }

    output
}

/// Elementwise subtracts a tensor with a const element.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::const_sub;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = const_sub(&x, k);
/// let expected = Tensor::<i32>::new(Some(&[0, -1, 0, -1, -1, -1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn const_sub<T: TensorType + Sub<Output = T>>(a: &Tensor<T>, b: T) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();

    for i in 0..output.len() {
        output[i] = output[i].clone() - b.clone();
    }

    output
}

/// Elementwise multiplies two tensors.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::mult;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[2, 3, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = mult(&vec![x, k]);
/// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn mult<T: TensorType + Mul<Output = T>>(t: &Vec<Tensor<T>>) -> Tensor<T> {
    // determines if we're multiplying by a 1D const
    if t.len() == 2 && t[1].dims().len() == 1 && t[1].dims()[0] == 1 {
        return const_mult(&t[0], t[1][0].clone());
    }

    for e in t.iter() {
        assert_eq!(t[0].dims(), e.dims());
    }
    // calculate value of output
    let mut output: Tensor<T> = t[0].clone();

    for e in t[1..].iter() {
        for (i, e_i) in e.iter().enumerate() {
            output[i] = output[i].clone() * e_i.clone()
        }
    }

    output
}

/// Elementwise divide a tensor with another tensor.
/// # Arguments
///
/// * `t` - Tensor
/// * `d` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::div;
/// let x = Tensor::<i32>::new(
///     Some(&[4, 1, 4, 1, 1, 4]),
///     &[2, 3],
/// ).unwrap();
/// let y = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = div(x, y);
/// let expected = Tensor::<i32>::new(Some(&[2, 1, 2, 1, 1, 4]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn div<T: TensorType + Div<Output = T>>(t: Tensor<T>, d: Tensor<T>) -> Tensor<T> {
    assert_eq!(t.dims(), d.dims());
    // calculate value of output
    let mut output: Tensor<T> = t;

    for (i, d_i) in d.iter().enumerate() {
        output[i] = output[i].clone() / d_i.clone()
    }
    output
}

/// Elementwise divides a tensor with a const element.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::const_div;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 7, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = const_div(&x, k);
/// let expected = Tensor::<i32>::new(Some(&[1, 0, 1, 3, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn const_div<T: TensorType + Div<Output = T>>(a: &Tensor<T>, b: T) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();

    for (i, a_i) in a.iter().enumerate() {
        output[i] = a_i.clone() / b.clone()
    }

    output
}

/// Elementwise multiplies a tensor with a const element.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::const_mult;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = const_mult(&x, k);
/// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn const_mult<T: TensorType + Mul<Output = T>>(a: &Tensor<T>, b: T) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();

    for i in 0..output.len() {
        output[i] = output[i].clone() * b.clone();
    }

    output
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
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = rescale(&x, k);
/// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn rescale<T: TensorType + Add<Output = T>>(a: &Tensor<T>, mult: usize) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();
    for (i, a_i) in a.iter().enumerate() {
        for _ in 1..mult {
            output[i] = output[i].clone() + a_i.clone();
        }
    }
    output
}

/// Elementwise raise a tensor to the nth power.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::pow;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = pow(&x, 3);
/// let expected = Tensor::<i32>::new(Some(&[8, 3375, 8, 1, 1, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn pow<T: TensorType + Mul<Output = T>>(a: &Tensor<T>, pow: usize) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();
    for (i, a_i) in a.iter().enumerate() {
        for _ in 1..pow {
            output[i] = output[i].clone() * a_i.clone();
        }
    }
    output
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
/// let x = Tensor::<i32>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = sum(&x);
/// let expected = 21;
/// assert_eq!(result[0], expected);
/// ```
pub fn sum<T: TensorType + Add<Output = T>>(a: &Tensor<T>) -> Tensor<T> {
    // calculate value of output
    let mut res = T::zero().unwrap();
    let _ = a.map(|a_i| res = res.clone() + a_i);
    Tensor::new(Some(&[res]), &[1]).unwrap()
}

/// Applies convolution over a 3D tensor of shape C x H x W (and adds a bias).
/// # Arguments
///
/// * `inputs` - A vector of tensors holding in order: input image, convolution kernel, convolution bias.
/// * `padding` - Tuple of padding values in x and y directions.
/// * `stride` - Tuple of stride values in x and y directions.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::convolution;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[5, 1, 1, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap();
/// let b = Tensor::<i32>::new(
///     Some(&[0]),
///     &[1],
/// ).unwrap();
/// let result = convolution::<i32>(&vec![x, k, b], (0, 0), (1, 1));
/// let expected = Tensor::<i32>::new(Some(&[31, 16, 8, 26]), &[1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn convolution<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<Tensor<T>>,
    padding: (usize, usize),
    stride: (usize, usize),
) -> Tensor<T> {
    let has_bias = inputs.len() == 3;
    let (image, kernel) = (inputs[0].clone(), inputs[1].clone());

    assert_eq!(image.dims().len(), 3);
    assert_eq!(kernel.dims().len(), 4);
    assert_eq!(image.dims()[0], kernel.dims()[1]);
    if has_bias {
        let bias = inputs[2].clone();
        assert_eq!(bias.dims().len(), 1);
        assert_eq!(bias.dims()[0], kernel.dims()[0]);
    }

    let image_dims = image.dims();
    let kernel_dims = kernel.dims();

    let (output_channels, input_channels, kernel_height, kernel_width) = (
        kernel_dims[0],
        kernel_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T>(image.clone(), padding);

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    // calculate value of output
    let mut output: Tensor<T> =
        Tensor::new(None, &[output_channels, vert_slides, horz_slides]).unwrap();

    for i in 0..output_channels {
        for j in 0..vert_slides {
            let rs = j * stride.0;
            for k in 0..horz_slides {
                let cs = k * stride.1;
                let mut res = dot(&vec![
                    &kernel.get_slice(&[i..i + 1]).clone(),
                    &padded_image.get_slice(&[
                        0..input_channels,
                        rs..(rs + kernel_height),
                        cs..(cs + kernel_width),
                    ]),
                ]);

                if has_bias {
                    // increment result by the bias
                    res[0] = res[0].clone() + inputs[2][i].clone();
                }

                output.set(&[i, j, k], res[0].clone());
            }
        }
    }
    output
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
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::sumpool;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let pooled = sumpool::<i32>(&x, (0, 0), (1, 1), (2, 2));
/// let expected: Tensor<i32> = Tensor::<i32>::new(Some(&[11, 8, 8, 10]), &[1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
/// ```
pub fn sumpool<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    image: &Tensor<T>,
    padding: (usize, usize),
    stride: (usize, usize),
    kernel_shape: (usize, usize),
) -> Tensor<T> {
    assert_eq!(image.dims().len(), 3);
    let image_dims = image.dims();

    let (image_channels, image_height, image_width) = (image_dims[0], image_dims[1], image_dims[2]);

    let (output_channels, kernel_height, kernel_width) =
        (image_channels, kernel_shape.0, kernel_shape.1);

    let padded_image = pad::<T>(image.clone(), padding);

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    // calculate value of output
    let mut output: Tensor<T> =
        Tensor::new(None, &[output_channels, vert_slides, horz_slides]).unwrap();

    for i in 0..output_channels {
        for j in 0..vert_slides {
            let rs = j * stride.0;
            for k in 0..horz_slides {
                let cs = k * stride.1;
                let thesum = sum(&padded_image.get_slice(&[
                    i..i + 1,
                    rs..(rs + kernel_height),
                    cs..(cs + kernel_width),
                ]));
                output.set(&[i, j, k], thesum[0].clone());
            }
        }
    }
    output
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
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::max_pool2d;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let pooled = max_pool2d::<i32>(&x, (0, 0), (1, 1), (2, 2));
/// let expected: Tensor<i32> = Tensor::<i32>::new(Some(&[5, 4, 4, 6]), &[1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
/// ```
pub fn max_pool2d<T: TensorType>(
    image: &Tensor<T>,
    padding: (usize, usize),
    stride: (usize, usize),
    pool_dims: (usize, usize),
) -> Tensor<T> {
    let image_dims = image.dims();
    assert_eq!(image_dims.len(), 3);

    let input_channels = image_dims[0];
    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T>(image.clone(), padding);

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

    for i in 0..input_channels {
        for j in 0..horz_slides {
            let rs = j * stride.0;
            for k in 0..vert_slides {
                let cs = k * stride.1;
                output.set(
                    &[i, j, k],
                    padded_image
                        .get_slice(&[i..(i + 1), rs..(rs + pool_dims.0), cs..(cs + pool_dims.1)])
                        .into_iter()
                        .fold(None, fmax)
                        .unwrap(),
                );
            }
        }
    }
    output
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
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let y = Tensor::<i32>::new(
///     Some(&[5, 5, 10, -4, 2, -1, 2, 0, 1]),
///     &[1, 3, 3],
/// ).unwrap();
/// assert_eq!(dot(&vec![&x, &y])[0], 86);
/// ```
pub fn dot<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<&Tensor<T>>,
) -> Tensor<T> {
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0].clone().len(), inputs[1].clone().len());
    let (a, b): (Tensor<T>, Tensor<T>) = (inputs[0].clone(), inputs[1].clone());
    let res = a
        .iter()
        .zip(b)
        .fold(T::zero().unwrap(), |acc, (k, i)| acc + k.clone() * i);
    Tensor::new(Some(&[res]), &[1]).unwrap()
}

/// Pads a 3D tensor of shape `C x H x W` to a tensor of shape `C x (H + 2xPADDING) x (W + 2xPADDING)` using 0 values.
/// # Arguments
///
/// * `image` - Tensor.
/// * `padding` - Tuple of padding values in x and y directions.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::pad;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let result = pad::<i32>(x, (1, 1));
/// let expected = Tensor::<i32>::new(
///     Some(&[0, 0, 0, 0, 0, 0, 5, 2, 3, 0, 0, 0, 4, -1, 0, 0, 3, 1, 6, 0, 0, 0, 0, 0, 0]),
///     &[1, 5, 5],
/// ).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn pad<T: TensorType>(image: Tensor<T>, padding: (usize, usize)) -> Tensor<T> {
    assert_eq!(image.dims().len(), 3);
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
    output
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
/// use ezkl::tensor::ops::sigmoid;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = sigmoid(&x, 1, 1);
/// let expected = Tensor::<i32>::new(Some(&[1, 1, 1, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn sigmoid(a: &Tensor<i32>, scale_input: usize, scale_output: usize) -> Tensor<i32> {
    // calculate value of output
    let mut output: Tensor<i32> = a.clone();

    for (i, a_i) in a.iter().enumerate() {
        let kix = (*a_i as f32) / (scale_input as f32);
        let fout = (scale_output as f32) / (1.0 + (-kix).exp());
        let rounded = fout.round();
        output[i] = rounded as i32;
    }
    output
}

/// Elementwise applies leaky relu to a tensor of integers.
/// # Arguments
///
/// * `a` - Tensor
/// * `scale` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::leakyrelu;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 15, 2, 1, 1, -5]),
///     &[2, 3],
/// ).unwrap();
/// let result = leakyrelu(&x, 1, 0.1);
/// let expected = Tensor::<i32>::new(Some(&[2, 15, 2, 1, 1, -1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn leakyrelu(a: &Tensor<i32>, scale: usize, slope: f32) -> Tensor<i32> {
    // calculate value of output
    let mut output: Tensor<i32> = a.clone();

    for (i, a_i) in a.iter().enumerate() {
        output[i] = if a_i < &0 {
            let d_inv_x = (slope) * (*a_i as f32) / (scale as f32);
            d_inv_x.round() as i32
        } else {
            let d_inv_x = (*a_i as f32) / (scale as f32);
            d_inv_x.round() as i32
        };
    }
    output
}
