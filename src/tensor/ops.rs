use crate::tensor::{Tensor, TensorType};
use itertools::Itertools;
pub use std::ops::{Add, Mul, Sub};

/// Matrix multiplies two 2D tensors (and adds an offset).
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
/// let result = affine(&vec![&x, &k, &b]);
/// let expected = Tensor::<i32>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn affine<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<&Tensor<T>>,
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

/// Matrix multiplies two 2D tensors.
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
/// let result = matmul(&vec![&k, &x]);
/// let expected = Tensor::<i32>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn matmul<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<&Tensor<T>>,
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
/// let result = add(&vec![&x, &k]);
/// let expected = Tensor::<i32>::new(Some(&[4, 4, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn add<T: TensorType + Add<Output = T>>(t: &Vec<&Tensor<T>>) -> Tensor<T> {
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

/// Subtracts multiple tensors.
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
/// let result = sub(&vec![&x, &k]);
/// let expected = Tensor::<i32>::new(Some(&[0, -2, 0, 0, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn sub<T: TensorType + Sub<Output = T>>(t: &Vec<&Tensor<T>>) -> Tensor<T> {
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

/// Elementwise multiplies two tensors.
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
/// let result = mult(&vec![&x, &k]);
/// let expected = Tensor::<i32>::new(Some(&[4, 3, 4, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn mult<T: TensorType + Mul<Output = T>>(t: &Vec<&Tensor<T>>) -> Tensor<T> {
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

/// Elementwise multiplies a tensor with a const element.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::const_mult;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let k = 2;
/// let result = const_mult(x, k);
/// let expected = Tensor::<i32>::new(Some(&[4, 2, 4, 2, 2, 2]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn const_mult<T: TensorType + Mul<Output = T> + Copy>(a: Tensor<T>, b: T) -> Tensor<T> {
    // calculate value of output
    let mut output: Tensor<T> = a.clone();

    for i in 0..output.len() {
        output[i] = output[i] * b;
    }

    output
}

/// Elementwise raise a tensor to the nth power.
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
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::tensor::ops::sum;
/// let x = Tensor::<i32>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap();
/// let result = sum(&x);
/// let expected = 21;
/// assert_eq!(result, expected);
/// ```
pub fn sum<T: TensorType + Add<Output = T>>(a: &Tensor<T>) -> Tensor<T> {
    // calculate value of output
    let mut res = T::zero().unwrap();
    a.map(|a_i| res = res.clone() + a_i.clone());
    Tensor::new(Some(&vec![res]), &[1]).unwrap()
}

/// Applies convolution over a 3D tensor of shape C x H x W (and adds a bias).
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
/// let result = convolution::<i32>(&vec![&x, &k, &b], (0, 0), (1, 1));
/// let expected = Tensor::<i32>::new(Some(&[31, 16, 8, 26]), &[1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn convolution<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    inputs: &Vec<&Tensor<T>>,
    padding: (usize, usize),
    stride: (usize, usize),
) -> Tensor<T> {
    assert_eq!(inputs.len(), 3);
    let (image, kernel, bias) = (inputs[0].clone(), inputs[1].clone(), inputs[2].clone());
    assert_eq!(image.dims().len(), 3);
    assert_eq!(kernel.dims().len(), 4);
    assert_eq!(bias.dims().len(), 1);
    assert_eq!(image.dims()[0], kernel.dims()[1]);
    assert_eq!(bias.dims()[0], kernel.dims()[0]);

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
                let prod = dot(&vec![
                    &kernel.get_slice(&[i..i + 1]).clone(),
                    &padded_image.get_slice(&[
                        0..input_channels,
                        rs..(rs + kernel_height),
                        cs..(cs + kernel_width),
                    ]),
                ]);
                output.set(&[i, j, k], prod[0].clone() + bias[i].clone());
            }
        }
    }
    output
}

/// Dot product of two tensors.
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
/// assert_eq!(dot(&vec![&x, &y]), 86);
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
    Tensor::new(Some(&vec![res]), &[1]).unwrap()
}

/// Pads a 3D tensor of shape `C x H x W` to a tensor of shape `C x (H + 2xPADDING) x (W + 2xPADDING)` using 0 values.
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
