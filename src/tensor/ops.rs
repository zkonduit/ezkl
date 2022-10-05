use crate::tensor::{Tensor, TensorType};
pub use std::ops::{Add, Mul};

/// Matrix multiplies two 2D tensors (and adds an offset).
/// ```
/// use halo2deeplearning::tensor::Tensor;
/// use halo2deeplearning::tensor::ops::matmul;
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
/// let result = matmul(k, b, x);
/// let expected = Tensor::<i32>::new(Some(&[26, 7, 11, 3, 15, 3, 7, 2]), &[2, 4]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn matmul<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    kernel: Tensor<T>,
    bias: Tensor<T>,
    mut input: Tensor<T>,
) -> Tensor<T> {
    assert_eq!(bias.dims()[0], kernel.dims()[0]);
    assert_eq!(input.dims()[0], kernel.dims()[1]);

    // does matrix to vector multiplication
    match input.dims().len() {
        1 => input.reshape(&[input.dims()[0], 1]),
        _ => {}
    }

    let input_dims = input.dims();
    let kernel_dims = kernel.dims();

    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &[kernel_dims[0], input_dims[1]]).unwrap();

    for i in 0..kernel_dims[0] {
        for j in 0..input_dims[1] {
            output.set(
                &[i, j],
                dot_product(
                    kernel.get_slice(&[i..i + 1]),
                    input.get_slice(&[0..input_dims[0], j..j + 1]),
                ) + bias[i].clone(),
            );
        }
    }
    output
}

/// Applies convolution over a 3D tensor of shape C x H x W (and adds a bias).
/// ```
/// use halo2deeplearning::tensor::Tensor;
/// use halo2deeplearning::tensor::ops::convolution;
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
/// let result = convolution::<i32>(k, b, x, (0, 0), (1, 1));
/// let expected = Tensor::<i32>::new(Some(&[31, 16, 8, 26]), &[1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn convolution<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    kernel: Tensor<T>,
    bias: Tensor<T>,
    image: Tensor<T>,
    padding: (usize, usize),
    stride: (usize, usize),
) -> Tensor<T> {
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
                output.set(
                    &[i, j, k],
                    dot_product(
                        kernel.get_slice(&[i..i + 1]).clone(),
                        padded_image.get_slice(&[
                            0..input_channels,
                            rs..(rs + kernel_height),
                            cs..(cs + kernel_width),
                        ]),
                    ) + bias[i].clone(),
                );
            }
        }
    }
    output
}

/// Applies 2D max pooling over a 3D tensor of shape C x H x W.
/// ```
/// use halo2deeplearning::tensor::Tensor;
/// use halo2deeplearning::tensor::ops::max_pool2d;
/// use halo2_proofs::circuit::Value;
/// use halo2_proofs::plonk::Assigned;
/// use halo2curves::pasta::Fp as F;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// const POOL_H: usize = 2;
/// const POOL_W: usize = 2;
/// let pooled = max_pool2d::<i32, POOL_H, POOL_W>(x, (0, 0), (1, 1));
/// let expected: Tensor<i32> = Tensor::<i32>::new(Some(&[5, 4, 4, 6]), &[1, 2, 2]).unwrap();
/// assert_eq!(pooled, expected);
/// ```
pub fn max_pool2d<
    T: TensorType,
    const POOL_H: usize,
    const POOL_W: usize,
>(
    image: Tensor<T>,
    padding: (usize, usize),
    stride: (usize, usize),
) -> Tensor<T> {
    let image_dims = image.dims();
    assert_eq!(image_dims.len(), 3);

    let input_channels = image_dims[0];
    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T>(image.clone(), padding);

    let horz_slides = (image_height + 2 * padding.0 - POOL_H) / stride.0 + 1;
    let vert_slides = (image_width + 2 * padding.1 - POOL_W) / stride.1 + 1;

    let mut output: Tensor<T> =
        Tensor::new(None, &[input_channels, horz_slides, vert_slides]).unwrap();

    let fmax = |acc: Option<T>, x: T| -> Option<T> {
        match (acc, x) {
            (None, x) => Some(x),
            (Some(a), x) => a.tmax(&x)
        }
    };

    for i in 0..input_channels {
        for j in 0..horz_slides {
            let rs = j * stride.0;
            for k in 0..vert_slides {
                let cs = k * stride.1;
                output.set(
                    &[i, j, k],
                    padded_image.get_slice(&[
                        i..(i + 1),
                        rs..(rs + POOL_H),
                        cs..(cs + POOL_W),
                    ]).into_iter()
                      .fold(None, fmax)
                      .unwrap(),
                );
            }
        }
    }
    output
}

/// Dot product of two tensors.
/// ```
/// use halo2deeplearning::tensor::Tensor;
/// use halo2deeplearning::tensor::ops::dot_product;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap();
/// let y = Tensor::<i32>::new(
///     Some(&[5, 5, 10, -4, 2, -1, 2, 0, 1]),
///     &[1, 3, 3],
/// ).unwrap();
/// assert_eq!(dot_product(x, y), 86);
/// ```
pub fn dot_product<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    w: Tensor<T>,
    x: Tensor<T>,
) -> T {
    w.iter()
        .zip(x)
        .fold(T::zero().unwrap(), |acc, (k, i)| acc + k.clone() * i)
}

/// Pads a 3D tensor of shape `C x H x W` to a tensor of shape `C x (H + 2xPADDING) x (W + 2xPADDING)` using 0 values.
/// ```
/// use halo2deeplearning::tensor::Tensor;
/// use halo2deeplearning::tensor::ops::pad;
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
