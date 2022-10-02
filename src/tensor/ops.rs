use crate::tensor::{Tensor, TensorType};
pub use std::ops::{Add, Mul};

/// Matrix multiplies two 2D tensors.
/// ```
/// use halo2deeplearning::tensor::Tensor;
/// use halo2deeplearning::tensor::ops::matmul;
///
/// let x = Tensor::<i32>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[3, 3],
/// ).unwrap();
/// let k = Tensor::<i32>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap();
/// let result = matmul(k, x);
/// let expected = Tensor::<i32>::new(Some(&[18, 2, 19, 10, 3, 10]), &[2, 3]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn matmul<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    kernel: Tensor<T>,
    input: Tensor<T>,
) -> Tensor<T> {
    let input_dims = input.dims();
    let kernel_dims = kernel.dims();

    assert!(input_dims[0] == kernel_dims[1]);

    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &[kernel_dims[0], input_dims[1]]).unwrap();

    for i in 0..kernel_dims[0] {
        for j in 0..input_dims[1] {
            output.set(
                &[i, j],
                dot_product(kernel.get_slice(&[i..i + 1]), input.get_slice(&[j..j + 1])),
            );
        }
    }
    output
}

/// Applies convolution over a 3D tensor of shape C x H x W.
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
/// let result = convolution::<i32>(k, x, &[0, 0, 1, 1]);
/// let expected = Tensor::<i32>::new(Some(&[31, 16, 8, 26]), &[1, 2, 2]).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn convolution<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    kernel: Tensor<T>,
    image: Tensor<T>,
    params: &[usize],
) -> Tensor<T> {
    assert_eq!(image.dims().len(), 3);
    assert_eq!(kernel.dims().len(), 4);
    assert_eq!(image.dims()[0], kernel.dims()[1]);
    assert_eq!(params.len(), 4);

    let image_dims = image.dims();
    let kernel_dims = kernel.dims();
    let padding = (params[0], params[1]);
    let stride = (params[2], params[3]);

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
                    ),
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
