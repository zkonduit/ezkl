pub mod eltwise;
use crate::tensor::{Tensor, TensorType};
pub use std::ops::{Add, Mul};

pub fn matmul<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    kernel: Tensor<T>,
    input: Tensor<T>,
) -> Tensor<T> {
    let input_dims = input.dims();
    let kernel_dims = kernel.dims();

    assert!(input_dims[1] == kernel_dims[1]);

    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &[1, kernel_dims[0]]).unwrap();

    for i in 0..kernel_dims[0] {
        output.set(
            &[0, i],
            dot_product(kernel.get_slice(&[i..i + 1]), input.get_slice(&[0..1])),
        );
    }
    output
}

pub fn convolution<
    T: TensorType + Mul<Output = T> + Add<Output = T>,
    const PADDING: usize,
    const STRIDE: usize,
>(
    kernel: Tensor<T>,
    image: Tensor<T>,
) -> Tensor<T> {
    let image_dims = image.dims();
    let kernel_dims = kernel.dims();
    assert_eq!(image_dims.len(), 3);
    assert_eq!(kernel_dims.len(), 4);
    assert_eq!(image_dims[0], kernel_dims[1]);

    let (output_channels, input_channels, kernel_height, kernel_width) = (
        kernel_dims[0],
        kernel_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let padded_image = pad::<T, PADDING>(image.clone());

    let horz_slides = (image_height + 2 * PADDING - kernel_height) / STRIDE + 1;
    let vert_slides = (image_width + 2 * PADDING - kernel_width) / STRIDE + 1;

    // calculate value of output
    let mut output: Tensor<T> =
        Tensor::new(None, &[output_channels, horz_slides, vert_slides]).unwrap();

    for i in 0..output_channels {
        for j in 0..horz_slides {
            let rs = j * STRIDE;
            for k in 0..vert_slides {
                let cs = k * STRIDE;
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

pub fn dot_product<T: TensorType + Mul<Output = T> + Add<Output = T>>(
    w: Tensor<T>,
    x: Tensor<T>,
) -> T {
    w.iter()
        .zip(x)
        .fold(T::zero().unwrap(), |acc, (k, i)| acc + k.clone() * i)
}

fn pad<T: TensorType, const PADDING: usize>(image: Tensor<T>) -> Tensor<T> {
    assert_eq!(image.dims().len(), 3);
    let (channels, height, width) = (image.dims()[0], image.dims()[1], image.dims()[2]);
    let padded_height = height + 2 * PADDING;
    let padded_width = width + 2 * PADDING;

    let mut output = Tensor::<T>::new(None, &[channels, padded_height, padded_width]).unwrap();

    for channel in 0..channels {
        for col in 0..height {
            for row in 0..width {
                output.set(
                    &[channel, col + PADDING, row + PADDING],
                    image.get(&[channel, col, row]).clone(),
                );
            }
        }
    }

    output.reshape(&[channels, padded_height, padded_width]);
    output
}

pub fn op<T: TensorType>(images: Vec<Tensor<T>>, f: impl Fn(T, T) -> T + Clone) -> Tensor<T> {
    images.iter().skip(1).fold(images[0].clone(), |acc, image| {
        acc.enum_map(|i, e| f(e, image[i].clone()))
    })
}
