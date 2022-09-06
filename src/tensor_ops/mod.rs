pub mod eltwise;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{AssignedCell, Value},
    plonk::Assigned,
};
pub use std::ops::{Add, Mul};

// for now assumes a batch size of 1
pub fn vec_matmul_field<F: FieldExt>(
    a: Tensor<AssignedCell<Assigned<F>, F>>,
    b: Tensor<AssignedCell<Assigned<F>, F>>,
    biases: Option<Tensor<AssignedCell<Assigned<F>, F>>>,
) -> Tensor<Value<Assigned<F>>> {
    // calculate value of output
    assert!(a.dims().len() == 1);
    assert!(b.dims().len() == 2);
    assert!(a.dims()[0] == b.dims()[0]);
    let out_dim = b.dims()[1];
    // calculate value of output
    let mut output: Tensor<Value<Assigned<F>>> = Tensor::new(None, &[out_dim]).unwrap();

    for (i, o) in output.iter_mut().enumerate() {
        for (j, x) in a.iter().enumerate() {
            *o = *o + b.get(&[i, j]).value_field() * x.value_field();
        }
        if let Some(ref bias) = biases {
            *o = *o + bias.get(&[i]).value_field()
        }
    }
    output
}

pub fn convolution<
    T: TensorType + Mul<Output = T> + Add<Output = T>,
    const KERNEL_HEIGHT: usize,
    const KERNEL_WIDTH: usize,
    const IMAGE_HEIGHT: usize,
    const IMAGE_WIDTH: usize,
    const PADDED_HEIGHT: usize,
    const PADDED_WIDTH: usize,
    const OUTPUT_HEIGHT: usize,
    const OUTPUT_WIDTH: usize,
    const PADDING: usize,
    const STRIDE: usize,
>(
    kernel: Tensor<T>,
    image: Tensor<T>,
) -> Tensor<T> {
    let padded_image = pad::<T, PADDED_HEIGHT, PADDED_WIDTH, PADDING>(image);

    let horz_slides = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
    let vert_slides = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

    // calculate value of output
    let mut output: Tensor<T> = Tensor::new(None, &[horz_slides, vert_slides]).unwrap();

    for horz_slide in 0..horz_slides {
        let col_start = horz_slide * STRIDE;
        for vert_slide in 0..vert_slides {
            let row_start = vert_slide * STRIDE;
            output.set(
                &[horz_slide, vert_slide],
                dot_product(
                    kernel.clone(),
                    padded_image.get_slice(&[
                        col_start..(col_start + KERNEL_WIDTH),
                        row_start..(row_start + KERNEL_HEIGHT),
                    ]),
                ),
            );
        }
    }
    output
}

fn dot_product<T: TensorType + Mul<Output = T> + Add<Output = T>>(w: Tensor<T>, x: Tensor<T>) -> T {
    w.iter()
        .zip(x)
        .fold(T::zero().unwrap(), |acc, (k, i)| acc + k.clone() * i)
}

fn pad<
    T: TensorType,
    const PADDED_HEIGHT: usize,
    const PADDED_WIDTH: usize,
    const PADDING: usize,
>(
    image: Tensor<T>,
) -> Tensor<T> {
    assert!(PADDED_HEIGHT == image.dims()[0] + 2 * PADDING);
    assert!(PADDED_WIDTH == image.dims()[1] + 2 * PADDING);

    println!("{:?}", image);
    let mut output = Tensor::<T>::new(None, &[PADDED_HEIGHT, PADDED_WIDTH]).unwrap();

    for col in 0..image.dims()[0] {
        for row in 0..image.dims()[1] {
            output.set(
                &[col + PADDING, row + PADDING],
                image.get(&[col, row]).clone(),
            );
        }
    }

    output.reshape(&[PADDED_HEIGHT, PADDED_WIDTH]);
    output
}

pub fn op<T: TensorType, const IMAGE_HEIGHT: usize, const IMAGE_WIDTH: usize>(
    images: Vec<Tensor<T>>,
    f: impl Fn(T, T) -> T + Clone,
) -> Tensor<T> {
    images.iter().skip(1).fold(images[0].clone(), |acc, image| {
        acc.enum_map(|i, e| f(e, image[i].clone()))
    })
}
