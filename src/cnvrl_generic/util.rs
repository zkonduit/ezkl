use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

use halo2_proofs::{arithmetic::FieldExt, circuit::Value, plonk::Expression};

type Matrix<T, const HEIGHT: usize, const WIDTH: usize> = [[T; HEIGHT]; WIDTH];

pub fn matrix<T: Debug, const HEIGHT: usize, const WIDTH: usize>(
    element: impl Fn() -> T,
) -> Matrix<T, HEIGHT, WIDTH> {
    (0..WIDTH)
        .map(|_| {
            (0..HEIGHT)
                .map(|_| element())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

pub trait Zero {
    fn zero() -> Self;
}

impl<F: FieldExt> Zero for Expression<F> {
    fn zero() -> Self {
        Expression::Constant(F::zero())
    }
}

impl<F: FieldExt> Zero for Value<F> {
    fn zero() -> Self {
        Value::known(F::zero())
    }
}

pub fn convolution<
    T,
    const KERNEL_HEIGHT: usize,
    const KERNEL_WIDTH: usize,
    const IMAGE_HEIGHT: usize,
    const IMAGE_WIDTH: usize,
    const PADDING: usize,
    const STRIDE: usize,
>(
    kernel: Matrix<T, KERNEL_HEIGHT, KERNEL_WIDTH>,
    image: Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>,
) -> Matrix<
    T,
    { (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1 },
    { (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1 },
>
where
    T: Clone + Debug + Zero + Add<Output = T> + Mul<Output = T>,
{
    let padded_image = pad::<T, IMAGE_HEIGHT, IMAGE_WIDTH, PADDING>(image);

    let horz_slides = (IMAGE_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1;
    let vert_slides = (IMAGE_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1;

    (0..horz_slides)
        .map(|horz_slide| {
            let col_start = horz_slide * STRIDE;
            (0..vert_slides)
                .map(|vert_slide| {
                    let row_start = vert_slide * STRIDE;
                    dot_product(
                        kernel.clone(),
                        slice(padded_image.clone(), col_start, row_start),
                    )
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn dot_product<T, const WIDTH: usize, const HEIGHT: usize>(
    kernel: Matrix<T, HEIGHT, WIDTH>,
    image_subset: Matrix<T, HEIGHT, WIDTH>,
) -> T
where
    T: Clone + Debug + Zero + Add<Output = T> + Mul<Output = T>,
{
    kernel
        .flatten()
        .iter()
        .zip(image_subset.flatten().iter())
        .fold(T::zero(), |acc, (k, i)| acc + k.clone() * i.clone())
}

fn slice<
    T,
    const KERNEL_HEIGHT: usize,
    const KERNEL_WIDTH: usize,
    const IMAGE_HEIGHT: usize,
    const IMAGE_WIDTH: usize,
>(
    image: Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>,
    col_start: usize,
    row_start: usize,
) -> Matrix<T, KERNEL_HEIGHT, KERNEL_WIDTH>
where
    T: Clone + Debug + Zero,
{
    (&image[col_start..(col_start + KERNEL_WIDTH)])
        .iter()
        .map(|col| {
            (&col[row_start..(row_start + KERNEL_HEIGHT)])
                .to_vec()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn pad<T, const IMAGE_HEIGHT: usize, const IMAGE_WIDTH: usize, const PADDING: usize>(
    image: Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>,
) -> Matrix<T, { IMAGE_HEIGHT + 2 * PADDING }, { IMAGE_WIDTH + 2 * PADDING }>
where
    T: Clone + Debug + Zero,
{
    let mut output: [[T; IMAGE_HEIGHT + 2 * PADDING]; IMAGE_WIDTH + 2 * PADDING] = (0
        ..(IMAGE_HEIGHT + 2 * PADDING))
        .map(|_| {
            (0..(IMAGE_WIDTH + 2 * PADDING))
                .map(|_| T::zero())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    for col in 0..IMAGE_WIDTH {
        for row in 0..IMAGE_HEIGHT {
            output[col + PADDING][row + PADDING] = image[col][row].clone();
        }
    }

    output
}

pub fn op<T, const IMAGE_HEIGHT: usize, const IMAGE_WIDTH: usize>(
    images: Vec<Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>>,
    op: impl Fn(T, T) -> T + Clone,
) -> Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>
where
    T: Clone + Debug + Zero + Add<Output = T>,
{
    images.iter().skip(1).fold(images[0].clone(), |acc, image| {
        op_pair(acc, image.clone(), op.clone())
    })
}

pub fn op_pair<T, const IMAGE_HEIGHT: usize, const IMAGE_WIDTH: usize>(
    image: Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>,
    other: Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>,
    op: impl Fn(T, T) -> T + Clone,
) -> Matrix<T, IMAGE_HEIGHT, IMAGE_WIDTH>
where
    T: Clone + Debug + Zero + Add<Output = T>,
{
    (0..IMAGE_WIDTH)
        .map(|col_idx| {
            (0..IMAGE_HEIGHT)
                .map(|row_idx| {
                    op(
                        image[col_idx][row_idx].clone(),
                        other[col_idx][row_idx].clone(),
                    )
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
