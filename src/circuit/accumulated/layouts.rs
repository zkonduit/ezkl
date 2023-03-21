use core::panic;
use std::error::Error;

use halo2_proofs::circuit::{Layouter, Value};

use crate::{
    circuit::{utils, CircuitError},
    tensor::{
        ops::{
            accumulated, add, affine as non_accum_affine, convolution as non_accum_conv,
            dot as non_accum_dot, matmul as non_accum_matmul, mult,
            scale_and_shift as ref_scale_and_shift, sub, sumpool as non_accum_sumpool,
        },
        Tensor, TensorError,
    },
};

use super::*;

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn dot<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 2],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != config.inputs.len() {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum dot layout".to_string(),
        )));
    }

    let t = match layouter.assign_region(
        || "assign inputs",
        |mut region| {
            let mut inputs = vec![];
            for (i, input) in values.iter().enumerate() {
                let inp = utils::value_muxer(
                    &config.inputs[i],
                    &{
                        let res = config.inputs[i].assign(&mut region, offset, input)?;
                        res.map(|e| e.value_field().evaluate())
                    },
                    input,
                );
                inputs.push(inp);
            }

            // Now we can assign the dot product
            let accumulated_dot = accumulated::dot(&inputs)
                .expect("accum poly: dot op failed")
                .into();
            let output = config
                .output
                .assign(&mut region, offset, &accumulated_dot)?;

            for i in 0..inputs[0].len() {
                let (_, y) = config.inputs[0].cartesian_coord(i);
                if y == 0 {
                    config
                        .selectors
                        .get(&BaseOp::InitDot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                } else {
                    config
                        .selectors
                        .get(&BaseOp::Dot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                }
            }

            let last_elem = output
                .get_slice(&[output.len() - 1..output.len()])
                .expect("accum poly: failed to fetch last elem");

            if matches!(config.check_mode, CheckMode::SAFE) {
                let safe_dot = non_accum_dot(&inputs.iter().map(|x| x).collect())
                    .map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

                assert_eq!(
                    Into::<Tensor<i32>>::into(last_elem.clone()),
                    Into::<Tensor<i32>>::into(safe_dot),
                )
            }
            // last element is the result
            Ok(last_elem)
        },
    ) {
        Ok(a) => a,
        Err(e) => {
            return Err(Box::new(e));
        }
    };

    Ok(ValTensor::from(t))
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn pairwise<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 2],
    offset: usize,
    op: BaseOp,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != config.inputs.len() {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum dot layout".to_string(),
        )));
    }

    let t = match layouter.assign_region(
        || "assign inputs",
        |mut region| {
            let mut inputs = vec![];
            for (i, input) in values.iter().enumerate() {
                let inp = utils::value_muxer(
                    &config.inputs[i],
                    &{
                        let res = config.inputs[i].assign(&mut region, offset, input)?;
                        res.map(|e| e.value_field().evaluate())
                    },
                    input,
                );
                inputs.push(inp);
            }

            // Now we can assign the dot product
            let op_result = match op {
                BaseOp::Add => add(&inputs),
                BaseOp::Sub => sub(&inputs),
                BaseOp::Mult => mult(&inputs),
                _ => panic!(),
            }
            .map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

            let output = config
                .output
                .assign(&mut region, offset, &op_result.into())?;

            for i in 0..inputs[0].len() {
                let (_, y) = config.inputs[0].cartesian_coord(i);
                config
                    .selectors
                    .get(&op)
                    .unwrap()
                    .enable(&mut region, offset + y)?;
            }

            Ok(output)
        },
    ) {
        Ok(a) => a,
        Err(e) => {
            return Err(Box::new(e));
        }
    };

    Ok(ValTensor::from(t))
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn matmul<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 2],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != 2 {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum matmul layout".to_string(),
        )));
    };

    let mut a = values[0].clone();
    let mut b = values[1].clone();
    b.transpose_2d()?;

    let num_a_repeats = b.dims()[0];
    let num_b_tiles = a.dims()[1];
    let b_row_len = b.dims()[1];

    a.repeat_rows(num_a_repeats)?;
    b.tile(num_b_tiles)?;

    let t = match layouter.assign_region(
        || "assign inputs",
        |mut region| {
            let mut inputs = vec![];

            for (i, elem) in vec![a.clone(), b.clone()].iter().enumerate() {
                let inp = utils::value_muxer(
                    &config.inputs[i],
                    &{
                        let res = config.inputs[i].assign(&mut region, offset, elem)?;
                        res.map(|e| e.value_field().evaluate())
                    },
                    elem,
                );
                inputs.push(inp);
            }

            // remove any repeats from the assignment
            if num_a_repeats > 1 {
                let dims = inputs[0].dims().to_vec();
                inputs[0].reshape(&[dims[0], dims[1..].iter().product()]);
                let mut rm_dup = vec![];
                for i in 0..dims[0] {
                    rm_dup.push(inputs[0].get_slice(&[i..i + 1, 0..dims[1]]).unwrap());
                }
                inputs[0] = Tensor::new(Some(&rm_dup), &[rm_dup.len()])
                    .unwrap()
                    .combine()
                    .unwrap();
            }

            inputs[0].reshape(values[0].dims());

            // transpose it back to its normal shape
            inputs[1] = inputs[1].get_slice(&[0..1]).unwrap();
            inputs[1].reshape(&[values[1].dims()[1], values[1].dims()[0]]);
            inputs[1].transpose_2d().unwrap();

            // now perform matrix multiplication on the processed tensors
            let accumulated_matmul =
                accumulated::matmul(&vec![inputs[0].clone(), inputs[1].clone()])
                    .expect("accum poly: matmul op failed");

            let output = config
                .output
                .assign(&mut region, offset, &accumulated_matmul.into())?;

            // these selectors map from
            for i in 0..a.dims().iter().product::<usize>() {
                let (_, y) = config.inputs[0].cartesian_coord(i);
                if (i) % b_row_len > 0 {
                    config
                        .selectors
                        .get(&BaseOp::Dot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                } else {
                    config
                        .selectors
                        .get(&BaseOp::InitDot)
                        .unwrap()
                        .enable(&mut region, offset + y)?;
                }
            }

            let dims = output.dims();
            let mut last_dims = vec![];

            for d in &dims[0..dims.len() - 1] {
                last_dims.push(0..*d);
            }
            let script_len = dims.last().unwrap();
            last_dims.push(script_len - 1..*script_len);

            let mut last_elem = output
                .get_slice(&last_dims)
                .expect("accum poly: failed to fetch last elem");

            last_elem.reshape(&[values[0].dims()[0], values[1].dims()[1]]);

            if matches!(config.check_mode, CheckMode::SAFE) {
                let safe_mm =
                    non_accum_matmul(&inputs).map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

                assert_eq!(
                    Into::<Tensor<i32>>::into(last_elem.clone()),
                    Into::<Tensor<i32>>::into(safe_mm),
                )
            }
            // Now we can assign the matmul op
            Ok(last_elem)
        },
    ) {
        Ok(a) => a,
        Err(e) => {
            return Err(Box::new(e));
        }
    };

    Ok(ValTensor::from(t))
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn affine<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 3],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut input, kernel, bias) = (values[0].clone(), values[1].clone(), values[2].clone());
    input.pad_row_ones()?;
    let params = kernel.append_to_row(bias)?;

    let mut last_elem = matmul(config, layouter, &[params, input], offset)?;
    last_elem.flatten();

    if matches!(config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_affine = non_accum_affine(
                &values
                    .iter()
                    .map(|x| x.get_inner().unwrap())
                    .collect::<Vec<Tensor<_>>>(),
            )
            .map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

            assert_eq!(
                Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?),
                Into::<Tensor<i32>>::into(safe_affine),
            )
        }
    }
    Ok(last_elem)
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn sumpool<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>],
    padding: (usize, usize),
    stride: (usize, usize),
    kernel_shape: (usize, usize),
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let image_channels = values[0].dims()[0];

    let mut kernel = Tensor::from(0..kernel_shape.0 * kernel_shape.1)
        .map(|_| Value::known(<F as TensorType>::one().unwrap()));
    kernel.reshape(&[1, 1, kernel_shape.0, kernel_shape.1]);

    let mut res = vec![];
    for i in 0..image_channels {
        res.push(conv(
            config,
            layouter,
            &[values[0].get_slice(&[i..i + 1])?, kernel.clone().into()],
            padding,
            stride,
            offset,
        )?);
    }
    let shape = &res[0].dims()[1..];
    let mut last_elem = res[1..].iter().fold(res[0].clone(), |acc, elem| {
        acc.concat(elem.clone()).unwrap()
    });
    last_elem.reshape(&[&[image_channels], shape].concat())?;

    if matches!(config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_sumpool =
                non_accum_sumpool(&values[0].get_inner()?, padding, stride, kernel_shape)
                    .map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

            assert_eq!(
                Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?),
                Into::<Tensor<i32>>::into(safe_sumpool),
            )
        }
    }
    Ok(last_elem)
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn conv<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>],
    padding: (usize, usize),
    stride: (usize, usize),
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assert!(stride.0 == 1);
    // assert!(stride.1 == 1);

    let has_bias = values.len() == 3;
    let (image, kernel) = (values[0].clone(), values[1].clone());

    if (image.dims().len() != 3)
        || (kernel.dims().len() != 4)
        || (image.dims()[0] != kernel.dims()[1])
    {
        return Err(Box::new(TensorError::DimMismatch("conv".to_string())));
    }

    let image_dims = image.dims();
    let kernel_dims = kernel.dims();

    let (output_channels, _input_channels, kernel_height, kernel_width) = (
        kernel_dims[0],
        kernel_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[1], image_dims[2]);
    let padded_height = image_height + 2 * padding.0;
    let padded_width = image_width + 2 * padding.1;

    let vert_slides = (padded_height - kernel_height) / stride.0 + 1;
    let horz_slides = (padded_width - kernel_width) / stride.1 + 1;

    let mut padded_image = image.clone();
    padded_image.pad(padding)?;
    padded_image.flatten();
    padded_image.reshape(&[padded_image.dims()[0], 1])?;

    let mut expanded_kernel = kernel.clone();

    expanded_kernel.multi_ch_blocked_toeplitz(
        vert_slides,
        padded_height,
        horz_slides,
        padded_width,
        stride.0,
        stride.1,
    )?;

    // expanded_kernel.downsample(0, stride.0)?;
    // expanded_kernel.downsample(0, stride.1)?;

    // println!("{:?}", expanded_kernel.dims());
    // println!("{:?}", padded_image.dims());

    let mut res = if has_bias {
        let mut tiled_bias = values[2].clone();
        if (tiled_bias.dims().len() != 1) || (tiled_bias.dims()[0] != kernel.dims()[0]) {
            return Err(Box::new(TensorError::DimMismatch("conv bias".to_string())));
        }
        // TODO: don't know if this correct
        tiled_bias.repeat_rows(vert_slides * horz_slides)?;
        tiled_bias.flatten();
        tiled_bias.reshape(&[tiled_bias.dims()[0], 1])?;

        affine(
            config,
            layouter,
            &[padded_image, expanded_kernel, tiled_bias],
            offset,
        )?
    } else {
        matmul(config, layouter, &[expanded_kernel, padded_image], offset)?
    };

    res.reshape(&[output_channels, vert_slides, horz_slides])?;

    if matches!(config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(res.clone().get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_conv = non_accum_conv(
                &values
                    .iter()
                    .map(|x| x.get_inner().unwrap())
                    .collect::<Vec<Tensor<_>>>(),
                padding,
                stride,
            )
            .map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(safe_conv),
            )
        }
    }

    Ok(res)
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn reshape<F: FieldExt + TensorType>(
    values: &[ValTensor<F>; 1],
    new_dims: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();
    t.reshape(new_dims)?;
    Ok(t)
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn identity<F: FieldExt + TensorType>(
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    Ok(values[0].clone())
}

/// Assigns variables to the regions created when calling `configure`.
/// # Arguments
/// * `values` - The explicit values to the operations.
/// * `layouter` - A Halo2 Layouter.
pub fn scale_and_shift<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    layouter: &mut impl Layouter<F>,
    values: &[ValTensor<F>; 3],
    offset: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (input, kernel, bias) = (values[0].clone(), values[1].clone(), values[2].clone());
    let prod = pairwise(config, layouter, &[input, kernel], offset, BaseOp::Mult)?;
    let res = pairwise(config, layouter, &[prod, bias], offset, BaseOp::Add)?;

    if matches!(config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(res.clone().get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let ref_scale_and_shift = ref_scale_and_shift(
                &values
                    .iter()
                    .map(|x| x.get_inner().unwrap())
                    .collect::<Vec<Tensor<_>>>(),
            )
            .map_err(|_| halo2_proofs::plonk::Error::Synthesis)?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(ref_scale_and_shift),
            )
        }
    };
    Ok(res)
}
