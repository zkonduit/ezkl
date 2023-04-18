use core::panic;
use std::error::Error;

use halo2_proofs::circuit::{Region, Value};
use itertools::Itertools;
use log::{error, trace};

use crate::{
    circuit::{ops::base::BaseOp, utils, BaseConfig, CheckMode, CircuitError},
    fieldutils::i128_to_felt,
    tensor::{
        ops::{
            accumulated, add, affine as non_accum_affine, convolution as non_accum_conv,
            dot as non_accum_dot, matmul as non_accum_matmul, max_pool2d as non_accum_max_pool2d,
            mult, nonlinearities::instance_norm as ref_instance_norm,
            nonlinearities::prelu as ref_prelu, pack as non_accum_pack, rescale as ref_rescaled,
            scale_and_shift as ref_scale_and_shift, sub, sum as non_accum_sum,
            sumpool as non_accum_sumpool,
        },
        Tensor, TensorError, ValType,
    },
};

use super::*;
use crate::circuit::ops::lookup::LookupOp;

/// Dot product accumulated layout
pub fn dot<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 2],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut inputs = vec![];
    let mut assigned_len = 0;
    for (i, input) in values.iter().enumerate() {
        let inp = {
            let (res, len) = config.inputs[i].assign_with_duplication(
                region,
                *offset,
                input,
                &config.check_mode,
            )?;
            assigned_len = len;
            res.map(|e| e.value_field().evaluate())
        };
        inputs.push(inp);
    }

    // Now we can assign the dot product
    let accumulated_dot = accumulated::dot(&[inputs[0].clone(), inputs[1].clone()])
        .expect("accum poly: dot op failed");
    let (output, output_assigned_len) = config.output.assign_with_duplication(
        region,
        *offset,
        &accumulated_dot.into(),
        &config.check_mode,
    )?;

    assert_eq!(assigned_len, output_assigned_len);

    for i in 0..assigned_len {
        let (x, y) = config.output.cartesian_coord(*offset + i);
        // hop over duplicates at start of column
        if y == 0 && i > 0 {
            continue;
        }
        if i == 0 {
            config
                .selectors
                .get(&(BaseOp::Mult, x))
                .unwrap()
                .enable(region, y)?;
        } else {
            config
                .selectors
                .get(&(BaseOp::Dot, x))
                .unwrap()
                .enable(region, y)?;
        }
    }

    let last_elem = output
        .get_slice(&[output.len() - 1..output.len()])
        .expect("accum poly: failed to fetch last elem");

    if matches!(&config.check_mode, CheckMode::SAFE) {
        let safe_dot = non_accum_dot(&inputs.iter().collect()).map_err(|e| {
            error!("{}", e);
            halo2_proofs::plonk::Error::Synthesis
        })?;

        assert_eq!(
            Into::<Tensor<i32>>::into(last_elem.clone()),
            Into::<Tensor<i32>>::into(safe_dot),
        );
    }
    *offset += assigned_len;
    // last element is the result
    Ok(ValTensor::from(last_elem))
}

/// Sum accumulated layout
pub fn sum<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let assigned_len: usize;
    let input = {
        let (res, len) = config.inputs[1].assign_with_duplication(
            region,
            *offset,
            &values[0],
            &config.check_mode,
        )?;
        assigned_len = len;
        res.map(|e| e.value_field().evaluate())
    };

    // Now we can assign the dot product
    let accumulated_sum = accumulated::sum(&input).expect("accum poly: sum op failed");

    let (output, output_assigned_len) = config.output.assign_with_duplication(
        region,
        *offset,
        &accumulated_sum.into(),
        &config.check_mode,
    )?;

    assert_eq!(assigned_len, output_assigned_len);

    for i in 0..assigned_len {
        let (x, y) = config.output.cartesian_coord(*offset + i);
        // skip over duplicates at start of column
        if y == 0 && i > 0 {
            continue;
        }
        if i == 0 {
            config
                .selectors
                .get(&(BaseOp::Identity, x))
                .unwrap()
                .enable(region, y)?;
        } else {
            config
                .selectors
                .get(&(BaseOp::Sum, x))
                .unwrap()
                .enable(region, y)?;
        }
    }

    let last_elem = output
        .get_slice(&[output.len() - 1..output.len()])
        .expect("accum poly: failed to fetch last elem");

    if matches!(&config.check_mode, CheckMode::SAFE) {
        let safe_dot = non_accum_sum(&input).map_err(|e| {
            error!("{}", e);
            halo2_proofs::plonk::Error::Synthesis
        })?;

        assert_eq!(
            Into::<Tensor<i32>>::into(last_elem.clone()),
            Into::<Tensor<i32>>::into(safe_dot),
        )
    }

    *offset += assigned_len;
    // last element is the result
    Ok(ValTensor::from(last_elem))
}

/// Pairwise (elementwise) op layout
pub fn pairwise<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 2],
    offset: &mut usize,
    op: BaseOp,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != config.inputs.len() {
        return Err(Box::new(CircuitError::DimMismatch(format!(
            "pairwise {} layout",
            op.as_str()
        ))));
    }

    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    // casts a 1D addition
    if rhs.dims().len() == 1 && rhs.dims()[0] == 1 {
        rhs.tile(lhs.dims().iter().product::<usize>())?;
        rhs.reshape(lhs.dims())?;
    }
    // make 1D casting commutative
    else if lhs.dims().len() == 1 && lhs.dims()[0] == 1 {
        lhs.tile(rhs.dims().iter().product::<usize>())?;
        lhs.reshape(rhs.dims())?;
    }

    let mut inputs = vec![];

    for (i, input) in [lhs.clone(), rhs.clone()].iter().enumerate() {
        let inp = {
            let res = config.inputs[i].assign(region, *offset, input)?;
            res.map(|e| e.value_field().evaluate())
        };
        inputs.push(inp);
    }

    // Now we can assign the dot product
    let op_result = match op {
        BaseOp::Add => add(&inputs),
        BaseOp::Sub => sub(&inputs),
        BaseOp::Mult => mult(&inputs),
        _ => panic!(),
    }
    .map_err(|e| {
        error!("{}", e);
        halo2_proofs::plonk::Error::Synthesis
    })?;

    let mut output = config.output.assign(region, *offset, &op_result.into())?;

    for i in 0..inputs[0].len() {
        let (x, y) = config.inputs[0].cartesian_coord(*offset + i);
        config
            .selectors
            .get(&(op.clone(), x))
            .unwrap()
            .enable(region, y)?;
    }

    *offset += output.len();

    output.reshape(lhs.dims().clone());

    Ok(ValTensor::from(output))
}

/// Matrix multiplication accumulated layout
pub fn matmul<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 2],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if values.len() != 2 {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum matmul layout".to_string(),
        )));
    };

    // number of stacked matrices
    let num_stacked_a = Vec::from(&values[0].dims()[0..values[0].dims().len() - 2])
        .iter()
        .product::<usize>();
    let num_stacked_b = Vec::from(&values[1].dims()[0..values[1].dims().len() - 2])
        .iter()
        .product::<usize>();
    if num_stacked_a != num_stacked_b {
        return Err(Box::new(CircuitError::DimMismatch(
            "accum matmul layout".to_string(),
        )));
    };

    let num_stacked = num_stacked_a;

    // [m,n]
    let mut a_stacked = values[0].clone();
    a_stacked.reshape(&[
        num_stacked,
        a_stacked.dims()[a_stacked.dims().len() - 2],
        a_stacked.dims()[a_stacked.dims().len() - 1],
    ])?;
    // [n,k]
    let mut b_stacked = values[1].clone();
    b_stacked.reshape(&[
        num_stacked,
        b_stacked.dims()[b_stacked.dims().len() - 2],
        b_stacked.dims()[b_stacked.dims().len() - 1],
    ])?;

    let mut res = vec![];

    for i in 0..num_stacked {
        let mut a = a_stacked.get_slice(&[i..i + 1])?;
        a.reshape(&[a.dims()[1], a.dims()[2]])?;
        let mut b = b_stacked.get_slice(&[i..i + 1])?;
        b.reshape(&[b.dims()[1], b.dims()[2]])?;

        let original_a_dims = a.dims().to_vec();
        let original_b_dims = b.dims().to_vec();

        // [k,n]
        b.transpose_2d()?;

        // [k]
        let num_a_repeats = b.dims()[0];
        // [m]
        let num_b_tiles = a.dims()[0];
        let b_row_len = b.dims()[1];

        a.repeat_rows(num_a_repeats)?;
        b.tile(num_b_tiles)?;

        let mut inputs = vec![];
        let mut assigned_len = 0;
        for (i, elem) in vec![a.clone(), b.clone()].iter().enumerate() {
            let inp = {
                let (res, len) = config.inputs[i].assign_with_duplication(
                    region,
                    *offset,
                    elem,
                    &config.check_mode,
                )?;
                assigned_len = len;
                res.map(|e| e.value_field().evaluate())
            };
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

        inputs[0].reshape(&original_a_dims);

        // transpose it back to its normal shape
        inputs[1] = inputs[1].get_slice(&[0..1]).unwrap();
        inputs[1].reshape(&[original_b_dims[1], original_b_dims[0]]);
        inputs[1].transpose_2d().unwrap();

        // now perform matrix multiplication on the processed tensors
        let accumulated_matmul = accumulated::matmul(&[inputs[0].clone(), inputs[1].clone()])
            .expect("accum poly: matmul op failed");

        let (output, output_assigned_len) = config.output.assign_with_duplication(
            region,
            *offset,
            &accumulated_matmul.into(),
            &config.check_mode,
        )?;

        assert_eq!(assigned_len, output_assigned_len);

        let mut idx_wo_duplicates = 0;
        for i in 0..assigned_len {
            let (x, y) = config.output.cartesian_coord(*offset + i);
            // skip over duplicates at start of column
            if y == 0 && i > 0 {
                continue;
            }
            if idx_wo_duplicates % b_row_len > 0 {
                config
                    .selectors
                    .get(&(BaseOp::Dot, x))
                    .unwrap()
                    .enable(region, y)?;
            } else {
                config
                    .selectors
                    .get(&(BaseOp::Mult, x))
                    .unwrap()
                    .enable(region, y)?;
            }
            idx_wo_duplicates += 1;
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

        last_elem.reshape(&[original_a_dims[0], original_b_dims[1]]);

        if matches!(&config.check_mode, CheckMode::SAFE) {
            let safe_mm = non_accum_matmul(&inputs).map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(last_elem.clone()),
                Into::<Tensor<i32>>::into(safe_mm),
            )
        }

        *offset += assigned_len;

        res.push(last_elem.clone());
    }

    let mut res = Tensor::new(Some(&res), &[res.len()])?.combine()?;
    let mut res_dims = Vec::from(&values[0].dims()[0..values[0].dims().len() - 2]);
    res_dims.push(values[0].dims()[values[0].dims().len() - 2]);
    res_dims.push(values[1].dims()[values[0].dims().len() - 1]);

    res.reshape(&res_dims);

    // Now we can assign the matmul op
    Ok(ValTensor::from(res))
}

/// Affine operation accumulated layout
pub fn affine<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 3],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut input, kernel, mut bias) = (values[0].clone(), values[1].clone(), values[2].clone());

    if input.dims().len() == 1 {
        input.reshape(&[input.len(), 1])?;
    }
    if bias.dims().len() == 1 {
        bias.reshape(&[bias.len(), 1])?;
    }
    input.pad_row_ones()?;
    let params = kernel.append_to_row(bias)?;

    let mut last_elem = matmul(config, region, &[params, input], offset)?;
    last_elem.flatten();

    if matches!(&config.check_mode, CheckMode::SAFE) {
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
            .map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?),
                Into::<Tensor<i32>>::into(safe_affine),
            )
        }
    }
    Ok(last_elem)
}

/// Negation operation accumulated layout
pub fn neg<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let input = {
        let res = config.inputs[1].assign(region, *offset, &values[0])?;
        res.map(|e| e.value_field().evaluate())
    };

    let neg = input.map(|e| -e);

    let output = config.output.assign(region, *offset, &neg.into())?;

    *offset += output.len();

    Ok(output.into())
}

/// Sumpool accumulated layout
pub fn sumpool<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>],
    padding: (usize, usize),
    stride: (usize, usize),
    kernel_shape: (usize, usize),
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let image_channels = values[0].dims()[0];

    let unit = config.inputs[1].assign_constant(region, *offset, F::from(1))?;
    *offset += 1;

    let mut kernel = Tensor::from(0..kernel_shape.0 * kernel_shape.1).map(|_| unit.clone());
    kernel.reshape(&[1, 1, kernel_shape.0, kernel_shape.1]);

    let mut res = vec![];
    for i in 0..image_channels {
        res.push(conv(
            config,
            region,
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

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_sumpool =
                non_accum_sumpool(&values[0].get_inner()?, padding, stride, kernel_shape).map_err(
                    |e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    },
                )?;

            assert_eq!(
                Into::<Tensor<i32>>::into(last_elem.clone().get_inner()?),
                Into::<Tensor<i32>>::into(safe_sumpool),
            )
        }
    }
    Ok(last_elem)
}

/// Convolution accumulated layout
pub fn max_pool2d<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    padding: (usize, usize),
    stride: (usize, usize),
    pool_dims: (usize, usize),
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let image = values[0].clone();

    if image.dims().len() != 3 {
        return Err(Box::new(TensorError::DimMismatch("max_pool2d".to_string())));
    }
    let image_dims = image.dims();

    let input_channels = image_dims[0];
    let (image_height, image_width) = (image_dims[1], image_dims[2]);

    let mut padded_image = image.clone();
    padded_image.pad(padding)?;

    let horz_slides = (image_height + 2 * padding.0 - pool_dims.0) / stride.0 + 1;
    let vert_slides = (image_width + 2 * padding.1 - pool_dims.1) / stride.1 + 1;

    let mut output: Tensor<ValType<F>> =
        Tensor::new(None, &[input_channels, horz_slides, vert_slides]).unwrap();

    for i in 0..input_channels {
        for j in 0..horz_slides {
            let rs = j * stride.0;
            for k in 0..vert_slides {
                let cs = k * stride.1;
                let slice = padded_image.get_slice(&[
                    i..(i + 1),
                    rs..(rs + pool_dims.0),
                    cs..(cs + pool_dims.1),
                ])?;
                let max_w = max(config, region, &[slice], offset)?;
                let max_w = &max_w.get_inner_tensor()?[0];
                output.set(&[i, j, k], max_w.clone());
            }
        }
    }

    let res: ValTensor<F> = output.into();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(res.clone().get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_max_pool =
                non_accum_max_pool2d(&image.get_inner()?, &padding, &stride, &pool_dims)?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(safe_max_pool),
            )
        }
    }

    Ok(res)
}

/// Convolution accumulated layout
pub fn conv<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>],
    padding: (usize, usize),
    stride: (usize, usize),
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
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

    let mut res = if has_bias {
        let mut tiled_bias = values[2].clone();
        if (tiled_bias.dims().len() != 1) || (tiled_bias.dims()[0] != kernel.dims()[0]) {
            return Err(Box::new(TensorError::DimMismatch("conv bias".to_string())));
        }
        tiled_bias.repeat_rows(vert_slides * horz_slides)?;
        tiled_bias.flatten();
        tiled_bias.reshape(&[tiled_bias.dims()[0], 1])?;

        affine(
            config,
            region,
            &[padded_image, expanded_kernel, tiled_bias],
            offset,
        )?
    } else {
        matmul(config, region, &[expanded_kernel, padded_image], offset)?
    };

    res.reshape(&[output_channels, vert_slides, horz_slides])?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
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
            .map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(safe_conv),
            )
        }
    }

    Ok(res)
}
/// Power accumulated layout
pub fn pow<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    exponent: u32,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();

    for _ in 1..exponent {
        t = pairwise(
            config,
            region,
            &[t, values[0].clone()],
            offset,
            BaseOp::Mult,
        )?;
    }

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(t.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_pow = values[0].get_inner().unwrap().pow(exponent).map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(t.get_inner()?),
                Into::<Tensor<i32>>::into(safe_pow),
            )
        }
    }

    Ok(t)
}

/// Rescaled op accumulated layout
pub fn rescale<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>],
    scales: &[(usize, usize)],
    offset: &mut usize,
) -> Result<Vec<ValTensor<F>>, Box<dyn Error>> {
    let mut rescaled_inputs = vec![];
    for (i, ri) in values.iter().enumerate() {
        let num_elems = ri.dims().iter().product::<usize>();
        let mult = Value::known(F::from(scales[i].1 as u64));
        let mult_tensor = Tensor::new(Some(&vec![mult; num_elems]), ri.dims())?;
        let scaled_input = pairwise(
            config,
            region,
            &[ri.clone(), mult_tensor.into()],
            offset,
            BaseOp::Mult,
        )?;
        if matches!(&config.check_mode, CheckMode::SAFE) {
            // during key generation this will be 0 so we use this as a flag to check
            // TODO: this isn't very safe and would be better to get the phase directly
            let is_assigned = !Into::<Tensor<i32>>::into(scaled_input.clone().get_inner()?)
                .iter()
                .all(|&x| x == 0);
            if is_assigned {
                let safe_rescale =
                    ref_rescaled(&ri.get_inner().unwrap(), scales[i].1).map_err(|e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    })?;

                assert_eq!(
                    Into::<Tensor<i32>>::into(scaled_input.get_inner()?),
                    Into::<Tensor<i32>>::into(safe_rescale),
                )
            }
        }
        rescaled_inputs.push(scaled_input);
    }

    Ok(rescaled_inputs)
}

/// Pack accumulated layout
pub fn pack<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    base: u32,
    scale: u32,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();
    t.flatten();

    // these unwraps should never ever fail if the Tensortypes are correctly implemented
    // if anything we want these to hard fail if not implemented
    let mut base_t = <F as TensorType>::zero().unwrap();
    for _ in 0..base {
        base_t += <F as TensorType>::one().unwrap();
    }
    let mut accum_base = vec![];
    let base_tensor = Tensor::new(Some(&[base_t]), &[1])?;
    for i in 0..t.dims().iter().product::<usize>() {
        accum_base.push(Value::known(base_tensor.pow((i as u32) * (scale + 1))?[0]));
    }

    let base_tensor = Tensor::new(Some(&accum_base), &[accum_base.len()])?;
    let base_prod = pairwise(
        config,
        region,
        &[t.clone(), base_tensor.into()],
        offset,
        BaseOp::Mult,
    )?;

    let res = sum(config, region, &[base_prod], offset)?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(res.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let safe_pow = non_accum_pack(&values[0].get_inner()?, Value::known(base_t), scale)
                .map_err(|e| {
                    error!("{}", e);
                    halo2_proofs::plonk::Error::Synthesis
                })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(safe_pow),
            )
        }
    }

    Ok(res)
}

/// Dummy (no contraints) reshape layout
pub fn reshape<F: FieldExt + TensorType>(
    values: &[ValTensor<F>; 1],
    new_dims: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();
    t.reshape(new_dims)?;
    Ok(t)
}

/// Identity constraint. Usually used to constrain an instance column to an advice so the returned cells / values can be operated upon.
pub fn identity<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let output = config.output.assign(region, *offset, &values[0].clone())?;

    *offset += output.len();

    Ok(output.into())
}

/// Scale and shift accumulated layout
pub fn scale_and_shift<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 3],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (input, kernel, bias) = (values[0].clone(), values[1].clone(), values[2].clone());
    let prod = pairwise(config, region, &[input, kernel], offset, BaseOp::Mult)?;
    let res = pairwise(config, region, &[prod, bias], offset, BaseOp::Add)?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(res.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let ref_scale_and_shift = ref_scale_and_shift(
                &values
                    .iter()
                    .map(|x| x.get_inner().unwrap())
                    .collect::<Vec<Tensor<_>>>(),
            )
            .map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(ref_scale_and_shift),
            )
        }
    };
    Ok(res)
}

/// Layout for range check.
pub fn range_check<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 2],
    offset: &mut usize,
    tol: i32,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assigns the instance to the advice.
    config.inputs[1].assign(region, *offset, &values[0])?;

    let output = config.output.assign(region, *offset, &values[1])?;

    for i in 0..values[0].len() {
        let (x, y) = config.inputs[1].cartesian_coord(*offset + i);
        config
            .selectors
            .get(&(BaseOp::Range { tol }, x))
            .unwrap()
            .enable(region, y)?;
    }

    *offset += output.len();

    Ok(output.into())
}

///
pub fn nonlinearity<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    nl: &LookupOp,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let region_name = format!("Lookup for {:#?}", nl);

    let x = &values[0];

    trace!("laying out {}", region_name);

    let w = ValTensor::from(config.lookup_input.assign(region, *offset, x)?);
    // extract integer_valuations
    let integer_evals: Tensor<i128> = w
        .get_int_evals()
        .map_err(|e| {
            error!("{}", e);
            halo2_proofs::plonk::Error::Synthesis
        })?
        .into_iter()
        .into();

    // for key generation integer_evals will be empty and we need to return a set of unassigned values
    let output: Tensor<Value<F>> = match integer_evals.len() {
        // if empty return an unknown val
        0 => Tensor::from((0..x.dims().iter().product::<usize>()).map(|_| Value::unknown())),
        // if not empty apply the nonlinearity !
        _ => {
            let x = Op::<F>::f(nl, &[integer_evals])?;
            x.map(|elem| Value::known(i128_to_felt(elem)))
        }
    };

    let mut output = config
        .lookup_output
        .assign(region, *offset, &output.into())?;

    for i in 0..x.len() {
        let (x, y) = config.lookup_input.cartesian_coord(*offset + i);
        config
            .lookup_selectors
            .get(&(nl.clone(), x))
            .unwrap()
            .enable(region, y)?;
    }

    output.reshape(x.dims());

    *offset += x.len();

    // constrain the calculated output to a column
    Ok(ValTensor::from(output))
}

/// PrElu layout
pub fn prelu<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 2],
    scale: usize,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut slopes = values[1].clone();
    if slopes.len() != 1 && slopes.len() != values[0].dims()[0] {
        return Err(
            "slope must be a scalar or a vector of length equal to the number of channels".into(),
        );
    }

    let diff = (values[0].len()) / slopes.len();
    slopes.repeat_rows(diff)?;
    slopes.reshape(values[0].dims())?;

    let relu = nonlinearity(
        config,
        region,
        &[values[0].clone()],
        &LookupOp::ReLU { scale },
        offset,
    )?;
    // -x
    let neg_x = neg(config, region, &[values[0].clone()], offset)?;
    // relu(-x)
    let relu_neg_x = nonlinearity(config, region, &[neg_x], &LookupOp::ReLU { scale }, offset)?;
    // relu(-x) * slope
    let scaled_relu_neg_x = pairwise(config, region, &[relu_neg_x, slopes], offset, BaseOp::Mult)?;

    let prelu = pairwise(
        config,
        region,
        &[relu, scaled_relu_neg_x],
        offset,
        BaseOp::Sub,
    )?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(prelu.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let mut int_input: Tensor<i128> = values[0].get_int_evals()?.into_iter().into();
            int_input.reshape(values[0].dims());
            let ref_prelu = ref_prelu(
                &int_input,
                scale,
                &values[1]
                    .get_int_evals()?
                    .into_iter()
                    .map(|e| e as f32)
                    .collect_vec(),
            )
            .map(|e| e as i32);

            assert_eq!(
                Into::<Tensor<i32>>::into(prelu.get_inner()?),
                Into::<Tensor<i32>>::into(ref_prelu),
            )
        }
    };

    Ok(prelu)
}

/// mean function layout
pub fn mean<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    scale: usize,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let x = &values[0];

    let sum_x = sum(config, region, &[x.clone()], offset)?;
    let nl = LookupOp::Div {
        denom: utils::F32((scale * x.len()) as f32),
    };
    nonlinearity(config, region, &[sum_x], &nl, offset)
}

/// variance function layout
pub fn variance<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    scale: usize,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let x = &values[0];

    let mean = mean(config, region, &[x.clone()], scale, offset)?;

    let sub = pairwise(
        config,
        region,
        &[x.clone(), mean.clone()],
        offset,
        BaseOp::Sub,
    )?;

    let square = pairwise(config, region, &[sub.clone(), sub], offset, BaseOp::Mult)?;

    let sum_square = sum(config, region, &[square], offset)?;

    // biased estimator
    let nl = LookupOp::Div {
        denom: utils::F32((scale * x.len()) as f32),
    };

    let variance = nonlinearity(config, region, &[sum_square], &nl, offset)?;

    Ok(variance)
}

/// max layout
pub fn max<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // this is safe because we later constrain it
    let max_int = values[0].get_int_evals()?.into_iter().max();
    let max_val: ValTensor<F> = match max_int {
        None => Tensor::new(Some(&vec![Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&vec![Value::known(i128_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_max_val: ValTensor<F> = config.inputs[1].assign(region, *offset, &max_val)?.into();
    *offset += 1;

    let unit: ValTensor<F> = Tensor::from(
        vec![config.inputs[1].assign_constant(region, *offset, F::from(1))?].into_iter(),
    )
    .into();
    *offset += 1;

    // max(x - 1)
    let max_minus_1 = pairwise(
        config,
        region,
        &[assigned_max_val.clone(), unit.clone()],
        offset,
        BaseOp::Sub,
    )?;

    // x - max(x - 1)
    let diff = pairwise(
        config,
        region,
        &[values[0].clone(), max_minus_1.clone()],
        offset,
        BaseOp::Sub,
    )?;
    // relu(x - max(x - 1))
    let relu = nonlinearity(
        config,
        region,
        &[diff],
        &LookupOp::ReLU { scale: 1 },
        offset,
    )?;

    let len = relu.dims().iter().product();

    // y_i*(1 - y_i) =0 // assert the values are either 0 or 1
    config.inputs[1].assign(region, *offset, &relu)?;
    for i in 0..len {
        let (x, y) = config.output.cartesian_coord(*offset + i);
        config
            .selectors
            .get(&(BaseOp::IsBoolean, x))
            .unwrap()
            .enable(region, y)?;
    }
    *offset += len;

    // sum(relu(x - max(x - 1)))
    let sum_relu = sum(config, region, &[relu], offset)?;
    // 1 - sum(relu(x - max(x - 1)))
    let one_minus_sum_relu = pairwise(
        config,
        region,
        &[unit.clone(), sum_relu.clone()],
        offset,
        BaseOp::Sub,
    )?;
    // relu(1 - sum(relu(x - max(x - 1))))
    let relu_one_minus_sum_relu = nonlinearity(
        config,
        region,
        &[one_minus_sum_relu],
        &LookupOp::ReLU { scale: 1 },
        offset,
    )?;

    // constraining relu(sum(relu(x - max(x - 1)) - len(x))) = 0
    config.inputs[1].assign(region, *offset, &relu_one_minus_sum_relu)?;

    let (x, y) = config.output.cartesian_coord(*offset);
    config
        .selectors
        .get(&(BaseOp::IsZero, x))
        .unwrap()
        .enable(region, y)?;
    *offset += relu_one_minus_sum_relu.len();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(assigned_max_val.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let ref_max: Tensor<i32> = Tensor::new(
                Some(&[values[0].get_int_evals()?.into_iter().max().unwrap() as i32]),
                &[1],
            )?;

            assert_eq!(Into::<Tensor<i32>>::into(max_val.get_inner()?), ref_max,)
        }
    };
    Ok(assigned_max_val)
}

/// min layout
pub fn min<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 1],
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // this is safe because we later constrain it

    let min_int = values[0].get_int_evals()?.into_iter().min();
    let min_val: ValTensor<F> = match min_int {
        None => Tensor::new(Some(&vec![Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&vec![Value::known(i128_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_min_val: ValTensor<F> = config.inputs[1].assign(region, *offset, &min_val)?.into();
    *offset += 1;

    let unit: ValTensor<F> = Tensor::from(
        vec![config.inputs[1].assign_constant(region, *offset, F::from(1))?].into_iter(),
    )
    .into();
    *offset += 1;

    // min(x + 1)
    let min_plus_1 = pairwise(
        config,
        region,
        &[assigned_min_val.clone(), unit.clone()],
        offset,
        BaseOp::Add,
    )?;

    // min(x + 1)  - x
    let diff = pairwise(
        config,
        region,
        &[min_plus_1.clone(), values[0].clone()],
        offset,
        BaseOp::Sub,
    )?;

    // relu(min(x + 1)  - x)
    let relu = nonlinearity(
        config,
        region,
        &[diff],
        &LookupOp::ReLU { scale: 1 },
        offset,
    )?;

    let len = relu.dims().iter().product();

    // y_i*(1 - y_i) =0 // assert the values are either 0 or 1
    config.inputs[1].assign(region, *offset, &relu)?;
    for i in 0..len {
        let (x, y) = config.output.cartesian_coord(*offset + i);
        config
            .selectors
            .get(&(BaseOp::IsBoolean, x))
            .unwrap()
            .enable(region, y)?;
    }

    *offset += len;

    // sum(relu(min(x + 1) - x))
    let sum_relu = sum(config, region, &[relu], offset)?;
    // 1 - sum(relu(min(x + 1) - x))
    let one_minus_sum_relu = pairwise(
        config,
        region,
        &[unit.into(), sum_relu.clone()],
        offset,
        BaseOp::Sub,
    )?;
    // relu(1 - sum(relu(min(x + 1) - x)))
    let relu_one_minus_sum_relu = nonlinearity(
        config,
        region,
        &[one_minus_sum_relu],
        &LookupOp::ReLU { scale: 1 },
        offset,
    )?;

    // constraining product to 0
    config.inputs[1].assign(region, *offset, &relu_one_minus_sum_relu)?;

    let (x, y) = config.output.cartesian_coord(*offset);
    config
        .selectors
        .get(&(BaseOp::IsZero, x))
        .unwrap()
        .enable(region, y)?;
    *offset += relu_one_minus_sum_relu.len();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(assigned_min_val.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let ref_min: Tensor<i32> = Tensor::new(
                Some(&[values[0].get_int_evals()?.into_iter().min().unwrap() as i32]),
                &[1],
            )?;

            assert_eq!(Into::<Tensor<i32>>::into(min_val.get_inner()?), ref_min,)
        }
    };
    Ok(assigned_min_val.into())
}

///
pub fn instance_norm<F: FieldExt + TensorType>(
    config: &mut BaseConfig<F>,
    region: &mut Region<F>,
    values: &[ValTensor<F>; 3],
    scale: usize,
    epsilon: u64,
    offset: &mut usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let gamma = values[1].clone();
    let beta = values[2].clone();

    if gamma.len() != values[0].dims()[0] {
        return Err("gamma and x channels must have the same length".into());
    };
    if beta.len() != values[0].dims()[0] {
        return Err("beta and x channels must have the same length".into());
    };

    let mut channel_norms = vec![];

    // iterate over inner channel
    for i in 0..values[0].dims()[0] {
        let x = values[0].get_slice(&[i..i + 1])?;
        let mean = mean(config, region, &[x.clone()], scale, offset)?;
        let variance = variance(config, region, &[x.clone()], scale, offset)?;

        let numerator = pairwise(
            config,
            region,
            &[x.clone(), mean.clone()],
            offset,
            BaseOp::Sub,
        )?;

        let denominator = pairwise(
            config,
            region,
            &[
                variance.clone(),
                // TODO: should we make this a constant ? doesn't matter I think
                Tensor::from(vec![F::from(epsilon)].into_iter()).into(),
            ],
            offset,
            BaseOp::Add,
        )?;

        let denominator = nonlinearity(
            config,
            region,
            &[denominator],
            &LookupOp::Sqrt {
                scales: (scale, scale),
            },
            offset,
        )?;

        let product = pairwise(
            config,
            region,
            &[numerator.clone(), denominator.clone()],
            offset,
            BaseOp::Mult,
        )?;

        let result = numerator
            .get_int_evals()?
            .iter()
            .zip(denominator.get_int_evals()?)
            .map(|(x, y)| F::from((x / y) as u64))
            .collect::<Vec<_>>();

        let result = Tensor::from(result.into_iter()).into();

        // constraining product to 0
        let result = config.inputs[1].assign(region, *offset, &result)?;
        config.output.assign(region, *offset, &product)?;

        let (x, y) = config.output.cartesian_coord(*offset);
        config
            .selectors
            .get(&(BaseOp::Identity, x))
            .unwrap()
            .enable(region, y)?;
        *offset += result.len();

        let scaled_fraction = pairwise(
            config,
            region,
            &[result.into(), gamma.clone()],
            offset,
            BaseOp::Mult,
        )?;

        let instance_norm = pairwise(
            config,
            region,
            &[scaled_fraction, beta.clone()],
            offset,
            BaseOp::Add,
        )?;

        channel_norms.push(instance_norm.get_inner_tensor()?);
    }

    let mut instance_norm = Tensor::from(channel_norms.into_iter()).combine()?;
    instance_norm.reshape(&values[0].dims());
    let instance_norm: ValTensor<F> = instance_norm.into();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !Into::<Tensor<i32>>::into(instance_norm.get_inner()?)
            .iter()
            .all(|&x| x == 0);
        if is_assigned {
            let ref_instance_norm: Tensor<i32> = ref_instance_norm(
                [
                    Tensor::from(values[0].get_int_evals()?.into_iter()),
                    Tensor::from(gamma.get_int_evals()?.into_iter()),
                    Tensor::from(beta.get_int_evals()?.into_iter()),
                ],
                epsilon as f32,
            )
            .map(|x| x as i32);
            assert_eq!(
                Into::<Tensor<i32>>::into(instance_norm.get_inner()?),
                ref_instance_norm,
            )
        }
    };

    Ok(instance_norm)
}
