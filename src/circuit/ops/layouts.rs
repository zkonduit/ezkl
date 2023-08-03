use core::panic;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::error;

use super::{
    chip::{BaseConfig, CheckMode, CircuitError},
    region::RegionCtx,
};
use crate::{
    circuit::{ops::base::BaseOp, utils},
    fieldutils::i128_to_felt,
    tensor::{
        get_broadcasted_shape,
        ops::{
            accumulated, add, conv as non_accum_conv, deconv as non_accum_deconv,
            dot as non_accum_dot, einsum as non_accum_einsum, max_pool2d as non_accum_max_pool2d,
            mult, pack as non_accum_pack, sub, sum as non_accum_sum, sumpool as non_accum_sumpool,
        },
        Tensor, TensorError, ValType,
    },
};

use super::*;
use crate::circuit::ops::lookup::LookupOp;

fn overflowed_len(starting_idx: usize, mut total_len: usize, column_len: usize) -> usize {
    let mut idx = starting_idx;
    // let x = idx / column_len;
    let y = idx % column_len;
    if y + total_len < column_len {
        return total_len;
    }
    // fill up first column
    idx += column_len - y;
    total_len += 1;
    loop {
        if idx >= starting_idx + total_len {
            break;
        }
        idx += column_len;
        total_len += 1;
    }
    total_len
}

/// Dot product accumulated layout
pub fn dot<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut values = values.clone();

    let mut removal_indices: HashSet<usize> = HashSet::new();
    removal_indices.extend(values[0].get_const_zero_indices()?);
    removal_indices.extend(values[1].get_const_zero_indices()?);
    let removal_indices = removal_indices.into_iter().collect_vec();
    values[0].remove_indices(&removal_indices)?;
    values[1].remove_indices(&removal_indices)?;

    if values[0].len() != values[1].len() {
        return Err(Box::new(TensorError::DimMismatch("dot".to_string())));
    }

    // if empty return a const
    if values[0].len() == 0 && values[1].len() == 0 {
        return Ok(Tensor::from([ValType::Constant(F::ZERO)].into_iter()).into());
    }

    if region.is_dummy() {
        let overflowed_len =
            overflowed_len(region.offset(), values[0].len(), config.output.col_size());
        region.increment(overflowed_len);
        return Ok(Tensor::from([ValType::from(Value::known(F::ZERO))].into_iter()).into());
    }

    let mut inputs = vec![];
    let mut assigned_len = 0;
    for (i, input) in values.iter().enumerate() {
        let inp = {
            let (res, len) =
                region.assign_with_duplication(&config.inputs[i], input, &config.check_mode)?;
            assigned_len = len;
            res.get_inner()?
        };
        inputs.push(inp);
    }

    // Now we can assign the dot product
    let accumulated_dot = accumulated::dot(&[inputs[0].clone(), inputs[1].clone()])
        .expect("accum poly: dot op failed");

    let (output, output_assigned_len) = region.assign_with_duplication(
        &config.output,
        &accumulated_dot.into(),
        &config.check_mode,
    )?;

    assert_eq!(assigned_len, output_assigned_len);

    // enable the selectors
    (0..assigned_len).for_each(|i| {
        let (x, y) = config.output.cartesian_coord(region.offset() + i);
        // hop over duplicates at start of column
        if y == 0 && i > 0 {
            return;
        }
        let selector = if i == 0 {
            config.selectors.get(&(BaseOp::Mult, x))
        } else {
            config.selectors.get(&(BaseOp::Dot, x))
        };
        region.enable(selector, y).unwrap();
    });

    let last_elem = output
        .get_slice(&[output.len() - 1..output.len()])
        .expect("dot: failed to fetch last elem");

    if matches!(&config.check_mode, CheckMode::SAFE) {
        let safe_dot = non_accum_dot(&inputs[..]).map_err(|e| {
            error!("{}", e);
            halo2_proofs::plonk::Error::Synthesis
        })?;

        assert_eq!(
            Into::<Tensor<i32>>::into(last_elem.get_inner()?),
            Into::<Tensor<i32>>::into(safe_dot),
        );
    }

    region.increment(assigned_len);
    // last element is the result
    Ok(last_elem)
}

/// Einsum
pub fn einsum<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    inputs: &mut [ValTensor<F>],
    equation: &str,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // Parse equation into an operation
    let original_eq = equation.to_string();

    let mut equation = equation.split("->");
    let inputs_eq = equation.next().unwrap();
    let output_eq = equation.next().unwrap();
    let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();

    for (i, input) in inputs.iter_mut().enumerate() {
        if input.dims().len() != inputs_eq[i].len()
            && input.dims().len() == 1
            && inputs_eq[i].len() == 2
        {
            input.reshape(&[1, input.dims()[0]])?;
        } else if input.dims().len() != inputs_eq[i].len() {
            return Err(Box::new(TensorError::DimMismatch("einsum".to_string())));
        }
    }

    // Check that the number of inputs matches the number of inputs in the equation
    if inputs.len() != inputs_eq.len() {
        return Err(Box::new(TensorError::DimMismatch("einsum".to_string())));
    }

    let mut indices_to_size = HashMap::new();
    for (i, input) in inputs.iter().enumerate() {
        for j in 0..inputs_eq[i].len() {
            let c = inputs_eq[i].chars().nth(j).unwrap();
            if let std::collections::hash_map::Entry::Vacant(e) = indices_to_size.entry(c) {
                e.insert(input.dims()[j]);
            } else if indices_to_size[&c] != input.dims()[j] {
                return Err(Box::new(TensorError::DimMismatch("einsum".to_string())));
            }
        }
    }

    // maps unrepresented indices in the output to a trivial 1
    for c in output_eq.chars() {
        indices_to_size.entry(c).or_insert(1);
    }

    // Compute the output tensor shape
    let mut output_shape: Vec<usize> = output_eq
        .chars()
        .map(|c| *indices_to_size.get(&c).unwrap())
        .collect();

    if output_shape.is_empty() {
        output_shape.push(1);
    }

    // Create a new output tensor with the computed shape
    let mut output: Tensor<ValType<F>> = Tensor::new(None, &output_shape)?;

    let mut seen = HashSet::new();
    let mut common_indices_to_inputs = vec![];
    for input in inputs_eq.iter().take(inputs.len()) {
        for c in input.chars() {
            if !seen.contains(&c) {
                seen.insert(c);
            } else {
                common_indices_to_inputs.push(c);
            }
        }
    }

    let non_common_indices = indices_to_size
        .keys()
        .filter(|&x| !common_indices_to_inputs.contains(x))
        .collect::<Vec<_>>();

    let non_common_coord_size = non_common_indices
        .iter()
        .map(|d| {
            // If the current index is in the output equation, then the slice should be the current coordinate
            if output_eq.contains(**d) {
                1
            // Otherwise, the slice should be the entire dimension of the input tensor
            } else {
                *indices_to_size.get(d).unwrap()
            }
        })
        .product::<usize>();

    let cartesian_coord = output_shape
        .iter()
        .map(|d| 0..*d)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    // Get the indices common accross input tensors
    let mut common_coord = common_indices_to_inputs
        .iter()
        .map(|d| {
            // If the current index is in the output equation, then the slice should be the current coordinate
            if output_eq.contains(*d) {
                0..1
            // Otherwise, the slice should be the entire dimension of the input tensor
            } else {
                0..*indices_to_size.get(d).unwrap()
            }
        })
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    // If there are no common indices, then we need to add an empty slice to force one iteration of the loop
    if common_coord.is_empty() {
        common_coord.push(vec![]);
    }

    output.iter_mut().enumerate().for_each(|(i, o)| {
        let coord = cartesian_coord[i].clone();
        // Compute the slice of each input tensor given the current coordinate of the output tensor
        let inputs = (0..inputs.len())
            .map(|idx| {
                let mut slice = vec![];
                for (i, c) in inputs_eq[idx].chars().enumerate() {
                    // If the current index is in the output equation, then the slice should be the current coordinate
                    if let Some(idx) = output_eq.find(c) {
                        slice.push(coord[idx]..coord[idx] + 1);
                    // Otherwise, the slice should be the entire dimension of the input tensor
                    } else {
                        slice.push(0..inputs[idx].dims()[i]);
                    }
                }
                // Get the slice of the input tensor
                inputs[idx].get_slice(&slice).unwrap()
            })
            .collect::<Vec<_>>();

        // in this case its just a dot product :)
        if non_common_coord_size == 1 && inputs.len() == 2 {
            *o = dot(config, region, inputs[..].try_into().unwrap())
                .unwrap()
                .get_inner_tensor()
                .unwrap()[0]
                .clone();
        } else {
            let mut prod = None;

            // Compute the cartesian product of all common indices
            for common_dim in &common_coord {
                let inputs = (0..inputs.len())
                    .map(|idx| {
                        let mut slice = vec![];
                        // Iterate over all indices in the input equation
                        for (i, c) in inputs_eq[idx].chars().enumerate() {
                            // If the current index is common to multiple inputs, then the slice should be the current coordinate
                            if let Some(j) = common_indices_to_inputs.iter().position(|&r| r == c) {
                                slice.push(common_dim[j]..common_dim[j] + 1);
                            } else {
                                slice.push(0..inputs[idx].dims()[i]);
                            }
                        }
                        // Get the slice of the input tensor
                        inputs[idx].get_slice(&slice).unwrap()
                    })
                    .collect::<Vec<_>>();

                let input_pairs = inputs
                    .iter()
                    .map(|d| d.get_inner_tensor().into_iter())
                    .multi_cartesian_product()
                    .collect::<Vec<_>>();

                // Compute the product of all input tensors
                for pair in input_pairs {
                    let product_across_pair =
                        pair[1..]
                            .iter()
                            .fold(ValTensor::from(pair[0].clone()), |acc, x| {
                                pairwise(config, region, &[acc, x.clone().into()], BaseOp::Mult)
                                    .unwrap()
                            });

                    if prod.is_none() {
                        prod = Some(product_across_pair);
                    } else {
                        prod = Some(
                            pairwise(
                                config,
                                region,
                                &[prod.unwrap(), product_across_pair],
                                BaseOp::Add,
                            )
                            .unwrap(),
                        );
                    }
                }
            }

            *o = prod.unwrap().get_inner_tensor().unwrap()[0].clone();
        }
    });

    let output: ValTensor<F> = output.into();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !output.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let safe_einsum = non_accum_einsum(
                &original_eq,
                &inputs
                    .iter()
                    .map(|x| x.get_inner().unwrap())
                    .collect::<Vec<Tensor<_>>>(),
            )
            .map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(output.clone().get_inner()?),
                Into::<Tensor<i32>>::into(safe_einsum),
            )
        }
    }

    Ok(output.into())
}

/// Sum accumulated layout
pub fn sum<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let assigned_len: usize;
    let input = {
        let (res, len) =
            region.assign_with_duplication(&config.inputs[1], &values[0], &config.check_mode)?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_sum = accumulated::sum(&input).expect("accum poly: sum op failed");

    let (output, output_assigned_len) = region.assign_with_duplication(
        &config.output,
        &accumulated_sum.into(),
        &config.check_mode,
    )?;

    assert_eq!(assigned_len, output_assigned_len);

    // enable the selectors
    (0..assigned_len).for_each(|i| {
        let (x, y) = config.output.cartesian_coord(region.offset() + i);
        // skip over duplicates at start of column
        if y == 0 && i > 0 {
            return;
        }
        let selector = if i == 0 {
            config.selectors.get(&(BaseOp::Identity, x))
        } else {
            config.selectors.get(&(BaseOp::Sum, x))
        };

        region.enable(selector, y).unwrap();
    });

    let last_elem = output
        .get_slice(&[output.len() - 1..output.len()])
        .expect("accum poly: failed to fetch last elem");

    if matches!(&config.check_mode, CheckMode::SAFE) {
        let safe_dot = non_accum_sum(&input).map_err(|e| {
            error!("{}", e);
            halo2_proofs::plonk::Error::Synthesis
        })?;

        assert_eq!(
            Into::<Tensor<i32>>::into(last_elem.get_inner()?),
            Into::<Tensor<i32>>::into(safe_dot),
        )
    }
    region.increment(assigned_len);

    // last element is the result
    Ok(last_elem)
}

/// Sum accumulated layout
pub fn sum_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output

    let a = &values[0];

    if axes.is_empty() {
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
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(*c..*c + 1);
            }
        }
        res.set(
            coord,
            sum(config, region, &[a.get_slice(&sum_dims)?])?.get_inner_tensor()?[0].clone(),
        );
    }

    Ok(res.into())
}

/// Max accumulated layout
pub fn max_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output

    let a = &values[0];

    if axes.is_empty() {
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
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(*c..*c + 1);
            }
        }
        res.set(
            coord,
            max(config, region, &[a.get_slice(&sum_dims)?])?.get_inner_tensor()?[0].clone(),
        );
    }

    Ok(res.into())
}

/// Min accumulated layout
pub fn min_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output

    let a = &values[0];

    if axes.is_empty() {
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
        for (i, c) in coord.iter().enumerate().take(a.dims().len()) {
            if axes.contains(&i) {
                sum_dims.push(0..a.dims()[i]);
            } else {
                sum_dims.push(*c..*c + 1);
            }
        }

        res.set(
            coord,
            min(config, region, &[a.get_slice(&sum_dims)?])?.get_inner_tensor()?[0].clone(),
        );
    }

    Ok(res.into())
}

/// Pairwise (elementwise) op layout
pub fn pairwise<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    op: BaseOp,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;
    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    let orig_lhs = lhs.clone();
    let orig_rhs = rhs.clone();

    let zero_indices_a = lhs.get_const_zero_indices()?;
    let zero_indices_b = rhs.get_const_zero_indices()?;
    let zero_indices = zero_indices_a
        .iter()
        .chain(zero_indices_b.iter())
        .cloned()
        .collect::<HashSet<_>>()
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    lhs.remove_indices(&zero_indices)?;
    rhs.remove_indices(&zero_indices)?;

    if lhs.len() != rhs.len() {
        return Err(Box::new(CircuitError::DimMismatch(format!(
            "pairwise {} layout",
            op.as_str()
        ))));
    }

    if region.is_dummy() {
        region.increment(lhs.len());
        let vals = vec![ValType::Value(Value::known(F::ZERO)); broadcasted_shape.iter().product()];
        let mut tensor: Tensor<ValType<F>> = Tensor::from(vals.into_iter());
        tensor.reshape(&broadcasted_shape);
        return Ok(tensor.into());
    }

    let mut inputs = vec![];

    for (i, input) in [lhs.clone(), rhs.clone()].iter().enumerate() {
        let inp = {
            let res = region.assign(&config.inputs[i], input)?;
            res.get_inner()?
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

    let output = region.assign(&config.output, &op_result.into())?;

    // Enable the selectors
    (0..inputs[0].len()).for_each(|i| {
        let (x, y) = config.inputs[0].cartesian_coord(region.offset() + i);
        let selector = config.selectors.get(&(op.clone(), x));

        region.enable(selector, y).unwrap();
    });

    region.increment(output.len());

    let mut j = 0;
    // infill the zero indices with the correct values from values[0] or values[1]
    let mut actual_output: ValTensor<F> =
        Into::<Tensor<ValType<F>>>::into((0..broadcasted_shape.iter().product()).map(|i| {
            if zero_indices.contains(&i) {
                let a = orig_lhs.get_inner_tensor().unwrap()[i].clone();
                let b = orig_rhs.get_inner_tensor().unwrap()[i].clone();
                let a_is_null = zero_indices_a.contains(&i);
                let b_is_null = zero_indices_b.contains(&i);
                let both_null = a_is_null && b_is_null;

                match op {
                    BaseOp::Add => {
                        if both_null {
                            ValType::Constant(F::ZERO)
                        } else if a_is_null {
                            b.clone()
                        } else if b_is_null {
                            a.clone()
                        } else {
                            ValType::Constant(F::ZERO)
                        }
                    }
                    BaseOp::Sub => {
                        if both_null {
                            ValType::Constant(F::ZERO)
                        } else if a_is_null {
                            let tensor = Tensor::new(Some(&[b]), &[1]).unwrap();
                            neg(config, region, &[tensor.into()])
                                .unwrap()
                                .get_inner_tensor()
                                .unwrap()[0]
                                .clone()
                        } else if b_is_null {
                            a.clone()
                        } else {
                            ValType::Constant(F::ZERO)
                        }
                    }
                    BaseOp::Mult => ValType::Constant(F::ZERO),
                    _ => panic!(),
                }
            } else {
                let val = output.get_inner_tensor().unwrap()[j].clone();
                j += 1;
                val
            }
        }))
        .into();

    actual_output.reshape(&broadcasted_shape)?;

    Ok(actual_output)
}

/// Iff
pub fn iff<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 3],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // if mask > 0 then output a else output b
    let (mask, b, a) = (&values[0], &values[1], &values[2]);

    let unit: ValTensor<F> =
        Tensor::from(vec![region.assign_constant(&config.inputs[1], F::from(1))?].into_iter())
            .into();
    region.next();

    // make sure mask is boolean
    let assigned_mask = region.assign(&config.inputs[1], mask)?;

    // Enable the selectors
    (0..assigned_mask.len()).for_each(|i| {
        let (x, y) = config.inputs[1].cartesian_coord(region.offset() + i);
        let selector = config.selectors.get(&(BaseOp::IsBoolean, x));

        region.enable(selector, y).unwrap();
    });

    region.increment(assigned_mask.len());

    let one_minus_mask = pairwise(config, region, &[unit, assigned_mask.clone()], BaseOp::Sub)?;

    let masked_a = pairwise(config, region, &[a.clone(), assigned_mask], BaseOp::Mult)?;
    let masked_b = pairwise(config, region, &[b.clone(), one_minus_mask], BaseOp::Mult)?;

    let output = pairwise(config, region, &[masked_a, masked_b], BaseOp::Add)?;

    // Now we can assign the matmul op
    Ok(output)
}

/// Negation operation accumulated layout
pub fn neg<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let input = {
        let res = region.assign(&config.inputs[1], &values[0])?;
        res.get_inner()?
    };

    let neg = input.map(|e| -e);

    let output = region.assign(&config.output, &neg.into())?;

    // Enable the selectors
    (0..values[0].len()).for_each(|i| {
        let (x, y) = config.inputs[1].cartesian_coord(region.offset() + i);
        let selector = config.selectors.get(&(BaseOp::Neg, x));

        region.enable(selector, y).unwrap();
    });

    region.increment(output.len());

    Ok(output)
}

/// Sumpool accumulated layout
pub fn sumpool<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    padding: (usize, usize),
    stride: (usize, usize),
    kernel_shape: (usize, usize),
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let batch_size = values[0].dims()[0];
    let image_channels = values[0].dims()[1];

    let unit = region.assign_constant(&config.inputs[1], F::from(1))?;
    region.next();

    let mut kernel = Tensor::from(0..kernel_shape.0 * kernel_shape.1).map(|_| unit.clone());
    kernel.reshape(&[1, 1, kernel_shape.0, kernel_shape.1]);

    let cartesian_coord = vec![(0..batch_size), (0..image_channels)]
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut res = vec![];

    cartesian_coord.iter().for_each(|coord| {
        let (b, i) = (coord[0], coord[1]);
        let input = values[0].get_slice(&[b..b + 1, i..i + 1]).unwrap();
        let output = conv(
            config,
            region,
            &[input, kernel.clone().into()],
            padding,
            stride,
        )
        .unwrap();
        res.push(output);
    });

    let shape = &res[0].dims()[2..];
    let mut last_elem = res[1..].iter().fold(res[0].clone(), |acc, elem| {
        acc.concat(elem.clone()).unwrap()
    });
    last_elem.reshape(&[&[batch_size, image_channels], shape].concat())?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !last_elem.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
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
pub fn max_pool2d<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    padding: (usize, usize),
    stride: (usize, usize),
    pool_dims: (usize, usize),
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let image = values[0].clone();

    if image.dims().len() != 4 {
        return Err(Box::new(TensorError::DimMismatch("max_pool2d".to_string())));
    }
    let image_dims = image.dims();

    let (batch, input_channels, image_height, image_width) =
        (image_dims[0], image_dims[1], image_dims[2], image_dims[3]);

    let mut padded_image = image.clone();
    padded_image.pad(padding)?;

    let vert_slides = (image_height + 2 * padding.0 - pool_dims.0) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - pool_dims.1) / stride.1 + 1;

    let mut output: Tensor<ValType<F>> =
        Tensor::new(None, &[batch, input_channels, horz_slides, vert_slides])?;

    let cartesian_coord = vec![
        (0..batch),
        (0..input_channels),
        (0..vert_slides),
        (0..horz_slides),
    ]
    .iter()
    .cloned()
    .multi_cartesian_product()
    .collect::<Vec<_>>();

    output.iter_mut().enumerate().for_each(|(flat_index, o)| {
        let coord = &cartesian_coord[flat_index];
        let (b, i, j, k) = (coord[0], coord[1], coord[2], coord[3]);
        let rs = j * stride.0;
        let cs = k * stride.1;
        let slice = padded_image
            .get_slice(&[
                b..(b + 1),
                i..(i + 1),
                rs..(rs + pool_dims.0),
                cs..(cs + pool_dims.1),
            ])
            .unwrap();
        let max_w = max(config, region, &[slice]).unwrap();
        *o = max_w.get_inner_tensor().unwrap()[0].clone();
    });

    let res: ValTensor<F> = output.into();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !res.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let raw_values = Into::<Tensor<i32>>::into(image.get_inner()?);
            let safe_max_pool = non_accum_max_pool2d(&raw_values, &padding, &stride, &pool_dims)?;

            assert_eq!(
                Into::<Tensor<i32>>::into(res.get_inner()?),
                Into::<Tensor<i32>>::into(safe_max_pool),
            )
        }
    }

    Ok(res)
}

/// DeConvolution accumulated layout
pub fn deconv<F: PrimeField + TensorType + PartialOrd + std::marker::Send + std::marker::Sync>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    inputs: &[ValTensor<F>],
    padding: (usize, usize),
    output_padding: (usize, usize),
    stride: (usize, usize),
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let has_bias = inputs.len() == 3;
    let (image, kernel) = (&inputs[0], &inputs[1]);

    if (image.dims().len() != 4) || (kernel.dims().len() != 4) {
        return Err(Box::new(TensorError::DimMismatch("deconv".to_string())));
    }

    if stride.0 == 0 || stride.1 == 0 {
        return Err(Box::new(TensorError::DimMismatch(
            "non-positive stride is not supported for deconv".to_string(),
        )));
    }

    if has_bias {
        let bias = &inputs[2];
        if (bias.dims().len() != 1) || (bias.dims()[0] != kernel.dims()[0]) {
            return Err(Box::new(TensorError::DimMismatch(
                "deconv bias".to_string(),
            )));
        }
    }

    let (kernel_height, kernel_width) = (kernel.dims()[2], kernel.dims()[3]);

    let null_val = ValType::Constant(F::ZERO);
    // region.assign_constant(&config.inputs[1], F::from(0))?;
    // region.next();

    let mut expanded_image = image.clone();
    expanded_image.intercalate_values(null_val.clone(), stride.0, 2)?;
    expanded_image.intercalate_values(null_val, stride.1, 3)?;
    expanded_image.pad((kernel_height - 1, kernel_width - 1))?;

    // flip order
    let channel_coord = (0..kernel.dims()[0])
        .cartesian_product(0..kernel.dims()[1])
        .collect::<Vec<_>>();

    let slice_coord = expanded_image
        .dims()
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if i == 2 {
                padding.0..d - padding.0 + output_padding.0
            } else if i == 3 {
                padding.1..d - padding.1 + output_padding.1
            } else {
                0..*d
            }
        })
        .collect::<Vec<_>>();

    let sliced_expanded_image = expanded_image.get_slice(&slice_coord)?;

    let mut inverted_kernels = vec![];

    for (i, j) in channel_coord {
        let channel = kernel.get_slice(&[i..i + 1, j..j + 1])?;
        let mut channel = Tensor::from(channel.get_inner_tensor()?.into_iter().rev());
        channel.reshape(&[kernel.dims()[2], kernel.dims()[3]]);
        inverted_kernels.push(channel);
    }

    let mut deconv_kernel =
        Tensor::new(Some(&inverted_kernels), &[inverted_kernels.len()])?.combine()?;
    deconv_kernel.reshape(kernel.dims());

    // tensorflow formatting patch
    if kernel.dims()[0] == sliced_expanded_image.dims()[1] {
        deconv_kernel.reshape(&[
            kernel.dims()[1],
            kernel.dims()[0],
            kernel.dims()[2],
            kernel.dims()[3],
        ]);
    }

    let conv_input = if has_bias {
        vec![
            sliced_expanded_image,
            deconv_kernel.clone().into(),
            inputs[2].clone(),
        ]
    } else {
        vec![sliced_expanded_image, deconv_kernel.clone().into()]
    };

    let output = conv(config, region, &conv_input, (0, 0), (1, 1))?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !output.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let safe_conv = non_accum_deconv(
                &inputs
                    .iter()
                    .map(|x| x.get_inner().unwrap())
                    .collect::<Vec<Tensor<_>>>(),
                padding,
                output_padding,
                stride,
            )
            .map_err(|e| {
                error!("{}", e);
                halo2_proofs::plonk::Error::Synthesis
            })?;

            assert_eq!(
                Into::<Tensor<i32>>::into(output.get_inner()?),
                Into::<Tensor<i32>>::into(safe_conv),
            )
        }
    }

    Ok(output)
}

/// Convolution accumulated layout
pub fn conv<F: PrimeField + TensorType + PartialOrd + std::marker::Send + std::marker::Sync>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    padding: (usize, usize),
    stride: (usize, usize),
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let has_bias = values.len() == 3;
    let (image, kernel) = (values[0].clone(), values[1].clone());

    // assign the kernel
    let kernel = region.assign(&config.inputs[0], &kernel)?;
    // assign the image
    let mut image = region.assign(&config.inputs[1], &image)?;
    // increment the region
    region.increment(std::cmp::max(image.len(), kernel.len()));

    let og_dims = image.dims().to_vec();

    // ensure inputs are 4D tensors
    if og_dims.len() == 3 {
        // adds a dummy batch dimension
        let mut new_dims = vec![1];
        new_dims.extend_from_slice(image.dims());
        image.reshape(&new_dims)?;
    }

    // if not 4D then error
    if (image.dims().len() != 4)
        || (kernel.dims().len() != 4)
        || ((image.dims()[1] != kernel.dims()[1]) && (kernel.dims()[1] != 1))
    {
        return Err(Box::new(TensorError::DimMismatch("conv".to_string())));
    }

    let image_dims = image.dims();
    let kernel_dims = kernel.dims();

    let mut padded_image = image.clone();
    padded_image.pad(padding)?;

    let (batch_size, output_channels, input_channels, kernel_height, kernel_width) = (
        image_dims[0],
        kernel_dims[0],
        image_dims[1],
        kernel_dims[2],
        kernel_dims[3],
    );

    let (image_height, image_width) = (image_dims[2], image_dims[3]);

    let vert_slides = (image_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    let num_groups = input_channels / kernel_dims[1];
    let input_channels_per_group = input_channels / num_groups;
    let output_channels_per_group = output_channels / num_groups;

    if output_channels_per_group == 0 {
        return Err(Box::new(TensorError::DimMismatch(format!(
            "Given groups={}, expected kernel to be at least {} at dimension 0 but got {} instead",
            num_groups, num_groups, output_channels_per_group
        ))));
    }

    let num_outputs =
        batch_size * num_groups * output_channels_per_group * vert_slides * horz_slides;

    let mut output = Tensor::new(None, &[num_outputs])?;

    let cartesian_coord = vec![
        (0..batch_size),
        (0..num_groups),
        (0..output_channels_per_group),
        (0..vert_slides),
        (0..horz_slides),
    ]
    .iter()
    .cloned()
    .multi_cartesian_product()
    .collect::<Vec<_>>();

    let reshape_output = |output: &mut Tensor<ValType<F>>| {
        // remove dummy batch dimension if we added one
        if og_dims.len() == 3 {
            output.reshape(&[output_channels, vert_slides, horz_slides]);
        } else {
            output.reshape(&[batch_size, output_channels, vert_slides, horz_slides]);
        }
    };

    output.iter_mut().enumerate().for_each(|(idx, o)| {
        let cartesian_coord_per_group = &cartesian_coord[idx];
        let (batch, group, i, j, k) = (
            cartesian_coord_per_group[0],
            cartesian_coord_per_group[1],
            cartesian_coord_per_group[2],
            cartesian_coord_per_group[3],
            cartesian_coord_per_group[4],
        );
        let rs = j * stride.0;
        let cs = k * stride.1;

        let start_channel = group * input_channels_per_group;
        let end_channel = start_channel + input_channels_per_group;

        let mut local_image = padded_image
            .get_slice(&[
                batch..batch + 1,
                start_channel..end_channel,
                rs..(rs + kernel_height),
                cs..(cs + kernel_width),
            ])
            .unwrap();

        local_image.flatten();

        let start_kernel_index = group * output_channels_per_group + i;
        let end_kernel_index = start_kernel_index + 1;
        let mut local_kernel = kernel
            .get_slice(&[start_kernel_index..end_kernel_index])
            .unwrap();

        local_kernel.flatten();

        // this is dot product notation in einsum format
        let mut res = einsum(config, region, &mut [local_image, local_kernel], "i,i->").unwrap();

        if has_bias {
            let bias = values[2]
                .get_inner_tensor()
                .unwrap()
                .get_slice(&[start_kernel_index..end_kernel_index])
                .unwrap();
            res = pairwise(config, region, &[res, bias.into()], BaseOp::Add).unwrap()
        }

        *o = res.get_inner_tensor().unwrap()[0].clone();
    });

    // remove dummy batch dimension if we added one
    reshape_output(&mut output);

    let output: ValTensor<_> = output.into();

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !output.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
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
                Into::<Tensor<i32>>::into(output.get_inner()?),
                Into::<Tensor<i32>>::into(safe_conv),
            )
        }
    }

    Ok(output)
}

/// Power accumulated layout
pub fn pow<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    exponent: u32,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();

    for _ in 1..exponent {
        t = pairwise(config, region, &[t, values[0].clone()], BaseOp::Mult)?;
    }

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !t.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
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
pub fn rescale<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    scales: &[(usize, u128)],
) -> Result<Vec<ValTensor<F>>, Box<dyn Error>> {
    let mut rescaled_inputs = vec![];
    for (i, ri) in values.iter().enumerate() {
        if scales[i].1 == 1 {
            rescaled_inputs.push(ri.clone());
            continue;
        }
        let scaled_input = nonlinearity(
            config,
            region,
            &[ri.clone()],
            &LookupOp::Div {
                denom: (scales[i].1 as f32).into(),
            },
        )?;

        rescaled_inputs.push(scaled_input);
    }

    Ok(rescaled_inputs)
}

/// Pack accumulated layout
pub fn pack<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    base: u32,
    scale: u32,
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
        BaseOp::Mult,
    )?;

    let res = sum(config, region, &[base_prod])?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !res.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
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
pub fn reshape<F: PrimeField + TensorType + PartialOrd>(
    values: &[ValTensor<F>; 1],
    new_dims: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();
    t.reshape(new_dims)?;
    Ok(t)
}

/// Dummy (no contraints) move_axis layout
pub fn move_axis<F: PrimeField + TensorType + PartialOrd>(
    values: &[ValTensor<F>; 1],
    source: usize,
    destination: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut t = values[0].clone();
    t.move_axis(source, destination)?;
    Ok(t)
}

/// resize layout
pub fn resize<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scales: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut output = region.assign(&config.output, &values[0])?;
    region.increment(output.len());
    output.resize(scales)?;

    Ok(output)
}

/// Slice layout
pub fn slice<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axis: &usize,
    start: &usize,
    end: &usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assigns the instance to the advice.
    let mut output = region.assign(&config.output, &values[0])?;
    region.increment(output.len());
    output.slice(axis, start, end)?;

    Ok(output)
}

/// Concat layout
pub fn concat<F: PrimeField + TensorType + PartialOrd>(
    values: &[ValTensor<F>],
    axis: &usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let collected_inner: Result<Vec<Tensor<_>>, _> =
        values.iter().map(|e| e.get_inner_tensor()).collect();
    Ok(tensor::ops::concat(&collected_inner?, *axis)?.into())
}

/// Identity constraint. Usually used to constrain an instance column to an advice so the returned cells / values can be operated upon.
pub fn identity<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let output = region.assign(&config.output, &values[0])?;
    region.increment(output.len());

    Ok(output)
}

/// Downsample layout
pub fn downsample<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axis: &usize,
    stride: &usize,
    modulo: &usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let input = region.assign(&config.inputs[0], &values[0])?;
    let processed_output =
        tensor::ops::downsample(&input.get_inner_tensor()?, *axis, *stride, *modulo)?;
    let output = region.assign(&config.output, &processed_output.into())?;
    region.increment(input.len());
    Ok(output)
}

/// Layout for range check.
pub fn range_check<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    tol: i32,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assigns the instance to the advice.
    region.assign(&config.inputs[1], &values[0])?;
    let output = region.assign(&config.output, &values[1])?;

    (0..values[0].len()).for_each(|i| {
        let (x, y) = config.inputs[1].cartesian_coord(region.offset() + i);
        let selector = config.selectors.get(&(BaseOp::Range { tol }, x));

        region.enable(selector, y).unwrap();
    });

    region.increment(output.len());

    Ok(output)
}

/// Layout for nonlinearity check.
pub fn nonlinearity<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    nl: &LookupOp,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let x = &values[0];

    if region.is_dummy() {
        region.increment(x.len());
        return Ok(x.clone());
    }

    let w = region.assign(&config.lookup_input, x)?;

    // extract integer_valuations
    let felt_evals: Tensor<F> = w.get_felt_evals().map_err(|e| {
        error!("{}", e);
        halo2_proofs::plonk::Error::Synthesis
    })?;

    // for key generation integer_evals will be empty and we need to return a set of unassigned values
    let output: Tensor<Value<F>> = match felt_evals.len() {
        // if empty return an unknown val
        0 => Tensor::from((0..x.dims().iter().product::<usize>()).map(|_| Value::unknown())),
        // if not empty apply the nonlinearity !
        _ => {
            let x = Op::<F>::f(nl, &[felt_evals])?;
            x.output.map(|elem| Value::known(elem))
        }
    };

    let mut output = region.assign(&config.lookup_output, &output.into())?;

    (0..x.len()).for_each(|i| {
        let (x, y) = config.lookup_input.cartesian_coord(region.offset() + i);
        let selector = config.lookup_selectors.get(&(nl.clone(), x));
        region.enable(selector, y).unwrap();
    });

    output.reshape(x.dims())?;

    region.increment(x.len());

    // constrain the calculated output to a column
    Ok(output)
}

/// mean function layout
pub fn mean<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let x = &values[0];

    let sum_x = sum(config, region, &[x.clone()])?;
    let nl = LookupOp::Div {
        denom: utils::F32((scale * x.len()) as f32),
    };
    nonlinearity(config, region, &[sum_x], &nl)
}

/// abs layout
pub fn abs<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let x = &values[0];
    // Negate the product
    let neg_x = neg(config, region, &[x.clone()])?;
    let relu_x = nonlinearity(config, region, &[x.clone()], &LookupOp::ReLU { scale: 1 })?;
    let relu_neg_x = nonlinearity(config, region, &[neg_x], &LookupOp::ReLU { scale: 1 })?;
    // abs(x) = relu(x) + relu(-x)
    let abs = pairwise(config, region, &[relu_x, relu_neg_x], BaseOp::Add)?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !abs.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let mut ref_abs: Tensor<i32> =
                tensor::ops::abs(&values[0].get_int_evals()?)?.map(|x| x as i32);
            ref_abs.reshape(values[0].dims());

            assert_eq!(Into::<Tensor<i32>>::into(abs.get_inner()?), ref_abs)
        }
    };

    Ok(abs)
}

/// max layout
pub fn max<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // this is safe because we later constrain it
    let max_int = values[0].get_int_evals()?.into_iter().max();
    let max_val: ValTensor<F> = match max_int {
        None => Tensor::new(Some(&[Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&[Value::known(i128_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_max_val = region.assign(&config.inputs[1], &max_val)?;
    region.next();

    let unit: ValTensor<F> =
        Tensor::from(vec![region.assign_constant(&config.inputs[1], F::from(1))?].into_iter())
            .into();
    region.next();

    // max(x - 1)
    let max_minus_1 = pairwise(
        config,
        region,
        &[assigned_max_val.clone(), unit.clone()],
        BaseOp::Sub,
    )?;

    // x - max(x - 1)
    let diff = pairwise(
        config,
        region,
        &[values[0].clone(), max_minus_1],
        BaseOp::Sub,
    )?;
    // relu(x - max(x - 1))
    let relu = nonlinearity(config, region, &[diff], &LookupOp::ReLU { scale: 1 })?;

    let len = relu.dims().iter().product();

    // y_i*(1 - y_i) =0 // assert the values are either 0 or 1
    region.assign(&config.inputs[1], &relu)?;

    (0..len).for_each(|i| {
        let (x, y) = config.inputs[1].cartesian_coord(region.offset() + i);
        let selector = config.selectors.get(&(BaseOp::IsBoolean, x));
        region.enable(selector, y).unwrap();
    });

    region.increment(len);

    // sum(relu(x - max(x - 1)))
    let sum_relu = sum(config, region, &[relu])?;
    // 1 - sum(relu(x - max(x - 1)))
    let one_minus_sum_relu = pairwise(config, region, &[unit, sum_relu], BaseOp::Sub)?;
    // relu(1 - sum(relu(x - max(x - 1))))
    let relu_one_minus_sum_relu = nonlinearity(
        config,
        region,
        &[one_minus_sum_relu],
        &LookupOp::ReLU { scale: 1 },
    )?;

    // constraining relu(sum(relu(x - max(x - 1)) - len(x))) = 0
    region.assign(&config.inputs[1], &relu_one_minus_sum_relu)?;

    let (x, y) = config.output.cartesian_coord(region.offset());
    let selector = config.selectors.get(&(BaseOp::IsZero, x));
    region.enable(selector, y)?;

    region.increment(relu_one_minus_sum_relu.len());

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !assigned_max_val.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
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
pub fn min<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // this is safe because we later constrain it

    let min_int = values[0].get_int_evals()?.into_iter().min();
    let min_val: ValTensor<F> = match min_int {
        None => Tensor::new(Some(&[Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&[Value::known(i128_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_min_val = region.assign(&config.inputs[1], &min_val)?;
    region.next();

    let unit: ValTensor<F> =
        Tensor::from(vec![region.assign_constant(&config.inputs[1], F::from(1))?].into_iter())
            .into();
    region.next();

    // free up lock

    // min(x + 1)
    let min_plus_1 = pairwise(
        config,
        region,
        &[assigned_min_val.clone(), unit.clone()],
        BaseOp::Add,
    )?;

    // min(x + 1)  - x
    let diff = pairwise(
        config,
        region,
        &[min_plus_1, values[0].clone()],
        BaseOp::Sub,
    )?;

    // relu(min(x + 1)  - x)
    let relu = nonlinearity(config, region, &[diff], &LookupOp::ReLU { scale: 1 })?;

    let len = relu.dims().iter().product();

    region.assign(&config.inputs[1], &relu)?;
    // y_i*(1 - y_i) =0 // assert the values are either 0 or 1
    (0..len).for_each(|i| {
        let (x, y) = config.inputs[1].cartesian_coord(region.offset() + i);
        let selector = config.selectors.get(&(BaseOp::IsBoolean, x));
        region.enable(selector, y).unwrap();
    });

    region.increment(len);

    // sum(relu(min(x + 1) - x))
    let sum_relu = sum(config, region, &[relu])?;
    // 1 - sum(relu(min(x + 1) - x))
    let one_minus_sum_relu = pairwise(config, region, &[unit, sum_relu], BaseOp::Sub)?;
    // relu(1 - sum(relu(min(x + 1) - x)))
    let relu_one_minus_sum_relu = nonlinearity(
        config,
        region,
        &[one_minus_sum_relu],
        &LookupOp::ReLU { scale: 1 },
    )?;

    region.assign(&config.inputs[1], &relu_one_minus_sum_relu)?;

    // constraining product to 0
    let (x, y) = config.output.cartesian_coord(region.offset());
    let selector = config.selectors.get(&(BaseOp::IsZero, x));
    region.enable(selector, y)?;

    region.increment(relu_one_minus_sum_relu.len());

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !assigned_min_val.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let ref_min: Tensor<i32> = Tensor::new(
                Some(&[values[0].get_int_evals()?.into_iter().min().unwrap() as i32]),
                &[1],
            )?;

            assert_eq!(Into::<Tensor<i32>>::into(min_val.get_inner()?), ref_min,)
        }
    };
    Ok(assigned_min_val)
}

/// softmax layout
pub fn multi_dim_softmax<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    input_scale: usize,
    output_scale: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // we want this to be as small as possible so we set the output scale to 1
    let dims = values[0].dims();

    if dims.len() == 1 {
        return softmax(config, region, values, input_scale, output_scale);
    }

    let cartesian_coord = dims[..dims.len() - 1]
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut outputs = vec![];

    for coord in cartesian_coord {
        let mut sum_dims = vec![];
        for c in coord {
            sum_dims.push(c..c + 1);
        }
        sum_dims.push(0..dims[dims.len() - 1]);

        let softmax_input = values[0].get_slice(&sum_dims)?;

        outputs.push(
            softmax(config, region, &[softmax_input], input_scale, output_scale)?
                .get_inner_tensor()?,
        );
    }

    let mut res = Tensor::new(Some(&outputs), &[outputs.len()])?.combine()?;
    res.reshape(dims);

    Ok(res.into())
}

/// softmax func
pub fn softmax<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    input_scale: usize,
    output_scale: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // we want this to be as small as possible so we set the output scale to 1
    let scales = (input_scale, output_scale);

    // elementwise exponential
    let ex = nonlinearity(config, region, values, &LookupOp::Exp { scales })?;

    // sum of exps
    let denom = sum(config, region, &[ex.clone()])?;
    // get the inverse

    let inv_denom = nonlinearity(
        config,
        region,
        &[denom],
        // we set to input scale + output_scale so the output scale is output)scale
        &LookupOp::Recip {
            scale: output_scale.pow(2),
        },
    )?;

    // product of num * (1 / denom) = 2*output_scale
    let softmax = pairwise(config, region, &[ex, inv_denom], BaseOp::Mult)?;

    if matches!(&config.check_mode, CheckMode::SAFE) {
        // during key generation this will be 0 so we use this as a flag to check
        // TODO: this isn't very safe and would be better to get the phase directly
        let is_assigned = !softmax.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let int_evals = Tensor::new(Some(&values[0].get_int_evals()?), values[0].dims())?;
            // scale is double the output
            let ref_sofmax: Tensor<i128> =
                tensor::ops::nonlinearities::softmax(&int_evals, input_scale, output_scale).0;

            let output_int_evals = Tensor::new(Some(&softmax.get_int_evals()?), values[0].dims())?;

            assert_eq!(output_int_evals, ref_sofmax,)
        }
    };

    Ok(softmax)
}

/// Checks that the percent error between the expected public output and the actual output value
/// is within the percent error expressed by the `tol` input, where `tol == 1.0` means the percent
/// error tolerance is 1 percent.
pub fn range_check_percent<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    input_scale: usize,
    output_scale: usize,
    tol: f32,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if tol == 0.0 {
        // regular equality constraint
        return range_check(config, region, values, 0);
    }

    // Calculate the difference between the expected output and actual output
    let diff = pairwise(config, region, values, BaseOp::Sub)?;

    // Calculate the reciprocal of the expected output tensor, scaling by double the scaling factor
    let scale = input_scale * output_scale;
    let recip = nonlinearity(
        config,
        region,
        &[values[0].clone()],
        &LookupOp::Recip { scale },
    )?;
    // Multiply the difference by the recip
    let product = pairwise(config, region, &[diff, recip], BaseOp::Mult)?;

    // Use the greater than look up table to check if the percent error is within the tolerance for upper bound
    let tol = tol / 100.0;
    let upper_bound = nonlinearity(
        config,
        region,
        &[product.clone()],
        &LookupOp::GreaterThan {
            a: utils::F32(tol * scale as f32),
        },
    )?;

    // Negate the product
    let neg_product = neg(config, region, &[product])?;

    // Use the greater than look up table to check if the percent error is within the tolerance for lower bound
    let lower_bound = nonlinearity(
        config,
        region,
        &[neg_product],
        &LookupOp::GreaterThan {
            a: utils::F32(tol * scale as f32),
        },
    )?;

    // Add the lower_bound and upper_bound
    let sum = pairwise(config, region, &[lower_bound, upper_bound], BaseOp::Add)?;

    // Assign the sum tensor to the inputs
    region.assign(&config.inputs[1], &sum)?;

    // Constrain the sum to be all zeros
    let (x, y) = config.output.cartesian_coord(region.offset());
    let selector = config.selectors.get(&(BaseOp::IsZero, x));
    region.enable(selector, y)?;

    region.increment(sum.len());

    if matches!(&config.check_mode, CheckMode::SAFE) {
        let is_assigned = !sum.get_inner()?.iter().any(|&x| {
            let mut is_empty = true;
            x.map(|f| is_empty = f == F::ZERO);
            is_empty
        });
        if is_assigned {
            let int_evals = &[
                Tensor::new(Some(&values[0].get_int_evals()?), values[0].dims())?,
                Tensor::new(Some(&values[1].get_int_evals()?), values[1].dims())?,
            ];
            let ref_range_check_percent: Tensor<i128> =
                tensor::ops::nonlinearities::range_check_percent(
                    int_evals,
                    input_scale,
                    output_scale,
                    tol,
                );
            let output_int_evals = Tensor::new(Some(&sum.get_int_evals()?), values[0].dims())?;
            assert_eq!(output_int_evals, ref_range_check_percent)
        }
    }
    Ok(sum)
}
