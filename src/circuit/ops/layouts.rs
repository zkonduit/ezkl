use std::{
    collections::{HashMap, HashSet},
    error::Error,
    ops::Range,
};

use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::{error, trace};
use maybe_rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use super::{
    chip::{BaseConfig, CircuitError},
    region::RegionCtx,
};
use crate::{
    circuit::{ops::base::BaseOp, utils},
    fieldutils::{felt_to_i128, i128_to_felt},
    tensor::{
        get_broadcasted_shape,
        ops::{accumulated, add, mult, sub},
        Tensor, TensorError, ValType,
    },
};

use super::*;
use crate::circuit::ops::lookup::LookupOp;

///
pub fn overflowed_len(starting_idx: usize, mut total_len: usize, column_len: usize) -> usize {
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

/// Same as div but splits the division into N parts
pub fn loop_div<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    divisor: F,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if divisor == F::ONE {
        return Ok(value[0].clone());
    }

    // if integer val is divisible by 2, we can use a faster method and div > F::S
    let mut divisor = divisor;
    let mut num_parts = 1;

    while felt_to_i128(divisor) % 2 == 0 && felt_to_i128(divisor) > (2_i128.pow(F::S - 4)) {
        divisor = i128_to_felt(felt_to_i128(divisor) / 2);
        num_parts += 1;
    }

    let output = div(config, region, value, divisor)?;
    if num_parts == 1 {
        return Ok(output);
    }

    let divisor_int = 2_i128.pow(num_parts - 1);
    let divisor_felt = i128_to_felt(divisor_int);
    if divisor_int <= 2_i128.pow(F::S - 3) {
        div(config, region, &[output], divisor_felt)
    } else {
        // keep splitting the divisor until it satisfies the condition
        loop_div(config, region, &[output], divisor_felt)
    }
}

/// Div accumulated layout
pub fn div<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    div: F,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if div == F::ONE {
        return Ok(value[0].clone());
    }

    let input = value[0].clone();
    let input_dims = input.dims();

    let range_check_bracket = felt_to_i128(div) / 2;

    let mut divisor = Tensor::from(vec![ValType::Constant(div)].into_iter());
    divisor.set_visibility(&crate::graph::Visibility::Fixed);
    let divisor = region.assign(&config.custom_gates.inputs[1], &divisor.into())?;
    region.increment(divisor.len());

    let is_assigned = !input.any_unknowns()? && !divisor.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.get_int_evals()?;
        tensor::ops::nonlinearities::const_div(&input_evals.clone(), felt_to_i128(div) as f64)
            .iter()
            .map(|x| Ok(Value::known(i128_to_felt(*x))))
            .collect::<Result<Tensor<Value<F>>, Box<dyn Error>>>()?
            .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };
    claimed_output.reshape(input_dims)?;
    region.assign(&config.custom_gates.output, &claimed_output)?;
    region.increment(claimed_output.len());

    let product = pairwise(
        config,
        region,
        &[claimed_output.clone(), divisor.clone()],
        BaseOp::Mult,
    )?;

    let diff_with_input = pairwise(
        config,
        region,
        &[product.clone(), input.clone()],
        BaseOp::Sub,
    )?;

    range_check(
        config,
        region,
        &[diff_with_input],
        &(-range_check_bracket, range_check_bracket),
    )?;

    Ok(claimed_output)
}

fn recip_int<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    input: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assert is boolean
    let zero_inverse_val = tensor::ops::nonlinearities::zero_recip(1.0)[0];
    // get values where input is 0
    let zero_mask = equals_zero(config, region, input)?;

    let one_minus_zero_mask = pairwise(
        config,
        region,
        &[
            zero_mask.clone(),
            ValTensor::from(Tensor::from([ValType::Constant(F::ONE)].into_iter())),
        ],
        BaseOp::Sub,
    )?;

    let zero_inverse_val = pairwise(
        config,
        region,
        &[
            zero_mask,
            ValTensor::from(Tensor::from(
                [ValType::Constant(i128_to_felt(zero_inverse_val))].into_iter(),
            )),
        ],
        BaseOp::Mult,
    )?;

    pairwise(
        config,
        region,
        &[one_minus_zero_mask, zero_inverse_val],
        BaseOp::Add,
    )
}

/// recip accumulated layout
pub fn recip<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    input_scale: F,
    output_scale: F,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if output_scale == F::ONE || output_scale == F::ZERO {
        return recip_int(config, region, value);
    }

    let input = value[0].clone();
    let input_dims = input.dims();

    let integer_input_scale = felt_to_i128(input_scale);
    let integer_output_scale = felt_to_i128(output_scale);

    // range_check_bracket is min of input_scale * output_scale and 2^F::S - 3
    let range_check_len = std::cmp::min(integer_output_scale, 2_i128.pow(F::S - 4));

    let input_scale_ratio =
        i128_to_felt(integer_input_scale * integer_output_scale / range_check_len);

    let range_check_bracket = range_check_len / 2;

    let is_assigned = !input.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.get_int_evals()?;
        tensor::ops::nonlinearities::recip(
            &input_evals,
            felt_to_i128(input_scale) as f64,
            felt_to_i128(output_scale) as f64,
        )
        .iter()
        .map(|x| Ok(Value::known(i128_to_felt(*x))))
        .collect::<Result<Tensor<Value<F>>, Box<dyn Error>>>()?
        .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };
    claimed_output.reshape(input_dims)?;
    let claimed_output = region.assign(&config.custom_gates.output, &claimed_output)?;
    region.increment(claimed_output.len());

    // this is now of scale 2 * scale
    let product = pairwise(
        config,
        region,
        &[claimed_output.clone(), input.clone()],
        BaseOp::Mult,
    )?;

    // divide by input_scale
    let rebased_div = loop_div(config, region, &[product], input_scale_ratio)?;

    let zero_inverse_val =
        tensor::ops::nonlinearities::zero_recip(felt_to_i128(output_scale) as f64)[0];
    let zero_inverse =
        Tensor::from([ValType::Constant(i128_to_felt::<F>(zero_inverse_val))].into_iter());

    let equal_zero_mask = equals_zero(config, region, &[input.clone()])?;

    let equal_inverse_mask = equals(
        config,
        region,
        &[claimed_output.clone(), zero_inverse.into()],
    )?;

    // assert the two masks are equal
    enforce_equality(
        config,
        region,
        &[equal_zero_mask.clone(), equal_inverse_mask],
    )?;

    let unit_scale = Tensor::from([ValType::Constant(i128_to_felt(range_check_len))].into_iter());

    let unit_mask = pairwise(
        config,
        region,
        &[equal_zero_mask, unit_scale.into()],
        BaseOp::Mult,
    )?;

    // now add the unit mask to the rebased_div
    let rebased_offset_div = pairwise(config, region, &[rebased_div, unit_mask], BaseOp::Add)?;

    // at most the error should be in the original unit scale's range
    range_check(
        config,
        region,
        &[rebased_offset_div],
        &(range_check_bracket, 3 * range_check_bracket),
    )?;

    Ok(claimed_output)
}

/// Dot product accumulated layout
pub fn dot<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    region.flush()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    let mut values = values.clone();

    // this section has been optimized to death, don't mess with it
    let mut removal_indices = values[0].get_const_zero_indices()?;
    let second_zero_indices = values[1].get_const_zero_indices()?;
    removal_indices.extend(second_zero_indices);
    removal_indices.par_sort_unstable();
    removal_indices.dedup();

    // is already sorted
    values[0].remove_indices(&mut removal_indices, true)?;
    values[1].remove_indices(&mut removal_indices, true)?;

    let elapsed = global_start.elapsed();
    trace!("filtering const zero indices took: {:?}", elapsed);

    if values[0].len() != values[1].len() {
        return Err(Box::new(TensorError::DimMismatch("dot".to_string())));
    }

    // if empty return a const
    if values[0].is_empty() && values[1].is_empty() {
        return Ok(Tensor::from([ValType::Constant(F::ZERO)].into_iter()).into());
    }

    let start = instant::Instant::now();
    let mut inputs = vec![];
    let block_width = config.custom_gates.output.num_inner_cols();

    let mut assigned_len = 0;
    for (i, input) in values.iter_mut().enumerate() {
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
        let inp = {
            let (res, len) = region.assign_with_duplication(
                &config.custom_gates.inputs[i],
                input,
                &config.check_mode,
                false,
            )?;
            assigned_len = len;
            res.get_inner()?
        };
        inputs.push(inp);
    }

    let elapsed = start.elapsed();
    trace!("assigning inputs took: {:?}", elapsed);

    // Now we can assign the dot product
    // time this step
    let start = instant::Instant::now();
    let accumulated_dot = accumulated::dot(&[inputs[0].clone(), inputs[1].clone()], block_width)?;
    let elapsed = start.elapsed();
    trace!("calculating accumulated dot took: {:?}", elapsed);

    let start = instant::Instant::now();
    let (output, output_assigned_len) = region.assign_with_duplication(
        &config.custom_gates.output,
        &accumulated_dot.into(),
        &config.check_mode,
        true,
    )?;
    let elapsed = start.elapsed();
    trace!("assigning output took: {:?}", elapsed);

    // enable the selectors
    if !region.is_dummy() {
        (0..output_assigned_len)
            .map(|i| {
                let (x, _, z) = config
                    .custom_gates
                    .output
                    .cartesian_coord(region.linear_coord() + i * block_width);
                // hop over duplicates at start of column
                if z == 0 && i > 0 {
                    return Ok(());
                }
                let selector = if i == 0 {
                    config.custom_gates.selectors.get(&(BaseOp::DotInit, x, 0))
                } else {
                    config.custom_gates.selectors.get(&(BaseOp::Dot, x, 0))
                };
                region.enable(selector, z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    let last_elem = output.get_slice(&[output.len() - 1..output.len()])?;

    region.increment(assigned_len);

    // last element is the result

    let elapsed = global_start.elapsed();
    trace!("dot layout took: {:?}, row {}", elapsed, region.row());
    trace!("----------------------------");
    Ok(last_elem)
}

/// Einsum
pub fn einsum<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    inputs: &[ValTensor<F>],
    equation: &str,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut equation = equation.split("->");
    let inputs_eq = equation.next().ok_or(CircuitError::InvalidEinsum)?;
    let output_eq = equation.next().ok_or(CircuitError::InvalidEinsum)?;
    let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();

    // Check that the number of inputs matches the number of inputs in the equation
    if inputs.len() != inputs_eq.len() {
        return Err(Box::new(TensorError::DimMismatch("einsum".to_string())));
    }

    let mut indices_to_size = HashMap::new();
    for (i, input) in inputs.iter().enumerate() {
        for j in 0..inputs_eq[i].len() {
            let c = inputs_eq[i]
                .chars()
                .nth(j)
                .ok_or(CircuitError::InvalidEinsum)?;
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
        .map(|c| {
            indices_to_size
                .get(&c)
                .ok_or(CircuitError::InvalidEinsum)
                .copied()
        })
        .collect::<Result<Vec<_>, _>>()?;

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
                Ok(1)
            // Otherwise, the slice should be the entire dimension of the input tensor
            } else {
                indices_to_size
                    .get(d)
                    .ok_or(CircuitError::InvalidEinsum)
                    .copied()
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .product::<usize>();

    let cartesian_coord = output_shape
        .iter()
        .map(|d| 0..*d)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    // Get the indices common across input tensors
    let mut common_coord = common_indices_to_inputs
        .iter()
        .map(|d| {
            // If the current index is in the output equation, then the slice should be the current coordinate
            if output_eq.contains(*d) {
                Ok(0..1)
            // Otherwise, the slice should be the entire dimension of the input tensor
            } else {
                Ok(0..*indices_to_size.get(d).ok_or(CircuitError::InvalidEinsum)?)
            }
        })
        .collect::<Result<Vec<Range<_>>, Box<dyn Error>>>()?
        .into_iter()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    // If there are no common indices, then we need to add an empty slice to force one iteration of the loop
    if common_coord.is_empty() {
        common_coord.push(vec![]);
    }

    let inner_loop_function = |i: usize, region: &mut RegionCtx<'_, F>| {
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
                inputs[idx].get_slice(&slice)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // in this case its just a dot product :)
        if non_common_coord_size == 1 && inputs.len() == 2 {
            Ok(dot(
                config,
                region,
                inputs[..].try_into().map_err(|e| {
                    error!("{}", e);
                    halo2_proofs::plonk::Error::Synthesis
                })?,
            )?
            .get_inner_tensor()?[0]
                .clone())
        } else {
            let mut prod_res = None;

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
                        inputs[idx].get_slice(&slice).map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let mut input_pairs = vec![];

                for input in inputs {
                    input_pairs.push(input.get_inner_tensor()?.clone().into_iter());
                }

                let input_pairs = input_pairs
                    .into_iter()
                    .multi_cartesian_product()
                    .collect::<Vec<_>>();

                // Compute the product of all input tensors
                for pair in input_pairs {
                    let product_across_pair = prod(
                        config,
                        region,
                        &[pair.try_into().map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?],
                    )?;

                    if let Some(product) = prod_res {
                        prod_res = Some(
                            pairwise(config, region, &[product, product_across_pair], BaseOp::Add)
                                .map_err(|e| {
                                    error!("{}", e);
                                    halo2_proofs::plonk::Error::Synthesis
                                })?,
                        );
                    } else {
                        prod_res = Some(product_across_pair);
                    }
                }
            }
            Ok::<_, region::RegionError>(
                prod_res
                    .ok_or(Into::<region::RegionError>::into("missing prod"))?
                    .get_inner_tensor()?[0]
                    .clone(),
            )
        }
    };

    region.flush()?;
    region.apply_in_loop(&mut output, inner_loop_function)?;

    let output: ValTensor<F> = output.into();

    Ok(output)
}

fn _sort_ascending<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let input = values[0].clone();

    // assert input is flat
    assert_eq!(input.dims().len(), 1);

    let is_assigned = !input.any_unknowns()?;

    let sorted = if is_assigned {
        input
            .get_int_evals()?
            .iter()
            .sorted_by(|a, b| a.cmp(b))
            .map(|x| Ok(Value::known(i128_to_felt(*x))))
            .collect::<Result<Tensor<Value<F>>, Box<dyn Error>>>()?
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
    };

    let assigned_sort = region.assign(&config.custom_gates.inputs[0], &sorted.into())?;

    let mut unit = Tensor::from(vec![F::from(1)].into_iter());
    unit.set_visibility(&crate::graph::Visibility::Fixed);
    let unit = region.assign(&config.custom_gates.inputs[1], &unit.try_into()?)?;

    region.increment(assigned_sort.len());

    for i in 0..assigned_sort.len() - 1 {
        // assert that each thing in turn is larger than the next
        let window_a = assigned_sort.get_slice(&[i..i + 1])?;
        let window_b = assigned_sort.get_slice(&[i + 1..i + 2])?;

        let diff = pairwise(
            config,
            region,
            &[window_b.clone(), window_a.clone()],
            BaseOp::Sub,
        )?;
        let greater_than = nonlinearity(
            config,
            region,
            &[diff],
            &LookupOp::GreaterThanEqual { a: 0.0.into() },
        )?;

        enforce_equality(config, region, &[unit.clone(), greater_than.clone()])?;
    }

    let mut zero_tensor = Tensor::from(vec![ValType::Constant(F::ZERO); input.len()].into_iter());
    zero_tensor.set_visibility(&crate::graph::Visibility::Fixed);
    let zero_tensor: ValTensor<F> = zero_tensor.try_into()?;

    // assert that this is a permutation/shuffle
    dynamic_lookup(
        config,
        region,
        &[assigned_sort.clone(), zero_tensor.clone()],
        &[input.clone(), zero_tensor],
    )?;

    Ok(assigned_sort)
}

///
fn _select_topk<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    k: usize,
    largest: bool,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut sorted = _sort_ascending(config, region, values)?;
    if largest {
        sorted.reverse()?;
    }
    Ok(sorted.get_slice(&[0..k])?)
}

/// Select top k elements
pub fn topk_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    k: usize,
    dim: usize,
    largest: bool,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let topk_at_k = move |config: &BaseConfig<F>,
                          region: &mut RegionCtx<F>,
                          values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, Box<dyn Error>> {
        _select_topk(config, region, values, k, largest)
    };

    let output: ValTensor<F> = multi_dim_axes_op(config, region, values, &[dim], topk_at_k)?;

    Ok(output)
}

fn select<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    dim_indices: ValTensor<F>,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut input, index) = (values[0].clone(), values[1].clone());
    input.flatten();

    if !(dim_indices.all_prev_assigned() || region.is_dummy()) {
        return Err("dim_indices must be assigned".into());
    }

    let is_assigned = !input.any_unknowns()? && !index.any_unknowns()?;

    let output: ValTensor<F> = if is_assigned {
        index
            .get_int_evals()?
            .iter()
            .map(|x| Ok(Value::known(input.get_felt_evals()?.get(&[*x as usize]))))
            .collect::<Result<Tensor<Value<F>>, Box<dyn Error>>>()?
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); index.len()]),
            &[index.len()],
        )?
    }
    .into();

    let (_, assigned_output) =
        dynamic_lookup(config, region, &[index, output], &[dim_indices, input])?;

    Ok(assigned_output)
}

fn one_hot<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    num_classes: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assert values is flat
    assert_eq!(values[0].dims().len(), 1);
    // assert its a single elelemnt
    assert_eq!(values[0].len(), 1);
    let input = values[0].clone();
    let is_assigned = !input.any_unknowns()?;

    let output: ValTensor<F> = if is_assigned {
        let int_evals = input.get_int_evals()?;
        let res = tensor::ops::one_hot(&int_evals, num_classes, 1)?;
        res.iter()
            .map(|x| Value::known(i128_to_felt(*x)))
            .collect::<Tensor<_>>()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); num_classes]),
            &[num_classes],
        )?
    }
    .into();

    let assigned_input = region.assign(&config.custom_gates.inputs[0], &input)?;

    // now assert all elems are 0 or 1
    let assigned_output = boolean_identity(config, region, &[output.clone()], true)?;
    region.increment(std::cmp::max(assigned_output.len(), assigned_input.len()));

    let sum = sum(config, region, &[assigned_output.clone()])?;
    // assert sum is 1
    let mut unit = Tensor::from(vec![F::from(1)].into_iter());
    unit.set_visibility(&crate::graph::Visibility::Fixed);
    let unit: ValTensor<F> = unit.try_into()?;

    enforce_equality(config, region, &[unit.clone(), sum])?;

    let gathered = gather(
        config,
        region,
        &[assigned_output.clone(), assigned_input.clone()],
        0,
    )?;

    enforce_equality(config, region, &[unit, gathered])?;

    Ok(assigned_output)
}

/// Dynamic lookup
pub fn dynamic_lookup<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    lookups: &[ValTensor<F>; 2],
    tables: &[ValTensor<F>; 2],
) -> Result<(ValTensor<F>, ValTensor<F>), Box<dyn Error>> {
    // if not all lookups same length err
    if lookups[0].len() != lookups[1].len() {
        return Err("lookups must be same length".into());
    }

    // if not all inputs same length err
    if tables[0].len() != tables[1].len() {
        return Err("tables must be same length".into());
    }

    let dynamic_lookup_index = region.dynamic_lookup_index();

    let (lookup_0, lookup_1) = (lookups[0].clone(), lookups[1].clone());
    let (table_0, table_1) = (tables[0].clone(), tables[1].clone());

    let table_0 = region.assign_dynamic_lookup(&config.dynamic_lookups.tables[0], &table_0)?;
    let _table_1 = region.assign_dynamic_lookup(&config.dynamic_lookups.tables[1], &table_1)?;
    let table_len = table_0.len();

    // now create a vartensor of constants for the dynamic lookup index
    let mut table_index = Tensor::from(
        vec![ValType::Constant(F::from(dynamic_lookup_index as u64)); table_len].into_iter(),
    );
    table_index.set_visibility(&crate::graph::Visibility::Fixed);
    let _table_index =
        region.assign_dynamic_lookup(&config.dynamic_lookups.tables[2], &table_index.into())?;

    let lookup_0 = region.assign(&config.custom_gates.inputs[0], &lookup_0)?;
    let lookup_1 = region.assign(&config.custom_gates.inputs[1], &lookup_1)?;
    let lookup_len = lookup_0.len();

    // now set the lookup index
    let mut lookup_index = Tensor::from(
        vec![ValType::Constant(F::from(dynamic_lookup_index as u64)); lookup_len].into_iter(),
    );
    lookup_index.set_visibility(&crate::graph::Visibility::Fixed);
    let _lookup_index = region.assign(&config.custom_gates.output, &lookup_index.into())?;

    if !region.is_dummy() {
        (0..table_len)
            .map(|i| {
                let table_selector = config.dynamic_lookups.table_selectors[0];
                let (_, _, z) = config.dynamic_lookups.tables[0]
                    .cartesian_coord(region.dynamic_lookup_col_coord() + i);
                region.enable(Some(&table_selector), z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    if !region.is_dummy() {
        // Enable the selectors
        (0..lookup_len)
            .map(|i| {
                let (x, y, z) =
                    config.custom_gates.inputs[0].cartesian_coord(region.linear_coord() + i);
                let lookup_selector = config
                    .dynamic_lookups
                    .lookup_selectors
                    .get(&(x, y))
                    .ok_or("missing selectors")?;

                region.enable(Some(&lookup_selector), z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    region.increment_dynamic_lookup_col_coord(table_len);
    region.increment_dynamic_lookup_index(1);
    region.increment(lookup_len);

    Ok((lookup_0, lookup_1))
}

/// One hot accumulated layout
pub fn one_hot_axis<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    num_classes: usize,
    dim: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let input = values[0].clone();
    let input_inner = input.get_inner_tensor()?;

    let mut output_dims = values[0].dims().to_vec();
    output_dims.insert(dim, num_classes);

    let mut op_tensors: Tensor<ValTensor<F>> = Tensor::new(None, input_inner.dims())?;

    let inner_loop_function =
        |i: usize, region: &mut RegionCtx<'_, F>| -> Result<ValTensor<F>, _> {
            let inp = input_inner[i].clone();
            let tensor = Tensor::new(Some(&[inp.clone()]), &[1])?;

            Ok(one_hot(config, region, &[tensor.into()], num_classes)?)
        };

    region.apply_in_loop(&mut op_tensors, inner_loop_function)?;

    // Allocate memory for the output tensor
    let cartesian_coord = output_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut output = Tensor::<ValType<F>>::new(None, &output_dims)?;

    output = output.par_enum_map(|i, _| {
        let coord = cartesian_coord[i].clone();
        let mut op_idx = coord.clone();
        let coord_at_dims = vec![coord[dim]];
        op_idx.remove(dim);

        let op_tensor = op_tensors.get(&op_idx);

        let op_tensor = op_tensor.get_inner_tensor()?;

        let one_hot_val = op_tensor.get(&coord_at_dims).clone();

        Ok::<_, region::RegionError>(one_hot_val)
    })?;

    Ok(output.into())
}

/// Gather accumulated layout
pub fn gather<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    dim: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut input, mut index_clone) = (values[0].clone(), values[1].clone());
    index_clone.flatten();
    if index_clone.is_singleton() {
        index_clone.reshape(&[1])?;
    }

    let mut assigned_len = vec![];
    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
        assigned_len.push(input.len());
    }
    if !index_clone.all_prev_assigned() {
        index_clone = region.assign(&config.custom_gates.inputs[1], &index_clone)?;
        assigned_len.push(index_clone.len());
    }

    if !assigned_len.is_empty() {
        // safe to unwrap since we've just checked it has at least one element
        region.increment(*assigned_len.iter().max().unwrap());
    }

    // Calculate the output tensor size
    let input_dims = input.dims();
    let mut output_size = input_dims.to_vec();

    output_size[dim] = index_clone.dims()[0];

    // these will be assigned as constants
    let mut indices = Tensor::from((0..input.dims()[dim] as u64).map(|x| F::from(x)));
    indices.set_visibility(&crate::graph::Visibility::Fixed);
    let indices = region.assign(&config.custom_gates.inputs[1], &indices.try_into()?)?;
    region.increment(indices.len());

    let mut iteration_dims = output_size.clone();
    iteration_dims[dim] = 1;

    // Allocate memory for the output tensor
    let cartesian_coord = iteration_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut results = HashMap::new();

    for coord in cartesian_coord {
        let mut slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
        slice[dim] = 0..input_dims[dim];

        let mut sliced_input = input.get_slice(&slice)?;
        sliced_input.flatten();

        let res = select(
            config,
            region,
            &[sliced_input, index_clone.clone()],
            indices.clone(),
        )?;

        results.insert(coord, res);
    }

    // Allocate memory for the output tensor
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut output = Tensor::new(None, &output_size)?.par_enum_map(|i, _: ValType<F>| {
        let coord = cartesian_coord[i].clone();
        let mut key = coord.clone();
        key[dim] = 0;
        let result = &results.get(&key).ok_or("missing result")?;
        let o = result.get_inner_tensor().map_err(|_| "missing tensor")?[coord[dim]].clone();
        Ok::<ValType<F>, region::RegionError>(o)
    })?;

    // Reshape the output tensor
    if index_clone.is_singleton() {
        output_size.remove(dim);
    }
    output.reshape(&output_size)?;

    Ok(output.into())
}

/// Gather accumulated layout
pub fn gather_elements<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    dim: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut input, mut index) = (values[0].clone(), values[1].clone());

    assert_eq!(input.dims().len(), index.dims().len());

    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
    }
    if !index.all_prev_assigned() {
        index = region.assign(&config.custom_gates.inputs[1], &index)?;
    }

    region.increment(std::cmp::max(input.len(), index.len()));

    // Calculate the output tensor size
    let input_dim = input.dims()[dim];
    let output_size = index.dims().to_vec();

    // these will be assigned as constants
    let mut indices = Tensor::from((0..input_dim as u64).map(|x| F::from(x)));
    indices.set_visibility(&crate::graph::Visibility::Fixed);
    let indices = region.assign(&config.custom_gates.inputs[1], &indices.try_into()?)?;
    region.increment(indices.len());

    // Allocate memory for the output tensor
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut output = Tensor::new(None, &output_size)?;

    let inner_loop_function = |i: usize, region: &mut RegionCtx<'_, F>| {
        let coord = cartesian_coord[i].clone();
        let index_val = index.get_inner_tensor()?.get(&coord);

        let mut slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
        slice[dim] = 0..input_dim;

        let mut sliced_input = input.get_slice(&slice)?;
        sliced_input.flatten();

        let index_valtensor: ValTensor<F> = Tensor::from([index_val.clone()].into_iter()).into();

        let res = select(
            config,
            region,
            &[sliced_input, index_valtensor],
            indices.clone(),
        )?;

        let res = res.get_inner_tensor()?;

        Ok(res[0].clone())
    };

    region.apply_in_loop(&mut output, inner_loop_function)?;

    Ok(output.into())
}

/// Gather accumulated layout
pub fn scatter_elements<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 3],
    dim: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut input, mut index, mut src) = (values[0].clone(), values[1].clone(), values[2].clone());

    assert_eq!(input.dims().len(), index.dims().len());

    let mut assigned_len = vec![];

    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
        assigned_len.push(input.len());
    }
    if !index.all_prev_assigned() {
        index = region.assign(&config.custom_gates.inputs[1], &index)?;
        assigned_len.push(index.len());
    }
    if !src.all_prev_assigned() {
        src = region.assign(&config.custom_gates.output, &src)?;
        assigned_len.push(src.len());
    }

    if !assigned_len.is_empty() {
        // safe to unwrap since we've just checked it has at least one element
        region.increment(*assigned_len.iter().max().unwrap());
    }

    // Calculate the output tensor size
    let input_dim = input.dims()[dim];
    let output_size = index.dims().to_vec();

    // these will be assigned as constants
    let mut indices = Tensor::from((0..input_dim as u64).map(|x| F::from(x)));
    indices.set_visibility(&crate::graph::Visibility::Fixed);
    let indices = region.assign(&config.custom_gates.inputs[1], &indices.try_into()?)?;
    region.increment(indices.len());

    // Allocate memory for the output tensor
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut unit = Tensor::from(vec![F::from(1)].into_iter());
    unit.set_visibility(&crate::graph::Visibility::Fixed);
    let unit: ValTensor<F> = unit.try_into()?;
    region.assign(&config.custom_gates.inputs[1], &unit)?;
    region.increment(1);

    let mut output: Tensor<()> = Tensor::new(None, &output_size)?;

    let mut inner_loop_function = |i: usize, region: &mut RegionCtx<'_, F>| {
        let coord = cartesian_coord[i].clone();
        let index_val = index.get_inner_tensor()?.get(&coord);

        let src_val = src.get_inner_tensor()?.get(&coord);
        let src_valtensor: ValTensor<F> = Tensor::from([src_val.clone()].into_iter()).into();

        let mut slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
        slice[dim] = 0..input_dim;

        let mut sliced_input = input.get_slice(&slice)?;
        sliced_input.flatten();

        let index_valtensor: ValTensor<F> = Tensor::from([index_val.clone()].into_iter()).into();

        let mask = equals(config, region, &[index_valtensor, indices.clone()])?;

        let one_minus_mask = pairwise(config, region, &[unit.clone(), mask.clone()], BaseOp::Sub)?;

        let pairwise_prod = pairwise(config, region, &[src_valtensor, mask], BaseOp::Mult)?;
        let pairwise_prod_2 = pairwise(
            config,
            region,
            &[sliced_input, one_minus_mask],
            BaseOp::Mult,
        )?;

        let res = pairwise(
            config,
            region,
            &[pairwise_prod, pairwise_prod_2],
            BaseOp::Add,
        )?;

        let input_cartesian_coord = slice.into_iter().multi_cartesian_product();

        let mutable_input_inner = input.get_inner_tensor_mut()?;

        for (i, r) in res.get_inner_tensor()?.iter().enumerate() {
            let coord = input_cartesian_coord
                .clone()
                .nth(i)
                .ok_or("invalid coord")?;
            *mutable_input_inner.get_mut(&coord) = r.clone();
        }
        Ok(())
    };

    output
        .iter_mut()
        .enumerate()
        .map(|(i, _)| inner_loop_function(i, region))
        .collect::<Result<Vec<()>, Box<dyn Error>>>()?;

    Ok(input)
}

/// sum accumulated layout
pub fn sum<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    region.flush()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    let mut values = values.clone();

    // this section has been optimized to death, don't mess with it
    let mut removal_indices = values[0].get_const_zero_indices()?;
    removal_indices.par_sort_unstable();
    removal_indices.dedup();

    // is already sorted
    values[0].remove_indices(&mut removal_indices, true)?;

    let elapsed = global_start.elapsed();
    trace!("filtering const zero indices took: {:?}", elapsed);

    // if empty return a const
    if values[0].is_empty() {
        return Ok(Tensor::from([ValType::Constant(F::ZERO)].into_iter()).into());
    }

    let block_width = config.custom_gates.output.num_inner_cols();

    let assigned_len: usize;
    let input = {
        let mut input = values[0].clone();
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
        let (res, len) = region.assign_with_duplication(
            &config.custom_gates.inputs[1],
            &input,
            &config.check_mode,
            false,
        )?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_sum = accumulated::sum(&input, block_width)?;

    let (output, output_assigned_len) = region.assign_with_duplication(
        &config.custom_gates.output,
        &accumulated_sum.into(),
        &config.check_mode,
        true,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        for i in 0..output_assigned_len {
            let (x, _, z) = config
                .custom_gates
                .output
                .cartesian_coord(region.linear_coord() + i * block_width);
            // skip over duplicates at start of column
            if z == 0 && i > 0 {
                continue;
            }
            let selector = if i == 0 {
                config.custom_gates.selectors.get(&(BaseOp::SumInit, x, 0))
            } else {
                config.custom_gates.selectors.get(&(BaseOp::Sum, x, 0))
            };

            region.enable(selector, z)?;
        }
    }

    let last_elem = output.get_slice(&[output.len() - 1..output.len()])?;

    region.increment(assigned_len);

    // last element is the result
    Ok(last_elem)
}

/// product accumulated layout
pub fn prod<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    region.flush()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    // this section has been optimized to death, don't mess with it
    let removal_indices = values[0].get_const_zero_indices()?;

    let elapsed = global_start.elapsed();
    trace!("finding const zero indices took: {:?}", elapsed);
    // if empty return a const
    if !removal_indices.is_empty() {
        return Ok(Tensor::from([ValType::Constant(F::ZERO)].into_iter()).into());
    }

    let block_width = config.custom_gates.output.num_inner_cols();
    let assigned_len: usize;
    let input = {
        let mut input = values[0].clone();
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ONE))?;
        let (res, len) = region.assign_with_duplication(
            &config.custom_gates.inputs[1],
            &input,
            &config.check_mode,
            false,
        )?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_prod = accumulated::prod(&input, block_width)?;

    let (output, output_assigned_len) = region.assign_with_duplication(
        &config.custom_gates.output,
        &accumulated_prod.into(),
        &config.check_mode,
        true,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        (0..output_assigned_len)
            .map(|i| {
                let (x, _, z) = config
                    .custom_gates
                    .output
                    .cartesian_coord(region.linear_coord() + i * block_width);
                // skip over duplicates at start of column
                if z == 0 && i > 0 {
                    return Ok(());
                }
                let selector = if i == 0 {
                    config
                        .custom_gates
                        .selectors
                        .get(&(BaseOp::CumProdInit, x, 0))
                } else {
                    config.custom_gates.selectors.get(&(BaseOp::CumProd, x, 0))
                };

                region.enable(selector, z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    let last_elem = output.get_slice(&[output.len() - 1..output.len()])?;

    region.increment(assigned_len);

    // last element is the result
    Ok(last_elem)
}

/// Axes wise op wrapper
fn axes_wise_op<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
    // generic layout op
    op: impl Fn(
            &BaseConfig<F>,
            &mut RegionCtx<F>,
            &[ValTensor<F>; 1],
        ) -> Result<ValTensor<F>, Box<dyn Error>>
        + Send
        + Sync,
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

    let inner_loop_function = |i: usize, region: &mut RegionCtx<'_, F>| {
        let coord = cartesian_coord[i].clone();
        let mut prod_dims = vec![];
        for (i, c) in coord.iter().enumerate() {
            if axes.contains(&i) {
                prod_dims.push(0..a.dims()[i]);
            } else {
                prod_dims.push(*c..*c + 1);
            }
        }
        let values = a.get_slice(&prod_dims)?;
        let op = op(config, region, &[values])?;

        Ok(op.get_inner_tensor()?[0].clone())
    };

    region.apply_in_loop(&mut res, inner_loop_function)?;

    Ok(res.into())
}

/// Sum accumulated layout
pub fn prod_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output
    axes_wise_op(config, region, values, axes, prod)
}

/// Sum accumulated layout
pub fn sum_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output
    axes_wise_op(config, region, values, axes, sum)
}

/// argmax layout
pub fn argmax_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    dim: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // these will be assigned as constants
    let mut indices = Tensor::from((0..values[0].dims()[dim] as u64).map(|x| F::from(x)));
    indices.set_visibility(&crate::graph::Visibility::Fixed);
    let indices = region.assign(&config.custom_gates.inputs[1], &indices.try_into()?)?;
    region.increment(indices.len());

    let argmax = move |config: &BaseConfig<F>,
                       region: &mut RegionCtx<F>,
                       values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, Box<dyn Error>> {
        argmax(config, region, values, indices.clone())
    };

    // calculate value of output
    axes_wise_op(config, region, values, &[dim], argmax)
}

/// Max accumulated layout
pub fn max_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output

    axes_wise_op(config, region, values, axes, max)
}

/// Argmin layout
pub fn argmin_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    dim: usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output
    // these will be assigned as constants
    let mut indices = Tensor::from((0..values[0].dims()[dim] as u64).map(|x| F::from(x)));
    indices.set_visibility(&crate::graph::Visibility::Fixed);
    let indices = region.assign(&config.custom_gates.inputs[1], &indices.try_into()?)?;
    region.increment(indices.len());

    let argmin = move |config: &BaseConfig<F>,
                       region: &mut RegionCtx<F>,
                       values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, Box<dyn Error>> {
        argmin(config, region, values, indices.clone())
    };

    axes_wise_op(config, region, values, &[dim], argmin)
}

/// Min accumulated layout
pub fn min_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // calculate value of output

    axes_wise_op(config, region, values, axes, min)
}

/// Pairwise (elementwise) op layout
pub fn pairwise<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    op: BaseOp,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // time to calculate the value of the output
    let global_start = instant::Instant::now();

    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;

    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    // original values
    let orig_lhs = lhs.clone();
    let orig_rhs = rhs.clone();

    // get indices of zeros
    let first_zero_indices = lhs.get_const_zero_indices()?;
    let second_zero_indices = rhs.get_const_zero_indices()?;
    let mut removal_indices = match op {
        BaseOp::Add | BaseOp::Mult => {
            let mut removal_indices = first_zero_indices.clone();
            removal_indices.extend(second_zero_indices.clone());
            removal_indices
        }
        BaseOp::Sub => second_zero_indices.clone(),
        _ => return Err(Box::new(CircuitError::UnsupportedOp)),
    };
    removal_indices.dedup();

    let removal_indices: HashSet<&usize> = HashSet::from_iter(removal_indices.iter());
    let removal_indices_ptr = &removal_indices;

    if lhs.len() != rhs.len() {
        return Err(Box::new(CircuitError::DimMismatch(format!(
            "pairwise {} layout",
            op.as_str()
        ))));
    }

    let mut inputs = vec![];
    for (i, input) in [lhs.clone(), rhs.clone()].iter().enumerate() {
        let inp = {
            let res = region.assign_with_omissions(
                &config.custom_gates.inputs[i],
                input,
                removal_indices_ptr,
            )?;

            res.get_inner()?
        };

        inputs.push(inp);
    }

    // Now we can assign the dot product
    // time the calc
    let start = instant::Instant::now();
    let op_result = match op {
        BaseOp::Add => add(&inputs),
        BaseOp::Sub => sub(&inputs),
        BaseOp::Mult => mult(&inputs),
        _ => return Err(Box::new(CircuitError::UnsupportedOp)),
    }
    .map_err(|e| {
        error!("{}", e);
        halo2_proofs::plonk::Error::Synthesis
    })?;
    let elapsed = start.elapsed();

    let assigned_len = inputs[0].len() - removal_indices.len();
    let mut output = region.assign_with_omissions(
        &config.custom_gates.output,
        &op_result.into(),
        removal_indices_ptr,
    )?;
    trace!("pairwise {} calc took {:?}", op.as_str(), elapsed);

    // Enable the selectors
    if !region.is_dummy() {
        (0..assigned_len)
            .map(|i| {
                let (x, y, z) =
                    config.custom_gates.inputs[0].cartesian_coord(region.linear_coord() + i);
                let selector = config.custom_gates.selectors.get(&(op.clone(), x, y));

                region.enable(selector, z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }
    region.increment(assigned_len);

    let a_tensor = orig_lhs.get_inner_tensor()?;
    let b_tensor = orig_rhs.get_inner_tensor()?;

    let first_zero_indices: HashSet<&usize> = HashSet::from_iter(first_zero_indices.iter());
    let second_zero_indices: HashSet<&usize> = HashSet::from_iter(second_zero_indices.iter());

    trace!("setting up indices took {:?}", start.elapsed());

    // infill the zero indices with the correct values from values[0] or values[1]
    if !removal_indices_ptr.is_empty() {
        output
            .get_inner_tensor_mut()?
            .par_enum_map_mut_filtered(removal_indices_ptr, |i| {
                let val = match op {
                    BaseOp::Add => {
                        let a_is_null = first_zero_indices.contains(&i);
                        let b_is_null = second_zero_indices.contains(&i);

                        if a_is_null && b_is_null {
                            ValType::Constant(F::ZERO)
                        } else if a_is_null {
                            b_tensor[i].clone()
                        } else {
                            a_tensor[i].clone()
                        }
                    }
                    BaseOp::Sub => {
                        let a_is_null = first_zero_indices.contains(&i);
                        // by default b is null in this case for sub
                        if a_is_null {
                            ValType::Constant(F::ZERO)
                        } else {
                            a_tensor[i].clone()
                        }
                    }
                    BaseOp::Mult => ValType::Constant(F::ZERO),
                    // can safely panic as the prior check ensures this is not called
                    _ => unreachable!(),
                };
                Ok::<_, TensorError>(val)
            })?;
    }

    output.reshape(&broadcasted_shape)?;

    let end = global_start.elapsed();
    trace!(
        "pairwise {} layout took {:?}, row: {}",
        op.as_str(),
        end,
        region.row()
    );

    Ok(output)
}

/// expand the tensor to the given shape
pub fn expand<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    shape: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut assigned_input = region.assign(&config.custom_gates.inputs[0], &values[0])?;
    assigned_input.expand(shape)?;
    region.increment(assigned_input.len());
    Ok(assigned_input)
}

///
pub fn greater<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;

    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    let diff = pairwise(config, region, &[lhs, rhs], BaseOp::Sub)?;

    nonlinearity(
        config,
        region,
        &[diff],
        &LookupOp::GreaterThan { a: utils::F32(0.) },
    )
}

///
pub fn greater_equal<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;

    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    let diff = pairwise(config, region, &[lhs, rhs], BaseOp::Sub)?;

    nonlinearity(
        config,
        region,
        &[diff],
        &LookupOp::GreaterThanEqual { a: utils::F32(0.) },
    )
}

///
pub fn less<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // just flip the order and use greater
    greater(config, region, &[values[1].clone(), values[0].clone()])
}

///
pub fn less_equal<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // just flip the order and use greater
    greater_equal(config, region, &[values[1].clone(), values[0].clone()])
}

/// And boolean operation
pub fn and<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let res = pairwise(config, region, values, BaseOp::Mult)?;

    Ok(res)
}

/// Or boolean operation
pub fn or<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let a = values[0].clone();
    let b = values[1].clone();

    let iff_values = &[a.clone(), a, b];

    let res = iff(config, region, iff_values)?;

    Ok(res)
}

/// Equality boolean operation
pub fn equals<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let diff = pairwise(config, region, values, BaseOp::Sub)?;
    equals_zero(config, region, &[diff])
}

/// Equality boolean operation
pub fn equals_zero<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let values = values[0].clone();
    let values_inverse = values.inverse()?;
    let product_values_and_invert = pairwise(
        config,
        region,
        &[values.clone(), values_inverse],
        BaseOp::Mult,
    )?;

    // constant of 1
    let mut ones = Tensor::from(vec![ValType::Constant(F::from(1))].into_iter());
    ones.set_visibility(&crate::graph::Visibility::Fixed);

    // subtract
    let output = pairwise(
        config,
        region,
        &[ones.into(), product_values_and_invert],
        BaseOp::Sub,
    )?;

    // take the product of diff and output
    let prod_check = pairwise(config, region, &[values, output.clone()], BaseOp::Mult)?;

    is_zero_identity(config, region, &[prod_check], false)?;

    Ok(output)
}

/// Xor boolean operation
pub fn xor<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let lhs = values[0].clone();
    let rhs = values[1].clone();

    let lhs_not = not(config, region, &[lhs.clone()])?;
    let rhs_not = not(config, region, &[rhs.clone()])?;

    let lhs_and_rhs_not = and(config, region, &[lhs, rhs_not.clone()])?;
    let lhs_not_and_rhs = and(config, region, &[rhs, lhs_not])?;

    // we can safely use add and not OR here because we know that lhs_and_rhs_not and lhs_not_and_rhs are =1 at different incices
    let res: ValTensor<F> = pairwise(
        config,
        region,
        &[lhs_and_rhs_not, lhs_not_and_rhs],
        BaseOp::Add,
    )?;

    Ok(res)
}

/// Not boolean operation
pub fn not<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mask = values[0].clone();

    let unit: ValTensor<F> = Tensor::from(
        vec![region.assign_constant(&config.custom_gates.inputs[0], F::from(1))?].into_iter(),
    )
    .into();

    // to leverage sparsity we don't assign this guy
    let nil: ValTensor<F> = Tensor::from(vec![ValType::Constant(F::from(0))].into_iter()).into();
    region.next();

    let res = iff(config, region, &[mask, nil, unit])?;

    Ok(res)
}

/// Iff
pub fn iff<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 3],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // if mask > 0 then output a else output b
    let (mask, a, b) = (&values[0], &values[1], &values[2]);

    let unit: ValTensor<F> = Tensor::from(
        vec![region.assign_constant(&config.custom_gates.inputs[0], F::from(1))?].into_iter(),
    )
    .into();

    // make sure mask is boolean
    let assigned_mask = boolean_identity(config, region, &[mask.clone()], true)?;

    let one_minus_mask = pairwise(config, region, &[unit, assigned_mask.clone()], BaseOp::Sub)?;

    let masked_a = pairwise(config, region, &[a.clone(), assigned_mask], BaseOp::Mult)?;

    let masked_b = pairwise(config, region, &[b.clone(), one_minus_mask], BaseOp::Mult)?;

    let res = pairwise(config, region, &[masked_a, masked_b], BaseOp::Add)?;

    Ok(res)
}

/// Negation operation accumulated layout
pub fn neg<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut nil = Tensor::from(vec![ValType::Constant(F::from(0))].into_iter());
    nil.set_visibility(&crate::graph::Visibility::Fixed);
    pairwise(
        config,
        region,
        &[nil.into(), values[0].clone()],
        BaseOp::Sub,
    )
}

/// Sumpool accumulated layout
pub fn sumpool<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    padding: [(usize, usize); 2],
    stride: (usize, usize),
    kernel_shape: (usize, usize),
    normalized: bool,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let batch_size = values[0].dims()[0];
    let image_channels = values[0].dims()[1];

    let unit = region.assign_constant(&config.custom_gates.inputs[1], F::from(1))?;
    region.next();

    let mut kernel = Tensor::from(0..kernel_shape.0 * kernel_shape.1).map(|_| unit.clone());
    kernel.reshape(&[1, 1, kernel_shape.0, kernel_shape.1])?;

    let cartesian_coord = [(0..batch_size), (0..image_channels)]
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut res = vec![];

    cartesian_coord
        .iter()
        .map(|coord| {
            let (b, i) = (coord[0], coord[1]);
            let input = values[0].get_slice(&[b..b + 1, i..i + 1])?;
            let output = conv(
                config,
                region,
                &[input, kernel.clone().into()],
                padding,
                stride,
            )?;
            res.push(output);
            Ok(())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let shape = &res[0].dims()[2..];
    let mut last_elem = res[1..]
        .iter()
        .try_fold(res[0].clone(), |acc, elem| acc.concat(elem.clone()))?;
    last_elem.reshape(&[&[batch_size, image_channels], shape].concat())?;

    if normalized {
        last_elem = loop_div(
            config,
            region,
            &[last_elem],
            F::from((kernel_shape.0 * kernel_shape.1) as u64),
        )?;
    }
    Ok(last_elem)
}

/// Convolution accumulated layout
pub fn max_pool2d<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    padding: [(usize, usize); 2],
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

    let vert_slides = (image_height + padding[0].0 + padding[1].0 - pool_dims.0) / stride.0 + 1;
    let horz_slides = (image_width + padding[0].1 + padding[1].1 - pool_dims.1) / stride.1 + 1;

    let mut output: Tensor<ValType<F>> =
        Tensor::new(None, &[batch, input_channels, horz_slides, vert_slides])?;

    let cartesian_coord = [
        (0..batch),
        (0..input_channels),
        (0..vert_slides),
        (0..horz_slides),
    ]
    .iter()
    .cloned()
    .multi_cartesian_product()
    .collect::<Vec<_>>();

    output
        .iter_mut()
        .enumerate()
        .map(|(flat_index, o)| {
            let coord = &cartesian_coord[flat_index];
            let (b, i, j, k) = (coord[0], coord[1], coord[2], coord[3]);
            let rs = j * stride.0;
            let cs = k * stride.1;
            let slice = padded_image.get_slice(&[
                b..(b + 1),
                i..(i + 1),
                rs..(rs + pool_dims.0),
                cs..(cs + pool_dims.1),
            ])?;
            let max_w = max(config, region, &[slice])?;
            *o = max_w.get_inner_tensor()?[0].clone();
            Ok(())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let res: ValTensor<F> = output.into();

    Ok(res)
}

/// DeConvolution accumulated layout
pub fn deconv<F: PrimeField + TensorType + PartialOrd + std::marker::Send + std::marker::Sync>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    inputs: &[ValTensor<F>],
    padding: [(usize, usize); 2],
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

    let (kernel_height, kernel_width) = (kernel.dims()[2], kernel.dims()[3]);

    let null_val = ValType::Constant(F::ZERO);
    // region.assign_constant(&config.custom_gates.inputs[1], F::from(0))?;
    // region.next();

    let mut expanded_image = image.clone();
    expanded_image.intercalate_values(null_val.clone(), stride.0, 2)?;
    expanded_image.intercalate_values(null_val, stride.1, 3)?;
    expanded_image.pad([(kernel_height - 1, kernel_width - 1); 2])?;

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
                padding[0].0..d - padding[1].0 + output_padding.0
            } else if i == 3 {
                padding[0].1..d - padding[1].1 + output_padding.1
            } else {
                0..*d
            }
        })
        .collect::<Vec<_>>();

    let sliced_expanded_image = expanded_image.get_slice(&slice_coord)?;

    let mut inverted_kernels = vec![];

    for (i, j) in channel_coord {
        let channel = kernel.get_slice(&[i..i + 1, j..j + 1])?;
        let mut channel = Tensor::from(channel.get_inner_tensor()?.clone().into_iter().rev());
        channel.reshape(&[kernel.dims()[2], kernel.dims()[3]])?;
        inverted_kernels.push(channel);
    }

    let mut deconv_kernel =
        Tensor::new(Some(&inverted_kernels), &[inverted_kernels.len()])?.combine()?;
    deconv_kernel.reshape(kernel.dims())?;

    // tensorflow formatting patch
    if kernel.dims()[0] == sliced_expanded_image.dims()[1] {
        deconv_kernel.reshape(&[
            kernel.dims()[1],
            kernel.dims()[0],
            kernel.dims()[2],
            kernel.dims()[3],
        ])?;
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

    let output = conv(config, region, &conv_input, [(0, 0); 2], (1, 1))?;

    Ok(output)
}

/// Convolution accumulated layout
pub fn conv<F: PrimeField + TensorType + PartialOrd + std::marker::Send + std::marker::Sync>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    padding: [(usize, usize); 2],
    stride: (usize, usize),
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let has_bias = values.len() == 3;
    let (mut image, mut kernel) = (values[0].clone(), values[1].clone());

    // we specifically want to use the same kernel and image for all the convolutions and need to enforce this by assigning them
    // 1. assign the kernel
    let mut assigned_len = vec![];

    if !kernel.all_prev_assigned() {
        kernel = region.assign(&config.custom_gates.inputs[0], &kernel)?;
        assigned_len.push(kernel.len());
    }
    // 2. assign the image
    if !image.all_prev_assigned() {
        image = region.assign(&config.custom_gates.inputs[1], &image)?;
        assigned_len.push(image.len());
    }

    if !assigned_len.is_empty() {
        // safe to unwrap since we've just checked it has at least one element
        region.increment(*assigned_len.iter().max().unwrap());
    }

    let og_image_dims = image.dims().to_vec();
    let og_kernel_dims = kernel.dims().to_vec();
    // ensure inputs are 4D tensors
    if og_image_dims.len() == 3 {
        // adds a dummy image_channels dimension
        let mut new_dims = image.dims().to_vec();
        // insert 1 at the input_channels pos
        if og_kernel_dims.len() == 3 {
            new_dims.insert(1, 1);
        } else {
            new_dims.insert(0, 1);
        }
        image.reshape(&new_dims)?;
    }

    // ensure kernel is 4D tensor
    if og_kernel_dims.len() == 3 && og_image_dims.len() == 3 {
        // adds a dummy image_channels dimension
        let mut new_dims = kernel.dims().to_vec();
        // insert 1 at the input_channels pos
        new_dims.insert(1, 1);
        kernel.reshape(&new_dims)?;
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

    let vert_slides = (image_height + padding[0].0 + padding[1].0 - kernel_height) / stride.0 + 1;
    let horz_slides = (image_width + padding[0].1 + padding[1].1 - kernel_width) / stride.1 + 1;

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

    let mut output: Tensor<ValType<F>> = Tensor::new(None, &[num_outputs])?;

    let cartesian_coord = [
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

    let inner_loop_function = |idx: usize, region: &mut RegionCtx<F>| {
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

        let mut local_image = padded_image.get_slice(&[
            batch..batch + 1,
            start_channel..end_channel,
            rs..(rs + kernel_height),
            cs..(cs + kernel_width),
        ])?;

        local_image.flatten();

        let start_kernel_index = group * output_channels_per_group + i;
        let end_kernel_index = start_kernel_index + 1;
        let mut local_kernel = kernel.get_slice(&[start_kernel_index..end_kernel_index])?;

        local_kernel.flatten();

        // this is dot product notation in einsum format
        let mut res = einsum(config, region, &[local_image, local_kernel], "i,i->")?;

        if has_bias {
            let bias_index = if values[2].len() > 1 {
                start_kernel_index
            } else {
                0
            };

            let bias = values[2].get_single_elem(bias_index)?;
            res = pairwise(config, region, &[res, bias], BaseOp::Add)?;
        }
        region.flush()?;

        Ok(res.get_inner_tensor()?[0].clone())
    };

    region.flush()?;
    region.apply_in_loop(&mut output, inner_loop_function)?;

    let reshape_output = |output: &mut Tensor<ValType<F>>| -> Result<(), TensorError> {
        // remove dummy batch dimension if we added one
        if og_image_dims.len() == 3 && vert_slides == 1 {
            output.reshape(&[batch_size, output_channels, horz_slides])?;
        } else if og_image_dims.len() == 3 {
            output.reshape(&[output_channels, vert_slides, horz_slides])?;
        } else {
            output.reshape(&[batch_size, output_channels, vert_slides, horz_slides])?;
        }
        Ok(())
    };

    // remove dummy batch dimension if we added one
    reshape_output(&mut output)?;

    let output: ValTensor<_> = output.into();

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

        let multiplier: ValTensor<F> =
            Tensor::from(vec![ValType::Constant(F::from(scales[i].1 as u64))].into_iter()).into();
        let scaled_input = pairwise(config, region, &[ri.clone(), multiplier], BaseOp::Mult)?;
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
    let mut base_t = <F as TensorType>::zero().ok_or(TensorError::FeltError)?;
    for _ in 0..base {
        base_t += <F as TensorType>::one().ok_or(TensorError::FeltError)?;
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
    let mut output = region.assign(&config.custom_gates.output, &values[0])?;
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
    let mut output = region.assign(&config.custom_gates.output, &values[0])?;
    region.increment(output.len());
    output.slice(axis, start, end)?;

    Ok(output)
}

/// Concat layout
pub fn concat<F: PrimeField + TensorType + PartialOrd>(
    values: &[ValTensor<F>],
    axis: &usize,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let collected_inner: Result<Vec<&Tensor<_>>, _> =
        values.iter().map(|e| e.get_inner_tensor()).collect();
    let collected_inner = collected_inner?;

    Ok(tensor::ops::concat(&collected_inner, *axis)?.into())
}

/// Identity constraint. Usually used to constrain an instance column to an advice so the returned cells / values can be operated upon.
pub fn identity<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut output = values[0].clone();
    if !output.all_prev_assigned() {
        output = region.assign(&config.custom_gates.output, &values[0])?;
        region.increment(output.len());
    }

    Ok(output)
}

/// is zero identity constraint.
pub fn is_zero_identity<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    assign: bool,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let output = if assign || !values[0].get_const_indices()?.is_empty() {
        let output = region.assign(&config.custom_gates.output, &values[0])?;
        region.increment(output.len());
        output
    } else {
        values[0].clone()
    };
    // Enable the selectors
    if !region.is_dummy() {
        (0..output.len())
            .map(|j| {
                let index = region.linear_coord() - j - 1;

                let (x, y, z) = config.custom_gates.output.cartesian_coord(index);
                let selector = config.custom_gates.selectors.get(&(BaseOp::IsZero, x, y));

                region.enable(selector, z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    Ok(output)
}

/// Boolean identity constraint. Usually used to constrain an instance column to an advice so the returned cells / values can be operated upon.
pub fn boolean_identity<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    assign: bool,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let output = if assign || !values[0].get_const_indices()?.is_empty() {
        // get zero constants indices
        let output = region.assign(&config.custom_gates.output, &values[0])?;
        region.increment(output.len());
        output
    } else {
        values[0].clone()
    };
    // Enable the selectors
    if !region.is_dummy() {
        (0..output.len())
            .map(|j| {
                let index = region.linear_coord() - j - 1;

                let (x, y, z) = config.custom_gates.output.cartesian_coord(index);
                let selector = config
                    .custom_gates
                    .selectors
                    .get(&(BaseOp::IsBoolean, x, y));

                region.enable(selector, z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

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
    let input = region.assign(&config.custom_gates.inputs[0], &values[0])?;
    let processed_output =
        tensor::ops::downsample(input.get_inner_tensor()?, *axis, *stride, *modulo)?;
    let output = region.assign(&config.custom_gates.output, &processed_output.into())?;
    region.increment(std::cmp::max(input.len(), output.len()));
    Ok(output)
}

/// layout for enforcing two sets of cells to be equal
pub fn enforce_equality<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // assert of same len

    // assigns the instance to the advice.
    let input = region.assign(&config.custom_gates.inputs[1], &values[0])?;
    let output = region.assign(&config.custom_gates.output, &values[1])?;

    if !region.is_dummy() {
        region.constrain_equal(&input, &output)?;
    }

    region.increment(output.len());

    Ok(output)
}

/// layout for range check.
pub fn range_check<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    range: &crate::circuit::table::Range,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    region.add_used_range_check(*range)?;

    // time the entire operation
    let timer = instant::Instant::now();

    let x = values[0].clone();

    let w = region.assign(&config.range_checks.input, &x)?;

    let assigned_len = x.len();

    let is_dummy = region.is_dummy();

    let table_index: ValTensor<F> = w
        .get_inner_tensor()?
        .par_enum_map(|_, e| {
            Ok::<ValType<F>, TensorError>(if let Some(f) = e.get_felt_eval() {
                let col_idx = if !is_dummy {
                    let table = config
                        .range_checks
                        .ranges
                        .get(range)
                        .ok_or(TensorError::TableLookupError)?;
                    table.get_col_index(f)
                } else {
                    F::ZERO
                };
                Value::known(col_idx).into()
            } else {
                Value::<F>::unknown().into()
            })
        })?
        .into();

    region.assign(&config.range_checks.index, &table_index)?;

    if !is_dummy {
        (0..assigned_len)
            .map(|i| {
                let (x, y, z) = config
                    .range_checks
                    .input
                    .cartesian_coord(region.linear_coord() + i);
                let selector = config.range_checks.selectors.get(&(*range, x, y));
                region.enable(selector, z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    if region.throw_range_check_error() {
        // assert is within range
        let int_values = w.get_int_evals()?;
        for v in int_values {
            if v < range.0 || v > range.1 {
                log::debug!("Value ({:?}) out of range: {:?}", v, range);
                return Err(Box::new(TensorError::TableLookupError));
            }
        }
    }

    region.increment(assigned_len);

    let elapsed = timer.elapsed();
    trace!(
        "range check {:?} layout took {:?}, row: {:?}",
        range,
        elapsed,
        region.row()
    );

    Ok(w)
}

/// layout for nonlinearity check.
pub fn nonlinearity<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    nl: &LookupOp,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    region.add_used_lookup(nl.clone(), values)?;

    // time the entire operation
    let timer = instant::Instant::now();

    let x = values[0].clone();

    let removal_indices = values[0].get_const_indices()?;
    let removal_indices: HashSet<&usize> = HashSet::from_iter(removal_indices.iter());
    let removal_indices_ptr = &removal_indices;

    let w = region.assign_with_omissions(&config.static_lookups.input, &x, removal_indices_ptr)?;

    let output = w.get_inner_tensor()?.par_enum_map(|i, e| {
        Ok::<_, TensorError>(if let Some(f) = e.get_felt_eval() {
            if !removal_indices.contains(&i) {
                Value::known(Op::<F>::f(nl, &[Tensor::from(vec![f].into_iter())])?.output[0]).into()
            } else {
                ValType::Constant(Op::<F>::f(nl, &[Tensor::from(vec![f].into_iter())])?.output[0])
            }
        } else {
            Value::<F>::unknown().into()
        })
    })?;

    let assigned_len = x.len() - removal_indices.len();
    let mut output = region.assign_with_omissions(
        &config.static_lookups.output,
        &output.into(),
        removal_indices_ptr,
    )?;

    let is_dummy = region.is_dummy();

    let table_index: ValTensor<F> = w
        .get_inner_tensor()?
        .par_enum_map(|i, e| {
            Ok::<_, TensorError>(if let Some(f) = e.get_felt_eval() {
                let col_idx = if !is_dummy {
                    let table = config
                        .static_lookups
                        .tables
                        .get(nl)
                        .ok_or(TensorError::TableLookupError)?;
                    table.get_col_index(f)
                } else {
                    F::ZERO
                };
                if !removal_indices.contains(&i) {
                    Value::known(col_idx).into()
                } else {
                    ValType::Constant(col_idx)
                }
            } else {
                Value::<F>::unknown().into()
            })
        })?
        .into();

    region.assign_with_omissions(
        &config.static_lookups.index,
        &table_index,
        removal_indices_ptr,
    )?;

    if !is_dummy {
        (0..assigned_len)
            .map(|i| {
                let (x, y, z) = config
                    .static_lookups
                    .input
                    .cartesian_coord(region.linear_coord() + i);
                let selector = config.static_lookups.selectors.get(&(nl.clone(), x, y));
                region.enable(selector, z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    }

    region.increment(assigned_len);

    output.reshape(x.dims())?;

    let elapsed = timer.elapsed();
    trace!(
        "nonlinearity {} layout took {:?}, row: {:?}",
        <LookupOp as Op<F>>::as_string(nl),
        elapsed,
        region.row()
    );

    // constrain the calculated output to a column
    Ok(output)
}

/// Argmax
pub fn argmax<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    indices: ValTensor<F>,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // this is safe because we later constrain it
    let argmax = values[0]
        .get_int_evals()?
        .into_par_iter()
        .enumerate()
        // we value the first index in the case of a tie
        .max_by_key(|(idx, value)| (*value, -(*idx as i64)))
        .map(|(idx, _)| idx as i128);
    let argmax_val: ValTensor<F> = match argmax {
        None => Tensor::new(Some(&[Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&[Value::known(i128_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_argmax: ValTensor<F> =
        region.assign(&config.custom_gates.inputs[1], &argmax_val)?;
    region.increment(assigned_argmax.len());

    let claimed_val = select(
        config,
        region,
        &[values[0].clone(), assigned_argmax.clone()],
        indices,
    )?;

    let max_val = max(config, region, &[values[0].clone()])?;

    enforce_equality(config, region, &[claimed_val, max_val])?;

    Ok(assigned_argmax)
}

/// Argmin
pub fn argmin<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    indices: ValTensor<F>,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // this is safe because we later constrain it
    let argmin = values[0]
        .get_int_evals()?
        .into_par_iter()
        .enumerate()
        // we value the first index in the case of a tie
        .min_by_key(|(idx, value)| (*value, (*idx as i64)))
        .map(|(idx, _)| idx as i128);
    let argmin_val: ValTensor<F> = match argmin {
        None => Tensor::new(Some(&[Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&[Value::known(i128_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_argmin: ValTensor<F> =
        region.assign(&config.custom_gates.inputs[1], &argmin_val)?;
    region.increment(assigned_argmin.len());

    // these will be assigned as constants
    let claimed_val = select(
        config,
        region,
        &[values[0].clone(), assigned_argmin.clone()],
        indices,
    )?;
    let min_val = min(config, region, &[values[0].clone()])?;

    enforce_equality(config, region, &[claimed_val, min_val])?;

    Ok(assigned_argmin)
}

/// max layout
pub fn max<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let input_len = values[0].len();
    _sort_ascending(config, region, values)?.get_slice(&[input_len - 1..input_len])
}

/// min layout
pub fn min<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    _sort_ascending(config, region, values)?.get_slice(&[0..1])
}

fn multi_dim_axes_op<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
    op: impl Fn(
            &BaseConfig<F>,
            &mut RegionCtx<F>,
            &[ValTensor<F>; 1],
        ) -> Result<ValTensor<F>, Box<dyn Error>>
        + Send
        + Sync,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let mut input = values[0].clone();

    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
        region.increment(input.len());
    }

    if input.dims().len() == 1 {
        return op(config, region, &[input]);
    }

    // Calculate the output tensor size
    let input_dims = input.dims();

    let mut sorted_axes = axes.to_vec();
    // descending order
    sorted_axes.sort_by(|x, y| y.cmp(x));

    let mut output_size_without_dim = input_dims.to_vec();
    for dim in &sorted_axes {
        output_size_without_dim.remove(*dim);
    }

    let mut op_tensors = Tensor::<ValTensor<F>>::new(None, &output_size_without_dim)?;

    // Allocate memory for the output tensor
    let cartesian_coord = output_size_without_dim
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let inner_loop_function = |i: usize, region: &mut RegionCtx<F>| {
        let coord = cartesian_coord[i].clone();
        let mut slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();

        for dim in &sorted_axes {
            slice.insert(*dim, 0..input_dims[*dim]);
        }

        let mut sliced_input = input.get_slice(&slice)?;
        sliced_input.flatten();

        Ok(op(config, region, &[sliced_input])?)
    };

    region.apply_in_loop(&mut op_tensors, inner_loop_function)?;

    // assert all op_tensors have the same dims
    let sample_op_output_size = op_tensors[0].dims();

    // now deduce the output size from the dims of the output tensors
    let mut output_size = input_dims.to_vec();
    for dim in axes.iter().enumerate() {
        output_size[*dim.1] = sample_op_output_size[dim.0];
    }

    // Allocate memory for the output tensor
    let cartesian_coord = output_size
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let mut output = Tensor::<ValType<F>>::new(None, &output_size)?;

    output = output.par_enum_map(|i, _| {
        let coord = cartesian_coord[i].clone();
        let mut op_idx = coord.clone();
        let mut coord_at_dims = vec![];
        for dim in &sorted_axes {
            op_idx.remove(*dim);
        }
        for dim in axes {
            coord_at_dims.push(coord[*dim]);
        }

        let topk_elem = op_tensors
            .get(&op_idx)
            .get_inner_tensor()?
            .get(&coord_at_dims)
            .clone();

        Ok::<_, region::RegionError>(topk_elem)
    })?;

    Ok(output.into())
}

/// softmax layout
pub fn softmax_axes<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
    axes: &[usize],
) -> Result<ValTensor<F>, Box<dyn Error>> {
    let soft_max_at_scale = move |config: &BaseConfig<F>,
                                  region: &mut RegionCtx<F>,
                                  values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, Box<dyn Error>> {
        softmax(config, region, values, scale)
    };

    let output = multi_dim_axes_op(config, region, values, axes, soft_max_at_scale)?;

    Ok(output)
}

/// softmax func
pub fn softmax<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    // elementwise exponential
    let ex = nonlinearity(config, region, values, &LookupOp::Exp { scale })?;

    // sum of exps
    let denom = sum(config, region, &[ex.clone()])?;
    // get the inverse

    let felt_scale = F::from(scale.0 as u64);
    let inv_denom = recip(config, region, &[denom], felt_scale, felt_scale)?;

    // product of num * (1 / denom) = 2*output_scale
    let softmax = pairwise(config, region, &[ex, inv_denom], BaseOp::Mult)?;

    Ok(softmax)
}

/// Checks that the percent error between the expected public output and the actual output value
/// is within the percent error expressed by the `tol` input, where `tol == 1.0` means the percent
/// error tolerance is 1 percent.
pub fn range_check_percent<F: PrimeField + TensorType + PartialOrd>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    scale: utils::F32,
    tol: f32,
) -> Result<ValTensor<F>, Box<dyn Error>> {
    if tol == 0.0 {
        // regular equality constraint
        return enforce_equality(config, region, values);
    }

    let mut values = [values[0].clone(), values[1].clone()];

    values[0] = region.assign(&config.custom_gates.inputs[0], &values[0])?;
    values[1] = region.assign(&config.custom_gates.inputs[1], &values[1])?;
    let total_assigned_0 = values[0].len();
    let total_assigned_1 = values[1].len();
    let total_assigned = std::cmp::max(total_assigned_0, total_assigned_1);
    region.increment(total_assigned);

    // Calculate the difference between the expected output and actual output
    let diff = pairwise(config, region, &values, BaseOp::Sub)?;

    // integer scale
    let int_scale = scale.0 as i128;
    // felt scale
    let felt_scale = i128_to_felt(int_scale);
    // range check len capped at 2^(S-3) and make it divisible 2
    let range_check_bracket = std::cmp::min(
        utils::F32(scale.0),
        utils::F32(2_f32.powf((F::S - 5) as f32)),
    )
    .0;

    let range_check_bracket_int = range_check_bracket as i128;

    // input scale ratio we multiply by tol such that in the new scale range_check_len represents tol percent
    let input_scale_ratio = ((scale.0.powf(2.0) / range_check_bracket) * tol) as i128 / 2 * 2;

    let recip = recip(
        config,
        region,
        &[values[0].clone()],
        felt_scale,
        felt_scale * F::from(100),
    )?;

    log::debug!("recip: {}", recip.show());

    // Multiply the difference by the recip
    let product = pairwise(config, region, &[diff, recip], BaseOp::Mult)?;

    log::debug!("product: {}", product.show());
    let rebased_product = loop_div(config, region, &[product], i128_to_felt(input_scale_ratio))?;
    log::debug!("rebased_product: {}", rebased_product.show());

    // check that it is within the tolerance range
    range_check(
        config,
        region,
        &[rebased_product],
        &(-range_check_bracket_int, range_check_bracket_int),
    )
}
