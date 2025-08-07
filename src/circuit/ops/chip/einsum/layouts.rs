use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::{error, trace};

use crate::{
    circuit::{base::BaseOp, region::RegionCtx, CircuitError},
    tensor::{
        get_broadcasted_shape,
        ops::{accumulated, add, mult, sub},
        TensorError, TensorType, ValTensor, ValType,
    },
};

use super::EinsumOpConfig;

/// Pairwise (elementwise) op layout
fn pairwise<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 2],
    op: BaseOp,
    min_phase: usize,
) -> Result<ValTensor<F>, CircuitError> {
    assert!(min_phase <= 1);
    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;

    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    if lhs.len() != rhs.len() {
        return Err(CircuitError::DimMismatch(format!(
            "pairwise {} layout",
            op.as_str()
        )));
    }

    let inputs = [lhs, rhs]
        .iter()
        .zip(config.inputs.iter().skip(min_phase))
        .map(|(val, var)| {
            let res = region.assign_einsum(var, val)?;

            Ok(res.get_inner()?)
        })
        .collect::<Result<Vec<_>, CircuitError>>()?;

    // Now we can assign the dot product
    // time the calc
    let op_result = match op {
        BaseOp::Add => add(&inputs),
        BaseOp::Sub => sub(&inputs),
        BaseOp::Mult => mult(&inputs),
        _ => return Err(CircuitError::UnsupportedOp),
    }
    .map_err(|e| {
        error!("{}", e);
        halo2_proofs::plonk::Error::Synthesis
    })?;

    let assigned_len = op_result.len();
    let mut output = region.assign_einsum(&config.output, &op_result.into())?;

    // Enable the selectors
    if !region.is_dummy() {
        (0..assigned_len)
            .map(|i| {
                let (x, y, z) = config.inputs[0].cartesian_coord(region.einsum_col_coord() + i);
                let selector = config.selectors.get(&(min_phase, op.clone(), x, y));

                region.enable(selector, z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }
    region.increment_einsum_col_coord(assigned_len);

    output.reshape(&broadcasted_shape)?;

    Ok(output)
}

fn sum<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 1],
    phase: usize,
) -> Result<ValTensor<F>, CircuitError> {
    if values[0].len() == 1 {
        return Ok(values[0].clone());
    }
    assert!(phase == 0 || phase == 1);

    region.flush_einsum()?;
    // time this entire function run
    let mut input = values[0].clone();

    let block_width = config.output.num_inner_cols();

    let assigned_len: usize;
    let input = {
        // FIXME : should pad with constant zero but currently this incurs an error
        // `NotEnoughColumnsForConstants` in halo2 because trying to assign constant
        // value to advice column, how to workaround this issue?
        input.pad_to_zero_rem(block_width, ValType::Value(Value::known(F::ZERO)))?;
        let (res, len) =
            region.assign_einsum_with_duplication_unconstrained(&config.inputs[phase], &input)?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_sum = accumulated::sum(&input, block_width)?;

    let (output, output_assigned_len) = region.assign_einsum_with_duplication_constrained(
        &config.output,
        &accumulated_sum.into(),
        &crate::circuit::CheckMode::UNSAFE,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        for i in 0..output_assigned_len {
            let (x, _, z) = config
                .output
                .cartesian_coord(region.einsum_col_coord() + i * block_width);
            // skip over duplicates at start of column
            if z == 0 && i > 0 {
                continue;
            }
            let selector = if i == 0 {
                config.selectors.get(&(phase, BaseOp::SumInit, x, 0))
            } else {
                config.selectors.get(&(phase, BaseOp::Sum, x, 0))
            };

            region.enable(selector, z)?;
        }
    }

    let last_elem = output.last()?;

    region.increment_einsum_col_coord(assigned_len);

    // last element is the result
    Ok(last_elem)
}

pub fn prod<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 1],
    phase: usize,
) -> Result<ValTensor<F>, CircuitError> {
    assert!(phase == 0 || phase == 1);
    region.flush_einsum()?;
    let block_width = config.output.num_inner_cols();
    let assigned_len: usize;
    let input = {
        let mut input = values[0].clone();
        // FIXME : should pad with constant one but currently this incurs an error
        // `NotEnoughColumnsForConstants` in halo2 because trying to assign constant
        // value to advice column, how to workaround this issue?
        input.pad_to_zero_rem(block_width, ValType::Value(Value::known(F::ONE)))?;
        let (res, len) = region
            .assign_einsum_with_duplication_unconstrained(&config.inputs[phase], &input)
            .expect("Failed here!");
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_prod = accumulated::prod(&input, block_width)?;

    let (output, output_assigned_len) = region.assign_einsum_with_duplication_constrained(
        &config.output,
        &accumulated_prod.into(),
        &crate::circuit::CheckMode::UNSAFE,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        (0..output_assigned_len)
            .map(|i| {
                let (x, _, z) = config
                    .output
                    .cartesian_coord(region.einsum_col_coord() + i * block_width);
                // skip over duplicates at start of column
                if z == 0 && i > 0 {
                    return Ok(());
                }
                let selector = if i == 0 {
                    config.selectors.get(&(phase, BaseOp::CumProdInit, x, 0))
                } else {
                    config.selectors.get(&(phase, BaseOp::CumProd, x, 0))
                };

                region.enable(selector, z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    let last_elem = output.last()?;

    region.increment_einsum_col_coord(assigned_len);

    // last element is the result
    Ok(last_elem)
}

pub fn dot<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 2],
    min_phase: usize,
) -> Result<ValTensor<F>, CircuitError> {
    assert!(min_phase <= 1);
    if values[0].len() != values[1].len() {
        return Err(TensorError::DimMismatch("dot".to_string()).into());
    }

    region.flush_einsum()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    let mut values = vec![values[0].clone(), values[1].clone()];

    let mut inputs = vec![];
    let block_width = config.output.num_inner_cols();

    let mut assigned_len = 0;
    for (val, var) in values.iter_mut().zip(config.inputs.iter().skip(min_phase)) {
        // FIXME : should pad with constant zero but currently this incurs an error
        // `NotEnoughColumnsForConstants` in halo2 because trying to assign constant
        // value to advice column, how to workaround this issue?
        val.pad_to_zero_rem(block_width, ValType::Value(Value::known(F::ZERO)))?;
        let inp = {
            let (res, len) = region.assign_einsum_with_duplication_unconstrained(var, &val)?;
            assigned_len = len;
            res.get_inner()?
        };
        inputs.push(inp);
    }

    // Now we can assign the dot product
    // time this step
    let accumulated_dot = accumulated::dot(&inputs[0], &inputs[1], block_width)?;
    let (output, output_assigned_len) = region.assign_einsum_with_duplication_constrained(
        &config.output,
        &accumulated_dot.into(),
        &crate::circuit::CheckMode::UNSAFE,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        (0..output_assigned_len)
            .map(|i| {
                let (x, _, z) = config
                    .output
                    .cartesian_coord(region.einsum_col_coord() + i * block_width);
                // hop over duplicates at start of column
                if z == 0 && i > 0 {
                    return Ok(());
                }
                let selector = if i == 0 {
                    config.selectors.get(&(min_phase, BaseOp::DotInit, x, 0))
                } else {
                    config.selectors.get(&(min_phase, BaseOp::Dot, x, 0))
                };
                region.enable(selector, z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    let last_elem = output.last()?;

    region.increment_einsum_col_coord(assigned_len);

    // last element is the result

    let elapsed = global_start.elapsed();
    trace!("dot layout took: {:?}, row {}", elapsed, region.row());
    trace!("----------------------------");
    Ok(last_elem)
}

/// Dot product of more than two tensors
pub fn multi_dot<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>],
    phases: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    assert!(phases.iter().all(|phase| *phase == 0 || *phase == 1));
    if !values.iter().all(|value| value.len() == values[0].len()) {
        return Err(TensorError::DimMismatch("dot".to_string()).into());
    }

    region.flush_einsum()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    let values: Vec<ValTensor<F>> = values.iter().copied().cloned().collect();
    // do pairwise dot product between intermediate tensor and the next tensor
    let intermediate = values
        .iter()
        .zip(phases)
        .skip(1)
        .fold((values[0].clone(), phases[0]), |acc, (input, phase)| {
            (
                pairwise(
                    config,
                    region,
                    &[&acc.0, input],
                    BaseOp::Mult,
                    std::cmp::min(acc.1, *phase),
                )
                .unwrap(),
                std::cmp::max(acc.1, *phase),
            )
        })
        .0;

    // Sum the final tensor
    // In current freivalds algorithm, there are no tensor contraction between phase 0 tensors,
    // so the phase of the resulting tensor is set to 1
    let accumulated_dot = sum(config, region, &[&intermediate], 1)?;
    let last_elem = accumulated_dot.last()?;

    let elapsed = global_start.elapsed();
    trace!("multi_dot layout took: {:?}, row {}", elapsed, region.row());
    trace!("----------------------------");
    Ok(last_elem)
}
