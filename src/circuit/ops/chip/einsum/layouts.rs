use halo2curves::ff::PrimeField;
use log::{error, trace};

use crate::{
    circuit::{base::BaseOp, einsum::BaseOpInfo, region::RegionCtx, CheckMode, CircuitError},
    tensor::{
        get_broadcasted_shape,
        ops::{accumulated, add, mult, sub},
        TensorError, TensorType, ValTensor, ValType,
    },
};

use super::ContractionConfig;

/// Pairwise (elementwise) op layout
pub fn pairwise<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 2],
    op: BaseOp,
    phases: &[usize; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let (mut lhs, mut rhs) = if phases[0] <= phases[1] {
        (values[0].clone(), values[1].clone())
    } else {
        (values[1].clone(), values[0].clone())
    };

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;

    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    if lhs.len() != rhs.len() {
        return Err(CircuitError::DimMismatch(format!(
            "pairwise {} layout",
            op.as_str()
        )));
    }

    region.flush_einsum()?;

    let input_vars = config.get_input_vars(phases.as_slice().into());
    let output_var = config.get_output_var(phases.as_slice().into());

    let inputs = [lhs, rhs]
        .iter()
        .zip(input_vars)
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
    let mut output = region.assign_einsum(output_var, &op_result.into())?;

    // Enable the selectors
    if !region.is_dummy() {
        (0..assigned_len)
            .map(|i| {
                let (x, y, z) = output_var.cartesian_coord(region.einsum_col_coord() + i);
                let op_info = BaseOpInfo {
                    op_kind: op.clone(),
                    input_phases: phases.as_slice().into(),
                };
                let selector = config.selectors.get(&(op_info, x, y));

                region.enable(selector, z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }
    region.increment_einsum_col_coord(assigned_len);

    output.reshape(&broadcasted_shape)?;

    Ok(output)
}

pub fn sum<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 1],
    phase: usize,
    check_mode: &CheckMode,
) -> Result<ValTensor<F>, CircuitError> {
    if values[0].len() == 1 {
        return Ok(values[0].clone());
    }
    assert!(phase == 0 || phase == 1);

    region.flush_einsum()?;
    let mut input = values[0].clone();

    let block_width = config.block_width();

    let assigned_len: usize;
    let input = {
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
        let var = config.get_input_vars([phase].as_slice().into())[0];
        let (res, len) = region.assign_einsum_with_duplication_unconstrained(var, &input)?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_sum = accumulated::sum(&input, block_width)?;

    let output_var = config.get_output_var([phase].as_slice().into());
    let (output, output_assigned_len) = region.assign_einsum_with_duplication_constrained(
        output_var,
        &accumulated_sum.into(),
        check_mode,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        for i in 0..output_assigned_len {
            let (x, _, z) = output_var.cartesian_coord(region.einsum_col_coord() + i * block_width);
            // skip over duplicates at start of column
            if z == 0 && i > 0 {
                continue;
            }
            let selector = if i == 0 {
                let op_info = BaseOpInfo {
                    op_kind: BaseOp::SumInit,
                    input_phases: [phase].as_slice().into(),
                };
                config.selectors.get(&(op_info, x, 0))
            } else {
                let op_info = BaseOpInfo {
                    op_kind: BaseOp::Sum,
                    input_phases: [phase].as_slice().into(),
                };
                config.selectors.get(&(op_info, x, 0))
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
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 1],
    phase: usize,
    check_mode: &CheckMode,
) -> Result<ValTensor<F>, CircuitError> {
    assert!(phase == 0 || phase == 1);
    region.flush_einsum()?;
    let block_width = config.block_width();
    let assigned_len: usize;
    let input = {
        let mut input = values[0].clone();
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ONE))?;
        let var = config.get_input_vars([phase].as_slice().into())[0];
        let (res, len) = region.assign_einsum_with_duplication_unconstrained(var, &input)?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_prod = accumulated::prod(&input, block_width)?;

    let output_var = config.get_output_var([phase].as_slice().into());
    let (output, output_assigned_len) = region.assign_einsum_with_duplication_constrained(
        output_var,
        &accumulated_prod.into(),
        check_mode,
    )?;

    // enable the selectors
    if !region.is_dummy() {
        (0..output_assigned_len)
            .map(|i| {
                let (x, _, z) =
                    output_var.cartesian_coord(region.einsum_col_coord() + i * block_width);
                // skip over duplicates at start of column
                if z == 0 && i > 0 {
                    return Ok(());
                }
                let selector = if i == 0 {
                    let op_info = BaseOpInfo {
                        op_kind: BaseOp::CumProdInit,
                        input_phases: [phase].as_slice().into(),
                    };
                    config.selectors.get(&(op_info, x, 0))
                } else {
                    let op_info = BaseOpInfo {
                        op_kind: BaseOp::CumProd,
                        input_phases: [phase].as_slice().into(),
                    };
                    config.selectors.get(&(op_info, x, 0))
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
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>; 2],
    phases: &[usize; 2],
    check_mode: &CheckMode,
) -> Result<ValTensor<F>, CircuitError> {
    if values[0].len() != values[1].len() {
        return Err(TensorError::DimMismatch("dot".to_string()).into());
    }

    region.flush_einsum()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    let mut values = if phases[0] <= phases[1] {
        [values[0].clone(), values[1].clone()]
    } else {
        [values[1].clone(), values[0].clone()]
    };
    let vars = config.get_input_vars(phases.as_slice().into());

    let mut inputs = vec![];
    let block_width = config.block_width();

    let mut assigned_len = 0;
    for (val, var) in values.iter_mut().zip(vars) {
        val.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
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
    let output_var = config.get_output_var(phases.as_slice().into());
    let (output, output_assigned_len) = region
        .assign_einsum_with_duplication_constrained(output_var, &accumulated_dot.into(), check_mode)
        .expect("failed to assign einsum with duplication constrained");

    // enable the selectors
    if !region.is_dummy() {
        (0..output_assigned_len)
            .map(|i| {
                let (x, _, z) =
                    output_var.cartesian_coord(region.einsum_col_coord() + i * block_width);
                // hop over duplicates at start of column
                if z == 0 && i > 0 {
                    return Ok(());
                }
                let selector = if i == 0 {
                    let op_info = BaseOpInfo {
                        op_kind: BaseOp::DotInit,
                        input_phases: phases.as_slice().into(),
                    };
                    config.selectors.get(&(op_info, x, 0))
                } else {
                    let op_info = BaseOpInfo {
                        op_kind: BaseOp::Dot,
                        input_phases: phases.as_slice().into(),
                    };
                    config.selectors.get(&(op_info, x, 0))
                };
                region.enable(selector, z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    let last_elem = output.last()?;

    region.increment_einsum_col_coord(assigned_len);

    let elapsed = global_start.elapsed();
    trace!("dot layout took: {:?}, row {}", elapsed, region.row());
    trace!("----------------------------");
    Ok(last_elem)
}

/// Dot product of more than two tensors
pub fn multi_dot<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &ContractionConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[&ValTensor<F>],
    phases: &[usize],
    check_mode: &CheckMode,
) -> Result<ValTensor<F>, CircuitError> {
    assert!(phases.iter().all(|phase| *phase == 0 || *phase == 1));
    if !values.iter().all(|value| value.len() == values[0].len()) {
        return Err(TensorError::DimMismatch("dot".to_string()).into());
    }
    // time this entire function run
    let global_start = instant::Instant::now();

    let values: Vec<ValTensor<F>> = values.iter().copied().cloned().collect();
    // do pairwise dot product between intermediate tensor and the next tensor
    let (intermediate, output_phase) = values
        .into_iter()
        .zip(phases.iter().cloned())
        .reduce(|(intermediate, intermediate_phase), (input, phase)| {
            (
                pairwise(
                    config,
                    region,
                    &[&intermediate, &input],
                    BaseOp::Mult,
                    &[intermediate_phase, phase],
                )
                .unwrap(),
                std::cmp::max(intermediate_phase, phase),
            )
        })
        .unwrap();

    let accumulated_dot = sum(config, region, &[&intermediate], output_phase, check_mode)?;
    let last_elem = accumulated_dot.last()?;

    let elapsed = global_start.elapsed();
    trace!("multi_dot layout took: {:?}, row {}", elapsed, region.row());
    trace!("----------------------------");
    Ok(last_elem)
}
