use std::{
    collections::{HashMap, HashSet},
    f64::consts::E,
    ops::Range,
};

use halo2_proofs::circuit::Value;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::{error, trace};
use maybe_rayon::{
    iter::IntoParallelRefIterator,
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
};

use self::tensor::{create_constant_tensor, create_zero_tensor};

use super::{chip::BaseConfig, region::RegionCtx};
use crate::{
    circuit::{ops::base::BaseOp, utils},
    fieldutils::{felt_to_integer_rep, integer_rep_to_felt, IntegerRep},
    tensor::{
        create_unit_tensor, get_broadcasted_shape,
        ops::{accumulated, add, mult, sub},
        Tensor, TensorError, ValType,
    },
    tensor::{DataFormat, KernelFormat},
};

use super::*;
use crate::circuit::ops::lookup::LookupOp;

const ASCII_ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

/// Calculate the L1 distance between two tensors.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::l1_distance;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
/// &[3, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
/// &[3, 3],
/// ).unwrap());
/// let result = l1_distance::<Fp>(&dummy_config, &mut dummy_region, &[x, k]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 1, 1, 1, 2, 2, 2]), &[3, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn l1_distance<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let diff = pairwise(config, region, values, BaseOp::Sub)?;
    let abs_diff = abs(config, region, &[diff])?;

    Ok(abs_diff)
}

/// Determines if from a set of 3 tensors the 1st is closest to a reference tensor.
/// should only be used in the context of a monotonic function like the product used in the division, recipe, and sqrt arguments;
/// or the increasing powers of 2 in the ln argument. Which is used to construct a convex error function.
fn optimum_convex_function<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    x: &ValTensor<F>,
    f: impl Fn(&BaseConfig<F>, &mut RegionCtx<F>, &ValTensor<F>) -> Result<ValTensor<F>, CircuitError>,
) -> Result<ValTensor<F>, CircuitError> {
    let one = create_constant_tensor(F::from(1), 1);

    let f_x = f(config, region, x)?;

    let x_plus_1 = pairwise(config, region, &[x.clone(), one.clone()], BaseOp::Add)?;
    let f_x_plus_1 = f(config, region, &x_plus_1)?;

    let x_minus_1 = pairwise(config, region, &[x.clone(), one.clone()], BaseOp::Sub)?;
    let f_x_minus_1 = f(config, region, &x_minus_1)?;

    // because the function is convex, the result should be the minimum of the three
    // note that we offset the x by 1 to get the next value
    // f(x) <= f(x+1) and f(x) < f(x-1)
    // the result is 1 if the function is optimal solely because of the convexity of the function
    // the distances can be equal but this is only possible if f(x) and f(x+1) are both optimal, but if (f(x) = f(x + 1))
    // f(x+1) is not smaller than f(x + 1 - 1) = f(x) and thus f(x) is unique
    let f_x_is_opt_rhs = less_equal(config, region, &[f_x.clone(), f_x_plus_1.clone()])?;
    let f_x_is_opt_lhs = less(config, region, &[f_x.clone(), f_x_minus_1.clone()])?;

    let is_opt = and(config, region, &[f_x_is_opt_lhs, f_x_is_opt_rhs])?;

    Ok(is_opt)
}

/// Err is less than some constant
pub fn diff_less_than<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    constant: F,
) -> Result<(), CircuitError> {
    let distance = l1_distance(config, region, values)?;

    let constant = create_constant_tensor(constant, 1);
    let is_less = less(config, region, &[distance.clone(), constant.clone()])?;

    // assert the result is 1
    let comparison_unit = create_constant_tensor(F::ONE, is_less.len());
    enforce_equality(config, region, &[is_less, comparison_unit])?;

    Ok(())
}

/// Div accumulated layout
pub(crate) fn div<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    div: F,
) -> Result<ValTensor<F>, CircuitError> {
    if div == F::ONE {
        return Ok(value[0].clone());
    }

    let input = value[0].clone();
    let input_dims = input.dims();

    let divisor = create_constant_tensor(div, 1);

    let is_assigned = !input.any_unknowns()? && !divisor.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.int_evals()?;
        tensor::ops::nonlinearities::const_div(
            &input_evals.clone(),
            felt_to_integer_rep(div) as f64,
        )
        .par_iter()
        .map(|x| Value::known(integer_rep_to_felt(*x)))
        .collect::<Tensor<Value<F>>>()
        .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };
    claimed_output.reshape(input_dims)?;
    // implicitly check if the prover provided output is within range
    let claimed_output = identity(config, region, &[claimed_output], true)?;

    let product = pairwise(
        config,
        region,
        &[claimed_output.clone(), divisor.clone()],
        BaseOp::Mult,
    )?;

    diff_less_than(config, region, &[input.clone(), product.clone()], div)?;

    Ok(claimed_output)
}

/// recip accumulated layout
pub(crate) fn recip<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    input_scale: F,
    output_scale: F,
) -> Result<ValTensor<F>, CircuitError> {
    let input = value[0].clone();
    let input_dims = input.dims();

    let unit_scale = create_constant_tensor(output_scale * input_scale, 1);

    let is_assigned = !input.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.int_evals()?;
        tensor::ops::nonlinearities::recip(
            &input_evals,
            felt_to_integer_rep(input_scale) as f64,
            felt_to_integer_rep(output_scale) as f64,
        )
        .par_iter()
        .map(|x| Value::known(integer_rep_to_felt(*x)))
        .collect::<Tensor<Value<F>>>()
        .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };
    claimed_output.reshape(input_dims)?;

    // implicitly check if the prover provided output is within range
    let claimed_output = identity(config, region, &[claimed_output], true)?;
    // divide by input_scale
    let zero_inverse_val =
        tensor::ops::nonlinearities::zero_recip(felt_to_integer_rep(output_scale) as f64)[0];
    let zero_inverse = create_constant_tensor(integer_rep_to_felt(zero_inverse_val), 1);

    let equal_zero_mask = equals_zero(config, region, &[input.clone()])?;
    let not_equal_zero_mask = not(config, region, &[equal_zero_mask.clone()])?;
    let equal_inverse_mask = equals(config, region, &[claimed_output.clone(), zero_inverse])?;

    let masked_unit_scale = pairwise(
        config,
        region,
        &[unit_scale.clone(), not_equal_zero_mask.clone()],
        BaseOp::Mult,
    )?;

    // assert the two masks are equal
    enforce_equality(
        config,
        region,
        &[equal_zero_mask.clone(), equal_inverse_mask],
    )?;

    let err_func = |config: &BaseConfig<F>,
                    region: &mut RegionCtx<F>,
                    x: &ValTensor<F>|
     -> Result<ValTensor<F>, CircuitError> {
        let product = pairwise(config, region, &[x.clone(), input.clone()], BaseOp::Mult)?;

        let distance = l1_distance(
            config,
            region,
            &[product.clone(), masked_unit_scale.clone()],
        )?;
        Ok(distance)
    };

    // we need to add 1 to the points where it is zero to ignore the cvx opt conditions at those points
    let mut is_opt = optimum_convex_function(config, region, &claimed_output, err_func)?;
    is_opt = pairwise(config, region, &[is_opt, equal_zero_mask], BaseOp::Add)?;

    let mut comparison_unit = create_constant_tensor(F::ONE, is_opt.len());
    comparison_unit.reshape(is_opt.dims())?;
    // assert that the result is 1
    enforce_equality(config, region, &[is_opt, comparison_unit])?;

    Ok(claimed_output)
}

/// Square root accumulated layout
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::sqrt;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3, 2, 3, 4, 3, 4, 9]),
///    &[3, 3],
/// ).unwrap());
/// let result = sqrt::<Fp>(&dummy_config, &mut dummy_region, &[x], 1.0.into()).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 2, 1, 2, 2, 2, 2, 3]), &[3, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn sqrt<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    input_scale: utils::F32,
) -> Result<ValTensor<F>, CircuitError> {
    let input = value[0].clone();
    let input_dims = input.dims();

    let unit_scale = create_constant_tensor(integer_rep_to_felt(input_scale.0 as IntegerRep), 1);

    let is_assigned = !input.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.int_evals()?;
        tensor::ops::nonlinearities::sqrt(&input_evals, input_scale.0 as f64)
            .par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
            .collect::<Tensor<Value<F>>>()
            .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };
    claimed_output.reshape(input_dims)?;
    // force the output to be positive or zero, also implicitly checks that the output is in range
    let claimed_output = abs(config, region, &[claimed_output.clone()])?;
    // rescaled input
    let rescaled_input = pairwise(config, region, &[input.clone(), unit_scale], BaseOp::Mult)?;

    let err_func = |config: &BaseConfig<F>,
                    region: &mut RegionCtx<F>,
                    x: &ValTensor<F>|
     -> Result<ValTensor<F>, CircuitError> {
        let product = pairwise(config, region, &[x.clone(), x.clone()], BaseOp::Mult)?;
        let distance = l1_distance(config, region, &[product.clone(), rescaled_input.clone()])?;
        Ok(distance)
    };

    let is_opt = optimum_convex_function(config, region, &claimed_output, err_func)?;

    let mut comparison_unit = create_constant_tensor(F::ONE, is_opt.len());
    comparison_unit.reshape(is_opt.dims())?;

    // assert that the result is 1
    enforce_equality(config, region, &[is_opt, comparison_unit])?;

    Ok(claimed_output)
}

/// Reciprocal square root accumulated layout
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::rsqrt;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
/// &[3, 3],
/// ).unwrap());
/// let result = rsqrt::<Fp>(&dummy_config, &mut dummy_region, &[x], 1.0.into(), 1.0.into()).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 1, 1, 1, 1, 1, 1]), &[3, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn rsqrt<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    value: &[ValTensor<F>; 1],
    input_scale: utils::F32,
    output_scale: utils::F32,
) -> Result<ValTensor<F>, CircuitError> {
    let sqrt = sqrt(config, region, value, input_scale)?;

    let felt_output_scale = integer_rep_to_felt(output_scale.0 as IntegerRep);
    let felt_input_scale = integer_rep_to_felt(input_scale.0 as IntegerRep);

    let recip = recip(config, region, &[sqrt], felt_input_scale, felt_output_scale)?;

    Ok(recip)
}

/// Dot product of two tensors.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::einsum;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
/// use ezkl::circuit::layouts::dot;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3],
/// ).unwrap());
/// let y = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 5, 10, -4, 2, -1, 2, 0, 1]),
///     &[1, 3, 3],
/// ).unwrap());
/// assert_eq!(dot::<Fp>(&dummy_config, &mut dummy_region, &[x, y]).unwrap().int_evals().unwrap()[0], 86);
/// ```
pub fn dot<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    if values[0].len() != values[1].len() {
        return Err(TensorError::DimMismatch("dot".to_string()).into());
    }

    region.flush()?;
    // time this entire function run
    let global_start = instant::Instant::now();

    let mut values = values.clone();

    let mut inputs = vec![];
    let block_width = config.custom_gates.output.num_inner_cols();

    let mut assigned_len = 0;
    for (i, input) in values.iter_mut().enumerate() {
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
        let inp = {
            let (res, len) = region
                .assign_with_duplication_unconstrained(&config.custom_gates.inputs[i], input)?;
            assigned_len = len;
            res.get_inner()?
        };
        inputs.push(inp);
    }

    // Now we can assign the dot product
    // time this step
    let accumulated_dot = accumulated::dot(&[inputs[0].clone(), inputs[1].clone()], block_width)?;
    let (output, output_assigned_len) = region.assign_with_duplication_constrained(
        &config.custom_gates.output,
        &accumulated_dot.into(),
        &config.check_mode,
    )?;

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
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    let last_elem = output.last()?;

    region.increment(assigned_len);

    // last element is the result

    let elapsed = global_start.elapsed();
    trace!("dot layout took: {:?}, row {}", elapsed, region.row());
    trace!("----------------------------");
    Ok(last_elem)
}

/// Computes the einstein sum of a set of tensors.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::einsum;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// // matmul case
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[2, 1, 2, 1, 1, 1]),
///  &[2, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[2, 3, 2, 1, 1, 1]),
/// &[3, 2],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "ij,jk->ik").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[8, 9, 5, 5]), &[2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // element wise multiplication
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
///  &[3, 3],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "ij,ij->ij").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 4, 9, 2, 6, 12, 3, 8, 15]), &[3, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
///
/// // dot product of A with the transpose of B.
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
///  &[3, 3],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "ik,jk->ij").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[14, 14, 14, 20, 20, 20, 26, 26, 26]), &[3, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // dot product
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 1, 2, 3, 1, 2, 3]),
///  &[3, 3],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "ik,ik->i").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[14, 20, 26]), &[3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
///
/// // dot product
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3]),
///  &[3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3]),
///  &[3],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "i,i->").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[14]), &[1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
///
/// // wut ?
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "anm,bm->ba").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[68, 80, 95, 113, 134, 158]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // wutttttt ?
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap());
/// let z =  ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5, 7, 8, 9, 9]),
///  &[2, 3],
/// ).unwrap());
///
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[z, x, k], "bn,anm,bm->ba").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[390, 414, 534, 994, 1153, 1384]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
///
/// // contraction with a single common axis
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "abc,cd->").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[648]), &[1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // contraction with no common axes (outer product)
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 2, 3, 2, 3, 4, 3, 4, 5, 1, 2, 3, 2, 3, 4, 3, 4, 5]),
///  &[3, 3, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap());
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "abc,ed->").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1296]), &[1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // trivial axes mapping
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5, 7, 8]),
///  &[2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[4, 5]),
///  &[2],
/// ).unwrap());
///
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x.clone(), k.clone()], "mk,k->m").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[41, 68]), &[2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x, k], "mk,k->mn").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[41, 68]), &[2, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[0, 0, 0, 3]),
///  &[1, 4],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[213, 227, 74, 77]),
///  &[4],
/// ).unwrap());
///
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x.clone(), k.clone()], "mk,k->ma").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[231]), &[1, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// // subtle difference
/// let result = einsum::<Fp>(&dummy_config, &mut dummy_region, &[x.clone(), k.clone()], "mk,n->ma").unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1773]), &[1, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// ```
///
pub fn einsum<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    inputs: &[ValTensor<F>],
    equation: &str,
) -> Result<ValTensor<F>, CircuitError> {
    let mut equation = equation.split("->");
    let inputs_eq = equation.next().ok_or(CircuitError::InvalidEinsum)?;
    let output_eq = equation.next().ok_or(CircuitError::InvalidEinsum)?;
    let inputs_eq = inputs_eq.split(',').collect::<Vec<_>>();

    // Check that the number of inputs matches the number of inputs in the equation
    if inputs.len() != inputs_eq.len() {
        return Err(TensorError::DimMismatch("einsum".to_string()).into());
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
                return Err(TensorError::DimMismatch("einsum".to_string()).into());
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
        .collect::<Result<Vec<Range<_>>, CircuitError>>()?
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
                    let product_across_pair = prod(config, region, &[pair.into()])?;

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
            Ok(prod_res
                .ok_or(CircuitError::MissingEinsumProduct)?
                .get_inner_tensor()?[0]
                .clone())
        }
    };

    region.flush()?;
    region.apply_in_loop(&mut output, inner_loop_function)?;

    let output: ValTensor<F> = output.into();

    Ok(output)
}

#[derive(Debug, Clone, Copy)]
/// Determines how to handle collisions in sorting.
pub enum SortCollisionMode {
    /// Do not sort (no rule)
    Unsorted,
    /// Sort by smallest index first
    SmallestIndexFirst,
    /// Sort by largest index first on collision
    LargestIndexFirst,
}

fn _sort_ascending<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    collision_handling: SortCollisionMode,
) -> Result<(ValTensor<F>, ValTensor<F>), CircuitError> {
    let mut input = values[0].clone();
    input.flatten();

    let is_assigned = !input.any_unknowns()?;

    let sorted = if is_assigned {
        let mut int_evals = input.int_evals()?;
        int_evals.sort_unstable();
        int_evals
            .par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
            .collect::<Tensor<Value<F>>>()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
    };

    let assigned_sort = region.assign(&config.custom_gates.inputs[0], &sorted.into())?;
    region.increment(assigned_sort.len());
    // assert that this is a permutation/shuffle
    let indices = shuffles(
        config,
        region,
        &[assigned_sort.clone()],
        &[input.clone()],
        collision_handling,
    )?;

    let window_a = assigned_sort.get_slice(&[0..assigned_sort.len() - 1])?;
    let window_b = assigned_sort.get_slice(&[1..assigned_sort.len()])?;

    let indices_a = indices.get_slice(&[0..indices.len() - 1])?;
    let indices_b = indices.get_slice(&[1..indices.len()])?;

    let unit = create_unit_tensor(window_a.len());

    match collision_handling {
        SortCollisionMode::Unsorted => {
            let is_greater = greater_equal(config, region, &[window_b.clone(), window_a.clone()])?;
            enforce_equality(config, region, &[unit, is_greater])?;
        }
        SortCollisionMode::SmallestIndexFirst => {
            let is_greater = greater(config, region, &[window_b.clone(), window_a.clone()])?;
            let is_equal = equals(config, region, &[window_b.clone(), window_a.clone()])?;
            let is_greater_indices =
                greater(config, region, &[indices_b.clone(), indices_a.clone()])?;

            let is_equal_and_is_greater_indices =
                and(config, region, &[is_equal, is_greater_indices])?;

            let is_greater_or_is_equal_and_is_greater_indices = or(
                config,
                region,
                &[is_greater, is_equal_and_is_greater_indices],
            )?;

            enforce_equality(
                config,
                region,
                &[unit, is_greater_or_is_equal_and_is_greater_indices],
            )?;
        }
        SortCollisionMode::LargestIndexFirst => {
            let is_greater = greater(config, region, &[window_b.clone(), window_a.clone()])?;
            let is_equal = equals(config, region, &[window_b.clone(), window_a.clone()])?;
            let is_lesser_indices = less(config, region, &[indices_b.clone(), indices_a.clone()])?;

            let is_equal_and_is_lesser_indices =
                and(config, region, &[is_equal, is_lesser_indices])?;

            let is_greater_or_is_equal_and_is_greater_indices = or(
                config,
                region,
                &[is_greater, is_equal_and_is_lesser_indices],
            )?;

            enforce_equality(
                config,
                region,
                &[unit, is_greater_or_is_equal_and_is_greater_indices],
            )?;
        }
    }

    Ok((assigned_sort, indices))
}

/// Returns top K values.
fn _select_topk<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    k: usize,
    largest: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let mut sorted = _sort_ascending(config, region, values, SortCollisionMode::Unsorted)?.0;
    if largest {
        sorted.reverse()?;
    }
    Ok(sorted.get_slice(&[0..k])?)
}

/// Returns top K values.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::topk_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2,3],
/// ).unwrap());
/// let result = topk_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], 2, 1, true).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[15, 2, 1, 1]),
///     &[2,2],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn topk_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    k: usize,
    dim: usize,
    largest: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let topk_at_k = move |config: &BaseConfig<F>,
                          region: &mut RegionCtx<F>,
                          values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, CircuitError> {
        _select_topk(config, region, values, k, largest)
    };

    let output: ValTensor<F> = multi_dim_axes_op(config, region, values, &[dim], topk_at_k)?;

    Ok(output)
}

fn select<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let (mut input, index) = (values[0].clone(), values[1].clone());
    input.flatten();

    // these will be assigned as constants
    let dim_indices: ValTensor<F> =
        Tensor::from((0..input.len() as u64).map(|x| ValType::Constant(F::from(x)))).into();

    let is_assigned = !input.any_unknowns()? && !index.any_unknowns()?;

    let output: ValTensor<F> = if is_assigned && region.witness_gen() {
        let felt_evals = input.get_felt_evals()?;
        index
            .int_evals()?
            .par_iter()
            .map(|x| Value::known(felt_evals.get(&[*x as usize])))
            .collect::<Tensor<Value<F>>>()
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

fn one_hot<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    num_classes: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // assert values is flat
    assert_eq!(values[0].dims().len(), 1);
    // assert its a single elelemnt
    assert_eq!(values[0].len(), 1);
    let input = values[0].clone();
    let is_assigned = !input.any_unknowns()?;

    let output: ValTensor<F> = if is_assigned {
        let int_evals = input.int_evals()?;
        let res = tensor::ops::one_hot(&int_evals, num_classes, 1)?;
        res.par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
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
    let unit = create_unit_tensor(1);

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
pub(crate) fn dynamic_lookup<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    lookups: &[ValTensor<F>; 2],
    tables: &[ValTensor<F>; 2],
) -> Result<(ValTensor<F>, ValTensor<F>), CircuitError> {
    // if not all lookups same length err
    if lookups[0].len() != lookups[1].len() {
        return Err(CircuitError::MismatchedLookupLength(
            lookups[0].len(),
            lookups[1].len(),
        ));
    }

    // if not all inputs same length err
    if tables[0].len() != tables[1].len() {
        return Err(CircuitError::MismatchedLookupTableLength(
            tables[0].len(),
            tables[1].len(),
        ));
    }

    let dynamic_lookup_index = region.dynamic_lookup_index();

    let (lookup_0, lookup_1) = (lookups[0].clone(), lookups[1].clone());
    let (table_0, table_1) = (tables[0].clone(), tables[1].clone());

    let (table_0, flush_len_0) =
        region.assign_dynamic_lookup(&config.dynamic_lookups.tables[0], &table_0)?;
    let (_table_1, flush_len_1) =
        region.assign_dynamic_lookup(&config.dynamic_lookups.tables[1], &table_1)?;
    if flush_len_0 != flush_len_1 {
        return Err(CircuitError::MismatchedLookupTableLength(
            flush_len_0,
            flush_len_1,
        ));
    }
    let table_len = table_0.len();

    // now create a vartensor of constants for the dynamic lookup index
    let table_index = create_constant_tensor(F::from(dynamic_lookup_index as u64), table_len);
    let _table_index =
        region.assign_dynamic_lookup(&config.dynamic_lookups.tables[2], &table_index)?;

    let lookup_0 = region.assign(&config.dynamic_lookups.inputs[0], &lookup_0)?;
    let lookup_1 = region.assign(&config.dynamic_lookups.inputs[1], &lookup_1)?;
    let lookup_len = lookup_0.len();

    // now set the lookup index
    let lookup_index = create_constant_tensor(F::from(dynamic_lookup_index as u64), lookup_len);

    let _lookup_index = region.assign(&config.dynamic_lookups.inputs[2], &lookup_index)?;

    let mut lookup_block = 0;

    if !region.is_dummy() {
        (0..table_len)
            .map(|i| {
                let (x, _, z) = config.dynamic_lookups.tables[0]
                    .cartesian_coord(region.combined_dynamic_shuffle_coord() + i + flush_len_0);

                if lookup_block != x {
                    lookup_block = x;
                }

                let table_selector = config.dynamic_lookups.table_selectors[lookup_block];
                region.enable(Some(&table_selector), z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    if !region.is_dummy() {
        // Enable the selectors
        (0..lookup_len)
            .map(|i| {
                let (x, y, z) =
                    config.dynamic_lookups.inputs[0].cartesian_coord(region.linear_coord() + i);

                let lookup_selector = config
                    .dynamic_lookups
                    .lookup_selectors
                    .get(&(lookup_block, (x, y)))
                    .ok_or(CircuitError::MissingSelectors(format!("{:?}", (x, y))))?;

                region.enable(Some(lookup_selector), z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    region.increment_dynamic_lookup_col_coord(table_len + flush_len_0);
    region.increment_dynamic_lookup_index(1);
    region.increment(lookup_len);

    Ok((lookup_0, lookup_1))
}

/// Shuffle arg
/// 1. the input is a set of pairs (index_input, value_input) -- looked up against a (dynamic) set of values (index_output, value_output).
/// 2. index_input is copy constrained to values in a fixed column and is thus a fixed set of incrementing values over the input
/// 3. value_input is just the input that we are ascertaining is shuffled
/// 4. index_output is (typically) a prover generated witness committed to in an advice column
/// 5. value_output is (typically) a prover generated witness committed to in an advice column
/// 6. Given the above, and given the fixed index_input , we go through every (index_input, value_input) pair and ascertain that it is contained in the input.
/// 7. Given the fixed incrementing index index_input, we avoid multiplicity in the output by leveraging this surrogate index: if index_output isn't matched to the exact value where for `index_input=index_output` -> `value_input=value_output`, then the lookup fails
pub(crate) fn shuffles<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    output: &[ValTensor<F>; 1],
    input: &[ValTensor<F>; 1],
    collision_handling: SortCollisionMode,
) -> Result<ValTensor<F>, CircuitError> {
    let shuffle_index = region.shuffle_index();
    let (output, input) = (output[0].clone(), input[0].clone());

    // assert input and reference are same length
    if output.len() != input.len() {
        return Err(CircuitError::MismatchedShuffleLength(
            output.len(),
            input.len(),
        ));
    }

    let (output, flush_len_ref) = region.assign_shuffle(&config.shuffles.outputs[0], &output)?;
    let output_len = output.len();
    let input = region.assign(&config.shuffles.inputs[0], &input)?;

    // now create a vartensor of constants for the shuffle index
    let index = create_constant_tensor(F::from(shuffle_index as u64), output_len);
    let (index, flush_len_index) = region.assign_shuffle(&config.shuffles.outputs[1], &index)?;
    region.assign(&config.shuffles.inputs[1], &index)?;

    if flush_len_index != flush_len_ref {
        return Err(CircuitError::MismatchedShuffleLength(
            flush_len_index,
            flush_len_ref,
        ));
    }

    // now found the position of each element of the reference to the input

    let is_known = !output.any_unknowns()? && !input.any_unknowns()?;

    let claimed_index_output = if is_known {
        let input = input.int_evals()?;
        let output = output.int_evals()?;

        // Keep track of which positions we've used for each value
        let mut used_positions: HashMap<usize, bool> = HashMap::new();

        let index_output = output
            .iter()
            .map(|x| {
                // Find all positions of the current element
                let mut positions: Vec<usize> = input
                    .iter()
                    .enumerate()
                    .filter(|(_, y)| *y == x)
                    .map(|(i, _)| i)
                    .collect();

                match collision_handling {
                    SortCollisionMode::Unsorted => {}
                    SortCollisionMode::SmallestIndexFirst => {
                        // Sort the positions by the index of the input element
                        positions.sort_unstable_by(|a, b| input[*a].cmp(&input[*b]));
                    }

                    SortCollisionMode::LargestIndexFirst => {
                        // Sort the positions by the index of the input element
                        positions.reverse();
                    }
                }

                // Find the first unused position for this element
                let pos = positions
                    .iter()
                    .find(|&&i| !used_positions.get(&i).unwrap_or(&false))
                    .unwrap_or(&0);

                // Mark this position as used
                used_positions.insert(*pos, true);

                Ok::<_, CircuitError>(Value::known(F::from(*pos as u64)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Verify all indices were used exactly once
        if !used_positions.values().all(|&x| x) {
            return Err(CircuitError::MissingShuffleElement);
        }

        Tensor::from(index_output.into_iter()).into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); output_len]),
            &[output_len],
        )?
        .into()
    };

    region.assign_shuffle(&config.shuffles.outputs[2], &claimed_index_output)?;

    // the incrementing index is the set of numbered values for the input tensor 0...n, and is FIXED
    let incrementing_index: ValTensor<F> =
        Tensor::from((0..output.len() as u64).map(|x| ValType::Constant(F::from(x)))).into();
    region.assign(&config.shuffles.inputs[2], &incrementing_index)?;

    let mut shuffle_block = 0;

    if !region.is_dummy() {
        (0..output_len)
            .map(|i| {
                let (x, _, z) = config.shuffles.outputs[0]
                    .cartesian_coord(region.combined_dynamic_shuffle_coord() + i + flush_len_ref);
                shuffle_block = x;
                let ref_selector = config.shuffles.output_selectors[shuffle_block];
                region.enable(Some(&ref_selector), z)?;
                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    if !region.is_dummy() {
        // Enable the selectors
        (0..output_len)
            .map(|i| {
                let (x, y, z) =
                    config.custom_gates.inputs[0].cartesian_coord(region.linear_coord() + i);
                let input_selector = config
                    .shuffles
                    .input_selectors
                    .get(&(shuffle_block, (x, y)))
                    .ok_or(CircuitError::MissingSelectors(format!("{:?}", (x, y))))?;

                region.enable(Some(input_selector), z)?;

                Ok(())
            })
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    region.increment_shuffle_col_coord(output_len + flush_len_ref);
    region.increment_shuffle_index(1);
    region.increment(output_len);

    Ok(claimed_index_output)
}

/// One hot accumulated layout
pub(crate) fn one_hot_axis<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    num_classes: usize,
    dim: usize,
) -> Result<ValTensor<F>, CircuitError> {
    let input = values[0].clone();
    let input_inner = input.get_inner_tensor()?;

    let mut output_dims = values[0].dims().to_vec();
    output_dims.insert(dim, num_classes);

    let mut op_tensors: Tensor<ValTensor<F>> = Tensor::new(None, input_inner.dims())?;

    let inner_loop_function =
        |i: usize, region: &mut RegionCtx<'_, F>| -> Result<ValTensor<F>, _> {
            let inp = input_inner[i].clone();
            let tensor = Tensor::new(Some(&[inp.clone()]), &[1])?;

            one_hot(config, region, &[tensor.into()], num_classes)
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

        Ok::<_, CircuitError>(one_hot_val)
    })?;

    Ok(output.into())
}

/// Gather accumulated layout
pub(crate) fn gather<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    dim: usize,
) -> Result<ValTensor<F>, CircuitError> {
    let (input, mut index_clone) = (values[0].clone(), values[1].clone());
    index_clone.flatten();
    if index_clone.is_singleton() {
        index_clone.reshape(&[1])?;
    }

    // Calculate the output tensor size
    let input_dims = input.dims();
    let mut output_size = input_dims.to_vec();
    output_size[dim] = index_clone.dims()[0];

    let linear_index =
        linearize_element_index(config, region, &[index_clone], input_dims, dim, true)?;

    let mut output = select(config, region, &[input, linear_index])?;

    output.reshape(&output_size)?;

    Ok(output)
}

/// Gather accumulated layout
pub(crate) fn gather_elements<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    dim: usize,
) -> Result<(ValTensor<F>, ValTensor<F>), CircuitError> {
    let (input, index) = (values[0].clone(), values[1].clone());

    assert_eq!(input.dims().len(), index.dims().len());

    // Calculate the output tensor size
    let output_size = index.dims().to_vec();

    let linear_index = linearize_element_index(config, region, &[index], input.dims(), dim, false)?;

    let mut output = select(config, region, &[input, linear_index.clone()])?;

    output.reshape(&output_size)?;

    Ok((output, linear_index))
}

/// Gather accumulated layout
pub(crate) fn gather_nd<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    batch_dims: usize,
) -> Result<(ValTensor<F>, ValTensor<F>), CircuitError> {
    let (input, index) = (values[0].clone(), values[1].clone());

    let index_dims = index.dims().to_vec();
    let input_dims = input.dims().to_vec();
    let last_value = index_dims
        .last()
        .ok_or(TensorError::DimMismatch("gather_nd".to_string()))?;
    if index_dims.last() > Some(&(input_dims.len() - batch_dims)) {
        return Err(TensorError::DimMismatch("gather_nd".to_string()).into());
    }

    let output_size =
    // If indices_shape[-1] == r-b, since the rank of indices is q,
    // indices can be thought of as N (q-b-1)-dimensional tensors containing 1-D tensors of dimension r-b,
    // where N is an integer equals to the product of 1 and all the elements in the batch dimensions of the indices_shape.
    // Let us think of each such r-b ranked tensor as indices_slice.
    // Each scalar value corresponding to data[0:b-1,indices_slice] is filled into
    // the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
     // if indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor containing 1-D tensors of dimension < r-b.
    // Let us think of each such tensors as indices_slice.
    // Each tensor slice corresponding to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor
    {
        let output_rank = input_dims.len() + index_dims.len() - 1 - batch_dims - last_value;

        let mut dims = index_dims[..index_dims.len() - 1].to_vec();
        let input_offset = batch_dims + last_value;
        dims.extend(input_dims[input_offset..input_dims.len()].to_vec());

        assert_eq!(output_rank, dims.len());
        dims

    };

    let linear_index = linearize_nd_index(config, region, &[index], input.dims(), batch_dims)?;

    let mut output = select(config, region, &[input, linear_index.clone()])?;

    output.reshape(&output_size)?;

    Ok((output, linear_index))
}

/// Takes a tensor representing a multi-dimensional index and returns a tensor representing the linearized index.
/// The linearized index is the index of the element in the flattened tensor.
/// FOr instance if the dims is [3,5,2], the linearized index of [2] at dim 1 is 2*5 + 3 = 13
pub(crate) fn linearize_element_index<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    dims: &[usize],
    dim: usize,
    is_flat_index: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let index = values[0].clone();
    if !is_flat_index {
        assert_eq!(index.dims().len(), dims.len());
        // if the index is already flat, return it
        if index.dims().len() == 1 {
            return Ok(index);
        }
    }

    let dim_multiplier: Tensor<usize> = Tensor::new(None, &[dims.len()])?;

    let dim_multiplier: Tensor<F> = dim_multiplier.par_enum_map(|i, _| {
        let mut res = 1;
        for dim in dims.iter().skip(i + 1) {
            res *= dim;
        }

        Ok::<_, CircuitError>(F::from(res as u64))
    })?;

    let iteration_dims = if is_flat_index {
        let mut dims = dims.to_vec();
        dims[dim] = index.len();
        dims
    } else {
        index.dims().to_vec()
    };

    let cartesian_coord = iteration_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let val_dim_multiplier: ValTensor<F> = dim_multiplier
        .get_slice(&[dim..dim + 1])?
        .map(|x| ValType::Constant(x))
        .into();

    let mut output = Tensor::new(None, &[cartesian_coord.len()])?;

    let inner_loop_function = |i: usize, region: &mut RegionCtx<'_, F>| {
        let coord = cartesian_coord[i].clone();
        let slice: Vec<Range<usize>> = if is_flat_index {
            coord[dim..dim + 1].iter().map(|x| *x..*x + 1).collect()
        } else {
            coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>()
        };

        let index_val = index.get_slice(&slice)?;

        let mut const_offset = F::ZERO;
        for i in 0..dims.len() {
            if i != dim {
                const_offset += F::from(coord[i] as u64) * dim_multiplier[i];
            }
        }
        let const_offset = create_constant_tensor(const_offset, 1);

        let res = pairwise(
            config,
            region,
            &[index_val, val_dim_multiplier.clone()],
            BaseOp::Mult,
        )?;

        let res = pairwise(config, region, &[res, const_offset], BaseOp::Add)?;

        Ok(res.get_inner_tensor()?[0].clone())
    };

    region.apply_in_loop(&mut output, inner_loop_function)?;

    Ok(output.into())
}

/// Takes a tensor representing a nd index and returns a tensor representing the linearized index.
/// The linearized index is the index of the element in the flattened tensor.
/// Given data tensor of rank r >= 1, indices tensor of rank q >= 1, and batch_dims integer b, this operator gathers slices of data into an output tensor of rank q + r - indices_shape[-1] - 1 - b.
/// indices is an q-dimensional integer tensor, best thought of as a (q-1)-dimensional tensor of index-tuples into data, where each element defines a slice of data
/// batch_dims (denoted as b) is an integer indicating the number of batch dimensions, i.e the leading b number of dimensions of data tensor and indices are representing the batches, and the gather starts from the b+1 dimension.
/// Some salient points about the inputs’ rank and shape:
///     r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks r and q
///     The first b dimensions of the shape of indices tensor and data tensor must be equal.
///     b < min(q, r) is to be honored.
///     The indices_shape[-1] should have a value between 1 (inclusive) and rank r-b (inclusive)
///     All values in indices are expected to be within bounds [-s, s-1] along axis of size s (i.e.) -data_shape[i] <= indices[...,i] <= data_shape[i] - 1. It is an error if any of the index values are out of bounds.
// The output is computed as follows:
/// The output tensor is obtained by mapping each index-tuple in the indices tensor to the corresponding slice of the input data.
///     If indices_shape[-1] > r-b => error condition
///     If indices_shape[-1] == r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensors containing 1-D tensors of dimension r-b, where N is an integer equals to the product of 1 and all the elements in the batch dimensions of the indices_shape.
///     Let us think of each such r-b ranked tensor as indices_slice. Each scalar value corresponding to data[0:b-1,indices_slice] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor (Example 1 below)
///     If indices_shape[-1] < r-b, since the rank of indices is q, indices can be thought of as N (q-b-1)-dimensional tensor containing 1-D tensors of dimension < r-b. Let us think of each such tensors as indices_slice. Each tensor slice corresponding to data[0:b-1, indices_slice , :] is filled into the corresponding location of the (q-b-1)-dimensional tensor to form the output tensor (Examples 2, 3, 4 and 5 below)
pub(crate) fn linearize_nd_index<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    dims: &[usize],
    batch_dims: usize,
) -> Result<ValTensor<F>, CircuitError> {
    let index = values[0].clone();
    let index_dims = index.dims().to_vec();

    let last_dim = index.dims().last().unwrap();
    let input_rank = dims[batch_dims..].len();

    let dim_multiplier: Tensor<usize> = Tensor::new(None, &[dims.len()])?;
    let dim_multiplier: Tensor<F> = dim_multiplier.par_enum_map(|i, _| {
        let mut res = 1;
        for dim in dims.iter().skip(i + 1) {
            res *= dim;
        }
        Ok::<_, CircuitError>(F::from(res as u64))
    })?;

    let iteration_dims = index.dims()[0..batch_dims].to_vec();

    let mut batch_cartesian_coord = iteration_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    if batch_cartesian_coord.is_empty() {
        batch_cartesian_coord.push(vec![]);
    }

    let index_dim_multiplier: ValTensor<F> = dim_multiplier
        .get_slice(&[batch_dims..dims.len()])?
        .map(|x| ValType::Constant(x))
        .into();

    let mut outer_results = vec![];

    for coord in batch_cartesian_coord {
        let slice: Vec<Range<usize>> = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();

        let mut index_slice = index.get_slice(&slice)?;
        index_slice.reshape(&index_dims[batch_dims..])?;

        // expand the index to the full dims by iterating over the rest of the dims and inserting constants
        // eg in the case
        // batch_dims = 0
        // data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
        // indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
        // output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
        // the index should be expanded to the shape [2,2,3]: [[0,1,0],[0,1,1],[1,0,0],[1,0,1]]

        let mut inner_cartesian_coord = index_slice.dims()[0..index_slice.dims().len() - 1]
            .iter()
            .map(|x| 0..*x)
            .multi_cartesian_product()
            .collect::<Vec<_>>();

        if inner_cartesian_coord.is_empty() {
            inner_cartesian_coord.push(vec![]);
        }

        let indices = if last_dim < &input_rank {
            inner_cartesian_coord
                .iter()
                .map(|x| {
                    let slice = x.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
                    let index = index_slice.get_slice(&slice)?;

                    // map over cartesian coord of rest of dims and insert constants
                    let grid = (*last_dim..input_rank)
                        .map(|x| 0..dims[x])
                        .multi_cartesian_product();

                    Ok(grid
                        .map(|x| {
                            let index = index.clone();
                            let constant_valtensor: ValTensor<F> = Tensor::from(
                                x.into_iter().map(|x| ValType::Constant(F::from(x as u64))),
                            )
                            .into();
                            index.concat(constant_valtensor)
                        })
                        .collect::<Result<Vec<_>, TensorError>>()?)
                })
                .collect::<Result<Vec<_>, CircuitError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        } else {
            inner_cartesian_coord
                .iter()
                .map(|x| {
                    let slice = x.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
                    index_slice.get_slice(&slice)
                })
                .collect::<Result<Vec<_>, TensorError>>()?
        };

        let mut const_offset = F::ZERO;
        for i in 0..batch_dims {
            const_offset += F::from(coord[i] as u64) * dim_multiplier[i];
        }

        let const_offset = create_constant_tensor(const_offset, 1);

        let mut results = vec![];

        for index_val in indices {
            let mut index_val = index_val.clone();
            index_val.flatten();
            let res = pairwise(
                config,
                region,
                &[index_val.clone(), index_dim_multiplier.clone()],
                BaseOp::Mult,
            )?;
            let res = res.concat(const_offset.clone())?;
            let res = sum(config, region, &[res])?;
            results.push(res.get_inner_tensor()?.clone());
            // assert than res is less than the product of the dims
            if region.witness_gen() {
                assert!(
                res.int_evals()?
                    .iter()
                    .all(|x| *x < dims.iter().product::<usize>() as IntegerRep),
                "res is greater than the product of the dims {} (coord={}, index_dim_multiplier={}, res={})",
                dims.iter().product::<usize>(),
                index_val.show(),
                index_dim_multiplier.show(),
                res.show()
            );
            }
        }

        let result_tensor = Tensor::from(results.into_iter());

        outer_results.push(result_tensor.combine()?);
    }

    let output = Tensor::from(outer_results.into_iter());
    let output = output.combine()?;

    Ok(output.into())
}

// assumes unique values in fullset
pub(crate) fn get_missing_set_elements<
    F: PrimeField + TensorType + PartialOrd + std::hash::Hash,
>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    ordered: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let (mut input, fullset) = (values[0].clone(), values[1].clone());
    let set_len = fullset.len();
    input.flatten();

    // while fullset is less than len of input concat

    let is_assigned = !input.any_unknowns()? && !fullset.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.int_evals()?;
        let mut fullset_evals = fullset.int_evals()?.into_iter().collect::<Vec<_>>();

        // get the difference between the two vectors
        for eval in input_evals.iter() {
            // delete first occurrence of that value
            if let Some(pos) = fullset_evals.iter().position(|x| x == eval) {
                fullset_evals.remove(pos);
            }
        }

        // if fullset + input is the same length, then input is a subset of fullset, else randomly delete elements, this is a patch for
        // the fact that we can't have a tensor of unknowns when using constant during gen-settings
        if fullset_evals.len() != set_len - input.len() {
            fullset_evals.truncate(set_len - input.len());
        }

        fullset_evals
            .par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
            .collect::<Tensor<Value<F>>>()
            .into()
    } else {
        let dim = fullset.len() - input.len();
        Tensor::new(Some(&vec![Value::<F>::unknown(); dim]), &[dim])?.into()
    };

    // assign the claimed output
    claimed_output = region.assign(&config.custom_gates.output, &claimed_output)?;

    region.increment(claimed_output.len());

    // input and claimed output should be the shuffles of fullset
    // concatenate input and claimed output
    let input_and_claimed_output = input.concat(claimed_output.clone())?;

    // assert that this is a permutation/shuffle
    shuffles(
        config,
        region,
        &[input_and_claimed_output.clone()],
        &[fullset.clone()],
        SortCollisionMode::Unsorted,
    )?;

    if ordered {
        // assert that the claimed output is sorted
        claimed_output = _sort_ascending(
            config,
            region,
            &[claimed_output],
            SortCollisionMode::Unsorted,
        )?
        .0;
    }

    Ok(claimed_output)
}

/// Gather accumulated layout
pub(crate) fn scatter_elements<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 3],
    dim: usize,
) -> Result<ValTensor<F>, CircuitError> {
    let (input, mut index, src) = (values[0].clone(), values[1].clone(), values[2].clone());

    assert_eq!(input.dims().len(), index.dims().len());

    if !index.all_prev_assigned() {
        index = region.assign(&config.custom_gates.inputs[1], &index)?;
        region.increment(index.len());
    }

    let is_assigned = !input.any_unknowns()? && !index.any_unknowns()? && !src.any_unknowns()?;

    let claimed_output: ValTensor<F> = if is_assigned && region.witness_gen() {
        let input_inner = input.int_evals()?;
        let index_inner = index.int_evals()?.map(|x| x as usize);
        let src_inner = src.int_evals()?;

        let res = tensor::ops::scatter(&input_inner, &index_inner, &src_inner, dim)?;

        res.par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
            .collect::<Tensor<Value<F>>>()
            .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };

    // assign the claimed output
    let mut claimed_output = region.assign(&config.custom_gates.output, &claimed_output)?;
    region.increment(claimed_output.len());
    claimed_output.reshape(input.dims())?;

    // scatter elements is the inverse of gather elements
    let (gather_src, linear_index) = gather_elements(
        config,
        region,
        &[claimed_output.clone(), index.clone()],
        dim,
    )?;

    // assert this is equal to the src
    enforce_equality(config, region, &[gather_src, src])?;

    let full_index_set: ValTensor<F> =
        Tensor::from((0..input.len() as u64).map(|x| ValType::Constant(F::from(x)))).into();
    let input_indices = get_missing_set_elements(
        config,
        region,
        &[linear_index, full_index_set.clone()],
        true,
    )?;

    claimed_output.flatten();
    let (gather_input, _) = gather_elements(
        config,
        region,
        &[claimed_output.clone(), input_indices.clone()],
        0,
    )?;
    // assert this is a subset of the input
    dynamic_lookup(
        config,
        region,
        &[input_indices, gather_input],
        &[full_index_set, input.clone()],
    )?;

    claimed_output.reshape(input.dims())?;

    Ok(claimed_output)
}

/// Scatter Nd
pub(crate) fn scatter_nd<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 3],
) -> Result<ValTensor<F>, CircuitError> {
    let (input, mut index, src) = (values[0].clone(), values[1].clone(), values[2].clone());

    if !index.all_prev_assigned() {
        index = region.assign(&config.custom_gates.inputs[1], &index)?;
        region.increment(index.len());
    }

    let is_assigned = !input.any_unknowns()? && !index.any_unknowns()? && !src.any_unknowns()?;

    let claimed_output: ValTensor<F> = if is_assigned && region.witness_gen() {
        let input_inner = input.int_evals()?;
        let index_inner = index.int_evals()?.map(|x| x as usize);
        let src_inner = src.int_evals()?;

        let res = tensor::ops::scatter_nd(&input_inner, &index_inner, &src_inner)?;

        res.par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
            .collect::<Tensor<Value<F>>>()
            .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };

    // assign the claimed output
    let mut claimed_output = region.assign(&config.custom_gates.output, &claimed_output)?;
    region.increment(claimed_output.len());
    claimed_output.reshape(input.dims())?;

    // scatter elements is the inverse of gather elements
    let (gather_src, linear_index) =
        gather_nd(config, region, &[claimed_output.clone(), index.clone()], 0)?;

    // assert this is equal to the src
    enforce_equality(config, region, &[gather_src, src])?;

    let full_index_set: ValTensor<F> =
        Tensor::from((0..input.len() as u64).map(|x| ValType::Constant(F::from(x)))).into();

    let input_indices = get_missing_set_elements(
        config,
        region,
        &[linear_index, full_index_set.clone()],
        true,
    )?;

    // now that it is flattened we can gather over elements on dim 0
    claimed_output.flatten();
    let (gather_input, _) = gather_elements(
        config,
        region,
        &[claimed_output.clone(), input_indices.clone()],
        0,
    )?;

    // assert this is a subset of the input
    dynamic_lookup(
        config,
        region,
        &[input_indices, gather_input],
        &[full_index_set, input.clone()],
    )?;

    claimed_output.reshape(input.dims())?;

    Ok(claimed_output)
}

/// Sums a tensor.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::sum;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = sum::<Fp>(&dummy_config, &mut dummy_region, &[x]).unwrap();
/// let expected = 21;
/// assert_eq!(result.int_evals().unwrap()[0], expected);
/// ```
pub fn sum<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    if values[0].len() == 1 {
        return Ok(values[0].clone());
    }

    region.flush()?;
    // time this entire function run
    let mut values = values.clone();

    // this section has been optimized to death, don't mess with it
    values[0].remove_const_zero_values();

    // if empty return a const
    if values[0].is_empty() {
        return Ok(create_zero_tensor(1));
    }

    let block_width = config.custom_gates.output.num_inner_cols();

    let assigned_len: usize;
    let input = {
        let mut input = values[0].clone();
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ZERO))?;
        let (res, len) =
            region.assign_with_duplication_unconstrained(&config.custom_gates.inputs[1], &input)?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_sum = accumulated::sum(&input, block_width)?;

    let (output, output_assigned_len) = region.assign_with_duplication_constrained(
        &config.custom_gates.output,
        &accumulated_sum.into(),
        &config.check_mode,
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

    let last_elem = output.last()?;

    region.increment(assigned_len);

    // last element is the result
    Ok(last_elem)
}

/// Takes prod of tensor's elements.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::prod;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = prod::<Fp>(&dummy_config, &mut dummy_region, &[x]).unwrap();
/// let expected = 0;
/// assert_eq!(result.int_evals().unwrap()[0], expected);
/// ```
pub fn prod<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    region.flush()?;
    // time this entire function run

    // this section has been optimized to death, don't mess with it
    let removal_indices = values[0].get_const_zero_indices();

    // if empty return a const
    if !removal_indices.is_empty() {
        return Ok(create_zero_tensor(1));
    }

    let block_width = config.custom_gates.output.num_inner_cols();
    let assigned_len: usize;
    let input = {
        let mut input = values[0].clone();
        input.pad_to_zero_rem(block_width, ValType::Constant(F::ONE))?;
        let (res, len) =
            region.assign_with_duplication_unconstrained(&config.custom_gates.inputs[1], &input)?;
        assigned_len = len;
        res.get_inner()?
    };

    // Now we can assign the dot product
    let accumulated_prod = accumulated::prod(&input, block_width)?;

    let (output, output_assigned_len) = region.assign_with_duplication_constrained(
        &config.custom_gates.output,
        &accumulated_prod.into(),
        &config.check_mode,
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
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    let last_elem = output.last()?;

    region.increment(assigned_len);

    // last element is the result
    Ok(last_elem)
}

/// Axes wise op wrapper
fn axes_wise_op<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
    // generic layout op
    op: impl Fn(
            &BaseConfig<F>,
            &mut RegionCtx<F>,
            &[ValTensor<F>; 1],
        ) -> Result<ValTensor<F>, CircuitError>
        + Send
        + Sync,
) -> Result<ValTensor<F>, CircuitError> {
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

/// Takes product of a tensor along specific axes.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::prod_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = prod_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], &[1]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[60, 0]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn prod_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    // calculate value of output
    axes_wise_op(config, region, values, axes, prod)
}

/// Sums a tensor along specific axes.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::sum_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = sum_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], &[1]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[19, 2]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn sum_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    // calculate value of output
    axes_wise_op(config, region, values, axes, sum)
}

/// Argmax of a tensor along specific axes.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::argmax_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = argmax_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[1, 0]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn argmax_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    dim: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // these will be assigned as constants
    let argmax = move |config: &BaseConfig<F>,
                       region: &mut RegionCtx<F>,
                       values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, CircuitError> { argmax(config, region, values) };

    // calculate value of output
    axes_wise_op(config, region, values, &[dim], argmax)
}

/// Max of a tensor along specific axes.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::max_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = max_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], &[1]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[15, 1]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn max_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    // calculate value of output

    axes_wise_op(config, region, values, axes, max)
}

/// Argmin of a tensor along specific axes.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::argmin_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = argmin_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], 1).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[0, 2]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn argmin_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    dim: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // calculate value of output

    let argmin = move |config: &BaseConfig<F>,
                       region: &mut RegionCtx<F>,
                       values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, CircuitError> { argmin(config, region, values) };

    axes_wise_op(config, region, values, &[dim], argmin)
}

/// Mins a tensor along specific axes.
/// # Arguments
///
/// * `a` - Tensor
/// * `b` - Single value
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::min_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 15, 2, 1, 1, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = min_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], &[1]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
///     Some(&[2, 0]),
///     &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn min_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    // calculate value of output

    axes_wise_op(config, region, values, axes, min)
}

/// Pairwise (elementwise) op layout
pub(crate) fn pairwise<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    op: BaseOp,
) -> Result<ValTensor<F>, CircuitError> {
    // time to calculate the value of the output
    let global_start = instant::Instant::now();

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

    let inputs = [lhs.clone(), rhs.clone()]
        .iter()
        .enumerate()
        .map(|(i, input)| {
            let res = region.assign(&config.custom_gates.inputs[i], input)?;

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
    let mut output = region.assign(&config.custom_gates.output, &op_result.into())?;

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
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }
    region.increment(assigned_len);

    output.reshape(&broadcasted_shape)?;

    let end = global_start.elapsed();
    trace!(
        "pairwise {} layout took {:?}, row: {}",
        op.as_str(),
        end,
        region.row()
    );
    trace!("----------------------------");

    Ok(output)
}

/// Mean of squares axes
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::mean_of_squares_axes;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[2, 15, 2, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let result = mean_of_squares_axes::<Fp>(&dummy_config, &mut dummy_region, &[x], &[1]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(
/// Some(&[78, 1]),
/// &[2, 1],
/// ).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn mean_of_squares_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    let squared = pow(config, region, values, 2)?;
    let sum_squared = sum_axes(config, region, &[squared], axes)?;

    let dividend: usize = values[0].len() / sum_squared.len();

    let mean_squared = div(config, region, &[sum_squared], F::from(dividend as u64))?;
    Ok(mean_squared)
}

/// expand the tensor to the given shape
pub(crate) fn expand<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    shape: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    let mut assigned_input = region.assign(&config.custom_gates.inputs[0], &values[0])?;
    assigned_input.expand(shape)?;
    region.increment(assigned_input.len());
    Ok(assigned_input)
}

/// Greater than operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::greater;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[1, 12, 6, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap());
/// let result = greater::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 1, 1, 0, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn greater<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let (mut lhs, mut rhs) = (values[0].clone(), values[1].clone());

    let broadcasted_shape = get_broadcasted_shape(lhs.dims(), rhs.dims())?;

    lhs.expand(&broadcasted_shape)?;
    rhs.expand(&broadcasted_shape)?;

    let diff = pairwise(config, region, &[lhs, rhs], BaseOp::Sub)?;
    let sign = sign(config, region, &[diff])?;
    let eq = equals(config, region, &[sign, create_unit_tensor(1)])?;
    Ok(eq)
}

/// Greater equals than operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::greater_equal;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[1, 12, 6, 4, 3, 2]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 2, 3, 4, 5, 4]),
/// &[2, 3],
/// ).unwrap());
/// let result = greater_equal::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 1, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn greater_equal<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let (lhs, rhs) = (values[0].clone(), values[1].clone());

    // add 1 to lhs
    let lhs_plus_one = pairwise(
        config,
        region,
        &[lhs.clone(), create_unit_tensor(1)],
        BaseOp::Add,
    )?;

    greater(config, region, &[lhs_plus_one, rhs])
}

/// Less than to operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::less;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 0, 5, 4, 5, 1]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap());
/// let result = less::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 1, 0, 0, 0, 1]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn less<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    // just flip the order and use greater
    greater(config, region, &[values[1].clone(), values[0].clone()])
}

/// Less equals than operation.
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::less_equal;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 0, 5, 4, 5, 1]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap());
/// let result = less_equal::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 0, 1, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn less_equal<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    // just flip the order and use greater
    greater_equal(config, region, &[values[1].clone(), values[0].clone()])
}

/// Elementwise applies and to two tensors
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::and;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let result = and::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 1, 0, 1, 0]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn and<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let a = boolean_identity(config, region, &[values[0].clone()], true)?;
    let b = boolean_identity(config, region, &[values[1].clone()], true)?;

    let res = pairwise(config, region, &[a, b], BaseOp::Mult)?;

    Ok(res)
}

/// Elementwise applies or to two tensors .
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::or;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let result = or::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 1, 1, 1, 0]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn or<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let a = values[0].clone();
    let b = values[1].clone();

    let b = boolean_identity(config, region, &[b], true)?;

    let iff_values = &[a.clone(), a, b];

    let res = iff(config, region, iff_values)?;

    Ok(res)
}

/// Elementwise applies equals to two tensors .
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::equals;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let result = equals::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 0, 1, 0, 1, 1]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn equals<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let diff = pairwise(config, region, values, BaseOp::Sub)?;
    equals_zero(config, region, &[diff])
}

/// Equality boolean operation
pub(crate) fn equals_zero<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    let values = values[0].clone();
    let values_inverse = values.inverse()?;

    let product_values_and_invert = pairwise(
        config,
        region,
        &[values.clone(), values_inverse],
        BaseOp::Mult,
    )?;

    // constant of 1
    let ones = create_unit_tensor(1);
    // subtract
    let output = pairwise(
        config,
        region,
        &[ones, product_values_and_invert],
        BaseOp::Sub,
    )?;

    // take the product of diff and output
    let prod_check = pairwise(config, region, &[values, output.clone()], BaseOp::Mult)?;

    let zero_tensor = create_zero_tensor(prod_check.len());
    enforce_equality(config, region, &[prod_check, zero_tensor])?;

    Ok(output)
}

/// Elementwise applies xor to two tensors
/// # Arguments
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::xor;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[1, 1, 1, 1, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let result = xor::<Fp>(&dummy_config, &mut dummy_region, &[a,b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 1, 0, 1, 0, 0]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn xor<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let lhs = values[0].clone();
    let rhs = values[1].clone();

    let lhs_not = not(config, region, &[lhs.clone()])?;
    let rhs_not = not(config, region, &[rhs.clone()])?;

    let lhs_and_rhs_not = and(config, region, &[lhs, rhs_not.clone()])?;
    let lhs_not_and_rhs = and(config, region, &[rhs, lhs_not])?;

    // we can safely use add and not OR here because we know that lhs_and_rhs_not and lhs_not_and_rhs are =1 at different indices
    let res: ValTensor<F> = pairwise(
        config,
        region,
        &[lhs_and_rhs_not, lhs_not_and_rhs],
        BaseOp::Add,
    )?;

    Ok(res)
}

/// Elementwise applies not to a tensor .
/// # Arguments
/// * `a` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::not;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 1, 1, 1, 1, 0]),
///   &[2, 3],
/// ).unwrap());
/// let result = not::<Fp>(&dummy_config, &mut dummy_region, &[x]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0, 0, 0, 0, 1]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn not<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    let mask = values[0].clone();

    let unit = create_unit_tensor(1);
    let nil = create_zero_tensor(1);

    let res = iff(config, region, &[mask, nil, unit])?;

    Ok(res)
}

/// IFF operation.
/// # Arguments
/// * `mask` - Tensor of 0s and 1s
/// * `a` - Tensor
/// * `b` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::iff;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let mask = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[1, 0, 1, 0, 1, 0]),
/// &[2, 3],
/// ).unwrap());
/// let a = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[1, 2, 3, 4, 5, 6]),
/// &[2, 3],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[7, 8, 9, 10, 11, 12]),
/// &[2, 3],
/// ).unwrap());
/// let result = iff::<Fp>(&dummy_config, &mut dummy_region, &[mask, a, b]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 8, 3, 10, 5, 12]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn iff<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 3],
) -> Result<ValTensor<F>, CircuitError> {
    // if mask > 0 then output a else output b
    let (mask, a, b) = (&values[0], &values[1], &values[2]);

    let unit = create_unit_tensor(1);
    // make sure mask is boolean
    let assigned_mask = boolean_identity(config, region, &[mask.clone()], true)?;

    let one_minus_mask = pairwise(config, region, &[unit, assigned_mask.clone()], BaseOp::Sub)?;

    let masked_a = pairwise(config, region, &[a.clone(), assigned_mask], BaseOp::Mult)?;

    let masked_b = pairwise(config, region, &[b.clone(), one_minus_mask], BaseOp::Mult)?;

    let res = pairwise(config, region, &[masked_a, masked_b], BaseOp::Add)?;

    Ok(res)
}

/// Negates a tensor.
/// # Arguments
///
/// * `a` - Tensor
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::neg;
///
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 1, 2, 1, 1, 1]),
///     &[2, 3],
/// ).unwrap());
/// let result = neg::<Fp>(&dummy_config, &mut dummy_region, &[x]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[-2, -1, -2, -1, -1, -1]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn neg<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    let nil = create_zero_tensor(1);
    pairwise(config, region, &[nil, values[0].clone()], BaseOp::Sub)
}

/// Applies sum pooling over ND tensor of shape B x C x D1 x D2 x ... x DN.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::sumpool;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
/// use ezkl::tensor::DataFormat;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap());
/// let pooled = sumpool::<Fp>(&dummy_config, &mut dummy_region, &[x.clone()], &vec![(0, 0); 2], &vec![1;2], &vec![2, 2], false, DataFormat::default()).unwrap();
/// let expected: Tensor<IntegerRep> = Tensor::<IntegerRep>::new(Some(&[11, 8, 8, 10]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled.int_evals().unwrap(), expected);
///
/// // This time with normalization
/// let pooled = sumpool::<Fp>(&dummy_config, &mut dummy_region, &[x], &vec![(0, 0); 2], &vec![1;2],  &vec![2, 2], true, DataFormat::default()).unwrap();
/// let expected: Tensor<IntegerRep> = Tensor::<IntegerRep>::new(Some(&[3, 2, 2, 3]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled.int_evals().unwrap(), expected);
/// ```
pub fn sumpool<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    padding: &[(usize, usize)],
    stride: &[usize],
    kernel_shape: &[usize],
    normalized: bool,
    data_format: DataFormat,
) -> Result<ValTensor<F>, CircuitError> {
    let mut image = values[0].clone();
    data_format.to_canonical(&mut image)?;

    if data_format.has_no_batch() {
        let mut dims = image.dims().to_vec();
        dims.insert(0, 1);
        image.reshape(&dims)?;
    }

    let batch_size = image.dims()[0];
    let image_channels = image.dims()[1];

    let kernel_len = kernel_shape.iter().product();

    let mut kernel = create_unit_tensor(kernel_len);
    let mut kernel_dims = vec![1, 1];
    kernel_dims.extend(kernel_shape);
    kernel.reshape(&kernel_dims)?;

    let kernel = region.assign(&config.custom_gates.inputs[1], &kernel)?;
    region.increment(kernel.len());

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
                &[input, kernel.clone()],
                padding,
                stride,
                1,
                DataFormat::default(),
                KernelFormat::default(),
            )?;
            res.push(output);
            Ok(())
        })
        .collect::<Result<Vec<_>, CircuitError>>()?;

    let shape = &res[0].dims()[2..];
    let mut last_elem = res[1..]
        .iter()
        .try_fold(res[0].clone(), |acc, elem| acc.concat(elem.clone()))?;
    last_elem.reshape(&[&[batch_size, image_channels], shape].concat())?;

    if normalized {
        last_elem = div(config, region, &[last_elem], F::from(kernel_len as u64))?;
    }

    data_format.from_canonical(&mut last_elem)?;

    Ok(last_elem)
}

/// Applies  max pooling over a ND tensor of shape B x C x D1 x D2 x ... x DN.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::max_pool;
/// use ezkl::tensor::DataFormat;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// use ezkl::tensor::ValTensor;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap());
/// let pooled = max_pool::<Fp>(&dummy_config, &mut dummy_region, &[x], &vec![(0, 0); 2], &vec![1;2], &vec![2;2], DataFormat::default()).unwrap();
/// let expected: Tensor<IntegerRep> = Tensor::<IntegerRep>::new(Some(&[5, 4, 4, 6]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(pooled.int_evals().unwrap(), expected);
///
/// ```
pub fn max_pool<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    padding: &[(usize, usize)],
    stride: &[usize],
    pool_dims: &[usize],
    data_format: DataFormat,
) -> Result<ValTensor<F>, CircuitError> {
    let mut image = values[0].clone();
    data_format.to_canonical(&mut image)?;

    if data_format.has_no_batch() {
        let mut dims = image.dims().to_vec();
        dims.insert(0, 1);
        image.reshape(&dims)?;
    }

    let image_dims = image.dims();

    let (batch, input_channels) = (image_dims[0], image_dims[1]);

    let mut padded_image = image.clone();
    padded_image.pad(padding.to_vec(), 2)?;

    let slides = image_dims[2..]
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let d = padding[i].0 + d + padding[i].1;
            d.checked_sub(pool_dims[i])
                .ok_or_else(|| TensorError::Overflow("max_pool".to_string()))?
                .checked_div(stride[i])
                .ok_or_else(|| TensorError::Overflow("max_pool".to_string()))?
                .checked_add(1)
                .ok_or_else(|| TensorError::Overflow("max_pool".to_string()))
        })
        .collect::<Result<Vec<_>, TensorError>>()?;

    let mut output_dims = vec![batch, input_channels];
    output_dims.extend(slides);

    let mut output: Tensor<ValType<F>> = Tensor::new(None, &output_dims)?;

    let cartesian_coord = output_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let inner_loop_function = |idx: usize, region: &mut RegionCtx<F>| {
        let coord = &cartesian_coord[idx];
        let (b, i) = (coord[0], coord[1]);

        let mut slice = vec![b..b + 1, i..i + 1];
        slice.extend(
            coord[2..]
                .iter()
                .zip(stride.iter())
                .zip(pool_dims.iter())
                .map(|((c, s), k)| {
                    let start = c * s;
                    let end = start + k;
                    start..end
                }),
        );

        let slice = padded_image.get_slice(&slice)?;
        let max_w = max(config, region, &[slice])?;

        Ok::<_, CircuitError>(max_w.get_inner_tensor()?[0].clone())
    };

    region.apply_in_loop(&mut output, inner_loop_function)?;

    let mut res: ValTensor<F> = output.into();

    data_format.from_canonical(&mut res)?;

    Ok(res)
}

/// Performs a deconvolution on the given input tensor.
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::deconv;
/// use ezkl::tensor::{val::ValTensor, DataFormat, KernelFormat};
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// // Original test case 1: Channel expansion
/// let c = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(Some(&[6, 0, 12, 4, 0, 8, 0, 0, 3, 0, 0, 2]), &[1, 2, 2, 3]).unwrap());
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, c], &vec![(1, 1); 2], &vec![0;2], &vec![2;2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 32, 0, 0, 6, 0, 0, 4, 0, 0, 0, 0]), &[1, 2, 2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 2: Basic deconvolution
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![0;2], &vec![1;2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[6, 14, 4, 2, 17, 21, 0, 1, 5]), &[1, 1, 3, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 3: With padding
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(1, 1); 2], &vec![0;2], &vec![1;2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[17]), &[1, 1, 1, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 4: With stride
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(1, 1); 2], &vec![0;2], &vec![2; 2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[10, 4, 0, 3]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 5: Zero padding with stride
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 1, 1, 5]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![0;2], &vec![2; 2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[6, 2, 12, 4, 2, 10, 4, 20, 0, 0, 3, 1, 0, 0, 1, 5]), &[1, 1, 4, 4]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 6: Different kernel shape
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 2]),
///     &[1, 1, 2, 1],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(1, 1); 2], &vec![0;2], &vec![2; 2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 0]), &[1, 1, 2, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 7: Different kernel shape without padding
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 2]),
///     &[1, 1, 2, 1],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![0;2], &vec![2; 2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[6, 0, 12, 4, 0, 8, 0, 0, 3, 0, 0, 2]), &[1, 1, 4, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 8: Channel expansion with stride
/// let c = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(Some(&[6, 0, 12, 4, 0, 8, 0, 0, 3, 0, 0, 2]), &[1, 2, 2, 3]).unwrap());
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, c], &vec![(1, 1); 2], &vec![0;2], &vec![2;2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[0, 32, 0, 0, 6, 0, 0, 4, 0, 0, 0, 0]), &[1, 2, 2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Original test case 9: With bias
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[3, 8, 0, 8, 4, 9, 8, 1, 8]),
///     &[1, 1, 3, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 0, 4, 6]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1]),
///     &[1],
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k, b], &vec![(1, 1); 2], &vec![0;2], &vec![1;2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[55, 58, 66, 69]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Additional test case 1: NHWC format with HWIO kernel
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 2, 2, 1],  // NHWC format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 1, 5, 3]),
///     &[2, 2, 1, 1],  // HWIO format
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(1, 1); 2], &vec![0;2], &vec![1;2], 1, DataFormat::NHWC, KernelFormat::HWIO).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[27]), &[1, 1, 1, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Additional test case 2: 1D deconvolution with NCHW format
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3]),
///     &[1, 1, 3],  // NCH format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2]),
///     &[1, 1, 2],  // OIH format
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0)], &vec![0], &vec![1], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 4, 7, 6]), &[1, 1, 4]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Additional test case 3: 3D deconvolution with NCHW format
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3, 4]),
///     &[1, 1, 2, 2, 1],  // NCDHW format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 1]),
///     &[1, 1, 1, 1, 2],  // OIDHW format
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 3], &vec![0; 3], &vec![1; 3], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[1, 1, 2, 2, 3, 3, 4, 4]), &[1, 1, 2, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Additional test case 4: Multi-channel with NHWC format and OHWI kernel
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1, 3, 2, 1, 4]),  // 2 channels, 2x2 spatial
///     &[1, 2, 2, 2],  // NHWC format [batch, height, width, channels]
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
///     &[1, 2, 2, 2],  // OHWI format [out_channels, height, width, in_channels]
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![0;2], &vec![1;2], 1, DataFormat::NHWC, KernelFormat::OHWI).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[10, 24, 4, 41, 78, 27, 27, 66, 39]), &[1, 3, 3, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Additional test case 5: CHW format (no batch dimension)
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 4, 0, 1]),
///     &[1, 2, 2],  // CHW format [channels, height, width]
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3, 4]),
///     &[1, 1, 2, 2],  // OIHW format [out_channels, in_channels, height, width]
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![0;2], &vec![1;2], 1, DataFormat::CHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[6, 6, 6]), &[1, 1, 1, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Additional test case 6: HWC format with HWIO kernel
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 3, 4, 1]),
///     &[2, 2, 1],  // HWC format [height, width, channels]
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 1, 2]),
///     &[2, 2, 1, 1],  // HWIO format [height, width, in_channels, out_channels]
/// ).unwrap());
/// let result = deconv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![0;2], &vec![1;2], 1, DataFormat::HWC, KernelFormat::HWIO).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[6, 6, 6]), &[1, 1, 3, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn deconv<
    F: PrimeField + TensorType + PartialOrd + std::hash::Hash + std::marker::Send + std::marker::Sync,
>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    inputs: &[ValTensor<F>],
    padding: &[(usize, usize)],
    output_padding: &[usize],
    stride: &[usize],
    num_groups: usize,
    data_format: DataFormat,
    kernel_format: KernelFormat,
) -> Result<ValTensor<F>, CircuitError> {
    let has_bias = inputs.len() == 3;
    let (mut working_image, mut working_kernel) = (inputs[0].clone(), inputs[1].clone());

    data_format.to_canonical(&mut working_image)?;
    kernel_format.to_canonical(&mut working_kernel)?;

    if stride.iter().any(|&s| s == 0) {
        return Err(TensorError::DimMismatch(
            "non-positive stride is not supported for deconv".to_string(),
        )
        .into());
    }

    let null_val = ValType::Constant(F::ZERO);
    let mut expanded_image = working_image.clone();

    // Expand image by inserting zeros according to stride
    for (i, s) in stride.iter().enumerate() {
        expanded_image.intercalate_values(null_val.clone(), *s, 2 + i)?;
    }

    // Pad to kernel size for each spatial dimension
    expanded_image.pad(
        working_kernel.dims()[2..]
            .iter()
            .map(|d| (d - 1, d - 1))
            .collect::<Vec<_>>(),
        2,
    )?;

    // Calculate slice coordinates considering padding and output padding
    let slice_coord = expanded_image
        .dims()
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if i >= 2 {
                padding[i - 2].0..d - padding[i - 2].1 + output_padding[i - 2]
            } else {
                0..*d
            }
        })
        .collect::<Vec<_>>();

    let sliced_expanded_image = expanded_image.get_slice(&slice_coord)?;

    // Generate channel coordinates for kernel transformation
    let (in_ch_dim, out_ch_dim) =
        KernelFormat::default().get_channel_dims(working_kernel.dims().len());
    let channel_coord = (0..working_kernel.dims()[out_ch_dim])
        .cartesian_product(0..working_kernel.dims()[in_ch_dim])
        .collect::<Vec<_>>();

    // Invert kernels for deconvolution
    let mut inverted_kernels = vec![];
    for (i, j) in channel_coord {
        let channel = working_kernel.get_slice(&[i..i + 1, j..j + 1])?;
        let mut channel = Tensor::from(channel.get_inner_tensor()?.clone().into_iter().rev());
        channel.reshape(&working_kernel.dims()[2..])?;
        inverted_kernels.push(channel);
    }

    let mut deconv_kernel =
        Tensor::new(Some(&inverted_kernels), &[inverted_kernels.len()])?.combine()?;
    deconv_kernel.reshape(working_kernel.dims())?;

    // Handle tensorflow-style input/output channel ordering
    if working_kernel.dims()[0] == sliced_expanded_image.dims()[1] {
        let mut dims = deconv_kernel.dims().to_vec();
        dims.swap(0, 1);
        deconv_kernel.reshape(&dims)?;
    }

    // Prepare inputs for convolution
    let conv_input = if has_bias {
        vec![
            sliced_expanded_image,
            deconv_kernel.clone().into(),
            inputs[2].clone(),
        ]
    } else {
        vec![sliced_expanded_image, deconv_kernel.clone().into()]
    };

    let conv_dim = working_kernel.dims()[2..].len();

    // Perform convolution with canonical formats
    let mut output = conv(
        config,
        region,
        &conv_input,
        &vec![(0, 0); conv_dim],
        &vec![1; conv_dim],
        num_groups,
        data_format.canonical(),   // Use canonical format
        kernel_format.canonical(), // Use canonical format
    )?;

    // Convert output back to requested format
    data_format.from_canonical(&mut output)?;

    Ok(output)
}

/// Applies convolution over a ND tensor of shape C x H x D1...DN (and adds a bias).
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::conv;
/// use ezkl::tensor::{val::ValTensor, DataFormat, KernelFormat};
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// // Test case 1: Basic 2D convolution with NCHW format (default)
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 1, 3, 3],
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 1, 1, 1]),
///     &[1, 1, 2, 2],
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[0]),
///     &[1],
/// ).unwrap());
/// let result = conv::<Fp>(&dummy_config, &mut dummy_region, &[x, k, b], &vec![(0, 0); 2], &vec![1;2], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[31, 16, 8, 26]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Test case 2: NHWC format with HWIO kernel
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3, 1],  // NHWC format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 1, 5, 1]),
///     &[2, 2, 1, 1],  // HWIO format
/// ).unwrap());
/// let result = conv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 2], &vec![1;2], 1, DataFormat::NHWC, KernelFormat::HWIO).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[11, 24, 20, 14]), &[1, 2, 2, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Test case 3: Multi-channel NHWC with OHWI kernel
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 2, 3, 0, 4, -1, 3, 1, 6, 5, 2, 3, 0, 4, -1, 3, 1, 6]),
///     &[1, 3, 3, 2],  // NHWC format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[5, 1, 1, 2, 5, 2, 1, 2]),
///     &[1, 2, 2, 2],  // OHWI format
/// ).unwrap());
/// let b = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 1]),
///     &[2],
/// ).unwrap());
/// let result = conv::<Fp>(&dummy_config, &mut dummy_region, &[x, k, b], &vec![(0, 0); 2], &vec![1;2], 1, DataFormat::NHWC, KernelFormat::OHWI).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[64, 66, 46, 58]), &[1, 2, 2, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Test case 4: 1D convolution with NCHW format
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3, 4, 5]),
///     &[1, 1, 5],  // NCHW format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3]),
///     &[1, 1, 3],  // OIHW format
/// ).unwrap());
/// let result = conv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0)], &vec![1], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[14, 20, 26]), &[1, 1, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// // Test case 5: 3D convolution with NCHW format
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 2, 3, 4, 5, 6, 7, 8]),
///     &[1, 1, 2, 2, 2],  // NCDHW format
/// ).unwrap());
/// let k = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[1, 1]),
///     &[1, 1, 1, 1, 2],  // OIDHW format
/// ).unwrap());
/// let result = conv::<Fp>(&dummy_config, &mut dummy_region, &[x, k], &vec![(0, 0); 3], &vec![1; 3], 1, DataFormat::NCHW, KernelFormat::OIHW).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[3, 7, 11, 15]), &[1, 1, 2, 2, 1]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn conv<
    F: PrimeField + TensorType + PartialOrd + std::hash::Hash + std::marker::Send + std::marker::Sync,
>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    padding: &[(usize, usize)],
    stride: &[usize],
    num_groups: usize,
    data_format: DataFormat,
    kernel_format: KernelFormat,
) -> Result<ValTensor<F>, CircuitError> {
    let has_bias = values.len() == 3;
    let (mut working_image, mut working_kernel) = (values[0].clone(), values[1].clone());

    data_format.to_canonical(&mut working_image)?;
    kernel_format.to_canonical(&mut working_kernel)?;

    if stride.iter().any(|&s| s == 0) {
        return Err(TensorError::DimMismatch(
            "non-positive stride is not supported for conv".to_string(),
        )
        .into());
    }

    // Assign tensors
    let mut assigned_len = vec![];
    if !working_kernel.all_prev_assigned() {
        working_kernel = region.assign(&config.custom_gates.inputs[0], &working_kernel)?;
        assigned_len.push(working_kernel.len());
    }
    if !working_image.all_prev_assigned() {
        working_image = region.assign(&config.custom_gates.inputs[1], &working_image)?;
        assigned_len.push(working_image.len());
    }

    if !assigned_len.is_empty() {
        region.increment(*assigned_len.iter().max().unwrap());
    }

    if data_format.has_no_batch() {
        let mut dim = working_image.dims().to_vec();
        dim.insert(0, 1);
        working_image.reshape(&dim)?;
    }

    let image_dims = working_image.dims();
    let kernel_dims = working_kernel.dims();

    // Apply padding
    let mut padded_image = working_image.clone();
    padded_image.pad(padding.to_vec(), 2)?;

    // Extract dimensions
    let batch_size = image_dims[0];
    let input_channels = image_dims[1];
    let output_channels = kernel_dims[0];

    // Calculate slides for each spatial dimension
    let slides = image_dims[2..]
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let d = padding[i].0 + d + padding[i].1;
            d.checked_sub(kernel_dims[i + 2])
                .ok_or_else(|| TensorError::Overflow("conv".to_string()))?
                .checked_div(stride[i])
                .ok_or_else(|| TensorError::Overflow("conv".to_string()))?
                .checked_add(1)
                .ok_or_else(|| TensorError::Overflow("conv".to_string()))
        })
        .collect::<Result<Vec<_>, TensorError>>()?;

    let input_channels_per_group = input_channels / num_groups;
    let output_channels_per_group = output_channels / num_groups;

    if output_channels_per_group == 0 || input_channels_per_group == 0 {
        return Err(TensorError::DimMismatch(format!(
            "Given groups={}, expected input channels and output channels to be divisible by groups, but got input_channels={}, output_channels={}",
            num_groups, input_channels, output_channels
        )).into());
    }

    let num_outputs =
        batch_size * num_groups * output_channels_per_group * slides.iter().product::<usize>();

    let mut output: Tensor<ValType<F>> = Tensor::new(None, &[num_outputs])?;

    // Create iteration space
    let mut iterations = vec![0..batch_size, 0..num_groups, 0..output_channels_per_group];
    for slide in slides.iter() {
        iterations.push(0..*slide);
    }

    let cartesian_coord = iterations
        .iter()
        .cloned()
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let batch_offset = if data_format.has_no_batch() {
        2 // No batch dimension, start coordinates after channels
    } else {
        3 // Has batch dimension, start coordinates after batch and channels
    };

    // Main convolution loop
    let inner_loop_function = |idx: usize, region: &mut RegionCtx<F>| {
        let cartesian_coord_per_group = &cartesian_coord[idx];
        let (batch, group, i) = (
            cartesian_coord_per_group[0],
            cartesian_coord_per_group[1],
            cartesian_coord_per_group[2],
        );

        let start_channel = group * input_channels_per_group;
        let end_channel = start_channel + input_channels_per_group;

        let mut slices = vec![batch..batch + 1, start_channel..end_channel];
        for (i, stride) in stride.iter().enumerate() {
            let coord = cartesian_coord_per_group[batch_offset + i] * stride;
            let kernel_dim = kernel_dims[2 + i];
            slices.push(coord..(coord + kernel_dim));
        }

        let mut local_image = padded_image.get_slice(&slices)?;
        local_image.flatten();

        let start_kernel_index = group * output_channels_per_group + i;
        let end_kernel_index = start_kernel_index + 1;
        let mut local_kernel = working_kernel.get_slice(&[start_kernel_index..end_kernel_index])?;
        local_kernel.flatten();

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

    // Reshape output
    let mut dims = vec![batch_size, output_channels];
    dims.extend(slides.iter().cloned());
    output.reshape(&dims)?;

    // Convert output back to requested format
    let mut final_output: ValTensor<F> = output.into();
    data_format.from_canonical(&mut final_output)?;

    Ok(final_output)
}

/// Power accumulated layout
pub(crate) fn pow<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    exponent: u32,
) -> Result<ValTensor<F>, CircuitError> {
    let mut t = values[0].clone();

    for _ in 1..exponent {
        t = pairwise(config, region, &[t, values[0].clone()], BaseOp::Mult)?;
    }

    Ok(t)
}

/// Rescaled op accumulated layout
pub(crate) fn rescale<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>],
    scales: &[(usize, u128)],
) -> Result<Vec<ValTensor<F>>, CircuitError> {
    let mut rescaled_inputs = vec![];
    for (i, ri) in values.iter().enumerate() {
        if scales[i].1 == 1 {
            rescaled_inputs.push(ri.clone());
            continue;
        }

        let multiplier = create_constant_tensor(F::from(scales[i].1 as u64), 1);
        let scaled_input = pairwise(config, region, &[ri.clone(), multiplier], BaseOp::Mult)?;
        rescaled_inputs.push(scaled_input);
    }

    Ok(rescaled_inputs)
}

/// Dummy (no constraints) reshape layout
pub(crate) fn reshape<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    values: &[ValTensor<F>; 1],
    new_dims: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    let mut t = values[0].clone();
    t.reshape(new_dims)?;
    Ok(t)
}

/// Dummy (no constraints) move_axis layout
pub(crate) fn move_axis<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    values: &[ValTensor<F>; 1],
    source: usize,
    destination: usize,
) -> Result<ValTensor<F>, CircuitError> {
    let mut t = values[0].clone();
    t.move_axis(source, destination)?;
    Ok(t)
}

/// resize layout
pub(crate) fn resize<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scales: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    let mut output = region.assign(&config.custom_gates.output, &values[0])?;
    region.increment(output.len());
    output.resize(scales)?;

    Ok(output)
}

/// Slice layout
pub(crate) fn slice<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axis: &usize,
    start: &usize,
    end: &usize,
) -> Result<ValTensor<F>, CircuitError> {
    // assigns the instance to the advice.
    let mut output = values[0].clone();

    let is_assigned = output.all_prev_assigned();
    if !is_assigned {
        output = region.assign(&config.custom_gates.output, &values[0])?;
        region.increment(output.len());
    }

    output.slice(axis, start, end)?;

    Ok(output)
}

/// Trilu layout
pub(crate) fn trilu<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    k: &i32,
    upper: &bool,
) -> Result<ValTensor<F>, CircuitError> {
    // assigns the instance to the advice.
    let mut output = values[0].clone();

    let is_assigned = output.all_prev_assigned();
    if !is_assigned {
        output = region.assign(&config.custom_gates.inputs[0], &values[0])?;
    }

    let res = tensor::ops::trilu(output.get_inner_tensor()?, *k, *upper)?;

    let output = region.assign(&config.custom_gates.output, &res.into())?;
    region.increment(output.len());

    Ok(output)
}

/// Concat layout
pub(crate) fn concat<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    values: &[ValTensor<F>],
    axis: &usize,
) -> Result<ValTensor<F>, CircuitError> {
    let collected_inner: Result<Vec<&Tensor<_>>, _> =
        values.iter().map(|e| e.get_inner_tensor()).collect();
    let collected_inner = collected_inner?;

    Ok(tensor::ops::concat(&collected_inner, *axis)?.into())
}

/// Identity constraint. Usually used to constrain an instance column to an advice so the returned cells / values can be operated upon.
pub(crate) fn identity<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    decomp: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let mut output = values[0].clone();
    if !output.all_prev_assigned() {
        // checks they are in range
        if decomp {
            output = decompose(
                config,
                region,
                &[output.clone()],
                &region.base(),
                &region.legs(),
            )?
            .1;
        } else {
            output = region.assign(&config.custom_gates.output, &values[0])?;
            region.increment(output.len());
        }
    }

    Ok(output)
}

/// Boolean identity constraint. Usually used to constrain an instance column to an advice so the returned cells / values can be operated upon.
pub(crate) fn boolean_identity<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    assign: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let output = if assign || !values[0].get_const_indices().is_empty() {
        // get zero constants indices
        let output = region.assign(&config.custom_gates.output, &values[0])?;
        region.increment(output.len());
        output
    } else {
        values[0].clone()
    };

    range_check(config, region, values, &(0, 1))?;

    Ok(output)
}

/// Downsample layout
pub(crate) fn downsample<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axis: &usize,
    stride: &usize,
    modulo: &usize,
) -> Result<ValTensor<F>, CircuitError> {
    let input = region.assign(&config.custom_gates.inputs[0], &values[0])?;
    let processed_output =
        tensor::ops::downsample(input.get_inner_tensor()?, *axis, *stride, *modulo)?;
    let output = region.assign(&config.custom_gates.output, &processed_output.into())?;
    region.increment(std::cmp::max(input.len(), output.len()));
    Ok(output)
}

/// layout for enforcing two sets of cells to be equal
pub(crate) fn enforce_equality<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    // assert of same len
    if values[0].len() != values[1].len() {
        return Err(TensorError::DimMismatch("enforce_equality".to_string()).into());
    }

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
pub(crate) fn range_check<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    range: &crate::circuit::table::Range,
) -> Result<ValTensor<F>, CircuitError> {
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
            Ok::<ValType<F>, CircuitError>(if let Some(f) = e.get_felt_eval() {
                let col_idx = if !is_dummy {
                    let table = config.range_checks.ranges.get(range).ok_or(
                        CircuitError::RangeCheckNotConfigured(format!("{:?}", range)),
                    )?;
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
            .collect::<Result<Vec<_>, CircuitError>>()?;
    }

    let is_assigned = !w.any_unknowns()?;
    if is_assigned && region.check_range() && config.check_mode.is_safe() {
        // assert is within range
        let int_values = w.int_evals()?;
        for v in int_values.iter() {
            if v < &range.0 || v > &range.1 {
                return Err(CircuitError::TableOOR(*v, range.0, range.1));
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
pub(crate) fn nonlinearity<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    nl: &LookupOp,
) -> Result<ValTensor<F>, CircuitError> {
    region.add_used_lookup(nl.clone(), values)?;

    // time the entire operation
    let timer = instant::Instant::now();

    let x = values[0].clone();

    let removal_indices = values[0].get_const_indices();
    let removal_indices: HashSet<usize> = HashSet::from_iter(removal_indices);

    let w = region.assign_with_omissions(&config.static_lookups.input, &x, &removal_indices)?;

    let output = w.get_inner_tensor()?.par_enum_map(|i, e| {
        Ok::<_, TensorError>(if let Some(f) = e.get_felt_eval() {
            if !removal_indices.contains(&i) {
                Value::known(nl.f(&[Tensor::from(vec![f].into_iter())])?.output[0]).into()
            } else {
                ValType::Constant(nl.f(&[Tensor::from(vec![f].into_iter())])?.output[0])
            }
        } else {
            Value::<F>::unknown().into()
        })
    })?;

    let assigned_len = x.len() - removal_indices.len();
    let mut output = region.assign_with_omissions(
        &config.static_lookups.output,
        &output.into(),
        &removal_indices,
    )?;

    let is_dummy = region.is_dummy();

    let table_index: ValTensor<F> = w
        .get_inner_tensor()?
        .par_enum_map(|i, e| {
            Ok::<_, CircuitError>(if let Some(f) = e.get_felt_eval() {
                let col_idx = if !is_dummy {
                    let table = config
                        .static_lookups
                        .tables
                        .get(nl)
                        .ok_or(CircuitError::LookupNotConfigured(Op::<F>::as_string(nl)))?;
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

    region.assign_with_omissions(&config.static_lookups.index, &table_index, &removal_indices)?;

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
            .collect::<Result<Vec<_>, CircuitError>>()?;
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
pub(crate) fn argmax<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    // this is safe because we later constrain it
    let argmax = values[0]
        .int_evals()?
        .into_par_iter()
        .enumerate()
        // we value the first index in the case of a tie
        .max_by_key(|(idx, value)| (*value, -(*idx as IntegerRep)))
        .map(|(idx, _)| idx as IntegerRep);
    let argmax_val: ValTensor<F> = match argmax {
        None => Tensor::new(Some(&[Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&[Value::known(integer_rep_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_argmax: ValTensor<F> =
        region.assign(&config.custom_gates.inputs[1], &argmax_val)?;
    region.increment(assigned_argmax.len());

    let claimed_val = select(
        config,
        region,
        &[values[0].clone(), assigned_argmax.clone()],
    )?;

    let (sorted_val, indices) =
        _sort_ascending(config, region, values, SortCollisionMode::LargestIndexFirst)?;

    enforce_equality(config, region, &[claimed_val, sorted_val.last()?])?;
    enforce_equality(config, region, &[assigned_argmax.clone(), indices.last()?])?;

    Ok(assigned_argmax)
}

/// Argmin
pub(crate) fn argmin<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    // this is safe because we later constrain it
    let argmin = values[0]
        .int_evals()?
        .into_par_iter()
        .enumerate()
        // we value the first index in the case of a tie
        .min_by_key(|(idx, value)| (*value, (*idx as IntegerRep)))
        .map(|(idx, _)| idx as IntegerRep);
    let argmin_val: ValTensor<F> = match argmin {
        None => Tensor::new(Some(&[Value::<F>::unknown()]), &[1])?.into(),
        Some(i) => Tensor::new(Some(&[Value::known(integer_rep_to_felt::<F>(i))]), &[1])?.into(),
    };

    let assigned_argmin: ValTensor<F> =
        region.assign(&config.custom_gates.inputs[1], &argmin_val)?;
    region.increment(assigned_argmin.len());

    // these will be assigned as constants
    let claimed_val = select(
        config,
        region,
        &[values[0].clone(), assigned_argmin.clone()],
    )?;
    let (min_val, indices) = _sort_ascending(
        config,
        region,
        values,
        SortCollisionMode::SmallestIndexFirst,
    )?;
    enforce_equality(config, region, &[claimed_val, min_val.first()?])?;
    enforce_equality(config, region, &[assigned_argmin.clone(), indices.first()?])?;

    Ok(assigned_argmin)
}

/// Max layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 2]
/// # Returns
/// * ValTensor<F>
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::max_comp;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[5, 2, 3, 0]),
///   &[1, 1, 2, 2],
/// ).unwrap());
/// let y = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[5, 1, 1, 1]),
///  &[1, 1, 2, 2],
/// ).unwrap());
///
/// let result = max_comp::<Fp>(&dummy_config, &mut dummy_region, &[x, y]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[5, 2, 3, 1]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn max_comp<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let is_greater = greater(config, region, values)?;
    let is_less = not(config, region, &[is_greater.clone()])?;

    let max_val_p1 = pairwise(
        config,
        region,
        &[values[0].clone(), is_greater],
        BaseOp::Mult,
    )?;

    let max_val_p2 = pairwise(config, region, &[values[1].clone(), is_less], BaseOp::Mult)?;

    pairwise(config, region, &[max_val_p1, max_val_p2], BaseOp::Add)
}

/// Min comp layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 2]
/// # Returns
/// * ValTensor<F>
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::min_comp;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///   Some(&[5, 2, 3, 0]),
///  &[1, 1, 2, 2],
/// ).unwrap());
/// let y = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[5, 1, 1, 1]),
/// &[1, 1, 2, 2],
/// ).unwrap());
/// let result = min_comp::<Fp>(&dummy_config, &mut dummy_region, &[x, y]).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[5, 1, 1, 0]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn min_comp<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let is_greater = greater(config, region, values)?;
    let is_less = not(config, region, &[is_greater.clone()])?;

    let min_val_p1 = pairwise(config, region, &[values[0].clone(), is_less], BaseOp::Mult)?;

    let min_val_p2 = pairwise(
        config,
        region,
        &[values[1].clone(), is_greater],
        BaseOp::Mult,
    )?;

    pairwise(config, region, &[min_val_p1, min_val_p2], BaseOp::Add)
}

/// max layout
pub(crate) fn max<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    Ok(
        _sort_ascending(config, region, values, SortCollisionMode::Unsorted)?
            .0
            .last()?,
    )
}

/// min layout
pub(crate) fn min<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    Ok(
        _sort_ascending(config, region, values, SortCollisionMode::Unsorted)?
            .0
            .first()?,
    )
}

/// floor layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 1]
/// * `scale` - utils::F32
/// * `legs` - usize
/// # Returns
/// * ValTensor<F>
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::floor;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[3, -2, -3, 1]),
/// &[1, 1, 2, 2],
/// ).unwrap());
/// let result = floor::<Fp>(&dummy_config, &mut dummy_region, &[x], 2.0.into(), 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[2, -2, -4, 0]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn floor<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
    legs: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // decompose with base scale and then set the last element to zero
    let decomposition = decompose(config, region, values, &(scale.0 as usize), &legs)?.0;
    // set the last element to zero and then recompose, we don't actually need to assign here
    // as this will automatically be assigned in the recompose function and uses the constant caching of RegionCtx
    let zero = ValType::Constant(F::ZERO);

    let negative_one = create_constant_tensor(integer_rep_to_felt(-1), 1);
    let assigned_negative_one = region.assign(&config.custom_gates.inputs[1], &negative_one)?;

    region.increment(1);

    let dims = decomposition.dims().to_vec();
    let first_dims = decomposition.dims().to_vec()[..decomposition.dims().len() - 1].to_vec();

    let mut incremented_tensor = Tensor::new(None, &first_dims)?;

    let cartesian_coord = first_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let inner_loop_function =
        |i: usize, region: &mut RegionCtx<F>| -> Result<Tensor<ValType<F>>, CircuitError> {
            let coord = cartesian_coord[i].clone();
            let slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let mut sliced_input = decomposition.get_slice(&slice)?;
            sliced_input.flatten();
            let last_elem = sliced_input.last()?;

            let last_elem_is_zero = equals_zero(config, region, &[last_elem.clone()])?;
            let last_elem_is_not_zero = not(config, region, &[last_elem_is_zero.clone()])?;

            let sign = sliced_input.first()?;
            let is_negative = equals(config, region, &[sign, assigned_negative_one.clone()])?;

            let is_negative_and_not_zero = and(
                config,
                region,
                &[last_elem_is_not_zero.clone(), is_negative.clone()],
            )?;

            // increment the penultimate element
            let incremented_elem = pairwise(
                config,
                region,
                &[
                    sliced_input.get_slice(&[sliced_input.len() - 2..sliced_input.len() - 1])?,
                    is_negative_and_not_zero.clone(),
                ],
                BaseOp::Add,
            )?;

            let mut inner_tensor = sliced_input.get_inner_tensor()?.clone();
            inner_tensor[sliced_input.len() - 2] =
                incremented_elem.get_inner_tensor()?.clone()[0].clone();

            // set the last elem to zero
            inner_tensor[sliced_input.len() - 1] = zero.clone();

            Ok(inner_tensor.clone())
        };

    region.apply_in_loop(&mut incremented_tensor, inner_loop_function)?;

    let mut incremented_tensor = incremented_tensor.combine()?;
    incremented_tensor.reshape(&dims)?;

    recompose(
        config,
        region,
        &[incremented_tensor.into()],
        &(scale.0 as usize),
    )
}

/// ceil layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 1]
/// * `scale` - utils::F32
/// * `legs` - usize
/// # Returns
/// * ValTensor<F>
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::ceil;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///  Some(&[3, -2, 3, 1]),
/// &[1, 1, 2, 2],
/// ).unwrap());
/// let result = ceil::<Fp>(&dummy_config, &mut dummy_region, &[x], 2.0.into(), 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, -2, 4, 2]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn ceil<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
    legs: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // decompose with base scale and then set the last element to zero
    let decomposition = decompose(config, region, values, &(scale.0 as usize), &legs)?.0;
    // set the last element to zero and then recompose, we don't actually need to assign here
    // as this will automatically be assigned in the recompose function and uses the constant caching of RegionCtx
    let zero = ValType::Constant(F::ZERO);

    let one = create_constant_tensor(integer_rep_to_felt(1), 1);
    let assigned_one = region.assign(&config.custom_gates.inputs[1], &one)?;

    region.increment(1);

    let dims = decomposition.dims().to_vec();
    let first_dims = decomposition.dims().to_vec()[..decomposition.dims().len() - 1].to_vec();

    let mut incremented_tensor = Tensor::new(None, &first_dims)?;

    let cartesian_coord = first_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let inner_loop_function =
        |i: usize, region: &mut RegionCtx<F>| -> Result<Tensor<ValType<F>>, CircuitError> {
            let coord = cartesian_coord[i].clone();
            let slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let mut sliced_input = decomposition.get_slice(&slice)?;
            sliced_input.flatten();
            let last_elem = sliced_input.last()?;

            let last_elem_is_zero = equals_zero(config, region, &[last_elem.clone()])?;
            let last_elem_is_not_zero = not(config, region, &[last_elem_is_zero.clone()])?;

            let sign = sliced_input.first()?;
            let is_positive = equals(config, region, &[sign, assigned_one.clone()])?;

            let is_positive_and_not_zero = and(
                config,
                region,
                &[last_elem_is_not_zero.clone(), is_positive.clone()],
            )?;

            // increment the penultimate element
            let incremented_elem = pairwise(
                config,
                region,
                &[
                    sliced_input.get_slice(&[sliced_input.len() - 2..sliced_input.len() - 1])?,
                    is_positive_and_not_zero.clone(),
                ],
                BaseOp::Add,
            )?;

            let mut inner_tensor = sliced_input.get_inner_tensor()?.clone();
            inner_tensor[sliced_input.len() - 2] =
                incremented_elem.get_inner_tensor()?.clone()[0].clone();

            // set the last elem to zero
            inner_tensor[sliced_input.len() - 1] = zero.clone();

            Ok(inner_tensor.clone())
        };

    region.apply_in_loop(&mut incremented_tensor, inner_loop_function)?;

    let mut incremented_tensor = incremented_tensor.combine()?;
    incremented_tensor.reshape(&dims)?;

    recompose(
        config,
        region,
        &[incremented_tensor.into()],
        &(scale.0 as usize),
    )
}

/// integer ln layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 1]
/// * `scale` - utils::F32
/// # Returns
/// * ValTensor<F>
/// # Example
///
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::ln;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[3, 2, 3, 1]),
/// &[1, 1, 2, 2],
/// ).unwrap());
///
/// let result = ln::<Fp>(&dummy_config, &mut dummy_region, &[x], 2.0.into()).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, 0, 4, -8]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
///
/// ```
pub fn ln<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
) -> Result<ValTensor<F>, CircuitError> {
    // first generate the claimed val

    let mut input = values[0].clone();
    let scale_as_felt = integer_rep_to_felt(scale.0.round() as IntegerRep);

    let triple_scaled_as_felt_tensor =
        create_constant_tensor(scale_as_felt * scale_as_felt * scale_as_felt, 1);

    // natural ln is log2(x) * ln(2)
    let ln2 = utils::F32::from(2.0_f32.ln());
    // now create a constant tensor for ln2 with scale
    let ln2_tensor: ValTensor<F> = create_constant_tensor(
        integer_rep_to_felt((ln2.0 * scale.0).round() as IntegerRep),
        1,
    );
    let unit = create_constant_tensor(integer_rep_to_felt(1), 1);
    let negative_one = create_constant_tensor(integer_rep_to_felt(-1), 1);

    // 2. assign the image
    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
        // don't need to increment because the claimed output is assigned to output and incremented accordingly
    }

    let is_assigned = !input.any_unknowns()?;

    let mut claimed_output: ValTensor<F> = if is_assigned {
        let input_evals = input.int_evals()?;
        // returns an integer with the base 2 logarithm
        tensor::ops::nonlinearities::ilog2(&input_evals.clone(), scale.0 as f64)
            .par_iter()
            .map(|x| Value::known(integer_rep_to_felt(*x)))
            .collect::<Tensor<Value<F>>>()
            .into()
    } else {
        Tensor::new(
            Some(&vec![Value::<F>::unknown(); input.len()]),
            &[input.len()],
        )?
        .into()
    };
    claimed_output.reshape(input.dims())?;
    let claimed_output = identity(&config, region, &[claimed_output], true)?;
    region.increment(claimed_output.len());

    let pow2_of_claimed_output = nonlinearity(
        config,
        region,
        &[claimed_output.clone()],
        &LookupOp::PowersOfTwo { scale },
    )?;

    let num_bits = (std::mem::size_of::<IntegerRep>() * 8) as IntegerRep;

    region.update_max_min_lookup_inputs_force(-num_bits, num_bits)?;

    // now subtract 1 from the claimed output
    let claimed_output_minus_one = pairwise(
        config,
        region,
        &[claimed_output.clone(), unit.clone()],
        BaseOp::Sub,
    )?;

    // now add 1 to the claimed output
    let claimed_output_plus_one = pairwise(
        config,
        region,
        &[claimed_output.clone(), unit.clone()],
        BaseOp::Add,
    )?;

    // prior power of 2 is less than claimed output
    let prior_pow2 = nonlinearity(
        config,
        region,
        &[claimed_output_minus_one],
        &LookupOp::PowersOfTwo { scale },
    )?;

    // next power of 2 is greater than claimed output
    let next_pow2 = nonlinearity(
        config,
        region,
        &[claimed_output_plus_one],
        &LookupOp::PowersOfTwo { scale },
    )?;

    let distance_to_claimed = pairwise(
        config,
        region,
        &[input.clone(), pow2_of_claimed_output.clone()],
        BaseOp::Sub,
    )?;

    let abs_distance_to_claimed = abs(config, region, &[distance_to_claimed.clone()])?;

    let abs_distance_to_next_pow2 =
        l1_distance(config, region, &[input.clone(), next_pow2.clone()])?;

    let abs_distance_to_prior_pow2 =
        l1_distance(config, region, &[input.clone(), prior_pow2.clone()])?;

    // because we round up this can be equal
    let is_closest_to_0: ValTensor<F> = less(
        config,
        region,
        &[
            abs_distance_to_claimed.clone(),
            abs_distance_to_next_pow2.clone(),
        ],
    )?;

    let is_closest_to_1 = less(
        config,
        region,
        &[
            abs_distance_to_claimed.clone(),
            abs_distance_to_prior_pow2.clone(),
        ],
    )?;

    let is_closest = and(
        config,
        region,
        &[is_closest_to_0.clone(), is_closest_to_1.clone()],
    )?;

    let mut comparison_unit = create_constant_tensor(integer_rep_to_felt(1), is_closest.len());
    comparison_unit.reshape(is_closest.dims())?;
    let assigned_unit = region.assign(&config.custom_gates.inputs[1], &comparison_unit)?;

    enforce_equality(config, region, &[is_closest, assigned_unit])?;

    // get a linear interpolation now

    let sign_of_distance_to_claimed = sign(config, region, &[distance_to_claimed.clone()])?;
    let sign_of_distance_to_claimed_is_negative = equals(
        config,
        region,
        &[sign_of_distance_to_claimed.clone(), negative_one.clone()],
    )?;

    let sign_of_distance_to_claimed_is_positive = not(
        config,
        region,
        &[sign_of_distance_to_claimed_is_negative.clone()],
    )?;

    let pow2_prior_to_claimed_distance = pairwise(
        config,
        region,
        &[pow2_of_claimed_output.clone(), prior_pow2.clone()],
        BaseOp::Sub,
    )?;

    let pow2_next_to_claimed_distance = pairwise(
        config,
        region,
        &[next_pow2.clone(), pow2_of_claimed_output.clone()],
        BaseOp::Sub,
    )?;

    let recip_pow2_prior_to_claimed_distance = recip(
        config,
        region,
        &[pow2_prior_to_claimed_distance],
        scale_as_felt,
        scale_as_felt * scale_as_felt,
    )?;

    let interpolated_distance = pairwise(
        config,
        region,
        &[
            recip_pow2_prior_to_claimed_distance.clone(),
            distance_to_claimed.clone(),
        ],
        BaseOp::Mult,
    )?;

    let gated_prior_interpolated_distance = pairwise(
        config,
        region,
        &[
            interpolated_distance.clone(),
            sign_of_distance_to_claimed_is_negative.clone(),
        ],
        BaseOp::Mult,
    )?;

    let recip_next_to_claimed_distance = recip(
        config,
        region,
        &[pow2_next_to_claimed_distance],
        scale_as_felt,
        scale_as_felt * scale_as_felt,
    )?;

    let interpolated_distance_next = pairwise(
        config,
        region,
        &[
            recip_next_to_claimed_distance.clone(),
            distance_to_claimed.clone(),
        ],
        BaseOp::Mult,
    )?;

    let gated_next_interpolated_distance = pairwise(
        config,
        region,
        &[
            interpolated_distance_next.clone(),
            sign_of_distance_to_claimed_is_positive.clone(),
        ],
        BaseOp::Mult,
    )?;

    let scaled_claimed_output = pairwise(
        config,
        region,
        &[claimed_output.clone(), triple_scaled_as_felt_tensor],
        BaseOp::Mult,
    )?;

    let claimed_output = pairwise(
        config,
        region,
        &[
            scaled_claimed_output.clone(),
            gated_prior_interpolated_distance.clone(),
        ],
        BaseOp::Add,
    )?;

    let claimed_output = pairwise(
        config,
        region,
        &[
            claimed_output.clone(),
            gated_next_interpolated_distance.clone(),
        ],
        BaseOp::Add,
    )?;

    // now multiply the claimed output by ln2
    pairwise(config, region, &[claimed_output, ln2_tensor], BaseOp::Mult)
}

/// round layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 1]
/// * `scale` - utils::F32
/// * `legs` - usize
/// # Returns
/// * ValTensor<F>
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::round;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[3, -2, 3, 1]),
/// &[1, 1, 2, 2],
/// ).unwrap());
/// let result = round::<Fp>(&dummy_config, &mut dummy_region, &[x], 4.0.into(), 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, -4, 4, 0]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn round<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
    legs: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // decompose with base scale and then set the last element to zero
    let decomposition = decompose(config, region, values, &(scale.0 as usize), &legs)?.0;
    // set the last element to zero and then recompose, we don't actually need to assign here
    // as this will automatically be assigned in the recompose function and uses the constant caching of RegionCtx
    let zero = ValType::Constant(F::ZERO);

    let one = create_constant_tensor(integer_rep_to_felt(1), 1);
    let negative_one = create_constant_tensor(integer_rep_to_felt(-1), 1);

    // if scale is not exactly divisible by 2 we warn
    if scale.0 % 2.0 != 0.0 {
        log::warn!("Scale is not exactly divisible by 2.0, rounding may not be accurate");
    }

    let midway_point: ValTensor<F> = create_constant_tensor(
        integer_rep_to_felt((scale.0 / 2.0).round() as IntegerRep),
        1,
    );
    let assigned_midway_point = region.assign(&config.custom_gates.inputs[1], &midway_point)?;
    region.increment(assigned_midway_point.len());

    let dims = decomposition.dims().to_vec();
    let first_dims = decomposition.dims().to_vec()[..decomposition.dims().len() - 1].to_vec();

    let mut incremented_tensor = Tensor::new(None, &first_dims)?;

    let cartesian_coord = first_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let inner_loop_function =
        |i: usize, region: &mut RegionCtx<F>| -> Result<Tensor<ValType<F>>, CircuitError> {
            let coord = cartesian_coord[i].clone();
            let slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let mut sliced_input = decomposition.get_slice(&slice)?;
            sliced_input.flatten();
            let last_elem = sliced_input.last()?;

            let sign = sliced_input.first()?;
            let is_positive = equals(config, region, &[sign.clone(), one.clone()])?;
            let is_negative = equals(config, region, &[sign, negative_one.clone()])?;

            let is_greater_than_midway = greater_equal(
                config,
                region,
                &[last_elem.clone(), assigned_midway_point.clone()],
            )?;

            // if greater than midway point and positive, increment
            let is_positive_and_more_than_midway = and(
                config,
                region,
                &[is_positive.clone(), is_greater_than_midway.clone()],
            )?;

            // is less than midway point and negative, decrement
            let is_negative_and_more_than_midway = and(
                config,
                region,
                &[is_negative.clone(), is_greater_than_midway],
            )?;

            let conditions_for_increment = or(
                config,
                region,
                &[
                    is_positive_and_more_than_midway.clone(),
                    is_negative_and_more_than_midway.clone(),
                ],
            )?;

            // increment the penultimate element
            let incremented_elem = pairwise(
                config,
                region,
                &[
                    sliced_input.get_slice(&[sliced_input.len() - 2..sliced_input.len() - 1])?,
                    conditions_for_increment.clone(),
                ],
                BaseOp::Add,
            )?;

            let mut inner_tensor = sliced_input.get_inner_tensor()?.clone();
            inner_tensor[sliced_input.len() - 2] =
                incremented_elem.get_inner_tensor()?.clone()[0].clone();

            // set the last elem to zero
            inner_tensor[sliced_input.len() - 1] = zero.clone();

            Ok(inner_tensor.clone())
        };

    region.apply_in_loop(&mut incremented_tensor, inner_loop_function)?;

    let mut incremented_tensor = incremented_tensor.combine()?;
    incremented_tensor.reshape(&dims)?;

    recompose(
        config,
        region,
        &[incremented_tensor.into()],
        &(scale.0 as usize),
    )
}

/// round half to even layout
/// # Arguments
/// * `config` - BaseConfig
/// * `region` - RegionCtx
/// * `values` - &[ValTensor<F>; 1]
/// * `scale` - utils::F32
/// * `legs` - usize
/// # Returns
/// * ValTensor<F>
/// # Example
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::round;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
/// Some(&[3, -2, 3, 1]),
/// &[1, 1, 2, 2],
/// ).unwrap());
/// let result = round::<Fp>(&dummy_config, &mut dummy_region, &[x], 4.0.into(), 2).unwrap();
/// let expected = Tensor::<IntegerRep>::new(Some(&[4, -4, 4, 0]), &[1, 1, 2, 2]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
///
pub fn round_half_to_even<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    scale: utils::F32,
    legs: usize,
) -> Result<ValTensor<F>, CircuitError> {
    // decompose with base scale and then set the last element to zero
    let decomposition = decompose(config, region, values, &(scale.0 as usize), &legs)?.0;
    // set the last element to zero and then recompose, we don't actually need to assign here
    // as this will automatically be assigned in the recompose function and uses the constant caching of RegionCtx
    let zero = ValType::Constant(F::ZERO);

    // if scale is not exactly divisible by 2 we warn
    if scale.0 % 2.0 != 0.0 {
        log::warn!("Scale is not exactly divisible by 2.0, rounding may not be accurate");
    }

    let midway_point: ValTensor<F> = create_constant_tensor(
        integer_rep_to_felt((scale.0 / 2.0).round() as IntegerRep),
        1,
    );

    let dims = decomposition.dims().to_vec();
    let first_dims = decomposition.dims().to_vec()[..decomposition.dims().len() - 1].to_vec();

    let mut incremented_tensor = Tensor::new(None, &first_dims)?;

    let cartesian_coord = first_dims
        .iter()
        .map(|x| 0..*x)
        .multi_cartesian_product()
        .collect::<Vec<_>>();

    let inner_loop_function =
        |i: usize, region: &mut RegionCtx<F>| -> Result<Tensor<ValType<F>>, CircuitError> {
            let coord = cartesian_coord[i].clone();
            let slice = coord.iter().map(|x| *x..*x + 1).collect::<Vec<_>>();
            let mut sliced_input = decomposition.get_slice(&slice)?;
            sliced_input.flatten();
            let last_elem = sliced_input.last()?;

            let penultimate_elem =
                sliced_input.get_slice(&[sliced_input.len() - 2..sliced_input.len() - 1])?;

            let is_equal_to_midway =
                equals(config, region, &[last_elem.clone(), midway_point.clone()])?;
            // penultimate_elem is equal to midway point and even, do nothing
            let is_odd = nonlinearity(
                config,
                region,
                &[penultimate_elem.clone()],
                &LookupOp::IsOdd,
            )?;

            let is_odd_and_equal_to_midway = and(
                config,
                region,
                &[is_odd.clone(), is_equal_to_midway.clone()],
            )?;

            let is_greater_than_midway =
                greater(config, region, &[last_elem.clone(), midway_point.clone()])?;

            // if the number is equal to midway point and odd increment, or if it is is_greater_than_midway
            let is_odd_and_equal_to_midway_or_greater_than_midway = or(
                config,
                region,
                &[
                    is_odd_and_equal_to_midway.clone(),
                    is_greater_than_midway.clone(),
                ],
            )?;

            // increment the penultimate element
            let incremented_elem = pairwise(
                config,
                region,
                &[
                    sliced_input.get_slice(&[sliced_input.len() - 2..sliced_input.len() - 1])?,
                    is_odd_and_equal_to_midway_or_greater_than_midway.clone(),
                ],
                BaseOp::Add,
            )?;

            let mut inner_tensor = sliced_input.get_inner_tensor()?.clone();
            inner_tensor[sliced_input.len() - 2] =
                incremented_elem.get_inner_tensor()?.clone()[0].clone();

            // set the last elem to zero
            inner_tensor[sliced_input.len() - 1] = zero.clone();

            Ok(inner_tensor.clone())
        };

    region.update_max_min_lookup_inputs_force(0, scale.0 as IntegerRep)?;

    region.apply_in_loop(&mut incremented_tensor, inner_loop_function)?;

    let mut incremented_tensor = incremented_tensor.combine()?;
    incremented_tensor.reshape(&dims)?;

    recompose(
        config,
        region,
        &[incremented_tensor.into()],
        &(scale.0 as usize),
    )
}

pub(crate) fn recompose<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    base: &usize,
) -> Result<ValTensor<F>, CircuitError> {
    let mut input = values[0].clone();

    let first_dims = input.dims().to_vec()[..input.dims().len() - 1].to_vec();
    let num_first_dims = first_dims.iter().product::<usize>();
    let n = input.dims().last().unwrap() - 1;

    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
        region.increment(input.len());
    }

    // to force the bases to be assigned
    if input.is_singleton() {
        input.reshape(&[1])?;
    }

    let mut bases: ValTensor<F> = Tensor::from({
        (0..num_first_dims)
            .flat_map(|_| {
                (0..n).rev().map(|x| {
                    let base = (*base).checked_pow(x as u32);
                    if let Some(base) = base {
                        Ok(ValType::Constant(integer_rep_to_felt(base as IntegerRep)))
                    } else {
                        Err(CircuitError::DecompositionBaseOverflow)
                    }
                })
            })
            .collect::<Result<Vec<_>, CircuitError>>()?
            .into_iter()
    })
    .into();
    let mut bases_dims = first_dims.clone();
    bases_dims.push(n);
    bases.reshape(&bases_dims)?;

    // equation needs to be constructed as ij,j->i but for arbitrary n dims we need to construct this dynamically
    // indices should map in order of the alphabet
    // start with lhs
    let lhs = ASCII_ALPHABET.chars().take(input.dims().len()).join("");
    let rhs = ASCII_ALPHABET.chars().take(input.dims().len() - 1).join("");

    let equation = format!("{},{}->{}", lhs, lhs, rhs);

    let mut sign_slice = first_dims.iter().map(|x| 0..*x).collect::<Vec<_>>();
    sign_slice.push(0..1);
    let mut rest_slice = first_dims.iter().map(|x| 0..*x).collect::<Vec<_>>();
    rest_slice.push(1..n + 1);

    let sign = input.get_slice(&sign_slice)?;
    let rest = input.get_slice(&rest_slice)?;

    // now add the rhs
    let prod_recomp = einsum(config, region, &[rest.clone(), bases], &equation)?;
    let mut signed_recomp = pairwise(config, region, &[prod_recomp, sign], BaseOp::Mult)?;
    signed_recomp.reshape(&first_dims)?;

    Ok(signed_recomp.into())
}

pub(crate) fn decompose<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    base: &usize,
    n: &usize,
) -> Result<(ValTensor<F>, ValTensor<F>), CircuitError> {
    let mut input = values[0].clone();

    if !input.all_prev_assigned() {
        input = region.assign(&config.custom_gates.inputs[0], &input)?;
    }

    // to force the bases to be assigned
    if input.is_singleton() {
        input.reshape(&[1])?;
    }

    let mut bases: ValTensor<F> = Tensor::from({
        (0..input.len())
            .flat_map(|_| {
                (0..*n).rev().map(|x| {
                    let base = (*base).checked_pow(x as u32);
                    if let Some(base) = base {
                        Ok(ValType::Constant(integer_rep_to_felt(base as IntegerRep)))
                    } else {
                        Err(CircuitError::DecompositionBaseOverflow)
                    }
                })
            })
            .collect::<Result<Vec<_>, CircuitError>>()?
            .into_iter()
    })
    .into();

    let mut bases_dims = input.dims().to_vec();
    bases_dims.push(*n);
    bases.reshape(&bases_dims)?;

    let mut decomposed_dims = input.dims().to_vec();
    decomposed_dims.push(*n + 1);

    let claimed_output = if region.witness_gen() {
        input.decompose(*base, *n)?
    } else {
        let decomposed_len = decomposed_dims.iter().product();
        let claimed_output = Tensor::new(
            Some(&vec![ValType::Value(Value::unknown()); decomposed_len]),
            &decomposed_dims,
        )?;

        claimed_output.into()
    };
    let claimed_output = region.assign(&config.custom_gates.output, &claimed_output)?;
    region.increment(claimed_output.len());

    let input_slice = input.dims().iter().map(|x| 0..*x).collect::<Vec<_>>();
    let mut sign_slice = input_slice.clone();
    sign_slice.push(0..1);
    let mut rest_slice = input_slice.clone();
    rest_slice.push(1..n + 1);

    let sign = claimed_output.get_slice(&sign_slice)?;
    let rest = claimed_output.get_slice(&rest_slice)?;

    let sign = range_check(config, region, &[sign], &(-1, 1))?;

    // isZero(input) * sign == 0.
    {
        let is_zero = equals_zero(config, region, &[input.clone()])?;
        // take the product of the sign and is_zero
        let sign_is_zero = pairwise(config, region, &[sign.clone(), is_zero], BaseOp::Mult)?;
        // constrain the sign_is_zero to be 0
        enforce_equality(
            config,
            region,
            &[
                sign_is_zero.clone(),
                create_constant_tensor(F::ZERO, sign_is_zero.len()),
            ],
        )?;
    }

    let rest = range_check(config, region, &[rest], &(0, (*base - 1) as IntegerRep))?;

    // equation needs to be constructed as ij,ij->i but for arbitrary n dims we need to construct this dynamically
    // indices should map in order of the alphabet
    // start with lhs
    let lhs = ASCII_ALPHABET.chars().take(rest.dims().len()).join("");
    let rhs = ASCII_ALPHABET.chars().take(rest.dims().len() - 1).join("");
    let equation = format!("{},{}->{}", lhs, lhs, rhs);

    // now add the rhs

    let prod_decomp = einsum(config, region, &[rest.clone(), bases], &equation)?;

    let signed_decomp = pairwise(config, region, &[prod_decomp, sign], BaseOp::Mult)?;

    enforce_equality(config, region, &[input.clone(), signed_decomp])?;

    Ok((claimed_output, input))
}

pub(crate) fn sign<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    let mut decomp = decompose(config, region, values, &region.base(), &region.legs())?.0;
    // get every n elements now, which correspond to the sign bit
    decomp.get_every_n(region.legs() + 1)?;
    decomp.reshape(values[0].dims())?;

    Ok(decomp)
}

pub(crate) fn abs<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
) -> Result<ValTensor<F>, CircuitError> {
    let sign = sign(config, region, values)?;

    pairwise(config, region, &[values[0].clone(), sign], BaseOp::Mult)
}

pub(crate) fn leaky_relu<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    alpha: &utils::F32,
    input_scale: &i32,
) -> Result<ValTensor<F>, CircuitError> {
    let sign = sign(config, region, values)?;

    let mut unit = create_unit_tensor(sign.len());
    unit.reshape(sign.dims())?;

    let relu_mask = equals(config, region, &[sign, unit])?;

    let positive = pairwise(
        config,
        region,
        &[values[0].clone(), relu_mask.clone()],
        BaseOp::Mult,
    )?;

    if alpha.0 == 0. {
        return Ok(positive);
    }

    if input_scale < &0 {
        return Err(CircuitError::NegativeScale("leaky_relu".to_string()));
    }

    let scale_constant = create_constant_tensor(F::from(2_i32.pow(*input_scale as u32) as u64), 1);

    let rescaled_positive = pairwise(config, region, &[positive, scale_constant], BaseOp::Mult)?;

    let neg_mask = not(config, region, &[relu_mask])?;

    let quantized_alpha = quantize_tensor(
        Tensor::from([alpha.0; 1].into_iter()),
        *input_scale,
        &crate::graph::Visibility::Fixed,
    )?;

    let alpha_tensor = create_constant_tensor(quantized_alpha[0], 1);

    let scaled_neg_mask = pairwise(config, region, &[neg_mask, alpha_tensor], BaseOp::Mult)?;

    let neg_part = pairwise(
        config,
        region,
        &[values[0].clone(), scaled_neg_mask],
        BaseOp::Mult,
    )?;

    pairwise(config, region, &[rescaled_positive, neg_part], BaseOp::Add)
}

fn multi_dim_axes_op<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    axes: &[usize],
    op: impl Fn(
            &BaseConfig<F>,
            &mut RegionCtx<F>,
            &[ValTensor<F>; 1],
        ) -> Result<ValTensor<F>, CircuitError>
        + Send
        + Sync,
) -> Result<ValTensor<F>, CircuitError> {
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

        op(config, region, &[sliced_input])
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

        Ok::<_, CircuitError>(topk_elem)
    })?;

    Ok(output.into())
}

/// softmax layout
pub(crate) fn softmax_axes<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    input_scale: utils::F32,
    output_scale: utils::F32,
    axes: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    let soft_max_at_scale = move |config: &BaseConfig<F>,
                                  region: &mut RegionCtx<F>,
                                  values: &[ValTensor<F>; 1]|
          -> Result<ValTensor<F>, CircuitError> {
        softmax(config, region, values, input_scale, output_scale)
    };

    let output = multi_dim_axes_op(config, region, values, axes, soft_max_at_scale)?;

    Ok(output)
}

/// percent func
pub(crate) fn percent<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    input_scale: utils::F32,
    output_scale: utils::F32,
) -> Result<ValTensor<F>, CircuitError> {
    let is_assigned = values[0].all_prev_assigned();
    let mut input = values[0].clone();
    if !is_assigned {
        input = region.assign(&config.custom_gates.inputs[0], &values[0])?;
        region.increment(input.len());
    };
    // sum of exps
    let denom = sum(config, region, &[input.clone()])?;

    let input_felt_scale = F::from(input_scale.0 as u64);
    let output_felt_scale = F::from(output_scale.0 as u64);
    let inv_denom = recip(
        config,
        region,
        &[denom],
        input_felt_scale,
        output_felt_scale,
    )?;
    // product of num * (1 / denom) = input_scale * output_scale
    pairwise(config, region, &[input, inv_denom], BaseOp::Mult)
}

/// Applies softmax
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::softmax;
/// use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[2, 2, 3, 2, 2, 0]),
///     &[2, 3],
/// ).unwrap());
/// let result = softmax::<Fp>(&dummy_config, &mut dummy_region, &[x], 128.0.into(), (128.0 * 128.0).into()).unwrap();
/// // doubles the scale of the input
/// let expected = Tensor::<IntegerRep>::new(Some(&[350012, 350012, 352768, 350012, 350012, 344500]), &[2, 3]).unwrap();
/// assert_eq!(result.int_evals().unwrap(), expected);
/// ```
pub fn softmax<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 1],
    input_scale: utils::F32,
    output_scale: utils::F32,
) -> Result<ValTensor<F>, CircuitError> {
    // get the max then subtract it
    let max_val = max(config, region, values)?;
    // rebase the input to 0
    let sub = pairwise(config, region, &[values[0].clone(), max_val], BaseOp::Sub)?;
    // elementwise exponential
    let ex = nonlinearity(
        config,
        region,
        &[sub],
        &LookupOp::Exp {
            scale: input_scale,
            base: E.into(),
        },
    )?;

    percent(config, region, &[ex.clone()], input_scale, output_scale)
}

/// Checks that the percent error between the expected public output and the actual output value
/// is within the percent error expressed by the `tol` input, where `tol == 1.0` means the percent
/// error tolerance is 1 percent.
/// # Examples
/// ```
/// use ezkl::tensor::Tensor;
/// use ezkl::fieldutils::IntegerRep;
/// use ezkl::circuit::ops::layouts::output;
///  use ezkl::tensor::val::ValTensor;
/// use halo2curves::bn256::Fr as Fp;
/// use ezkl::circuit::region::RegionCtx;
/// use ezkl::circuit::region::RegionSettings;
/// use ezkl::circuit::BaseConfig;
///
/// let dummy_config = BaseConfig::dummy(12, 2);
/// let mut dummy_region = RegionCtx::new_dummy(0,2,RegionSettings::all_true(65536, 4));
///
/// let x = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///     Some(&[100, 200, 300, 400, 500, 600]),
///     &[2, 3],
/// ).unwrap());
/// let y = ValTensor::from_integer_rep_tensor(Tensor::<IntegerRep>::new(
///    Some(&[101, 201, 302, 403, 503, 603]),
///   &[2, 3],
/// ).unwrap());
/// let result = output::<Fp>(&dummy_config, &mut dummy_region, &[x, y], false).unwrap();
/// ```
pub fn output<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
    region: &mut RegionCtx<F>,
    values: &[ValTensor<F>; 2],
    decomp: bool,
) -> Result<ValTensor<F>, CircuitError> {
    let mut values = [values[0].clone(), values[1].clone()];

    if !values[0].all_prev_assigned() {
        // range check the outputs
        values[0] = layouts::identity(config, region, &[values[0].clone()], decomp)?;
    }

    if !values[1].all_prev_assigned() {
        // range check the outputs
        values[1] = layouts::identity(config, region, &[values[1].clone()], decomp)?;
    }

    // regular equality constraint
    return enforce_equality(config, region, &[values[0].clone(), values[1].clone()]);
}
