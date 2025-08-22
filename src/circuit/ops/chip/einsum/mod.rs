use crate::circuit::base::BaseOp;
use crate::circuit::chip::einsum::analysis::{analyze_single_equation, EinsumAnalysis};
use crate::circuit::einsum::layouts::{pairwise, sum};
use crate::circuit::einsum::reduction_planner::Reduction;
use crate::circuit::region::RegionCtx;
use crate::circuit::CircuitError;
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor, ValType, VarTensor};
use halo2_proofs::circuit::Value;
use halo2_proofs::plonk::{
    Challenge, ConstraintSystem, Constraints, Expression, FirstPhase, Selector,
};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use layouts::{dot, multi_dot, prod};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

///
pub mod analysis;
mod layouts;
mod reduction_planner;

/// A struct representing reductions for the einsums
#[derive(Clone, Debug, Default)]
pub struct Einsums<F: PrimeField + TensorType + PartialOrd> {
    /// custom gate to constrain tensor contractions
    custom_gate: EinsumOpConfig<F>,
    /// custom gate to constrain random linear combinations used by Freivalds' argument
    rlc_gates: Vec<RLCConfig<F>>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Einsums<F> {
    ///
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        let dummy_custom_gate = EinsumOpConfig {
            inputs: [
                dummy_var.clone(),
                dummy_var.clone(),
                dummy_var.clone(),
                dummy_var.clone(),
            ],
            output: dummy_var.clone(),
            selectors: BTreeMap::default(),
            _marker: PhantomData,
        };
        Self {
            custom_gate: dummy_custom_gate,
            rlc_gates: vec![],
        }
    }

    ///
    pub fn challenges(&self) -> Vec<Challenge> {
        self.rlc_gates
            .iter()
            .map(|gate| gate.challenge)
            .collect_vec()
    }

    /// Configure the columns based on universal Einsum analysis
    pub fn configure_universal(
        meta: &mut ConstraintSystem<F>,
        analysis: &EinsumAnalysis,
        num_inner_cols: usize,
        logrows: usize,
    ) -> Self {
        let capacity = analysis.reduction_length;
        let inputs: [VarTensor; 4] = [
            VarTensor::new_advice(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity),
            VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity),
        ];
        let output = VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity);
        let custom_gate = EinsumOpConfig::new(meta, &inputs, &output);

        let mut rlc_gates = vec![];
        for _ in 0..analysis.max_num_output_axes {
            let rlc_gate = RLCConfig::new(meta, &[inputs[0].clone(), inputs[2].clone()], &output);
            rlc_gates.push(rlc_gate);
        }

        Self {
            custom_gate,
            rlc_gates,
        }
    }

    ///
    pub fn assign_einsum(
        &self,
        region: &mut RegionCtx<F>,
        input_tensors: &[&ValTensor<F>],
        output_tensor: &ValTensor<F>,
        equation: &str,
    ) -> Result<(), CircuitError> {
        region.set_num_einsum_inner_cols(self.custom_gate.output.num_inner_cols());

        let (input_exprs, _) = equation.split_once("->").unwrap();
        let input_exprs = input_exprs.split(",").collect_vec();
        assert_eq!(input_exprs.len(), input_tensors.len());

        let mut input_tensors = input_tensors.iter().copied().cloned().collect_vec();
        let mut output_tensor = output_tensor.clone();

        // Remove trivial axes from tensors
        input_tensors
            .iter_mut()
            .map(|tensor| tensor.remove_trivial_axes())
            .collect::<Result<Vec<_>, TensorError>>()?;
        output_tensor.remove_trivial_axes()?;

        let mut input_axes_to_dim: HashMap<char, usize> = HashMap::new();
        input_exprs
            .iter()
            .zip(input_tensors.iter())
            .for_each(|(indices, tensor)| {
                let tensor_dim = tensor.dims();
                indices
                    .chars()
                    .zip(tensor_dim.iter())
                    .for_each(|(index, dim)| {
                        if let std::collections::hash_map::Entry::Vacant(e) =
                            input_axes_to_dim.entry(index)
                        {
                            e.insert(*dim);
                        }
                    });
            });

        let equation_analysis = analyze_single_equation(&equation, &input_axes_to_dim)?;
        let equation = equation_analysis.equation;

        let output_shape = equation_analysis
            .output_indices
            .iter()
            .map(|c| input_axes_to_dim.get(c).copied().unwrap())
            .collect_vec();
        let squashed_output = self.assign_output(region, &output_tensor, output_shape)?;

        // reorder the reduction of input tensors and reduce
        let reordered_input_reductions = reduction_planner::input_reductions(&equation).unwrap();
        let mut tensors = input_tensors;

        for reduction in reordered_input_reductions.iter() {
            let (input_expr, output_expr) = reduction.expression().split_once("->").unwrap();
            let input_exprs = input_expr.split(",").collect_vec();

            let remaining_axes = output_expr.chars().collect_vec();
            let mut remaining_axes_indices = remaining_axes
                .iter()
                .map(|c| 0..input_axes_to_dim[c])
                .multi_cartesian_product()
                .collect_vec();

            // Dummy value to ensure the for loop runs at least once
            if remaining_axes.is_empty() {
                remaining_axes_indices.push(vec![]);
            }

            let input_tensors = reduction
                .input_indices()
                .iter()
                .map(|idx| tensors[*idx].clone())
                .collect_vec();

            let mut flattened_input_tensors: Vec<Vec<ValTensor<F>>> =
                vec![vec![]; input_tensors.len()];
            for remaining_axes_indices in remaining_axes_indices {
                // corresponds to 1 running sum of input tensors
                for (i, (input_tensor, input_expr)) in
                    input_tensors.iter().zip(input_exprs.iter()).enumerate()
                {
                    let mut sliced_dim = vec![];
                    input_expr.chars().for_each(|axis| {
                        if let Some(pos) = remaining_axes.iter().position(|c| *c == axis) {
                            sliced_dim
                                .push(remaining_axes_indices[pos]..remaining_axes_indices[pos] + 1);
                        } else {
                            // common axis
                            sliced_dim.push(0..input_axes_to_dim[&axis]);
                        }
                    });
                    let mut sliced_input_tensor = input_tensor.get_slice(&sliced_dim)?;
                    sliced_input_tensor.flatten();
                    flattened_input_tensors[i].push(sliced_input_tensor);
                }
            }
            let flattened_input_tensors = flattened_input_tensors
                .into_iter()
                .map(|tensors| {
                    ValTensor::from(
                        tensors
                            .into_iter()
                            .flat_map(|t| t.get_inner_tensor().unwrap().clone().into_iter())
                            .collect_vec(),
                    )
                })
                .collect_vec();

            let output_dims = output_expr
                .chars()
                .map(|c| input_axes_to_dim[&c])
                .collect_vec();

            let contracted_output = match reduction {
                Reduction::RLC {
                    axis,
                    input_phase,
                    challenge_index,
                    ..
                } => {
                    assert_eq!(flattened_input_tensors.len(), 1);
                    let rlc_len = input_axes_to_dim[axis];
                    let mut result = self.rlc_gates[*challenge_index].assign_rlc(
                        region,
                        &flattened_input_tensors[0],
                        region.challenges()[*challenge_index],
                        rlc_len,
                        *input_phase,
                    )?;
                    result.reshape(&output_dims)?;
                    result
                }
                Reduction::Contraction {
                    axis, input_phases, ..
                } => match axis {
                    Some(axis) => {
                        let dot_product_len = input_axes_to_dim[axis];
                        assign_input_contraction(
                            &self.custom_gate,
                            region,
                            flattened_input_tensors,
                            dot_product_len,
                            &output_dims,
                            input_phases,
                        )?
                    }
                    None => {
                        let mut result = assign_pairwise_mult(
                            &self.custom_gate,
                            region,
                            flattened_input_tensors,
                            input_phases,
                        )?;
                        result.reshape(&output_dims)?;
                        result
                    }
                },
            };
            tensors.push(contracted_output);
        }
        tensors.retain(|tensor| tensor.is_singleton());

        let scalars: ValTensor<F> = tensors
            .into_iter()
            .map(|t| t.get_inner_tensor().unwrap().get_scalar())
            .collect_vec()
            .into();
        let squashed_input = prod(&self.custom_gate, region, &[&scalars], 1)?;

        region.constrain_equal(&squashed_input, &squashed_output)
    }

    fn assign_output(
        &self,
        region: &mut RegionCtx<F>,
        output: &ValTensor<F>,
        mut output_shape: Vec<usize>,
    ) -> Result<ValTensor<F>, CircuitError> {
        let mut intermediate_values = output.clone();

        let challenges = region
            .challenges()
            .iter()
            .take(output_shape.len())
            .copied()
            .collect_vec();
        // Intermediate values output from the previous reduction
        // Loop over the output axes
        for (idx, (rlc_config, challenge)) in self
            .rlc_gates
            .iter()
            .take(output_shape.len())
            .zip(challenges)
            .rev()
            .enumerate()
        {
            let rlc_len = output_shape.last().copied().unwrap();
            intermediate_values.flatten();
            let phase = if idx > 0 { 1 } else { 0 };
            intermediate_values =
                rlc_config.assign_rlc(region, &intermediate_values, challenge, rlc_len, phase)?;
            output_shape.pop();
        }

        Ok(intermediate_values)
    }
}

fn assign_pairwise_mult<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    flattened_tensors: Vec<ValTensor<F>>,
    input_phases: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    assert_eq!(flattened_tensors.len(), input_phases.len());
    let (result, _) = flattened_tensors
        .into_iter()
        .zip(input_phases.iter().cloned())
        .reduce(|(acc, acc_phase), (input, phase)| {
            (
                pairwise(
                    config,
                    region,
                    &[&acc, &input],
                    BaseOp::Mult,
                    &[acc_phase, phase],
                )
                .unwrap(),
                std::cmp::max(acc_phase, phase),
            )
        })
        .unwrap();
    Ok(result)
}

fn assign_input_contraction<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    flattened_tensors: Vec<ValTensor<F>>,
    dot_product_len: usize,
    output_shape: &[usize],
    input_phases: &[usize],
) -> Result<ValTensor<F>, CircuitError> {
    assert_eq!(flattened_tensors.len(), input_phases.len());
    let num_dot_products = output_shape.iter().product();
    let mut dot_product_results = vec![];
    for chunk_idx in 0..num_dot_products {
        let start = chunk_idx * dot_product_len;
        let tensors: Vec<_> = flattened_tensors
            .iter()
            .map(|tensor| tensor.get_slice(&[start..(start + dot_product_len)]))
            .collect::<Result<Vec<_>, TensorError>>()?;
        let result = if tensors.len() == 1 {
            sum(config, region, &[&tensors[0]], input_phases[0])?
        } else if tensors.len() == 2 {
            dot(
                config,
                region,
                &[&tensors[0], &tensors[1]],
                &[input_phases[0], input_phases[1]],
            )?
        } else {
            multi_dot(
                config,
                region,
                tensors.iter().collect_vec().as_slice(),
                input_phases,
            )?
        };
        dot_product_results.push(result.get_inner_tensor()?.get_scalar());
    }
    let mut tensor = ValTensor::from(dot_product_results);
    tensor.reshape(output_shape)?;
    Ok(tensor)
}

/// `EinsumOpConfig` is the custom gate used for einsum contraction operations
#[derive(Clone, Debug, Default)]
struct EinsumOpConfig<F: PrimeField + TensorType + PartialOrd> {
    // [phase 0, phase 0, phase 1, phase 1]
    inputs: [VarTensor; 4],
    // phase 1
    output: VarTensor,
    // (phase, BaseOp, block index, inner column index) -> selector
    selectors: BTreeMap<(usize, BaseOp, usize, usize), Selector>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> EinsumOpConfig<F> {
    fn new(meta: &mut ConstraintSystem<F>, inputs: &[VarTensor; 4], output: &VarTensor) -> Self {
        let mut selectors = BTreeMap::new();
        for phase in [0, 1] {
            for i in 0..output.num_blocks() {
                for j in 0..output.num_inner_cols() {
                    selectors.insert((phase, BaseOp::Mult, i, j), meta.selector());
                }
            }
        }

        for phase in [0, 1] {
            for i in 0..output.num_blocks() {
                selectors.insert((phase, BaseOp::DotInit, i, 0), meta.selector());
                selectors.insert((phase, BaseOp::Dot, i, 0), meta.selector());
                selectors.insert((phase, BaseOp::SumInit, i, 0), meta.selector());
                selectors.insert((phase, BaseOp::Sum, i, 0), meta.selector());
            }
        }
        selectors.insert(
            (1, BaseOp::CumProdInit, output.num_blocks() - 1, 0),
            meta.selector(),
        );
        selectors.insert(
            (1, BaseOp::CumProd, output.num_blocks() - 1, 0),
            meta.selector(),
        );
        for ((phase, base_op, block_idx, inner_col_idx), selector) in selectors.iter() {
            match base_op {
                BaseOp::Mult => {
                    meta.create_gate(base_op.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);

                        let zero = Expression::<F>::Constant(F::ZERO);
                        let mut qis = vec![zero; 4];
                        for (i, q_i) in qis
                            .iter_mut()
                            .enumerate()
                            .skip(*phase * 2)
                            .take(base_op.num_inputs())
                        {
                            *q_i = inputs[i]
                                .query_rng(meta, *block_idx, *inner_col_idx, 0, 1)
                                .expect("einsum op config: input query failed")[0]
                                .clone()
                        }
                        // Get output expressions for each input channel
                        let (rotation_offset, rng) = base_op.query_offset_rng();
                        let constraints = {
                            let expected_output: Tensor<Expression<F>> = output
                                .query_rng(meta, *block_idx, *inner_col_idx, rotation_offset, rng)
                                .expect("einsum op config: output query failed");

                            let res = base_op
                                .nonaccum_f((qis[2 * *phase].clone(), qis[2 * *phase + 1].clone()));
                            vec![expected_output[base_op.constraint_idx()].clone() - res]
                        };
                        Constraints::with_selector(selector, constraints)
                    });
                }
                _ => {
                    meta.create_gate(base_op.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);
                        let mut qis = vec![vec![]; 4];
                        for (i, q_i) in qis
                            .iter_mut()
                            .enumerate()
                            .skip(*phase * 2)
                            .take(base_op.num_inputs())
                        {
                            *q_i = inputs[i]
                                .query_whole_block(meta, *block_idx, 0, 1)
                                .expect("einsum op config: input query failed")
                                .into_iter()
                                .collect()
                        }
                        // Get output expressions for each input channel
                        let (rotation_offset, rng) = base_op.query_offset_rng();
                        let expected_output: Tensor<Expression<F>> = output
                            .query_rng(meta, *block_idx, 0, rotation_offset, rng)
                            .expect("einsum op config: output query failed");

                        let res = base_op.accum_f(
                            expected_output[0].clone(),
                            qis[2 * phase + 1].clone(),
                            qis[2 * *phase].clone(),
                        );
                        let constraints =
                            vec![expected_output[base_op.constraint_idx()].clone() - res];

                        Constraints::with_selector(selector, constraints)
                    });
                }
            }
        }

        Self {
            inputs: inputs.clone(),
            output: output.clone(),
            selectors,
            _marker: PhantomData,
        }
    }
}

/// `RLCConfig` is the custom gate used for random linear combination with the specific challenge
#[derive(Clone, Debug)]
struct RLCConfig<F: PrimeField + TensorType + PartialOrd> {
    pub challenge: Challenge,
    /// [phase 0, phase 1]
    pub inputs: [VarTensor; 2],
    pub output: VarTensor,
    /// (phase of input, block index) -> (init selector, acc selector)
    pub selectors: BTreeMap<(usize, usize), (Selector, Selector)>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> RLCConfig<F> {
    fn new(meta: &mut ConstraintSystem<F>, inputs: &[VarTensor; 2], output: &VarTensor) -> Self {
        let challenge = meta.challenge_usable_after(FirstPhase);

        let mut selectors = BTreeMap::new();
        for (phase, input) in inputs.iter().enumerate() {
            for block_idx in 0..input.num_blocks() {
                let selector = (meta.selector(), meta.selector());
                selectors.insert((phase, block_idx), selector);
            }
        }
        let block_width = output.num_inner_cols();
        let powers_of_challenge = (0..block_width)
            .scan(Expression::Constant(F::ONE), |r_power, _| {
                *r_power = r_power.clone() * challenge.expr();
                Some(r_power.clone())
            })
            .collect_vec();
        for ((phase, block_idx), (init_selector, acc_selector)) in selectors.iter() {
            meta.create_gate("init", |meta| {
                let selector = meta.query_selector(*init_selector);
                let input_exprs = inputs[*phase]
                    .query_whole_block(meta, *block_idx, 0, 1)
                    .expect("rlc config: input query failed")
                    .into_iter()
                    .collect();
                let constraints = {
                    let expected_output: Tensor<Expression<F>> = output
                        .query_rng(meta, *block_idx, 0, 0, 1)
                        .expect("rlc config: output query failed");

                    let res = BaseOp::Dot.accum_f(
                        Expression::Constant(F::ZERO),
                        powers_of_challenge.iter().cloned().rev().collect_vec(),
                        input_exprs,
                    );
                    vec![expected_output[0].clone() - res]
                };
                Constraints::with_selector(selector, constraints)
            });
            meta.create_gate("acc", |meta| {
                let selector = meta.query_selector(*acc_selector);
                let input_exprs = inputs[*phase]
                    .query_whole_block(meta, *block_idx, 0, 1)
                    .expect("rlc config: input query failed")
                    .into_iter()
                    .collect();
                let constraints = {
                    let expected_output: Tensor<Expression<F>> = output
                        .query_rng(meta, *block_idx, 0, -1, 2)
                        .expect("rlc config: output query failed");

                    let res = BaseOp::Dot.accum_f(
                        expected_output[0].clone() * powers_of_challenge.last().cloned().unwrap(),
                        powers_of_challenge.iter().cloned().rev().collect_vec(),
                        input_exprs,
                    );
                    vec![expected_output[1].clone() - res]
                };
                Constraints::with_selector(selector, constraints)
            });
        }
        Self {
            inputs: inputs.clone(),
            output: output.clone(),
            selectors,
            challenge,
            _marker: PhantomData,
        }
    }

    fn assign_rlc(
        &self,
        region: &mut RegionCtx<F>,
        flattened_input: &ValTensor<F>,
        challenge: Value<F>,
        rlc_len: usize,
        phase: usize,
    ) -> Result<ValTensor<F>, CircuitError> {
        region.flush_einsum()?;
        let block_width = self.output.num_inner_cols();
        let powers_of_challenge = (0..block_width)
            .scan(Value::known(F::ONE), |challenge_power, _| {
                *challenge_power = challenge_power.clone() * challenge;
                Some(challenge_power.clone())
            })
            .collect_vec();
        let mut rlc_results: Vec<ValType<F>> = vec![];
        for tensor in flattened_input.get_inner_tensor()?.chunks_exact(rlc_len) {
            let running_sums = tensor
                .iter()
                .chunks(block_width)
                .into_iter()
                .scan(Value::known(F::ZERO), |state, val| {
                    let curr_sum: Value<F> = val
                        .into_iter()
                        .zip(powers_of_challenge.iter().rev())
                        .map(|(v, c_power)| {
                            c_power.and_then(|c_power| {
                                Value::known(c_power * v.get_felt_eval().unwrap())
                            })
                        })
                        .reduce(|acc, v| acc + v)
                        .unwrap();
                    *state = *state * powers_of_challenge.last().unwrap() + curr_sum;
                    Some(*state)
                })
                .collect_vec();

            let assigned_len = {
                let mut input: ValTensor<F> = tensor.iter().collect_vec().into();
                input.pad_to_zero_rem(block_width, ValType::Value(Value::known(F::ZERO)))?;
                let (_, len) = region
                    .assign_einsum_with_duplication_unconstrained(&self.inputs[phase], &input)?;
                len
            };
            let (assigned_output, assigned_output_len) = {
                let running_sums = running_sums.into_iter().map(ValType::from).collect_vec();
                region.assign_einsum_with_duplication_constrained(
                    &self.output,
                    &running_sums.into(),
                    &crate::circuit::CheckMode::UNSAFE,
                )?
            };

            (0..assigned_output_len)
                .map(|i| {
                    let (block_idx, _, z) = self
                        .output
                        .cartesian_coord(region.einsum_col_coord() + i * block_width);
                    if z == 0 && i > 0 {
                        return Ok(());
                    }
                    let selector = if i == 0 {
                        self.selectors
                            .get(&(phase, block_idx))
                            .map(|(init, _)| init)
                    } else {
                        self.selectors.get(&(phase, block_idx)).map(|(_, acc)| acc)
                    };
                    region.enable(selector, z)?;
                    Ok(())
                })
                .collect::<Result<Vec<_>, CircuitError>>()?;
            rlc_results.push(assigned_output.last()?.get_inner_tensor()?.get_scalar());

            region.increment_einsum_col_coord(assigned_len);
        }
        Ok(rlc_results.into())
    }
}
