use crate::circuit::base::BaseOp;
use crate::circuit::chip::einsum::analysis::{analyze_single_equation, EinsumAnalysis};
use crate::circuit::einsum::layouts::sum;
use crate::circuit::region::RegionCtx;
use crate::circuit::CircuitError;
use crate::tensor::ops::accumulated;
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor, VarTensor};
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
mod contraction_planner;
mod layouts;

/// A struct representing the contractions for the einsums
#[derive(Clone, Debug, Default)]
pub struct Einsums<F: PrimeField + TensorType + PartialOrd> {
    /// challenges
    pub challenges: Vec<Challenge>,
    /// Witness powers of challenges
    /// TODO : bake challenges into constraints
    pub challenge_columns: Vec<VarTensor>,
    ///
    pub power_series_selectors: Vec<Selector>,
    /// custom gate to constrain tensor contractions
    custom_gate: EinsumOpConfig<F>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Einsums<F> {
    ///
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        let dummy_custom_gate = EinsumOpConfig {
            inputs: [dummy_var.clone(), dummy_var.clone(), dummy_var.clone()],
            output: dummy_var.clone(),
            selectors: BTreeMap::default(),
            _marker: PhantomData,
        };
        Self {
            challenges: vec![],
            challenge_columns: vec![],
            power_series_selectors: vec![],
            custom_gate: dummy_custom_gate,
        }
    }

    /// Configure the columns based on universal Einsum analysis
    pub fn configure_universal(
        meta: &mut ConstraintSystem<F>,
        analysis: &EinsumAnalysis,
        num_inner_cols: usize,
        logrows: usize,
    ) -> Self {
        let challenges: Vec<_> = (0..analysis.max_num_output_axes)
            .map(|_| meta.challenge_usable_after(FirstPhase))
            .collect();

        let mut challenge_columns = vec![];
        let mut power_series_selectors = vec![];
        for challenge in challenges.iter() {
            let selector = meta.selector();
            let challenge_tensor = VarTensor::new_advice_in_second_phase(
                meta,
                logrows,
                1,
                analysis.longest_challenge_vector,
            );
            meta.create_gate("power series", |meta| {
                let selector = meta.query_selector(selector);
                let cells = challenge_tensor.query_rng(meta, 0, 0, -1, 2).unwrap();
                let [prev, curr] = cells.into_iter().collect_vec().try_into().unwrap();

                Constraints::with_selector(selector, vec![curr - prev * challenge.expr()])
            });
            challenge_columns.push(challenge_tensor);
            power_series_selectors.push(selector);
        }

        // tentatively add the space for witnessing powers of challenges
        let capacity = analysis.contraction_length
            + (1..=analysis.longest_challenge_vector)
                .map(|n| n + (num_inner_cols - (n % num_inner_cols)))
                .sum::<usize>()
                * analysis.max_num_output_axes;
        let custom_gate = EinsumOpConfig::new(meta, num_inner_cols, logrows, capacity);

        Self {
            challenges,
            challenge_columns,
            power_series_selectors,
            custom_gate,
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

        let challenges: Vec<Value<F>> = region
            .challenges()
            .iter()
            .cloned()
            .take(equation_analysis.num_output_axes)
            .collect();
        let mut challenge_tensors = vec![];
        for (idx, (challenge, output_axis)) in challenges
            .into_iter()
            .zip(equation_analysis.output_indices.iter())
            .enumerate()
        {
            let power = *input_axes_to_dim.get(output_axis).unwrap();
            let value = accumulated::prod(&vec![challenge.clone(); power].into_iter().into(), 1)?;
            let challenge_tensor =
                region.assign_einsum(&self.challenge_columns[idx], &value.into())?;
            // enable selector
            for i in 1..power {
                let (block, column, row) = self.challenge_columns[idx].cartesian_coord(region.einsum_col_coord() + i);
                assert!(block == 0 && column == 0);
                assert!(region.einsum_col_coord() + i == row);
                region.enable(Some(&self.power_series_selectors[idx]), row)?;
            }
            challenge_tensors.push(ValTensor::from(challenge_tensor));
        }

        let squashed_output = self.assign_output(
            region,
            &output_tensor,
            &challenge_tensors.iter().collect_vec(),
        )?;

        // reorder the contraction of input tensors and contract
        let reordered_input_contractions =
            contraction_planner::input_contractions(&equation).unwrap();
        let mut tensors = input_tensors;
        tensors.extend(challenge_tensors);

        for contraction in reordered_input_contractions.iter() {
            let (input_expr, output_expr) = contraction.expression.split_once("->").unwrap();
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

            let input_tensors = contraction
                .input_indices
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
            let dot_product_len = input_axes_to_dim[&contraction.axis];
            let output_dims = output_expr
                .chars()
                .map(|c| input_axes_to_dim[&c])
                .collect_vec();
            let contracted_output = assign_input_contraction(
                &self.custom_gate,
                region,
                flattened_input_tensors,
                dot_product_len,
                &output_dims,
                &contraction.input_phases,
            )?;

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
        challenge_vectors: &[&ValTensor<F>],
    ) -> Result<ValTensor<F>, CircuitError> {
        let mut intermediate_values = output.clone();

        // Intermediate values output from the previous contraction
        // Loop over the output axes
        for (idx, powers_of_challenge) in challenge_vectors.into_iter().rev().enumerate() {
            let phases = if idx > 0 { [1, 1] } else { [0, 1] };
            intermediate_values = assign_output_contraction(
                &self.custom_gate,
                region,
                &intermediate_values,
                &powers_of_challenge,
                &phases,
            )?;
        }

        Ok(intermediate_values)
    }
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
            let (tensors, phases): (Vec<_>, Vec<usize>) = tensors
                .into_iter()
                .zip(input_phases.iter().cloned())
                // reorder the tensors to ensure there is no contraction between phase 0 tensors
                .sorted_by(|(_, phase0), (_, phase1)| Ord::cmp(&phase1, &phase0))
                .unzip();
            multi_dot(
                config,
                region,
                tensors.iter().collect_vec().as_slice(),
                &phases,
            )?
        };
        dot_product_results.push(result.get_inner_tensor()?.get_scalar());
    }
    let mut tensor = ValTensor::from(dot_product_results);
    tensor.reshape(output_shape)?;
    Ok(tensor)
}

fn assign_output_contraction<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &EinsumOpConfig<F>,
    region: &mut RegionCtx<F>,
    tensor: &ValTensor<F>,
    powers_of_challenge: &ValTensor<F>,
    phases: &[usize; 2],
) -> Result<ValTensor<F>, CircuitError> {
    let dot_product_len = powers_of_challenge.len();
    // Split tensor and challenge vector into dot products
    let mut dot_product_results = vec![];
    for tensor in tensor.get_inner_tensor()?.chunks_exact(dot_product_len) {
        let tensor = ValTensor::from(tensor.to_vec());
        let result = dot(config, region, &[&tensor, powers_of_challenge], phases)?
            .get_inner_tensor()?
            .get_scalar();
        dot_product_results.push(result);
    }

    Ok(ValTensor::from(dot_product_results))
}

/// `EinsumOpConfig` is the custom gate used for einsum contraction operations
#[derive(Clone, Debug, Default)]
struct EinsumOpConfig<F: PrimeField + TensorType + PartialOrd> {
    // [phase 0, phase 1, phase 1]
    inputs: [VarTensor; 3],
    // phase 1
    output: VarTensor,
    // (min phase, BaseOp, block index, inner column index) -> selector
    selectors: BTreeMap<(usize, BaseOp, usize, usize), Selector>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> EinsumOpConfig<F> {
    fn new(
        meta: &mut ConstraintSystem<F>,
        num_inner_cols: usize,
        logrows: usize,
        capacity: usize,
    ) -> Self {
        let inputs: [VarTensor; 3] = (0..3)
            .map(|i| {
                if i == 0 {
                    VarTensor::new_advice(meta, logrows, num_inner_cols, capacity)
                } else {
                    VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity)
                }
            })
            .collect_vec()
            .try_into()
            .unwrap();

        let output = VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity);

        let mut selectors = BTreeMap::new();

        // Required `BaseOp`s in Einsum contraction
        for min_phase in [0, 1] {
            for i in 0..output.num_blocks() {
                for j in 0..output.num_inner_cols() {
                    selectors.insert((min_phase, BaseOp::Mult, i, j), meta.selector());
                }
            }
        }
        for min_phase in [0, 1] {
            for i in 0..output.num_blocks() {
                selectors.insert((min_phase, BaseOp::CumProd, i, 0), meta.selector());
                selectors.insert((min_phase, BaseOp::CumProdInit, i, 0), meta.selector());
                selectors.insert((min_phase, BaseOp::DotInit, i, 0), meta.selector());
                selectors.insert((min_phase, BaseOp::Dot, i, 0), meta.selector());
                selectors.insert((min_phase, BaseOp::Sum, i, 0), meta.selector());
                selectors.insert((min_phase, BaseOp::SumInit, i, 0), meta.selector());
            }
        }
        for ((min_phase, base_op, block_idx, inner_col_idx), selector) in selectors.iter() {
            match base_op {
                BaseOp::Mult => {
                    meta.create_gate(base_op.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);

                        let zero = Expression::<F>::Constant(F::ZERO);
                        let mut qis = vec![zero; 3];
                        for (i, q_i) in qis
                            .iter_mut()
                            .enumerate()
                            .skip(*min_phase)
                            .take(base_op.num_inputs())
                        {
                            *q_i = inputs[i]
                                .query_rng(meta, *block_idx, *inner_col_idx, 0, 1)
                                .expect("non accum: input query failed")[0]
                                .clone()
                        }
                        // Get output expressions for each input channel
                        let (rotation_offset, rng) = base_op.query_offset_rng();
                        let constraints = {
                            let expected_output: Tensor<Expression<F>> = output
                                .query_rng(meta, *block_idx, *inner_col_idx, rotation_offset, rng)
                                .expect("non accum: output query failed");

                            let res = base_op
                                .nonaccum_f((qis[*min_phase].clone(), qis[min_phase + 1].clone()));
                            vec![expected_output[base_op.constraint_idx()].clone() - res]
                        };
                        Constraints::with_selector(selector, constraints)
                    });
                }
                _ => {
                    meta.create_gate(base_op.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);
                        let mut qis = vec![vec![]; 3];
                        for (i, q_i) in qis
                            .iter_mut()
                            .enumerate()
                            .skip(*min_phase)
                            .take(base_op.num_inputs())
                        {
                            *q_i = inputs[i]
                                .query_whole_block(meta, *block_idx, 0, 1)
                                .expect("accum: input query failed")
                                .into_iter()
                                .collect()
                        }
                        // Get output expressions for each input channel
                        let (rotation_offset, rng) = base_op.query_offset_rng();
                        let expected_output: Tensor<Expression<F>> = output
                            .query_rng(meta, *block_idx, 0, rotation_offset, rng)
                            .expect("accum: output query failed");

                        let res = base_op.accum_f(
                            expected_output[0].clone(),
                            qis[min_phase + 1].clone(),
                            qis[*min_phase].clone(),
                        );
                        let constraints =
                            vec![expected_output[base_op.constraint_idx()].clone() - res];

                        Constraints::with_selector(selector, constraints)
                    });
                }
            }
        }

        Self {
            inputs,
            output,
            selectors,
            _marker: PhantomData,
        }
    }
}
