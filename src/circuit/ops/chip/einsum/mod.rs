use crate::circuit::base::BaseOp;
use crate::circuit::chip::einsum::analysis::{analyze_single_equation, EinsumAnalysis};
use crate::circuit::layouts::prod;
use crate::circuit::region::RegionCtx;
use crate::circuit::{BaseConfig, CircuitError};
use crate::tensor::{Tensor, TensorError, TensorType, ValTensor, VarTensor};
use halo2_proofs::plonk::{Challenge, ConstraintSystem, Constraints, Expression, FirstPhase, Selector};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use layouts::{dot, multi_dot};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

///
pub mod analysis;
mod contraction_planner;
mod layouts;

/// A struct representing the contractions for the einsums
#[derive(Clone, Debug, Default)]
pub struct Einsums<F: PrimeField + TensorType + PartialOrd> {
    /// input tensors
    pub inputs: Vec<VarTensor>,
    /// output tensor
    pub output: VarTensor,
    /// challenges
    pub challenges: Vec<Challenge>,
    ///
    pub challenge_columns: Vec<VarTensor>,
    /// max number of input tensors
    pub max_inputs: usize,
    /// max number of output tensor axes
    pub max_challenges: usize,
    input_contractions: Vec<EinsumOpConfig<F>>,
    output_contractions: Vec<EinsumOpConfig<F>>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> Einsums<F> {
    ///
    pub fn dummy(col_size: usize, num_inner_cols: usize) -> Self {
        let dummy_var = VarTensor::dummy(col_size, num_inner_cols);
        Self {
            inputs: vec![dummy_var.clone(), dummy_var.clone()],
            output: dummy_var.clone(),
            challenges: vec![],
            challenge_columns: vec![],
            max_inputs: 0,
            max_challenges: 0,
            input_contractions: vec![],
            output_contractions: vec![],
            _marker: PhantomData,
        }
    }

    /// configure the columns based on universal Einsum analysis
    pub fn configure_universal(meta: &mut ConstraintSystem<F>, analysis: &EinsumAnalysis) -> Self {
        let mut inputs = vec![];
        for _ in 0..analysis.max_num_inputs {
            let max_input_size = analysis.max_input_size;
            // FIXME calculate the no. of rows needed
            let k = max_input_size.ilog2() + 1;
            let input_tensor = VarTensor::new_advice(meta, k as usize, 1, max_input_size);
            inputs.push(input_tensor);
        }

        let output = {
            // FIXME
            let max_output_size = analysis.max_output_size;
            // FIXME calculate the no. of rows needed
            let k = max_output_size.ilog2() + 1;
            VarTensor::new_advice_in_second_phase(meta, k as usize, 1, max_output_size)
        };

        let mut challenge_columns = vec![];
        for _ in 0..analysis.max_num_output_axes {
            let longest_challenge_vector = analysis.max_output_size;
            // FIXME calculate the no. of rows needed
            let k = longest_challenge_vector.ilog2() + 1;
            let challenge_tensor = VarTensor::new_advice_in_second_phase(
                meta,
                k as usize,
                1,
                longest_challenge_vector,
            );
            challenge_columns.push(challenge_tensor);
        }

        inputs.extend_from_slice(&challenge_columns);

        let challenges: Vec<_> = (0..analysis.max_num_output_axes)
            .map(|_| meta.challenge_usable_after(FirstPhase))
            .collect();

        let mut output_contractions = vec![];
        for _ in 0..analysis.max_num_output_axes {
            // FIXME
            let num_inner_cols = 1;
            let logrows = analysis.max_output_size.ilog2() + 1;
            let capacity = analysis.max_output_size;
            output_contractions.push(EinsumOpConfig::new(
                meta,
                num_inner_cols,
                logrows as usize,
                capacity,
            ));
        }

        let mut input_contractions = vec![];
        for _ in 0..analysis.max_contraction_depth {
            // FIXME
            let num_inner_cols = 1;
            let logrows = analysis.max_input_size.ilog2() + 1;
            let capacity = analysis.max_input_size;
            input_contractions.push(EinsumOpConfig::new(
                meta,
                num_inner_cols,
                logrows as usize,
                capacity,
            ));
        }

        Self {
            inputs,
            output,
            challenges,
            challenge_columns,
            max_inputs: analysis.max_num_inputs,
            max_challenges: analysis.max_num_output_axes,
            output_contractions,
            input_contractions,
            _marker: PhantomData,
        }
    }

    ///
    pub fn assign_with_padding(
        &self,
        base_config: &BaseConfig<F>,
        region: &mut RegionCtx<F>,
        input_tensors: &[&ValTensor<F>],
        output_tensor: &ValTensor<F>,
        equation: &str,
    ) -> Result<(), CircuitError> {
        let (input_exprs, _) = equation.split_once("->").unwrap();
        let input_exprs = input_exprs.split(",").collect_vec();
        assert_eq!(input_exprs.len(), input_tensors.len());

        // Remove trivial axes from tensors
        let mut input_tensors = input_tensors.iter().copied().cloned().collect_vec();
        let mut output_tensor = output_tensor.clone();

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

        // // Zero-pad unused input columns
        // for i in input_tensors.len()..self.inputs.len() {
        //     self.assign_zero_tensor(region, &self.inputs[i])?;
        // }

        let challenge_vectors: Vec<ValTensor<F>> = region
            .challenges()
            .iter()
            .cloned()
            .take(equation_analysis.output_axes)
            .collect();

        let squashed_output = self.assign_output(
            region,
            &output_tensor,
            &challenge_vectors.iter().collect_vec(),
        )?;

        // reorder the contraction of input tensors and contract
        let reordered_input_contractions = contraction_planner::input_contractions(&equation).unwrap();
        assert_eq!(
            reordered_input_contractions.len(),
            equation_analysis.contraction_depth,
        );
        let mut tensors = input_tensors;
        tensors.extend(challenge_vectors);

        for (contraction, config) in reordered_input_contractions.iter().zip(self.input_contractions.iter()) {
            // region_ctx.set_offset(0);
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
                .input_indices()
                .into_iter()
                .map(|idx| tensors[idx].clone())
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
                config,
                region,
                flattened_input_tensors,
                dot_product_len,
                &output_dims,
            )?;

            tensors.push(contracted_output);
        }
        tensors.retain(|tensor| tensor.is_singleton());

        // FIXME constrain this to be a product
        let tensors: ValTensor<F> = tensors
            .into_iter()
            .map(|t| t.get_inner_tensor().unwrap().get_scalar())
            .collect_vec()
            .into();
        let squashed_input = prod(base_config, region, &[&tensors])?;

        region.constrain_equal(&squashed_input, &squashed_output)
    }

    fn assign_output(
        &self,
        region: &mut RegionCtx<F>,
        output: &ValTensor<F>,
        challenge_vectors: &[&ValTensor<F>],
    ) -> Result<ValTensor<F>, CircuitError> {
        // let initial_offset = region.offset();
        // Witness challenge vectors
        // FIXME constrain these to be a well-constructed power series
        let mut challenge_vectors_assigned = vec![];
        for (i, challenge_vector) in challenge_vectors.iter().enumerate() {
            challenge_vectors_assigned
                .push(region.assign(&self.challenge_columns[i], challenge_vector)?);
        }

        // Initialise `intermediate_values` to the original output tensor
        // region.set_offset(initial_offset);
        // Witness flattened output
        let mut intermediate_values = region.assign(&self.output, output)?;

        // Intermediate values output from the previous contraction
        // Loop over the output axes
        for (idx, powers_of_challenge) in challenge_vectors_assigned.into_iter().rev().enumerate() {
            // region.set_offset(initial_offset);
            intermediate_values = assign_output_contraction(
                &self.output_contractions[idx],
                region,
                &intermediate_values,
                &powers_of_challenge,
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
) -> Result<ValTensor<F>, CircuitError> {
    let num_dot_products = output_shape.iter().product();
    // klm
    // Contract along k, using lm dot products of length k
    let mut dot_product_results = vec![];
    for chunk_idx in 0..num_dot_products {
        let start = chunk_idx * dot_product_len;
        let tensors: Vec<_> = flattened_tensors
            .iter()
            .map(|tensor| tensor.get_slice(&[start..(start + dot_product_len)]))
            .collect::<Result<Vec<_>, TensorError>>()?;
        let result = multi_dot(config, region, tensors.iter().collect_vec().as_slice())?
            .get_inner_tensor()?
            .get_scalar();
        dot_product_results.push(result);
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
) -> Result<ValTensor<F>, CircuitError> {
    let dot_product_len = powers_of_challenge.len();
    // Split tensor and challenge vector into dot products
    let mut dot_product_results = vec![];
    for tensor in tensor.get_inner_tensor()?.chunks_exact(dot_product_len) {
        let tensor = ValTensor::from(tensor.to_vec());
        let result = dot(config, region, &[&tensor, powers_of_challenge])?
            .get_inner_tensor()?
            .get_scalar();
        dot_product_results.push(result);
    }

    Ok(ValTensor::from(dot_product_results))
}

/// `EinsumOpConfig` is the custom gate used for einsum contraction operations
#[derive(Clone, Debug, Default)]
struct EinsumOpConfig<F: PrimeField + TensorType + PartialOrd> {
    inputs: [VarTensor; 2],
    output: VarTensor,
    selectors: BTreeMap<(BaseOp, usize, usize), Selector>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> EinsumOpConfig<F> {
    fn new(
        meta: &mut ConstraintSystem<F>,
        num_inner_cols: usize,
        logrows: usize,
        capacity: usize,
    ) -> Self {
        // TODO optimise choice of advice columns globally
        let inputs = [(); 2]
            .map(|_| VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity));

        let output =
            VarTensor::new_advice_in_second_phase(meta, logrows, num_inner_cols, capacity);

        let mut selectors = BTreeMap::new();

        // Required `BaseOp`s in Einsum contraction
        for i in 0..output.num_blocks() {
            for j in 0..output.num_inner_cols() {
                selectors.insert((BaseOp::Mult, i, j), meta.selector());
            }
        }
        for i in 0..output.num_blocks() {
            selectors.insert((BaseOp::DotInit, i, 0), meta.selector());
            selectors.insert((BaseOp::Dot, i, 0), meta.selector());
            selectors.insert((BaseOp::Sum, i, 0), meta.selector());
            selectors.insert((BaseOp::SumInit, i, 0), meta.selector());
        }

        for ((base_op, block_idx, inner_col_idx), selector) in selectors.iter() {
            match base_op {
                BaseOp::Mult => {
                    meta.create_gate(base_op.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);

                        let zero = Expression::<F>::Constant(F::ZERO);
                        let mut qis = vec![zero; 2];
                        for (i, q_i) in qis
                            .iter_mut()
                            .enumerate()
                            .take(2)
                            .skip(2 - base_op.num_inputs())
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
        
                            let res = base_op.nonaccum_f((qis[0].clone(), qis[1].clone()));
                            vec![expected_output[base_op.constraint_idx()].clone() - res]
                        };
        
                        Constraints::with_selector(selector, constraints)
                    });
                },
                _ => {
                    meta.create_gate(base_op.as_str(), |meta| {
                        let selector = meta.query_selector(*selector);
                        let mut qis = vec![vec![]; 2];
                        for (i, q_i) in qis
                            .iter_mut()
                            .enumerate()
                            .take(2)
                            .skip(2 - base_op.num_inputs())
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
        
                        let res = base_op.accum_f(expected_output[0].clone(), qis[0].clone(), qis[1].clone());
                        let constraints = vec![expected_output[base_op.constraint_idx()].clone() - res];
        
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
