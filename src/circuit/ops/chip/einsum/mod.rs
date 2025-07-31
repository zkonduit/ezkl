use crate::circuit::chip::einsum::analysis::{analyze_single_equation, EinsumAnalysis};
use crate::circuit::chip::einsum::contraction_planner::input_contractions;
use crate::circuit::layouts::{dot, multi_dot, prod};
use crate::circuit::region::RegionCtx;
use crate::circuit::{BaseConfig, CircuitError};
use crate::tensor::{TensorError, TensorType, ValTensor, VarTensor};
use halo2_proofs::plonk::{Challenge, ConstraintSystem, FirstPhase};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use std::collections::HashMap;
use std::marker::PhantomData;

pub mod analysis;
mod contraction_planner;

/// A struct representing the contractions for the einsums
#[derive(Clone, Debug, Default)]
pub struct Einsums<F: PrimeField + TensorType + PartialOrd> {
    pub inputs: Vec<VarTensor>,
    pub output: VarTensor,
    pub challenges: Vec<Challenge>,
    pub challenge_columns: Vec<VarTensor>,
    pub max_inputs: usize,
    pub max_challenges: usize,
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
            _marker: PhantomData,
        }
    }

    /// configure the columns based on universal Einsum analysis
    pub fn configure_universal(
        meta: &mut ConstraintSystem<F>,
        analysis: &EinsumAnalysis,
    ) -> Self {
        let mut inputs = vec![];
        for _ in 0..analysis.max_inputs {
            let max_input_size = analysis.max_input_size;
            // FIXME calculate the no. of rows needed
            let k = max_input_size.ilog2() + 1;
            let input_tensor = VarTensor::new_advice(meta, k as usize, 1, max_input_size);
            inputs.push(input_tensor);
        }

        let output = {
            let max_output_size = analysis.max_output_size;
            // FIXME calculate the no. of rows needed
            let k = max_output_size.ilog2() + 1;
            VarTensor::new_advice_in_second_phase(meta, k as usize, 1, max_output_size)
        };

        let mut challenge_columns = vec![];
        for _ in 0..analysis.max_output_axes {
            let longest_challenge_vector = analysis.longest_challenge_vector;
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

        let challenges: Vec<_> = (0..analysis.max_output_axes)
            .map(|_| meta.challenge_usable_after(FirstPhase))
            .collect();

        Self {
            inputs,
            output,
            challenges,
            challenge_columns,
            max_inputs: analysis.max_inputs,
            max_challenges: analysis.max_output_axes,
            _marker: PhantomData,
        }
    }

    // // FIXME: Dante, do we need this?
    // // We are currently not creating constraints for unused cells
    // fn assign_zero_tensor(
    //     &self,
    //     region: &mut RegionCtx<F>,
    //     tensor: &VarTensor,
    // ) -> Result<ValTensor<F>, CircuitError> {
    //     // | 0 | 1 | 2 | 3 | 4 |

    //     for column in tensor.inner.flatten() {
    //         region.assign_elem(input);
    //     }
    //     todo!()
    // }

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

        // Assign actual inputs
        for (i, input) in input_tensors.iter().enumerate() {
            region.assign(&self.inputs[i], input)?;
        }

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

        let squashed_output =
            self.assign_output(base_config, region, &output_tensor, &challenge_vectors.iter().collect_vec())?;

        // reorder the contraction of input tensors and contract
        let reordered_input_contractions = input_contractions(&equation).unwrap();
        assert_eq!(
            reordered_input_contractions.len(),
            equation_analysis.contraction_depth,
        );
        let mut tensors = vec![];
        for (i, input_tensor) in input_tensors.iter().chain(challenge_vectors.iter()).enumerate() {
            // region.set_offset(0);
            let witnessed_tensor = region.assign(&self.inputs[i], input_tensor)?;
            tensors.push(witnessed_tensor);
        }

        for contraction in reordered_input_contractions.iter() {
            // region.set_offset(0);
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
                base_config,
                region,
                flattened_input_tensors,
                dot_product_len,
                &output_dims,
            )?;

            tensors.push(contracted_output);
        }
        tensors.retain(|tensor| tensor.dims() == &[1]);

        // FIXME constrain this to be a product
        let tensors: ValTensor<F> = tensors
            .into_iter()
            .map(|t| t.get_inner_tensor().unwrap().get_scalar())
            .collect_vec()
            .into();
        let squashed_input = prod(base_config, region, &[&tensors])?;

        // region.set_offset(5);
        region.constrain_equal(&squashed_input, &squashed_output)
    }

    fn assign_output(
        &self,
        config: &BaseConfig<F>,
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
        for powers_of_challenge in challenge_vectors_assigned.into_iter().rev() {
            // region.set_offset(initial_offset);
            intermediate_values = assign_output_contraction(
                config,
                region,
                &intermediate_values,
                &powers_of_challenge,
            )?;
        }

        Ok(intermediate_values)
    }
}

fn assign_input_contraction<F: PrimeField + TensorType + PartialOrd + std::hash::Hash>(
    config: &BaseConfig<F>,
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
    config: &BaseConfig<F>,
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
