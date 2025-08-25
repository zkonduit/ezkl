use std::collections::HashMap;

use itertools::Itertools;

use crate::circuit::{
    einsum::reduction_planner::{self, Reduction},
    CircuitError,
};

///
#[derive(Debug, Clone)]
pub struct EinsumAnalysis {
    /// max size of input tensors
    pub max_input_size: usize,
    /// max size of output tensors
    pub max_output_size: usize,
    /// max number of input tensors
    pub max_num_inputs: usize,
    /// max number of output axes
    pub max_num_output_axes: usize,
    ///
    pub longest_challenge_vector: usize,
    ///
    pub reduction_length: usize,
}

///
#[derive(Debug, Clone)]
pub struct SingleEquationAnalysis {
    ///
    pub equation: String,
    ///
    pub num_inputs: usize,
    ///
    pub max_input_size: usize,
    ///
    pub output_size: usize,
    ///
    pub num_output_axes: usize,
    ///
    pub output_indices: Vec<char>,
    ///
    pub longest_challenge_vector: usize,
    /// the length of dot product to compute all the reductions
    pub reduction_length: usize,
}

///
pub fn analyze_einsum_usage(
    equations: &HashMap<String, HashMap<char, usize>>,
) -> Result<EinsumAnalysis, CircuitError> {
    let mut max_num_inputs = 0;
    let mut max_input_size = 0;
    let mut max_output_size = 0;
    let mut max_num_output_axes = 0;
    let mut longest_challenge_vector = 0;
    let mut reduction_length = 0;

    for (equation, input_axes_to_dim) in equations.iter() {
        let analysis = analyze_single_equation(equation, input_axes_to_dim)?;
        max_input_size = max_input_size.max(analysis.max_input_size);
        longest_challenge_vector = longest_challenge_vector.max(analysis.longest_challenge_vector);
        max_output_size = max_output_size.max(analysis.output_size);
        max_num_inputs = max_num_inputs.max(analysis.num_inputs);
        max_num_output_axes = max_num_output_axes.max(analysis.num_output_axes);
        reduction_length += analysis.reduction_length;
    }

    Ok(EinsumAnalysis {
        max_input_size,
        longest_challenge_vector,
        max_output_size,
        max_num_inputs,
        max_num_output_axes,
        reduction_length,
    })
}

///
pub fn analyze_single_equation(
    equation: &str,
    input_axes_to_dim: &HashMap<char, usize>,
) -> Result<SingleEquationAnalysis, CircuitError> {
    // Sanitise equation to remove trivial axes
    let equation = {
        let (inputs_str, output_str) = equation.split_once("->").unwrap();
        let input_equations: Vec<&str> = inputs_str.split(',').collect();

        let inputs: Vec<String> = input_equations
            .iter()
            .map(|input| {
                input
                    .chars()
                    .filter(|char| {
                        input_axes_to_dim.get(char).is_some() && *input_axes_to_dim.get(char).unwrap() > 1
                    })
                    .collect()
            })
            .collect();

        let output = output_str
            .chars()
            .filter(|c| {
                input_axes_to_dim.get(c).is_some() && *input_axes_to_dim.get(c).unwrap() > 1
            })
            .collect();

        [inputs.join(","), output].join("->")
    };

    let (inputs_str, output_str) = equation.split_once("->").unwrap();
    let input_equations: Vec<&str> = inputs_str.split(',').collect();

    let max_input_size = input_equations
        .iter()
        .map(|eqn| {
            eqn.chars()
                .map(|c| input_axes_to_dim.get(&c).unwrap())
                .product()
        })
        .max()
        .unwrap();

    let output_indices: Vec<char> = output_str.chars().collect();
    let output_dims = output_indices
        .iter()
        .map(|c| input_axes_to_dim.get(&c).unwrap());
    let output_size = output_dims.clone().product();
    let longest_challenge_vector = *output_dims.clone().max().unwrap();

    let output_reduction_length = {
        let mut output_dims = output_dims.rev().cloned().collect_vec();
        let mut total_length = 0;
        for _ in 0..output_dims.len() {
            let dot_product_len = output_dims.remove(0);
            let num_dot_products: usize = output_dims.iter().product();
            total_length += dot_product_len * num_dot_products;
        }
        total_length
    };

    let input_reductions_length = {
        let input_reductions = reduction_planner::input_reductions(&equation)?;
        input_reductions
            .into_iter()
            .map(|reduction| {
                let (_, output_expr) = reduction.expression().split_once("->").unwrap();
                let num_inputs = reduction.input_indices().len();
                let dot_product_len = match reduction {
                    Reduction::RLC { axis, .. } => *input_axes_to_dim.get(&axis).unwrap(),
                    Reduction::Contraction { axis, .. } => *axis
                        .and_then(|axis| input_axes_to_dim.get(&axis))
                        .unwrap_or(&1),
                };
                let num_dot_products: usize = output_expr
                    .chars()
                    .map(|c| input_axes_to_dim.get(&c).unwrap())
                    .product();
                // since `multi_dot` does pairwise mult between input pairs and final summation
                if num_inputs <= 2 {
                    num_dot_products * dot_product_len
                } else {
                    num_dot_products * (dot_product_len * num_inputs)
                }
            })
            .sum::<usize>()
    };

    Ok(SingleEquationAnalysis {
        output_size,
        longest_challenge_vector,
        max_input_size,
        equation: equation.to_string(),
        num_inputs: input_equations.len(),
        num_output_axes: output_indices.len(),
        output_indices,
        reduction_length: output_reduction_length + input_reductions_length,
    })
}
