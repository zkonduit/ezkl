use std::collections::HashMap;

use crate::circuit::{chip::einsum::contraction_planner, CircuitError};

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
    pub max_contraction_depth: usize,
    ///
    pub universal_gate_size: usize,
    ///
    pub total_challenge_columns: usize,
    ///
    pub longest_challenge_vector: usize,
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
    pub output_axes: usize,
    ///
    pub contraction_depth: usize,
    ///
    pub common_indices: Vec<char>,
    ///
    pub output_indices: Vec<char>,
    ///
    pub longest_challenge_vector: usize,
}

///
pub fn analyze_einsum_usage(
    equations: &HashMap<String, HashMap<char, usize>>,
) -> Result<EinsumAnalysis, CircuitError> {
    let mut max_num_inputs = 0;
    let mut max_input_size = 0;
    let mut max_output_size = 0;
    let mut max_num_output_axes = 0;
    let mut max_contraction_depth = 0;
    let mut longest_challenge_vector = 0;

    for (equation, input_axes_to_dim) in equations.iter() {
        let analysis = analyze_single_equation(equation, input_axes_to_dim)?;
        max_input_size = max_input_size.max(analysis.max_input_size);
        longest_challenge_vector = longest_challenge_vector.max(analysis.longest_challenge_vector);
        max_output_size = max_output_size.max(analysis.output_size);
        max_num_inputs = max_num_inputs.max(analysis.num_inputs);
        max_num_output_axes = max_num_output_axes.max(analysis.output_axes);
        max_contraction_depth = max_contraction_depth.max(analysis.contraction_depth);
    }

    Ok(EinsumAnalysis {
        max_input_size,
        longest_challenge_vector,
        max_output_size,
        max_num_inputs,
        max_num_output_axes,
        max_contraction_depth,
        universal_gate_size: max_num_inputs + max_num_output_axes, // For padding with zeros
        total_challenge_columns: max_num_output_axes,
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
                    .filter(|char| input_axes_to_dim.get(char).is_some())
                    .collect()
            })
            .collect();

        let output = output_str
            .chars()
            .filter(|c| input_axes_to_dim.get(c).is_some())
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

    let mut all_indices = std::collections::HashSet::new();
    let mut common_indices = Vec::new();

    // Find common indices across inputs
    for input_eq in &input_equations {
        for c in input_eq.chars() {
            if !all_indices.insert(c) {
                common_indices.push(c);
            }
        }
    }

    let output_indices: Vec<char> = output_str.chars().collect();
    let output_dims = output_indices
        .iter()
        .map(|c| input_axes_to_dim.get(&c).unwrap());
    let output_size = output_dims.clone().product();
    let longest_challenge_vector = *output_dims.max().unwrap();

    // Contraction depth is determined by the number of sequential reductions needed
    let contraction_depth = contraction_planner::input_contractions(&equation)?.len();

    Ok(SingleEquationAnalysis {
        output_size,
        longest_challenge_vector,
        max_input_size,
        equation: equation.to_string(),
        num_inputs: input_equations.len(),
        output_axes: output_indices.len(),
        contraction_depth,
        common_indices,
        output_indices,
    })
}
