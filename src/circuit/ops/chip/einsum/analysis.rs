use std::collections::HashSet;

use crate::circuit::{chip::einsum::contraction_planner, CircuitError};

#[derive(Debug, Clone)]
pub struct EinsumAnalysis {
    pub max_inputs: usize,
    pub max_output_axes: usize,
    pub max_contraction_depth: usize,
    pub universal_gate_size: usize,
    pub total_challenge_columns: usize,
}

#[derive(Debug, Clone)]
pub struct SingleEquationAnalysis {
    pub num_inputs: usize,
    pub output_axes: usize,
    pub contraction_depth: usize,
    pub common_indices: Vec<char>,
    pub output_indices: Vec<char>,
}

pub fn analyze_einsum_usage(equations: &HashSet<String>) -> Result<EinsumAnalysis, CircuitError> {
    let mut max_inputs = 0;
    let mut max_output_axes = 0;
    let mut max_contraction_depth = 0;

    for equation in equations {
        let analysis = analyze_single_equation(equation)?;
        max_inputs = max_inputs.max(analysis.num_inputs);
        max_output_axes = max_output_axes.max(analysis.output_axes);
        max_contraction_depth = max_contraction_depth.max(analysis.contraction_depth);
    }

    Ok(EinsumAnalysis {
        max_inputs,
        max_output_axes,
        max_contraction_depth,
        universal_gate_size: max_inputs + max_output_axes, // For padding with zeros
        total_challenge_columns: max_output_axes,
    })
}

// FIXME : In the analysis phase, we should consider trivial output axes
pub fn analyze_single_equation(equation: &str) -> Result<SingleEquationAnalysis, CircuitError> {
    let (inputs_str, output_str) = equation.split_once("->").unwrap();
    let input_equations: Vec<&str> = inputs_str.split(',').collect();

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

    // Contraction depth is determined by the number of sequential reductions needed
    let contraction_depth = contraction_planner::input_contractions(equation)?.len();

    Ok(SingleEquationAnalysis {
        num_inputs: input_equations.len(),
        output_axes: output_indices.len(),
        contraction_depth,
        common_indices,
        output_indices,
    })
}
