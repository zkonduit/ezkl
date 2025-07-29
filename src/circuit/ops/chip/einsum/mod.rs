use std::marker::PhantomData;
use std::collections::HashSet;

use crate::tensor::{Tensor, TensorType, ValTensor};
use halo2_proofs::circuit::{Layouter, AssignedCell, Value};
use halo2_proofs::plonk::{Advice, Challenge, Column, FirstPhase, SecondPhase, Selector, ConstraintSystem, Expression};
use halo2_wrong_ecc::maingate::RegionCtx;
use halo2curves::ff::{Field, PrimeField};
use itertools::Itertools;

mod contraction_planner;

/// A struct representing the contractions for the einsums
#[derive(Debug, Clone)]
pub struct Einsums<F: PrimeField + TensorType + PartialOrd> {
    equation: String,
    // FIXME use VarTensor::Advice
    pub inputs: Vec<Column<Advice>>,
    // FIXME use VarTensor::Advice
    pub output: Column<Advice>,
    pub challenges: Vec<Challenge>,
    pub challenge_columns: Vec<Column<Advice>>,
    pub input_summations: Vec<ContractionConfig<F>>,
    pub output_summations: Vec<ContractionConfig<F>>,
    /// Universal gate configuration for different einsum sizes
    pub max_inputs: usize,
    pub max_challenges: usize,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> Einsums<F> {
    fn dummy(_a: usize, _b: usize) -> Self {
        todo!()
    }

    fn configure_universal(
        equation: String,
        meta: &mut ConstraintSystem<F>,
        analysis: &EinsumAnalysis,
    ) -> Self {
        // Allocate maximum number of columns needed
        let inputs: Vec<_> = (0..analysis.max_inputs)
            .map(|_| meta.advice_column_in(FirstPhase))
            .collect();

        let challenge_columns: Vec<_> = (0..analysis.max_output_axes)
            .map(|_| meta.advice_column_in(SecondPhase))
            .collect();

        let challenges: Vec<_> = (0..analysis.max_output_axes)
            .map(|_| meta.challenge_usable_after(FirstPhase))
            .collect();

        // Configure contraction configs for maximum depth
        let input_summations = self.configure_contraction_tree(
            meta,
            analysis.max_contraction_depth,
            analysis.max_inputs,
        );

        // Configure output summations for challenge-based compression
        let output_summations = self.configure_output_summations(meta, analysis.max_output_axes);

        let output = meta.advice_column_in(SecondPhase);

        Self {
            equation,
            inputs,
            challenges,
            challenge_columns,
            input_summations,
            output,
            output_summations,
            max_inputs: analysis.max_inputs,
            max_challenges: analysis.max_output_axes,
            _marker: PhantomData,
        }
    }

    fn configure_output_summations(
        &self,
        meta: &mut ConstraintSystem<F>,
        max_output_axes: usize,
    ) -> Vec<ContractionConfig<F>> {
        let mut output_summations = vec![];
        output_dims.reverse();
        for _ in 0..max_output_axes {
            output_dims.remove(0);
            let num_dot_products = output_dims.iter().product();
            let num_inputs = 2;
            output_summations.push(ContractionConfig::new(meta, num_dot_products, num_inputs));
        }
        output_summations
    }

    fn configure_contraction_tree(
        &self,
        meta: &mut ConstraintSystem<F>,
        max_contraction_depth: usize,
        max_inputs: usize,
    ) -> Vec<ContractionConfig<F>> {
        todo!()
    }

    fn assign_tensor_to_column(
        &self,
        region: &mut RegionCtx<F>,
        input: ValTensor<F>,
        column: Column<Advice>,
    ) -> Result<ValTensor<F>, CircuitError> {
        todo!()
    }

    fn assign_zero_tensor(
        &self,
        region: &mut RegionCtx<F>,
        column: Column<Advice>,
    ) -> Result<ValTensor<F>, CircuitError> {
        todo!()
    }

    fn assign_with_padding(
        &self,
        region: &mut RegionCtx<F>,
        input_tensors: &[&ValTensor<F>],
        output_tensor: ValTensor<F>,
        equation: &str,
    ) -> Result<ValTensor<F>, CircuitError> {
        let (inputs, output) = self.equation.split_once("->").unwrap();
        let inputs = inputs.split(",").collect_vec();
        assert_eq!(inputs.len(), input_tensors.len());

        // Remove trivial axes from tensors
        let input_tensors: Vec<_> = input_tensors
            .iter()
            .map(|tensor| tensor.remove_trivial_axes())
            .collect();
        let output_tensor = output_tensor.remove_trivial_axes();

        let mut input_axes_to_dim: HashMap<char, usize> = HashMap::new();
        inputs
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

        // Sanitise equation to remove trivial axes
        let equation = {
            let inputs = inputs.iter().map(|input| {
                let mut chars = input.chars();
                chars.filter_map(|char| input.get(char).is_some());
                chars.collect()
            });

            let output = output
                .chars()
                .filter(|c| input_axes_to_dim.get(c).is_some())
                .collect();

            [inputs.join(","), output].join("->")
        };

        let equation_analysis = analyze_single_equation(equation);

        // Assign actual inputs
        for (i, input) in inputs.iter().enumerate() {
            self.assign_tensor_to_column(region, input, self.inputs[i])?;
        }

        // Zero-pad unused input columns
        for i in inputs.len()..self.inputs.len() {
            self.assign_zero_tensor(region, self.inputs[i])?;
        }

        // Configure active contractions only
        let active_contractions = &self.input_summations[..equation_analysis.contraction_depth];

        // Assign challenges and witness challenge vectors
        let assigned_challenges: Vec<Value<F>> = self
            .challenges
            .iter()
            .take(equation_analysis.output_axes)
            .map(|challenge| region.get_challenge(*challenge))
            .collect();

        // Create challenge vectors as powers of challenge (following prototype pattern)
        let non_trivial_output_dims = equation_analysis
            .output_indices
            .iter()
            .map(|&c| get_axis_size_for_index(c, inputs))
            .collect::<Vec<_>>();

        let challenge_vectors: Vec<Tensor<Value<F>>> = non_trivial_output_dims
            .into_iter()
            .enumerate()
            .map(|(i, size)| {
                let challenge = assigned_challenges[i];
                let powers_of_challenge = (0..size)
                    .scan(Value::known(F::ONE), |state, _| {
                        *state = *state * challenge;
                        Some(*state)
                    })
                    .collect_vec();
                Tensor::from(
                    powers_of_challenge
                        .into_iter()
                        .map(Tensor::Scalar)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        let squashed_output = self.assign_output(
            region,
            output_tensor,
            challenge_vectors.iter().cloned().collect_vec(),
        )?;

        let mut input_tensors = input_tensors.clone();
        input_tensors.extend_from_slice(&challenge_vectors);

        // reorder the contraction of input tensors and contract
        let reordered_input_contractions = input_contractions(&self.equation).unwrap();
        assert_eq!(
            reordered_input_contractions.len(),
            self.input_summations.len()
        );
        let mut tensors = vec![];
        for (i, input_tensor) in input_tensors.iter().enumerate() {
            region.set_offset(0);
            let witnessed_tensor = input_tensor.witness(region, self.inputs[i])?;
            tensors.push(witnessed_tensor);
        }

        for (contraction, config) in reordered_input_contractions
            .iter()
            .zip(self.input_summations.iter())
        {
            region.set_offset(0);
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

            let mut flattened_input_tensors: Vec<Vec<Tensor<AssignedCell<F, F>>>> =
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
                    let sliced_input_tensor = input_tensor.slice(&sliced_dim);
                    flattened_input_tensors[i].push(Tensor::array(sliced_input_tensor.flatten()));
                }
            }
            let flattened_input_tensors = flattened_input_tensors
                .into_iter()
                .map(|tensors| {
                    tensors
                        .iter()
                        .flat_map(|tensor| tensor.flatten())
                        .collect_vec()
                })
                .collect_vec();
            let dot_product_len = input_axes_to_dim[&contraction.axis];
            let output_dims = output_expr
                .chars()
                .map(|c| input_axes_to_dim[&c])
                .collect_vec();
            let contracted_output = config.assign(
                region,
                flattened_input_tensors,
                dot_product_len,
                &output_dims,
            )?;

            tensors.push(contracted_output);
        }
        tensors.retain(|tensor| tensor.scalar().is_ok());

        // FIXME constrain this to be a product
        let squashed_input = tensors
            .iter()
            .map(|tensor| tensor.scalar().unwrap().value())
            .fold(Value::known(F::ONE), |acc, v| acc * v);
        region.set_offset(5);
        let squashed_input = region.assign_advice(|| "", self.output, squashed_input)?;
        region.constrain_equal(squashed_input.cell(), squashed_output.cell())
    }

    fn assign_output(
        &self,
        region: &mut RegionCtx<F>,
        output: Tensor<Value<F>>,
        challenge_vectors: Vec<Tensor<Value<F>>>,
    ) -> Result<AssignedCell<F, F>, CircuitError> {
        let initial_offset = region.offset();
        // Witness challenge vectors
        // FIXME constrain these to be a well-constructed power series
        let mut challenge_vectors_assigned = vec![];
        for (i, challenge_vector) in challenge_vectors.iter().enumerate() {
            region.set_offset(initial_offset);
            let mut challenge_vector_assigned = vec![];
            for value in challenge_vector.flatten().into_iter() {
                challenge_vector_assigned.push(region.assign_advice(
                    || "",
                    self.challenge_columns[i],
                    value,
                )?);
                region.next();
            }
            challenge_vectors_assigned.push(challenge_vector_assigned);
        }

        // Initialise `intermediate_values` to the original output tensor
        region.set_offset(initial_offset);
        let mut intermediate_values: Vec<AssignedCell<F, F>> = vec![];
        // Witness flattened output
        for value in output.flatten().into_iter() {
            let value = region.assign_advice(|| "", self.output, value)?;
            region.next();
            intermediate_values.push(value);
        }

        // Intermediate values output from the previous summation
        // Loop over the output axes
        for (summation_config, powers_of_challenge) in self
            .output_summations
            .iter()
            .zip(challenge_vectors_assigned.into_iter().rev())
        {
            region.set_offset(initial_offset);
            intermediate_values = summation_config.assign_output(
                region,
                intermediate_values,
                &powers_of_challenge,
            )?;
        }

        Ok(intermediate_values[0].clone())
    }
}

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

pub fn analyze_einsum_usage(equations: &HashSet<String>) -> EinsumAnalysis {
    let mut max_inputs = 0;
    let mut max_output_axes = 0;
    let mut max_contraction_depth = 0;

    for equation in equations {
        let analysis = analyze_single_equation(equation);
        max_inputs = max_inputs.max(analysis.num_inputs);
        max_output_axes = max_output_axes.max(analysis.output_axes);
        max_contraction_depth = max_contraction_depth.max(analysis.contraction_depth);
    }

    EinsumAnalysis {
        max_inputs,
        max_output_axes,
        max_contraction_depth,
        universal_gate_size: max_inputs + max_output_axes, // For padding with zeros
        total_challenge_columns: max_output_axes,
    }
}

fn analyze_single_equation(equation: &str) -> SingleEquationAnalysis {
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
    let contraction_depth = contraction_planner::input_contractions(equation)
        .map(|contractions| contractions.len())
        .unwrap_or(0);

    SingleEquationAnalysis {
        num_inputs: input_equations.len(),
        output_axes: output_indices.len(),
        contraction_depth,
        common_indices,
        output_indices,
    }
}

/// Each `ContractionConfig` corresponds to a summation in the einsum argument,
/// i.e. contraction along a single axis.
/// This consists of multiple running sum (i.e. dot product) arguments.
/// Each `ContractionConfig` constraints element-wise multiplication and summation between input tensors.
#[derive(Debug, Clone)]
struct ContractionConfig<F: Field> {
    dot_products: Vec<DotProductConfig<F>>,
}

impl<F: PrimeField> ContractionConfig<F> {
    fn new(
        meta: &mut ConstraintSystem<F>,
        // The number of dot products
        num_dot_products: usize,
        // The number of tensors in the dot product
        num_inputs: usize,
    ) -> Self {
        // TODO optimise choice of advice columns globally
        let inputs: Vec<_> = (0..num_inputs)
            .map(|_| meta.advice_column_in(SecondPhase))
            .collect();
        inputs.iter().for_each(|c| meta.enable_equality(*c));

        let running_sum = meta.advice_column_in(SecondPhase);
        meta.enable_equality(running_sum);

        let mut dot_products = vec![];
        for _ in 0..num_dot_products {
            dot_products.push(DotProductConfig::new(meta, &inputs, running_sum));
        }

        Self { dot_products }
    }

    fn assign(
        &self,
        region: &mut RegionCtx<F>,
        flattened_tensors: Vec<Vec<AssignedCell<F, F>>>,
        dot_product_len: usize,
        output_shape: &[usize],
    ) -> Result<Tensor<AssignedCell<F, F>>, CircuitError> {
        let num_dot_products = self.dot_products.len();
        let mut dot_product_results = vec![];
        for chunk_idx in 0..num_dot_products {
            let start = chunk_idx * dot_product_len;
            let tensors: Vec<_> = flattened_tensors
                .iter()
                .map(|tensor| tensor[start..(start + dot_product_len)].to_vec())
                .collect();
            let result = self.dot_products[chunk_idx].assign(region, tensors)?;
            dot_product_results.push(result);
        }
        let tensor = Tensor::array(dot_product_results).reshape(output_shape);
        Ok(tensor)
    }

    fn assign_output(
        &self,
        region: &mut RegionCtx<F>,
        tensor: Vec<AssignedCell<F, F>>,
        powers_of_challenge: &[AssignedCell<F, F>],
    ) -> Result<Vec<AssignedCell<F, F>>, CircuitError> {
        let num_dot_products = self.dot_products.len();
        let dot_product_len = powers_of_challenge.len();
        assert_eq!(tensor.len(), num_dot_products * dot_product_len);
        // Split tensor and challenge vector into dot products
        let mut dot_product_results = vec![];
        for (idx, tensor) in tensor.chunks_exact(dot_product_len).enumerate() {
            let tensors = vec![tensor.to_vec(), powers_of_challenge.to_vec()];
            let result = self.dot_products[idx].assign(region, tensors)?;
            dot_product_results.push(result);
        }

        Ok(dot_product_results)
    }
}

#[derive(Debug, Clone)]
struct DotProductConfig<F: Field> {
    selector: (Selector, Selector),
    inputs: Vec<Column<Advice>>,
    running_sum: Column<Advice>,
    _marker: PhantomData<F>,
}

impl<F: Field> DotProductConfig<F> {
    fn assign(
        &self,
        region: &mut RegionCtx<F>,
        tensors: Vec<Vec<AssignedCell<F, F>>>,
    ) -> Result<AssignedCell<F, F>, CircuitError> {
        assert_eq!(tensors.len(), self.inputs.len());
        assert!(tensors.iter().map(|t| t.len()).all_equal());
        let dot_product_len = tensors[0].len();
        let initial_offset = region.offset();
        // Copy `tensors` values into appropriate cells
        for (col, tensor) in self.inputs.iter().zip(tensors.iter()) {
            region.set_offset(initial_offset);
            for cell in tensor.iter() {
                println!(
                    "cell : ({:?}, {})",
                    cell.cell().column,
                    cell.cell().row_offset
                );
                println!("(col, offset) : ({}, {})", col.index(), region.offset());
                region.copy_advice(cell, *col)?;
                region.next();
            }
        }

        let mut transposed_tensors = vec![];
        for idx in 0..tensors[0].len() {
            transposed_tensors.push(
                tensors
                    .iter()
                    .map(|tensor| tensor[idx].clone())
                    .collect_vec(),
            );
        }
        let running_sum = transposed_tensors
            .iter()
            .scan(Value::known(F::ZERO), |state, inputs| {
                let multiplied = inputs
                    .iter()
                    .map(|input| input.value())
                    .fold(Value::known(F::ONE), |acc, v| acc * v);
                *state = *state + multiplied;
                Some(*state)
            });

        region.set_offset(initial_offset);
        let mut result = None;
        for (idx, running_sum) in running_sum.enumerate() {
            let running_sum = region.assign_advice(|| "", self.running_sum, running_sum)?;

            if idx == 0 {
                region.enable(self.selector.0)?;
            } else {
                region.enable(self.selector.1)?;
            }

            if idx == dot_product_len - 1 {
                result = Some(running_sum)
            }
            region.next();
        }
        Ok(result.unwrap())
    }

    fn new(
        meta: &mut ConstraintSystem<F>,
        inputs: &[Column<Advice>],
        running_sum: Column<Advice>,
    ) -> Self {
        // TODO cache and retrieve selectors if using repeated advice columns
        let selector = (meta.selector(), meta.selector());

        let config = Self {
            selector,
            inputs: inputs.to_vec(),
            running_sum,
            _marker: PhantomData,
        };

        config.dot_product_gate(meta);

        config
    }

    // Helper dot product gate
    fn dot_product_gate(&self, meta: &mut ConstraintSystem<F>) {
        meta.create_gate("initialization", |_| {
            let s = self.selector.0.expr();
            let init = self
                .inputs
                .iter()
                .fold(Expression::Constant(F::ONE), |acc, input| acc * input.cur());
            vec![s * (self.running_sum.cur() - init)]
        });

        meta.create_gate("accumulation", |_| {
            let s = self.selector.1.expr();
            let acc = self.running_sum.prev();
            let curr = self
                .inputs
                .iter()
                .fold(Expression::Constant(F::ONE), |acc, input| acc * input.cur());
            vec![s * (self.running_sum.cur() - (acc + curr))]
        });
    }
}
