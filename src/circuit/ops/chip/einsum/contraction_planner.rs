use std::{collections::BTreeSet, ops::Index};

use itertools::Itertools;

use crate::{
    circuit::CircuitError,
    tensor::{Tensor, TensorType},
};

#[derive(Clone, Debug)]
pub struct TensorIndex(usize);

/// `Contraction` expresses a tensor contraction between input tensors
#[derive(Debug)]
pub struct Contraction {
    pub expression: String,
    pub axis: char,
    /// Uniquely identifying indices of input tensors to be contracted
    input_indices: Vec<TensorIndex>,
}

impl<T: TensorType> Index<TensorIndex> for Vec<Tensor<T>> {
    type Output = Tensor<T>;

    fn index(&self, index: TensorIndex) -> &Self::Output {
        &self[index.0]
    }
}

impl Contraction {
    pub fn num_inputs(&self) -> usize {
        let (input_exprs, _) = self.expression.split_once("->").unwrap();
        let input_exprs: Vec<_> = input_exprs.split(",").map(|eq| eq.to_string()).collect();
        input_exprs.len()
    }

    pub fn input_indices(&self) -> Vec<TensorIndex> {
        self.input_indices.clone()
    }
}

pub fn input_contractions(expression: &str) -> Result<Vec<Contraction>, CircuitError> {
    let (input_exprs, output_expr) = expression.split_once("->").unwrap();
    let input_exprs: Vec<_> = input_exprs.split(",").map(|eq| eq.to_string()).collect();
    // augment `input_exprs` with output axes
    let mut input_exprs = input_exprs.clone();
    input_exprs.extend(output_expr.chars().map(|c| c.to_string()).collect_vec());
    let mut input_tensor_counter = input_exprs.len();
    let mut augmented_input_exprs: Vec<(String, TensorIndex)> = input_exprs
        .into_iter()
        .zip((0..input_tensor_counter).map(TensorIndex))
        .collect();
    let mut contractions: Vec<Contraction> = vec![];

    // Contract input_exprs along given axis
    let mut contract = |input_exprs: Vec<(String, TensorIndex)>,
                        axis: char|
     -> (Contraction, Vec<(String, TensorIndex)>) {
        // Note all input_exprs that contain `axis`
        // [bn,bm,b]
        let contracted_inputs = input_exprs
            .iter()
            .filter(|(eq, _)| eq.chars().contains(&axis))
            .cloned()
            .collect_vec();
        let (contracted_inputs_axes, contracted_inputs_indices): (Vec<String>, Vec<TensorIndex>) =
            contracted_inputs.into_iter().unzip();

        // nm
        let contracted_output: BTreeSet<char> = contracted_inputs_axes
            .iter()
            .flat_map(|input| input.chars().filter(|&c| c != axis))
            .collect();
        let contracted_output: String = contracted_output.iter().collect();

        let mut expression = contracted_inputs_axes.join(",");
        expression.push_str("->");
        // bn,bm,b->nm
        expression.push_str(&contracted_output);

        let contraction = Contraction {
            expression,
            axis,
            input_indices: contracted_inputs_indices,
        };

        // Mutate input_exprs
        let mut input_exprs = input_exprs.clone();
        // [anm]
        input_exprs.retain(|(input_eq, _)| !contracted_inputs_axes.contains(input_eq));
        // [anm,nm]
        input_exprs.push((contracted_output.clone(), TensorIndex(input_tensor_counter)));
        input_tensor_counter += 1;

        (contraction, input_exprs)
    };

    for axis in output_expr.chars() {
        let (contraction, new_input_exprs) = contract(augmented_input_exprs, axis);
        contractions.push(contraction);
        augmented_input_exprs = new_input_exprs;
    }

    // These are not output axes and were not contracted with random vectors
    let remaining_axes: BTreeSet<_> = augmented_input_exprs
        .iter()
        .flat_map(|(eq, _)| eq.chars())
        .collect();

    for axis in remaining_axes.iter() {
        let (contraction, new_input_exprs) = contract(augmented_input_exprs, *axis);
        contractions.push(contraction);
        augmented_input_exprs = new_input_exprs;
    }

    Ok(contractions)
}
