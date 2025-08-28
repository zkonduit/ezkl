use std::{collections::BTreeSet, ops::Index};

use halo2curves::ff::PrimeField;
use itertools::Itertools;

use crate::{
    circuit::CircuitError,
    tensor::{TensorType, ValTensor},
};

/// inj,jk->ik                [inj,jk]
/// inj,i->nj => RLC          [jk,nj]
/// jk,k->j   => RLC          [nj,j]
/// nj,j->n   => Contraction  [n]
/// n->       => Contraction  []
///
/// bn,anm,bm->ba                         [bn,anm,bm]
/// bn,bm->bnm             => Contraction [anm,bnm]
/// bnm,b->nm              => RLC         [anm,nm]
/// anm,a->nm              => RLC         [nm,nm]
/// nm,nm->m               => Contraction [m]
/// m->                    => Contraction []

#[derive(Debug)]
pub enum Reduction {
    /// Random linear combination with powers of challenge along the axis
    RLC {
        expression: String,
        axis: char,
        /// Uniquely identifying index of input tensor to be reduced
        input_index: TensorIndex,
        /// phase of input tensor
        input_phase: usize,
        /// phase of output tensor
        output_phase: usize,
        challenge_index: usize,
    },
    Contraction {
        expression: String,
        /// when axis is `None`, the contraction is pairwise multiplication
        axis: Option<char>,
        /// Uniquely identifying indices of input tensors to be contracted
        input_indices: Vec<TensorIndex>,
        /// phases of input tensors
        input_phases: Vec<usize>,
        /// phase of output tensor
        output_phase: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct TensorIndex(usize);

impl<T: PrimeField + TensorType + PartialOrd> Index<TensorIndex> for Vec<ValTensor<T>> {
    type Output = ValTensor<T>;

    fn index(&self, index: TensorIndex) -> &Self::Output {
        &self[index.0]
    }
}

impl Reduction {
    pub fn expression(&self) -> &str {
        match self {
            Reduction::Contraction { expression, .. } => expression,
            Reduction::RLC { expression, .. } => &expression,
        }
    }

    pub fn input_indices(&self) -> Vec<TensorIndex> {
        match self {
            Reduction::Contraction { input_indices, .. } => input_indices.clone(),
            Reduction::RLC { input_index, .. } => vec![*input_index],
        }
    }

    pub fn output_phase(&self) -> usize {
        match self {
            Reduction::Contraction { output_phase, .. } => *output_phase,
            Reduction::RLC { output_phase, .. } => *output_phase,
        }
    }
}

pub fn input_reductions(expression: &str) -> Result<Vec<Reduction>, CircuitError> {
    let (input_exprs, output_expr) = expression.split_once("->").unwrap();
    let input_exprs: Vec<_> = input_exprs.split(",").map(|eq| eq.to_string()).collect();
    // (phase, expression)
    let input_exprs: Vec<(usize, String)> =
        input_exprs.into_iter().map(|expr| (0, expr)).collect_vec();

    let mut input_tensor_counter = input_exprs.len();
    let mut input_exprs: Vec<((usize, String), TensorIndex)> = input_exprs
        .into_iter()
        .zip((0..input_tensor_counter).map(TensorIndex))
        .collect();
    let mut reductions: Vec<Reduction> = vec![];

    // Reduce input_exprs along given axis
    let mut reduce = |input_exprs: Vec<((usize, String), TensorIndex)>,
                      axis: char|
     -> (Reduction, Vec<((usize, String), TensorIndex)>) {
        let inputs = input_exprs
            .iter()
            .filter(|((_, eq), _)| eq.chars().contains(&axis))
            .cloned()
            .collect_vec();
        let (inputs_axes, input_indices): (Vec<(usize, String)>, Vec<TensorIndex>) =
            inputs.iter().cloned().unzip();
        let (input_phases, inputs_axes): (Vec<usize>, Vec<String>) =
            inputs_axes.into_iter().unzip();

        let is_output_axis = output_expr.chars().contains(&axis);
        let output: String = if is_output_axis == true && inputs.len() > 1 {
            let output: BTreeSet<char> =
                inputs_axes.iter().flat_map(|input| input.chars()).collect();
            output.iter().collect()
        } else {
            let output: BTreeSet<char> = inputs_axes
                .iter()
                .flat_map(|input| input.chars().filter(|&c| c != axis))
                .collect();
            output.iter().collect()
        };

        let reduction = if is_output_axis == true && inputs.len() == 1 {
            let mut expression = inputs_axes.join(",");
            expression.push_str(format!(",{axis}").as_str());
            expression.push_str("->");
            expression.push_str(&output);
            Reduction::RLC {
                expression,
                axis,
                input_index: input_indices[0],
                input_phase: input_phases[0],
                output_phase: 1,
                challenge_index: output_expr.chars().position(|c| c == axis).unwrap(),
            }
        } else if is_output_axis == true {
            let mut expression = inputs_axes.join(",");
            let output_phase = input_phases.iter().copied().max().unwrap();
            expression.push_str("->");
            expression.push_str(&output);
            Reduction::Contraction {
                expression,
                axis: None,
                input_indices: input_indices,
                input_phases,
                output_phase,
            }
        } else {
            let mut expression = inputs_axes.join(",");
            let output_phase = input_phases.iter().copied().max().unwrap();
            expression.push_str("->");
            expression.push_str(&output);
            Reduction::Contraction {
                expression,
                axis: Some(axis),
                input_indices: input_indices,
                input_phases,
                output_phase,
            }
        };

        // Mutate input_exprs
        let mut input_exprs = input_exprs.clone();
        input_exprs.retain(|((_, input_eq), _)| !inputs_axes.contains(input_eq));
        input_exprs.push((
            (reduction.output_phase(), output.clone()),
            TensorIndex(input_tensor_counter),
        ));
        input_tensor_counter += 1;

        (reduction, input_exprs)
    };

    let mut output_axes = output_expr.chars().collect_vec();
    while let Some(axis) = output_axes.first().cloned() {
        let num_inputs = input_exprs
            .iter()
            .filter(|((_, eq), _)| eq.chars().contains(&axis))
            .count();
        if num_inputs == 0 {
            output_axes.remove(0);
        } else {
            let (reduction, new_input_exprs) = reduce(input_exprs, axis);
            reductions.push(reduction);
            input_exprs = new_input_exprs;
        }
    }

    // These are not output axes and were not contracted with random vectors
    let remaining_axes: BTreeSet<_> = input_exprs
        .iter()
        .flat_map(|((_, eq), _)| eq.chars())
        .collect();

    for axis in remaining_axes.iter() {
        let (reduction, new_input_exprs) = reduce(input_exprs, *axis);
        reductions.push(reduction);
        input_exprs = new_input_exprs;
    }

    Ok(reductions)
}
