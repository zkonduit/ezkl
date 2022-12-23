use crate::tensor::{Tensor, TensorError};
use anyhow::Result;
use tract_onnx::prelude::{InferenceFact, Node};
use tract_onnx::tract_hir::internal::InferenceOp;

// Warning: currently ignores stride information
pub fn vector_to_quantized(
    vec: &[f32],
    dims: &[usize],
    shift: f32,
    scale: i32,
) -> Result<Tensor<i32>, TensorError> {
    let mult = scale_to_multiplier(scale);
    let scaled: Vec<i32> = vec
        .iter()
        .map(|e| (mult * e + shift).round() as i32)
        .collect();
    Tensor::new(Some(&scaled), dims)
}

pub fn scale_to_multiplier(scale: i32) -> f32 {
    i32::pow(2, scale as u32) as f32
}

pub fn node_output_shapes(
    node: &Node<InferenceFact, Box<dyn InferenceOp>>,
) -> Result<Vec<Option<Vec<usize>>>> {
    let mut shapes = Vec::new();
    let outputs = node.outputs.to_vec();
    for output in outputs {
        let mv = output
            .fact
            .shape
            .clone()
            .as_concrete_finite()?
            .map(|x| x.to_vec());
        shapes.push(mv)
    }
    Ok(shapes)
}
