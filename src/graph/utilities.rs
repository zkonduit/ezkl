use crate::tensor::{Tensor, TensorError};
use anyhow::Result;
use tract_onnx::prelude::{InferenceFact, Node};
use tract_onnx::tract_hir::internal::InferenceOp;

// Warning: currently ignores stride information
/// Quantizes an iterable of f32s to a [Tensor] of i32s using a fixed point representation.
/// Arguments
///
/// * `vec` - the vector to quantize.
/// * `dims` - the dimensionality of the resulting [Tensor].
/// * `shift` - offset used in the fixed point representation.
/// * `scale` - `2^scale` used in the fixed point representation.
pub fn vector_to_quantized(
    vec: &[f32],
    dims: &[usize],
    shift: f32,
    scale: u32,
) -> Result<Tensor<i128>, TensorError> {
    let mult = scale_to_multiplier(scale);
    let scaled: Vec<i128> = vec
        .iter()
        .map(|e| (mult * e + shift).round() as i128)
        .collect();
    Tensor::new(Some(&scaled), dims)
}

/// Converts a scale (log base 2) to a fixed point multiplier.
pub fn scale_to_multiplier(scale: u32) -> f32 {
    i128::pow(2, scale) as f32
}

/// Gets the shape of a onnx node's outlets.
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
