use crate::tensor::{Tensor, TensorError};
use anyhow::Result;
use tract_onnx::prelude::{InferenceFact, Node};
use tract_onnx::tract_hir::internal::InferenceOp;

// Warning: currently ignores stride information
pub fn ndarray_to_quantized(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
    shift: f32,
    scale: f32,
) -> Result<Tensor<i32>, TensorError> {
    let dims: Vec<usize> = arr.shape().to_vec();
    let scaled = scale * arr + shift;
    let inner: Vec<i32> = scaled
        .into_raw_vec()
        .iter()
        .map(|float| unsafe { float.round().to_int_unchecked::<i32>() })
        .collect();
    Tensor::new(Some(&inner), &dims)
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
