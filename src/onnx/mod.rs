use crate::tensor::{Tensor, TensorError};
use std::path::Path;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact};
use tract_onnx::tract_hir::infer::Factoid;
use tract_onnx::tract_hir::internal::InferenceOp;

pub struct OnnxModel {
    model: Graph<InferenceFact, Box<dyn InferenceOp>>,
}

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

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();
        OnnxModel { model }
    }

    pub fn get_ndarray_by_node_name(
        &self,
        name: impl AsRef<str>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> {
        let node = self.model.node_by_name(name).unwrap();
        let fact = &node.outputs[0].fact;

        let nav = fact
            .value
            .concretize()
            .unwrap()
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();
        nav
    }

    pub fn get_tensor_by_node_name(
        &self,
        name: impl AsRef<str>,
        shift: f32,
        scale: f32,
    ) -> Tensor<i32> {
        let arr = self.get_ndarray_by_node_name(name);
        ndarray_to_quantized(arr, shift, scale).unwrap()
    }
}
