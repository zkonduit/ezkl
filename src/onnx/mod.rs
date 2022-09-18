use crate::fieldutils::felt_to_i32;
use crate::tensor::{Tensor, TensorError};
use std::{fs, marker::PhantomData, path::Path};
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
    let order = arr.ndim();
    let dims: Vec<usize> = arr.shape().to_vec();
    let scaled = scale * arr + shift;
    let inner: Vec<i32> = scaled
        .into_raw_vec()
        .iter()
        .map(|float| unsafe { float.round().to_int_unchecked::<i32>().into() })
        .collect();
    Tensor::new(Some(&inner), &dims)
}

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();
        println!("loaded model {:?}", model);
        OnnxModel { model }
    }

    pub fn get_ndarray_by_node_name(
        &self,
        name: impl AsRef<str>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> {
        //        let model = tract_onnx::onnx().model_for_path("data/ff.o").unwrap();

        let node = self.model.node_by_name(name).unwrap();
        //        let fact = &self.model.nodes[1].outputs[0].fact;
        let fact = &node.outputs[0].fact;
        let shape = fact.shape.clone().as_concrete_finite().unwrap().unwrap();
        println!("{:?}", shape);
        let nav = fact
            // let nav = self.model.nodes[1].outputs[0]
            //     .fact
            .value
            .concretize()
            .unwrap()
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();
        nav
        //    println!("{:?}", nav);
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
