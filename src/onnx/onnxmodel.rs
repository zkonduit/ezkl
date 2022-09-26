use crate::nn::affine::Affine1dConfig;
use crate::nn::cnvrl::ConvConfig;
use crate::tensor::{Tensor, ValTensor, VarTensor};
use crate::tensor_ops::eltwise::{EltwiseConfig, ReLu, Sigmoid};
use anyhow::Result;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{Column, ConstraintSystem, Instance},
};
use std::env;
use std::path::Path;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, Node, OutletId};
use tract_onnx::tract_hir::{infer::Factoid, internal::InferenceOp};

use crate::nn::LayerConfig;
use crate::tensor::TensorType;

use super::utilities::{ndarray_to_quantized, node_output_shapes};

// Initially, some of these OpKinds will be folded into others (for example, Const nodes that
// contain parameters will be handled at the consuming node.
// Eventually, though, we probably want to keep them and treat them directly (layouting and configuring
// at each type of node)
#[derive(Clone, Debug, Copy)]
pub enum OpKind {
    Affine,
    Convolution,
    ReLU,
    Sigmoid,
    Source,
    Const,
    Input,
    Unknown,
}

#[derive(Clone)]
pub enum OnnxNodeConfig<F: FieldExt + TensorType, const BITS: usize> {
    Affine(Affine1dConfig<F>),
    Conv(ConvConfig<F, 2, 0>),
    ReLU(EltwiseConfig<F, BITS, ReLu<F>>),
    Sigmoid(EltwiseConfig<F, BITS, Sigmoid<F, 128, 128>>),
    Const,
    Source,
    Input,
    NotConfigured,
}

#[derive(Clone)]
pub struct OnnxModelConfig<F: FieldExt + TensorType, const BITS: usize> {
    configs: Vec<OnnxNodeConfig<F, BITS>>,
    pub public_output: Column<Instance>,
}

#[derive(Clone, Debug)]
pub struct OnnxNode {
    node: Node<InferenceFact, Box<dyn InferenceOp>>,
}

impl OnnxNode {
    pub fn new(node: Node<InferenceFact, Box<dyn InferenceOp>>) -> Self {
        OnnxNode { node }
    }

    pub fn output_shapes(&self) -> Result<Vec<Option<Vec<usize>>>> {
        let mut shapes = Vec::new();
        let outputs = self.node.outputs.to_vec();
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

    pub fn opkind(&self) -> OpKind {
        let res = match self.node.op().name().as_ref() {
            "Gemm" => OpKind::Affine,
            "Conv" => OpKind::Convolution,
            "Clip" => OpKind::ReLU,
            "Sigmoid" => OpKind::Sigmoid,
            "Const" => OpKind::Const,
            "Source" => OpKind::Input,
            "input" => OpKind::Input,
            _ => OpKind::Unknown,
        };

        // println!(
        //     "Classified {:?} as {:?}",
        //     self.node.op().name().as_ref(),
        //     res
        // );
        res
    }

    pub fn name(&self) -> String {
        self.node.name.clone().into()
    }

    pub fn output_ndarray_by_slot(
        &self,
        slot: usize,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> {
        let fact = &self.node.outputs[slot].fact;
        let nav = fact
            .value
            .concretize()
            .unwrap()
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();
        nav
    }

    pub fn output_tensor_by_slot(&self, slot: usize, shift: f32, scale: f32) -> Tensor<i32> {
        let arr = self.output_ndarray_by_slot(slot);
        ndarray_to_quantized(arr, shift, scale).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct OnnxModel {
    pub model: Graph<InferenceFact, Box<dyn InferenceOp>>, // The raw Tract data structure
    pub onnx_nodes: Vec<OnnxNode>, // Wrapped nodes with additional methods and potentially data (e.g. quantization)
}

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();
        let onnx_nodes: Vec<OnnxNode> = model
            .nodes()
            .iter()
            .map(|n| OnnxNode::new(n.clone()))
            .collect();
        OnnxModel { model, onnx_nodes }
    }
    pub fn from_arg() -> Self {
        let args: Vec<String> = env::args().collect();
        let filename: String = args[1].clone();
        OnnxModel::new(filename)
    }

    pub fn configure<F: FieldExt + TensorType, const BITS: usize>(
        &self,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
        fixeds: VarTensor,
    ) -> Result<OnnxModelConfig<F, BITS>> {
        // Note that the order of the nodes, and the eval_order, is not stable between model loads
        let order = self.eval_order()?;
        let mut configs: Vec<OnnxNodeConfig<F, BITS>> =
            vec![OnnxNodeConfig::NotConfigured; order.len()];
        for node_idx in order {
            configs[node_idx] =
                self.configure_node(node_idx, meta, advices.clone(), fixeds.clone())?;
        }

        let public_output: Column<Instance> = meta.instance_column();
        meta.enable_equality(public_output);

        Ok(OnnxModelConfig {
            configs,
            public_output,
        })
    }

    /// Infer the params, input, and output, and configure against the provided meta and Advice and Fixed columns.
    /// Note that we require the context of the Graph to complete this task.
    fn configure_node<F: FieldExt + TensorType, const BITS: usize>(
        &self,
        node_idx: usize,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
        fixeds: VarTensor, // Should use fixeds, but currently buggy
    ) -> Result<OnnxNodeConfig<F, BITS>> {
        let node = &self.onnx_nodes[node_idx];
        //        println!("Configure Node {}, a {:?}", node_idx, node.opkind());

        // Figure out, find, and load the params
        match node.opkind() {
            OpKind::Affine => {
                // The parameters are assumed to be fixed kernel and bias. This node should have three inputs in total:
                // two inputs which are Const(..) that have the f32s, and one variable input which are the activations.
                // The first input is the activations, second is the weight matrix, and the third the bias.
                // Consider using shape information only here, rather than loading the param tensor (although loading
                // the tensor guarantees that assign will work if there are errors or ambiguityies in the shape
                // data).
                let input_outlets = &node.node.inputs;
                let (_input_node_ix, _input_node_slot) =
                    (input_outlets[0].node, input_outlets[0].slot);

                let (weight_node_ix, weight_node_slot) =
                    (input_outlets[1].node, input_outlets[1].slot);
                let (_bias_node_ix, _bias_node_slot) =
                    (input_outlets[2].node, input_outlets[2].slot);
                let weight_node = OnnxNode::new(self.nodes()[weight_node_ix].clone());
                let weight_value =
                    weight_node.output_tensor_by_slot(weight_node_slot, 0f32, 256f32);

                let in_dim = weight_value.dims()[1];
                let out_dim = weight_value.dims()[0];
                // If we don't want to resue the columns:
                // let mut weight: Tensor<Column<Fixed>> =
                //     (0..in_dim * out_dim).map(|_| meta.fixed_column()).into();
                // weight.reshape(&weight_value.dims());
                // let mut bias: Tensor<Column<Fixed>> =
                //     (0..out_dim).map(|_| meta.fixed_column()).into();
                // bias.reshape?
                let weight_fixeds = advices.get_slice(&[0..out_dim], &[out_dim, in_dim]); //&[0..out_dim], &[out_dim, in_dim]
                let bias_fixeds = advices.get_slice(&[out_dim + 1..out_dim + 2], &[out_dim]);
                let params = [weight_fixeds, bias_fixeds];
                let input = advices.get_slice(&[out_dim + 2..out_dim + 3], &[in_dim]);
                let output = advices.get_slice(&[out_dim + 3..out_dim + 4], &[out_dim]);
                let conf = Affine1dConfig::configure(meta, &params, input, output);
                Ok(OnnxNodeConfig::Affine(conf))
            }
            OpKind::ReLU => {
                // Here,   node.output_shapes().unwrap()[0].as_ref().unwrap() == vec![1,LEN]
                let length = node.output_shapes().unwrap()[0].as_ref().unwrap()[1];
                let conf: EltwiseConfig<F, BITS, ReLu<F>> = EltwiseConfig::configure(
                    meta,
                    advices.get_slice(&[0..length], &[length]),
                    None,
                );
                Ok(OnnxNodeConfig::ReLU(conf))
            }
            OpKind::Sigmoid => {
                // Here,   node.output_shapes().unwrap()[0].as_ref().unwrap() == vec![1,LEN]
                let length = node.output_shapes().unwrap()[0].as_ref().unwrap()[1];
                let conf: EltwiseConfig<F, BITS, Sigmoid<F, 128, 128>> = EltwiseConfig::configure(
                    meta,
                    advices.get_slice(&[0..length], &[length]),
                    None,
                );
                Ok(OnnxNodeConfig::Sigmoid(conf))
            }
            OpKind::Const => {
                // Typically parameters for one or more layers.
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(OnnxNodeConfig::Const)
            }
            OpKind::Source => {
                // This is the input to the model (e.g. the image).
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(OnnxNodeConfig::Source)
            }
            OpKind::Input => {
                // This is the input to the model (e.g. the image).
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(OnnxNodeConfig::Input)
            }

            _ => {
                unimplemented!()
            }
        }
    }

    pub fn layout<F: FieldExt + TensorType, const BITS: usize>(
        &self,
        config: OnnxModelConfig<F, BITS>,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
    ) -> Result<ValTensor<F>> {
        let order = self.eval_order()?;
        let mut x = input;
        for node_idx in order {
            x = match self.layout_node(
                node_idx,
                layouter,
                x.clone(),
                config.configs[node_idx].clone(),
            ) {
                Some(vt) => {
                    // This is just to log the layer output
                    match vt.clone() {
                        ValTensor::PrevAssigned { inner: v, dims: _ } => {
                            let r: Tensor<i32> = v.clone().into();
                            println!("Node {} out: {:?}", node_idx, r);
                        }
                        _ => panic!("Should be assigned"),
                    };

                    vt
                }
                None => x, // Some nodes don't produce tensor output, we skip these
            }
        }
        Ok(x)
    }

    // Takes an input ValTensor; alternatively we could recursively layout all the predecessor tensors
    // (which may be more correct for some graphs).
    // Does not take parameters, instead looking them up in the network.
    // At the Source level, the input will be fed by the prover.
    fn layout_node<F: FieldExt + TensorType, const BITS: usize>(
        &self,
        node_idx: usize,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        config: OnnxNodeConfig<F, BITS>,
    ) -> Option<ValTensor<F>> {
        let node = &self.onnx_nodes[node_idx];
        let input_outlets = &node.node.inputs;

        //        println!("Layout Node {}, {:?}", node_idx, node.opkind());
        //        let nice: Tensor<i32> = input.into();
        //        println!("Node {} input tensor {:?}", node_idx, &input.clone());

        // The node kind and the config should be the same.
        match (node.opkind(), config) {
            (OpKind::Affine, OnnxNodeConfig::Affine(ac)) => {
                // This node should have three inputs in total:
                // two inputs which are Const(..) that have the f32s,
                // and one variable input which are the activations.
                // The first input is the activations, second is the weight matrix, and the third the bias.
                // Determine the input, and recursively assign the input node.

                let (_input_node_ix, _input_node_slot) =
                    (input_outlets[0].node, input_outlets[0].slot);
                // Properly we should check that weight and bias are Const ops
                let (weight_node_ix, weight_node_slot) =
                    (input_outlets[1].node, input_outlets[1].slot);
                let (bias_node_ix, bias_node_slot) = (input_outlets[2].node, input_outlets[2].slot);
                let weight_node = OnnxNode::new(self.nodes()[weight_node_ix].clone());
                let weight_value =
                    weight_node.output_tensor_by_slot(weight_node_slot, 0f32, 256f32);
                //                println!("Weight: {:?}", weight_value);
                // let in_dim = weight_value.dims()[1];
                // let out_dim = weight_value.dims()[0];
                let weight_vt =
                    ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(weight_value));
                //                let weight_vt = ValTensor::from(weight_value);
                let bias_node = OnnxNode::new(self.nodes()[bias_node_ix].clone());
                let bias_value = bias_node.output_tensor_by_slot(bias_node_slot, 0f32, 256f32);
                let bias_vt =
                    ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(bias_value));
                // println!(
                //     "input {:?} W {:?} b {:?}",
                //     input.dims(),
                //     weight_vt.dims(),
                //     bias_vt.dims()
                // );
                let out = ac.layout(layouter, input, &[weight_vt, bias_vt]);
                //                println!("Node {} out {:?}", node_idx, out);
                Some(out)
            }
            (OpKind::Convolution, OnnxNodeConfig::Conv(cc)) => {
                todo!()
            }
            (OpKind::ReLU, OnnxNodeConfig::ReLU(rc)) => {
                // For activations and elementwise operations, the dimensions are sometimes only in one or the other of input and output.
                //                let length = node.output_shapes().unwrap()[0].as_ref().unwrap()[1]; //  shape is vec![1,LEN]
                Some(rc.layout(layouter, input))
            }
            (OpKind::Sigmoid, OnnxNodeConfig::Sigmoid(sc)) => Some(sc.layout(layouter, input)),

            _ => {
                None //panic!("Node Op and Config mismatch, or unknown Op.")
            }
        }
    }

    /// Get a linear extension of the model (an evaluation order), for example to feed to circuit construction.
    /// Note that this order is not stable over multiple reloads of the model.  For example, it will freely
    /// interchange the order of evaluation of fixed parameters.   For example weight could have id 1 on one load,
    /// and bias id 2, and vice versa on the next load of the same file. The ids are also not stable.
    pub fn eval_order(&self) -> Result<Vec<usize>> {
        self.model.eval_order()
    }

    /// Note that this order is not stable.
    pub fn nodes(&self) -> Vec<Node<InferenceFact, Box<dyn InferenceOp>>> {
        self.model.nodes().clone().to_vec()
    }

    pub fn input_outlets(&self) -> Result<Vec<OutletId>> {
        Ok(self.model.input_outlets()?.to_vec())
    }

    pub fn output_outlets(&self) -> Result<Vec<OutletId>> {
        Ok(self.model.output_outlets()?.to_vec())
    }

    // In general we need to determine the input shapes by looking up the input tensors in the predeccesor nodes
    pub fn input_shapes_by_node_idx(&self, node_idx: usize) -> Result<Vec<Option<Vec<usize>>>> {
        let mut shapes = Vec::new();

        for OutletId { node, slot } in &self.model.nodes()[node_idx].inputs {
            let prec = OnnxNode::new(self.model.nodes()[*node].clone());
            let shapevec = prec.output_shapes()?;
            shapes.push(shapevec[*slot].clone());
        }
        Ok(shapes)
    }

    pub fn max_fixeds_width(&self) -> Result<usize> {
        self.max_advices_width() //todo, improve this computation
    }

    pub fn max_advices_width(&self) -> Result<usize> {
        let mut max: usize = 1;
        for node in &self.model.nodes {
            for shape in node_output_shapes(&node)? {
                match shape {
                    None => {}
                    Some(vs) => {
                        for v in vs {
                            if v > max {
                                max = v
                            }
                        }
                    }
                }
            }
        }
        Ok(max + 5)
    }

    pub fn get_node_output_shape_by_name_and_rank(
        &self,
        name: impl AsRef<str>,
        rank: usize,
    ) -> Vec<usize> {
        let node = self.model.node_by_name(name).unwrap();
        let fact = &node.outputs[rank].fact;
        let shape = fact.shape.clone().as_concrete_finite().unwrap().unwrap();
        shape.to_vec()
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
