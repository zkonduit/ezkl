use crate::nn::affine::Affine1dConfig;
use crate::nn::cnvrl::ConvConfig;
use crate::nn::eltwise::{EltwiseConfig, ReLu, Sigmoid};
use crate::nn::LayerConfig;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor, VarTensor};
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
pub enum OnnxNodeConfig<F: FieldExt + TensorType> {
    Affine(Affine1dConfig<F>),
    Conv(ConvConfig<F>),
    ReLU(EltwiseConfig<F, ReLu<F>>),
    Sigmoid(EltwiseConfig<F, Sigmoid<F, 128, 128>>),
    Const,
    Source,
    Input,
    NotConfigured,
}

#[derive(Clone)]
pub struct OnnxModelConfig<F: FieldExt + TensorType> {
    configs: Vec<OnnxNodeConfig<F>>,
    pub public_output: Column<Instance>,
}

/// Fields:
/// node is the raw Tract Node data structure
/// output_max is an inferred maximum value that can appear in the output tensor given previous quantization choices.
/// opkind: OpKind
/// scale: usize, // Scale begins at 1.0. When a weight or activation is multiplied by a, scale*=a. Except that scale is always a power of two, and this is that power; multiplication is usually multiplication, while division typically happens in a lookup.  Scale is a runtime type, in that tensors of differing scales should not be combined.
/// input_shapes and output_shapes  are Option<Vec<Option<Vec<usize>>>>.  These are the inferred shapes for input and output tensors. The first coordinate is the Onnx "slot" and the second is the tensor.
/// None indicates unknown, so input_shapes = Some(vec![None, Some(vec![3,4])]) indicates that we
/// know something, there are two slots, and the first tensor has unknown shape, while the second has shape [3,4].
#[derive(Clone, Debug)]
pub struct OnnxNode {
    node: Node<InferenceFact, Box<dyn InferenceOp>>, // the raw Tract Node data structure
    pub opkind: OpKind,
    output_max: f32, //
    qscale: usize, // Scale begins at 1.0. When a weight or activation is multiplied by a, scale*=a. Except that scale is always a power of two, and this qscale is that power.
    // Inferred shapes for input and output tensors. The first coordinate is the Onnx "slot" and the second is the tensor.
    // None indicates unknown, so input_shapes = Some(vec![None, Some(vec![3,4])]) indicates that we
    // know something, there are two slots, and the first tensor has unknown shape, while the second has shape [3,4].
    input_shapes: Option<Vec<Option<Vec<usize>>>>,
    output_shapes: Option<Vec<Option<Vec<usize>>>>,
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    in_dim: Option<Vec<usize>>,
    out_dim: Option<Vec<usize>>,
}

impl OnnxNode {
    pub fn new(node: Node<InferenceFact, Box<dyn InferenceOp>>) -> Self {
        let opkind = match node.op().name().as_ref() {
            "Gemm" => OpKind::Affine,
            "Conv" => OpKind::Convolution,
            "Clip" => OpKind::ReLU,
            "Sigmoid" => OpKind::Sigmoid,
            "Const" => OpKind::Const,
            "Source" => OpKind::Input,
            "input" => OpKind::Input,
            _ => OpKind::Unknown,
        };
        let output_shapes = match node_output_shapes(&node) {
            Ok(s) => Some(s),
            _ => None,
        };
        let on = OnnxNode {
            node,
            opkind,
            output_max: f32::INFINITY,
            qscale: 0,
            input_shapes: None,
            output_shapes,
	    None,
	    None,
        };
        println!("{:?}", on);
        on
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
    pub onnx_nodes: Vec<OnnxNode>, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    pub bits: usize,
    pub last_shape: Vec<usize>,
}

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();

        let onnx_nodes: Vec<OnnxNode> = model
            .nodes()
            .iter()
            .map(|n| OnnxNode::new(n.clone()))
            .collect();
        OnnxModel {
            model,
            onnx_nodes,
            bits: 14,
            last_shape: Vec::from([0]),
        }
    }
    pub fn from_arg() -> Self {
        let args: Vec<String> = env::args().collect();
        let filename: String = args[1].clone();
        OnnxModel::new(filename)
    }

    pub fn configure<F: FieldExt + TensorType>(
        &mut self,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
        fixeds: VarTensor,
    ) -> Result<OnnxModelConfig<F>> {
        // Note that the order of the nodes, and the eval_order, is not stable between model loads
        let order = self.eval_order()?;
        let mut configs: Vec<OnnxNodeConfig<F>> = vec![OnnxNodeConfig::NotConfigured; order.len()];
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
    fn configure_node<F: FieldExt + TensorType>(
        &mut self,
        node_idx: usize,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
        fixeds: VarTensor, // Should use fixeds, but currently buggy
    ) -> Result<OnnxNodeConfig<F>> {
        let node = &self.onnx_nodes[node_idx];
        //        println!("Configure Node {}, a {:?}", node_idx, node.opkind());

        // Figure out, find, and load the params
        match node.opkind {
            OpKind::Affine => {
                // The parameters are assumed to be fixed kernel and bias. This node should have three inputs in total:
                // two inputs which are Const(..) that have the f32s, and one variable input which are the activations.
                // The first input is the activations, second is the weight matrix, and the third the bias.
                // Consider using shape information only here, rather than loading the param tensor (although loading
                // the tensor guarantees that assign will work if there are errors or ambiguities in the shape
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
                let input = advices.get_slice(&[out_dim + 2..out_dim + 3], &[in_dim]);
                let output = advices.get_slice(&[out_dim + 3..out_dim + 4], &[out_dim]);
                let conf = Affine1dConfig::configure(
                    meta,
                    &[weight_fixeds, bias_fixeds, input, output],
                    None,
                );
                self.last_shape = Vec::from([out_dim]);
                Ok(OnnxNodeConfig::Affine(conf))
            }
            OpKind::ReLU => {
                let length = self.last_shape.clone().into_iter().product();

                let conf: EltwiseConfig<F, ReLu<F>> = EltwiseConfig::configure(
                    meta,
                    &[advices.get_slice(&[0..length], &[length])],
                    Some(&[self.bits]),
                );
                Ok(OnnxNodeConfig::ReLU(conf))
            }
            OpKind::Sigmoid => {
                // Here,   node.output_shapes().unwrap()[0].as_ref().unwrap() == vec![1,LEN]
                let length = node.output_shapes().unwrap()[0].as_ref().unwrap()[1];
                let conf: EltwiseConfig<F, Sigmoid<F, 128, 128>> = EltwiseConfig::configure(
                    meta,
                    &[advices.get_slice(&[0..length], &[length])],
                    Some(&[self.bits]),
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

    pub fn layout<F: FieldExt + TensorType>(
        &self,
        config: OnnxModelConfig<F>,
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
                    println!("Applying {:?}", self.onnx_nodes[node_idx]);
                    println!("Node {} out: {}", node_idx, vt.show());
                    // This is just to log the layer output
                    // match vt.clone() {
                    //     ValTensor::PrevAssigned { inner: v, dims: _ } => {
                    //         let r: Tensor<i32> = v.clone().into();
                    //         println!("Node {} out: {:?}", node_idx, r);
                    //     }
                    //     _ => panic!("Should be assigned"),
                    // };

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
    fn layout_node<F: FieldExt + TensorType>(
        &self,
        node_idx: usize,
        layouter: &mut impl Layouter<F>,
        input: ValTensor<F>,
        config: OnnxNodeConfig<F>,
    ) -> Option<ValTensor<F>> {
        let node = &self.onnx_nodes[node_idx];
        let input_outlets = &node.node.inputs;

        //        println!("Layout Node {}, {:?}", node_idx, node.opkind());
        //        let nice: Tensor<i32> = input.into();
        //        println!("Node {} input tensor {:?}", node_idx, &input.clone());

        // The node kind and the config should be the same.
        match (node.opkind, config) {
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
                let out = ac.layout(layouter, &[weight_vt, bias_vt, input]);
                //                println!("Node {} out {:?}", node_idx, out);
                Some(out)
            }
            (OpKind::Convolution, OnnxNodeConfig::Conv(cc)) => {
                todo!()
            }
            (OpKind::ReLU, OnnxNodeConfig::ReLU(rc)) => {
                // For activations and elementwise operations, the dimensions are sometimes only in one or the other of input and output.
                //                let length = node.output_shapes().unwrap()[0].as_ref().unwrap()[1]; //  shape is vec![1,LEN]
                Some(rc.layout(layouter, &[input]))
            }
            (OpKind::Sigmoid, OnnxNodeConfig::Sigmoid(sc)) => Some(sc.layout(layouter, &[input])),

            _ => {
                None //panic!("Node Op and Config mismatch, or unknown Op.")
            }
        }
    }

    /// Make a forward pass over the graph to determine tensor shapes and quantization strategy
    pub fn forward_shape_and_quantize_pass(&self) -> Result<()> {
        let order = self.eval_order()?;
        let mut last_output: Vec<usize> = Vec::new();
        let mut qscale = 0usize;
        let mut maximum_activation = 1.0f32;
        for node_idx in order {
            let mut this_node = &self.onnx_nodes[node_idx];
	    match this_node.opkind {
		OpKind::Affine => {
		    
		    maximum_activation = maximum_activation*this_node.qscale*;
		    
		}


	    };



	    
        }

        Ok(())
    }

    // Make a recursive backward pass to shape and quantize?

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
            //            let prec = OnnxNode::new(self.model.nodes()[*node].clone());
            let prec = &self.onnx_nodes[*node];
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
