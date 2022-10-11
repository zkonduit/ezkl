use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::circuit::basic::*;
use crate::circuit::eltwise::{DivideBy, EltwiseConfig, ReLu, Sigmoid};
use crate::commands::{model_path, Cli, Commands};
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor, VarTensor};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::{Column, ConstraintSystem, Instance},
};
use itertools::Itertools;
use log::{debug, error, info, trace, warn};
use std::cmp::max;
use std::fmt;
use std::path::Path;
use tabled::{Table, Tabled};
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, Node, OutletId};
use tract_onnx::tract_hir::{
    infer::Factoid,
    internal::InferenceOp,
    ops::cnn::Conv,
    ops::expandable::Expansion,
    ops::nn::DataFormat,
    tract_core::ops::{
        cnn::{conv::KernelFormat, PaddingSpec},
        konst::Const,
    },
};
// Initially, some of these OpKinds will be folded into others (for example, Const nodes that
// contain parameters will be handled at the consuming node.
// Eventually, though, we probably want to keep them and treat them directly (layouting and configuring
// at each type of node)
#[derive(Clone, Debug)]
pub enum OpKind {
    Rescaled(Box<OpKind>, usize),
    ReLU(usize),
    Sigmoid(usize),
    Const,
    Input,
    Fused(FusedOp),
    Unknown(String),
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpKind::Rescaled(l, _) => write!(f, "rescaled {}", (*l)),
            OpKind::ReLU(s) => write!(f, "relu {}", s),
            OpKind::Sigmoid(s) => write!(f, "sigmoid {}", s),
            OpKind::Const => write!(f, "const"),
            OpKind::Input => write!(f, "input"),
            OpKind::Fused(s) => write!(f, "{}", s),
            OpKind::Unknown(c) => write!(f, "? {}", c),
        }
    }
}

#[derive(Clone, Debug)]
pub enum NodeConfigTypes<F: FieldExt + TensorType> {
    Rescaled(EltwiseConfig<F, DivideBy<F>>, Box<NodeConfigTypes<F>>),
    ReLU(EltwiseConfig<F, ReLu<F>>),
    Sigmoid(EltwiseConfig<F, Sigmoid<F>>),
    Divide(EltwiseConfig<F, DivideBy<F>>),
    Fused(FusedConfig<F>, Vec<usize>),
    Const,
    Input,
    NotConfigured,
}

#[derive(Clone, Debug)]
pub struct NodeConfig<F: FieldExt + TensorType> {
    config: NodeConfigTypes<F>,
    onnx_idx: Vec<usize>,
}

#[derive(Clone)]
pub struct OnnxModelConfig<F: FieldExt + TensorType> {
    configs: Vec<NodeConfig<F>>,
    pub model: OnnxModel,
    pub public_output: Column<Instance>,
}

fn display_option<T: fmt::Debug>(o: &Option<T>) -> String {
    match o {
        Some(s) => format!("{:?}", s),
        None => String::new(),
    }
}

fn display_inputs(o: &Vec<OutletId>) -> String {
    if o.len() > 0 {
        let mut nodes = vec![];
        for id in o.iter() {
            nodes.push(id.node);
        }
        format!("{:?}", nodes)
    } else {
        String::new()
    }
}

fn display_tensor(o: &Option<Tensor<i32>>) -> String {
    match o {
        Some(s) => format!("[{:#?}...]", s[0]),
        None => String::new(),
    }
}

/// Fields:
/// node is the raw Tract Node data structure.
/// opkind: OpKind is our op enum.
/// output_max is an inferred maximum value that can appear in the output tensor given previous quantization choices.
/// in_scale and out_scale track the denominator in the fixed point representation. Tensors of differing scales should not be combined.
/// in_dims and out_dims are the shape of the activations only which enter and leave the node.
#[derive(Clone, Debug, Tabled)]
pub struct OnnxNode {
    node: Node<InferenceFact, Box<dyn InferenceOp>>,
    pub opkind: OpKind,
    output_max: f32,
    min_cols: usize,
    in_scale: i32,
    out_scale: i32,
    #[tabled(display_with = "display_tensor")]
    const_value: Option<Tensor<i32>>, // float value * 2^qscale if applicable.
    // Usually there is a simple in and out shape of the node as an operator.  For example, an Affine node has three input_shapes (one for the input, weight, and bias),
    // but in_dim is [in], out_dim is [out]
    #[tabled(display_with = "display_inputs")]
    inputs: Vec<OutletId>,
    #[tabled(display_with = "display_option")]
    in_dims: Option<Vec<usize>>,
    #[tabled(display_with = "display_option")]
    out_dims: Option<Vec<usize>>,
    idx: usize,
    #[tabled(display_with = "display_option")]
    inter_idx: Option<usize>,
}

impl OnnxNode {
    pub fn new(node: Node<InferenceFact, Box<dyn InferenceOp>>, scale: i32, idx: usize) -> Self {
        let mut opkind = match node.op().name().as_ref() {
            "Clip" => OpKind::ReLU(1),
            "Sigmoid" => OpKind::Sigmoid(1),
            "Const" => OpKind::Const,
            "Source" => OpKind::Input,
            "Add" => OpKind::Fused(FusedOp::Add),
            "Sub" => OpKind::Fused(FusedOp::Sub),
            "Mul" => OpKind::Fused(FusedOp::Mult),
            "Gemm" => OpKind::Fused(FusedOp::Affine),
            "MatMulInference" => OpKind::Fused(FusedOp::Matmul),
            "Dot" => OpKind::Fused(FusedOp::Dot),
            "Reduce<Sum>" => OpKind::Fused(FusedOp::Sum),
            "Pow" => OpKind::Fused(FusedOp::Pow(1)),
            "Conv" => OpKind::Fused(FusedOp::Conv((1, 1), (1, 1))),
            "ConvHir" => OpKind::Fused(FusedOp::Conv((1, 1), (1, 1))),

            c => {
                warn!("{:?} is not currently supported", c);
                OpKind::Unknown(c.to_string())
            }
        };
        let output_shapes = match node_output_shapes(&node) {
            Ok(s) => Some(s),
            _ => None,
        };

        // Set some default values, then figure out more specific values if possible based on the opkind.
        let min_cols = 1;
        let mut const_value = None;
        let in_scale = scale;
        let mut out_scale = 0i32;
        let in_dims = None;
        let mut out_dims = None;
        let mut output_max = f32::INFINITY;
        let inter_idx = None;

        match opkind {
            OpKind::Const => {
                // Extract the padding and stride layer hyperparams
                let op = Box::new(node.op());
                let const_node: &Const = match op.as_any().downcast_ref() {
                    Some(b) => b,
                    None => {
                        error!("op is not a const!");
                        panic!()
                    }
                };
                let vec = const_node.0.as_slice::<f32>().unwrap().to_vec();
                let mut dims = const_node.0.shape().to_vec();
                if dims.is_empty() {
                    dims.push(1)
                }
                out_scale = in_scale;
                let t = vector_to_quantized(&vec, &dims, 0f32, out_scale).unwrap();
                out_dims = Some(t.dims().to_vec());
                output_max = t.iter().map(|x| x.abs()).max().unwrap() as f32;
                const_value = Some(t);
            }
            OpKind::Input => {
                let dims = if let Some([Some(v)]) = output_shapes.as_deref() {
                    v.to_vec()
                } else {
                    // Turn  `outputs: [?,3,32,32,F32 >3/0]` into `vec![3,32,32]`  in two steps
                    let the_shape: Result<Vec<i64>> = node.outputs[0]
                        .fact
                        .shape
                        .dims()
                        .filter_map(|x| x.concretize())
                        .map(|x| x.to_i64())
                        .collect();

                    the_shape
                        .unwrap()
                        .iter()
                        .map(|x| (*x as i32) as usize)
                        .collect()
                };
                // remove batch dim for now
                if dims[0] == 1 && dims.len() > 1 {
                    out_dims = Some(dims[1..].to_vec())
                } else {
                    out_dims = Some(dims)
                }

                output_max = 256.0;
                out_scale = in_scale;
            }
            OpKind::Fused(FusedOp::Conv(_, _)) => {
                // Extract the padding and stride layer hyperparams
                let op = Box::new(node.op());

                let conv_node: &Conv = match op.downcast_ref::<Box<dyn Expansion>>() {
                    Some(b) => match (*b).as_any().downcast_ref() {
                        Some(b) => b,
                        None => {
                            error!("not a conv!");
                            panic!()
                        }
                    },
                    None => {
                        error!("op is not a Tract Expansion!");
                        panic!()
                    }
                };

                // only support pytorch type formatting for now
                assert_eq!(conv_node.data_format, DataFormat::NCHW);
                assert_eq!(conv_node.kernel_fmt, KernelFormat::OIHW);

                let stride = conv_node.strides.clone().unwrap();
                let padding = match &conv_node.padding {
                    PaddingSpec::Explicit(p, _, _) => p,
                    _ => panic!("padding is not explicitly specified"),
                };

                opkind = OpKind::Fused(FusedOp::Conv(
                    (padding[0], padding[1]),
                    (stride[0], stride[1]),
                ));
            }
            _ => {}
        }

        OnnxNode {
            inputs: node.inputs.clone(),
            node,
            opkind,
            output_max,
            min_cols,
            in_scale,
            out_scale,
            const_value,
            in_dims,
            out_dims,
            idx,
            inter_idx,
        }
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
        self.node.name.clone()
    }
}

#[derive(Clone, Debug, Default)]
struct ConfigureParams<F: FieldExt + TensorType> {
    basic_nodes: Vec<usize>,
    rescale_conf: Option<EltwiseConfig<F, DivideBy<F>>>,
    configs: Vec<NodeConfig<F>>,
}

#[derive(Clone, Debug, Default)]
struct FuseParams {
    shapes: Vec<Vec<usize>>,
    nodes: Vec<FusedNode>,
    inputs: Vec<usize>,
    out_dim: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct OnnxModel {
    pub model: Graph<InferenceFact, Box<dyn InferenceOp>>, // The raw Tract data structure
    pub onnx_nodes: Vec<OnnxNode>, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    pub bits: usize,
    pub scale: i32,
}

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>, scale: i32, bits: usize) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();

        let onnx_nodes: Vec<OnnxNode> = model
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| OnnxNode::new(n.clone(), scale, i))
            .collect();

        let mut om = OnnxModel {
            model,
            scale,
            onnx_nodes,
            bits,
        };
        om.forward_shape_and_quantize_pass().unwrap();

        debug!("{}", Table::new(om.onnx_nodes.clone()).to_string());

        om
    }
    pub fn from_arg() -> Self {
        let args = Cli::parse();

        match args.command {
            Commands::Table { model } => OnnxModel::new(model_path(model), args.scale, args.bits),
            Commands::Mock { data: _, model } => {
                OnnxModel::new(model_path(model), args.scale, args.bits)
            }
            Commands::Fullprove {
                data: _,
                model,
                pfsys: _,
            } => OnnxModel::new(model_path(model), args.scale, args.bits),
            // Commands::Vkey {
            //     model,
            //     output: _,
            //     params: _,
            //     pfsys: _,
            // } => OnnxModel::new(model_path(model), args.scale, args.bits),
            // Commands::Pkey {
            //     model,
            //     output: _,
            //     params: _,
            //     pfsys: _,
            // } => OnnxModel::new(model_path(model), args.scale, args.bits),
            Commands::Prove {
                data: _,
                model,
                output: _,
                pfsys: _,
            } => OnnxModel::new(model_path(model), args.scale, args.bits),
            Commands::Verify {
                model,
                proof: _,
                pfsys: _,
            } => OnnxModel::new(model_path(model), args.scale, args.bits),
        }
    }

    pub fn configure<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> Result<OnnxModelConfig<F>> {
        info!("configuring model");
        // Note that the order of the nodes, and the eval_order, is not stable between model loads
        let order = self.eval_order()?;
        let mut results = ConfigureParams::<F>::default();
        for node_idx in order.clone() {
            let node = &self.onnx_nodes[node_idx];
            debug!("configuring node {} a {:?}", node_idx, &node.opkind);
            self.configure_muxer(&node.opkind, meta, &mut results, node, advices.clone());
        }
        // one last cleanup
        if !results.basic_nodes.is_empty() {
            let config = self.configure_basic_ops(&results.basic_nodes, meta, advices);
            results.configs.push(NodeConfig {
                config,
                onnx_idx: vec![*order.last().unwrap()],
            });
        }
        // rescale output just in case final operation doesn't do it
        // let scale_diff = self.onnx_nodes.last().unwrap().out_scale - self.scale;
        // if scale_diff > 0 {
        //     let node = self.onnx_nodes.last().unwrap();
        //     let mult = scale_to_multiplier(scale_diff);
        //     let divconf = self.configure_divide_by(node, meta, advices.clone(), &(mult as usize));
        //     results.configs.push(NodeConfig {
        //         config: NodeConfigTypes::Divide(divconf),
        //         onnx_idx: vec![0],
        //     });
        // }
        //
        let public_output: Column<Instance> = meta.instance_column();
        meta.enable_equality(public_output);

        Ok(OnnxModelConfig {
            configs: results.configs,
            model: self.clone(),
            public_output,
        })
    }

    fn configure_muxer<F: FieldExt + TensorType>(
        &self,
        op: &OpKind,
        meta: &mut ConstraintSystem<F>,
        params: &mut ConfigureParams<F>,
        node: &OnnxNode,
        advices: VarTensor,
    ) {
        match op {
            OpKind::Rescaled(inner_op, denom) => {
                params.rescale_conf =
                    Some(self.configure_divide_by(node, meta, advices.clone(), denom));
                // when we divide by we start a new "basic" group of operations to fuse together
                if !params.basic_nodes.is_empty() {
                    let config =
                        self.configure_basic_ops(&params.basic_nodes, meta, advices.clone());
                    let config = NodeConfigTypes::Rescaled(
                        params.rescale_conf.clone().unwrap(),
                        Box::new(config),
                    );
                    params.configs.push(NodeConfig {
                        config,
                        onnx_idx: params.basic_nodes.clone(),
                    });
                    params.basic_nodes = vec![];
                }
                self.configure_muxer(inner_op, meta, params, node, advices);
            }
            OpKind::Fused(_) => params.basic_nodes.push(node.idx),
            OpKind::Input => params.configs.push(NodeConfig {
                config: self.configure_node(op, node, meta, advices),
                onnx_idx: vec![node.idx],
            }),
            OpKind::Const => params.configs.push(NodeConfig {
                config: self.configure_node(op, node, meta, advices),
                onnx_idx: vec![node.idx],
            }),
            OpKind::Unknown(c) => {
                error!("{:?} not yet implemented", c);
                unimplemented!()
            }
            _ => {
                if !params.basic_nodes.is_empty() {
                    let config =
                        self.configure_basic_ops(&params.basic_nodes, meta, advices.clone());
                    params.configs.push(NodeConfig {
                        config,
                        onnx_idx: params.basic_nodes.clone(),
                    });
                    params.basic_nodes = vec![];
                }
                let config = self.configure_node(op, node, meta, advices);
                let config = match params.rescale_conf.clone() {
                    Some(s) => {
                        params.rescale_conf = None;
                        NodeConfigTypes::Rescaled(s, Box::new(config))
                    }
                    None => config,
                };

                params.configs.push(NodeConfig {
                    config,
                    onnx_idx: vec![node.idx],
                });
            }
        }
    }

    fn extract_node_inputs(&self, node: &OnnxNode) -> Vec<&OnnxNode> {
        // The parameters are assumed to be fixed kernel and bias. Affine and Conv nodes should have three inputs in total:
        // two inputs which are Const(..) that have the f32s, and one variable input which are the activations.
        // The first input is the activations, second is the weight matrix, and the third the bias.
        // Consider using shape information only here, rather than loading the param tensor (although loading
        // the tensor guarantees that assign will work if there are errors or ambiguities in the shape
        // data).
        // Other layers such as non-linearities only have a single input (activations).
        let input_outlets = &node.node.inputs;
        let mut inputs = Vec::<&OnnxNode>::new();
        for i in input_outlets.iter() {
            inputs.push(&self.onnx_nodes[i.node]);
        }
        inputs
    }

    fn configure_basic_ops<F: FieldExt + TensorType>(
        &self,
        indices: &[usize],
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        let mut params = FuseParams::default();
        let mut offset = 0;
        for i in indices.iter() {
            let node = &self.onnx_nodes[*i];
            self.basic_op_muxer(
                &node.opkind,
                &mut params,
                node,
                indices,
                meta,
                advices.clone(),
                &mut offset,
            )
        }

        let mut variables = vec![];
        let mut start = 0;
        params.shapes.push(params.out_dim);
        // final iteration generates the output
        for s in params.shapes.iter() {
            let end: usize = if s.len() > 1 {
                s[0..s.len() - 1].iter().product()
            } else {
                1
            };
            variables.push(advices.get_slice(&[start..start + end], s));
            start += end;
        }

        NodeConfigTypes::Fused(
            FusedConfig::configure(meta, &variables, &params.nodes),
            params.inputs,
        )
    }

    fn basic_op_muxer<F: FieldExt + TensorType>(
        &self,
        op: &OpKind,
        params: &mut FuseParams,
        node: &OnnxNode,
        indices: &[usize],
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
        input_offset: &mut usize,
    ) {
        params.out_dim = node.out_dims.clone().unwrap();
        match &op {
            OpKind::Rescaled(inner_op, _) => {
                self.basic_op_muxer(inner_op, params, node, indices, meta, advices, input_offset)
            }
            OpKind::Fused(op) => {
                let mut input_order = vec![];
                let mut inter_counter = 0;
                for (idx, input) in node.node.inputs.iter().enumerate() {
                    if indices.contains(&input.node) {
                        input_order.push(FusedInputType::Inter(inter_counter));
                        inter_counter += 1;
                    } else if !params.inputs.contains(&input.node) {
                        params.inputs.push(input.node);
                        params
                            .shapes
                            .push(self.onnx_nodes[input.node].clone().out_dims.unwrap());
                        input_order.push(FusedInputType::Input(*input_offset + idx));
                    } else {
                        input_order.push(FusedInputType::Input(
                            params.inputs.iter().position(|&r| r == input.node).unwrap(),
                        ));
                    }
                }
                *input_offset = *input_offset + params.inputs.len() - 1;
                if let FusedOp::Pow(_) = op {
                    // the last node is just the const node for the power, which we have info for already
                    input_order.pop();
                    params.inputs.pop();
                    params.shapes.pop();
                }

                params.nodes.push(FusedNode {
                    op: *op,
                    input_order,
                })
            }
            s => {
                error!("For {:?} call configure_node instead", s);
                panic!();
            }
        }
    }

    fn configure_divide_by<F: FieldExt + TensorType>(
        &self,
        node: &OnnxNode,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
        denom: &usize,
    ) -> EltwiseConfig<F, DivideBy<F>> {
        let dims = match &node.in_dims {
            Some(v) => v,
            None => {
                error!("layer has no input shape");
                panic!()
            }
        };
        let length = dims.clone().into_iter().product();
        EltwiseConfig::configure(
            meta,
            &[advices.get_slice(&[0..length], &[length])],
            //&[advices.get_slice(&[0..length], dims)],
            Some(&[self.bits, *denom]),
        )
    }

    /// Infer the params, input, and output, and configure against the provided meta and Advice and Fixed columns.
    /// Note that we require the context of the Graph to complete this task.
    fn configure_node<F: FieldExt + TensorType>(
        &self,
        op: &OpKind,
        node: &OnnxNode,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        match op {
            OpKind::Rescaled(inner_op, _) => self.configure_node(inner_op, node, meta, advices),
            OpKind::ReLU(s) => {
                let dims = match &node.in_dims {
                    Some(v) => v,
                    None => {
                        error!("relu layer has no input shape");
                        panic!()
                    }
                };

                let length = dims.clone().into_iter().product();

                let conf: EltwiseConfig<F, ReLu<F>> = EltwiseConfig::configure(
                    meta,
                    &[advices.get_slice(&[0..length], &[length])],
                    Some(&[self.bits, *s]),
                );
                NodeConfigTypes::ReLU(conf)
            }
            OpKind::Sigmoid(denominator) => {
                let dims = match &node.in_dims {
                    Some(v) => v,
                    None => {
                        let _ = anyhow!("sigmoid layer has no input shape");
                        panic!()
                    }
                };

                let length = dims.clone().into_iter().product();
                let conf: EltwiseConfig<F, Sigmoid<F>> = EltwiseConfig::configure(
                    meta,
                    &[advices.get_slice(&[0..length], &[length])],
                    Some(&[
                        self.bits,
                        *denominator,
                        scale_to_multiplier(self.scale) as usize,
                    ]),
                );
                NodeConfigTypes::Sigmoid(conf)
            }
            OpKind::Const => {
                // Typically parameters for one or more layers.
                // Currently this is handled in the consuming node(s), but will be moved here.
                NodeConfigTypes::Const
            }
            OpKind::Input => {
                // This is the input to the model (e.g. the image).
                // Currently this is handled in the consuming node(s), but will be moved here.
                NodeConfigTypes::Input
            }
            OpKind::Fused(s) => {
                error!("For {:?} call fuse_basic_ops instead", s);
                panic!();
            }
            OpKind::Unknown(c) => {
                error!("{:?} not yet implemented", c);
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
        info!("model layout");
        let mut x = vec![input];
        for node_config in config.configs.iter() {
            let mut display: String = "".to_string();
            for (i, idx) in node_config.onnx_idx[0..].iter().enumerate() {
                let node = &self.onnx_nodes[*idx];
                if i > 0 {
                    display.push_str(&format!(
                        "| combined with node {}, a {:?} |",
                        idx,
                        node.node.op().name()
                    ));
                } else {
                    display.push_str(&format!(
                        "laying out node {}, a {:?} ",
                        idx,
                        node.node.op().name()
                    ));
                }
            }
            info!("{}", display);
            let node = &self.onnx_nodes[*node_config.onnx_idx.last().unwrap()];
            // assert!(self.config_matches(&node.opkind, &node_config.config));
            match self.layout_config(node, layouter, &mut x, &node_config.config)? {
                Some(vt) => x.push(vt),
                None => {} // Some nodes don't produce tensor output, we skip these
            };
            trace!("  output {}", x.last().unwrap().show()); //only use with mock prover
        }

        Ok(x.last().unwrap().clone())
    }

    // Takes an input ValTensor; alternatively we could recursively layout all the predecessor tensors
    // (which may be more correct for some graphs).
    // Does not take parameters, instead looking them up in the network.
    // At the Source level, the input will be fed by the prover.
    fn layout_config<F: FieldExt + TensorType>(
        &self,
        node: &OnnxNode,
        layouter: &mut impl Layouter<F>,
        inputs: &mut Vec<ValTensor<F>>,
        config: &NodeConfigTypes<F>,
    ) -> Result<Option<ValTensor<F>>> {
        // The node kind and the config should be the same.
        let res = match config.clone() {
            NodeConfigTypes::Fused(mut ac, idx) => {
                println!("{:?} {:?}", inputs.len(), idx);
                let mut values = self
                    .idx_to_inter_idx(&idx)
                    .iter()
                    .map(|i| inputs[*i].clone())
                    .collect_vec();

                println!("{:?} {:?}", values.len(), idx);

                for i in idx.iter() {
                    let node = &self.onnx_nodes[*i];
                    if let OpKind::Const = node.opkind {
                        let val = node
                            .const_value
                            .clone()
                            .context("Tensor<i32> should already be loaded")?;
                        values.push(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(val).into());
                    }
                }
                Some(ac.layout(layouter, &values))
            }
            NodeConfigTypes::Rescaled(dc, op) => {
                let last = inputs.last_mut().unwrap();
                *last = dc.layout(layouter, &[last.clone()]);
                return self.layout_config(node, layouter, inputs, &*op);
            }
            NodeConfigTypes::Divide(dc) => {
                Some(dc.layout(layouter, &[inputs.last().unwrap().clone()]))
            }
            NodeConfigTypes::ReLU(rc) => {
                // For activations and elementwise operations, the dimensions are sometimes only in one or the other of input and output.
                Some(rc.layout(layouter, &[inputs.last().unwrap().clone()]))
            }
            NodeConfigTypes::Sigmoid(sc) => {
                Some(sc.layout(layouter, &[inputs.last().unwrap().clone()]))
            }

            NodeConfigTypes::Input => None,
            NodeConfigTypes::Const => None,
            _ => {
                panic!("Node Op and Config mismatch, or unknown Op ",)
            }
        };
        Ok(res)
    }

    // remove for now as matching has become more complex
    // fn config_matches<F: FieldExt + TensorType>(
    //     &self,
    //     op: &OpKind,
    //     config: &NodeConfigTypes<F>,
    // ) -> bool {
    //     // The node kind and the config should be the same.
    //     match (op, config.clone()) {
    //         (OpKind::Rescaled(op, _), NodeConfigTypes::Rescaled(_, config)) => {
    //             self.config_matches(op, &*config)
    //         }
    //         (OpKind::ReLU(_), NodeConfigTypes::ReLU(_)) => true,
    //         (OpKind::Sigmoid(_), NodeConfigTypes::Sigmoid(_)) => true,
    //         (OpKind::Input, NodeConfigTypes::Input) => true,
    //         (OpKind::Fused(_), NodeConfigTypes::Fused(_, _)) => true,
    //         (OpKind::Const, NodeConfigTypes::Const) => true,
    //         (_, NodeConfigTypes::Divide(_)) => true,
    //         _ => false,
    //     }
    // }

    /// Make a forward pass over the graph to determine tensor shapes and quantization strategy
    /// Mutates the nodes.
    pub fn forward_shape_and_quantize_pass(&mut self) -> Result<()> {
        info!("quantizing model activations");
        let order = self.eval_order()?;
        let order_len = order.len();

        let mut inter_idx = 0;

        for node_idx in order {
            // mutate a copy of the node, referring to other nodes in the vec, then swap modified node in at the end
            let mut node = self.onnx_nodes[node_idx].clone();

            let inputs = self.extract_node_inputs(&node);

            match node.opkind {
                OpKind::Sigmoid(_) => {
                    let input_node = inputs[0];
                    node.in_dims = input_node.out_dims.clone();
                    node.out_dims = input_node.out_dims.clone();
                    node.in_scale = input_node.out_scale;
                    node.out_scale = self.scale;
                    let scale_diff = node.in_scale;
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        node.opkind = OpKind::Sigmoid(mult as usize);
                    }

                    node.output_max = scale_to_multiplier(node.out_scale);

                    node.min_cols = max(1, node.in_dims.as_ref().unwrap().iter().product());
                }

                OpKind::ReLU(_) => {
                    let input_node = inputs[0];
                    node.in_dims = input_node.out_dims.clone();
                    node.out_dims = input_node.out_dims.clone();
                    node.output_max = input_node.output_max;
                    node.in_scale = input_node.out_scale;
                    node.out_scale = self.scale;
                    let scale_diff = node.in_scale - node.out_scale;
                    // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        node.opkind = OpKind::ReLU(mult as usize); // now the input will be scaled down to match
                        node.output_max = input_node.output_max / mult;
                    }
                    node.min_cols = max(1, node.in_dims.as_ref().unwrap().iter().product());
                }
                OpKind::Fused(s) => {
                    let input_node = inputs[0];
                    node.in_dims = input_node.out_dims.clone();
                    node.out_dims = input_node.out_dims.clone();
                    node.min_cols = inputs
                        .iter()
                        .map(|input| {
                            input.out_dims.clone().unwrap().iter().product::<usize>() as f32
                        })
                        .sum::<f32>() as usize;

                    inputs
                        .iter()
                        .tuple_windows()
                        .all(|(a, b)| a.in_scale == b.in_scale);
                    match s {
                        FusedOp::Dot => todo!(),
                        FusedOp::Conv(padding, stride) => {
                            let (input_node, weight_node, bias_node) =
                                (inputs[0], inputs[1], inputs[2]);

                            let oihw = weight_node.out_dims.as_ref().unwrap();
                            let (out_channels, _, kernel_height, kernel_width) =
                                (oihw[0], oihw[1], oihw[2], oihw[3]);

                            let (padding_h, padding_w, stride_h, stride_w) =
                                (padding.0, padding.1, stride.0, stride.1);

                            node.in_dims = input_node.out_dims.clone();

                            let input_height = node.in_dims.as_ref().unwrap()[1];
                            let input_width = node.in_dims.as_ref().unwrap()[2];

                            let out_height =
                                (input_height + 2 * padding_h - kernel_height) / stride_h + 1;
                            let out_width =
                                (input_width + 2 * padding_w - kernel_width) / stride_w + 1;

                            node.out_dims = Some(vec![out_channels, out_height, out_width]);

                            node.output_max = input_node.output_max
                                * weight_node.output_max
                                * ((kernel_height * kernel_width) as f32);

                            node.in_scale = input_node.out_scale;
                            assert_eq!(weight_node.out_scale, bias_node.out_scale);

                            node.out_scale = weight_node.out_scale + input_node.out_scale;
                        }
                        FusedOp::Matmul => {
                            let (a_node, b_node) = (inputs[0], inputs[1]);

                            let in_dim = a_node.out_dims.as_ref().unwrap()[1];
                            node.in_dims = Some(vec![in_dim]);

                            let a_dims = a_node.clone().out_dims.unwrap();
                            let b_dims = b_node.clone().out_dims.unwrap();
                            let mut dims = Vec::from(&a_dims[0..a_dims.len() - 2]);
                            dims.push(a_dims[a_dims.len() - 2]);
                            dims.push(b_dims[a_dims.len() - 1]);

                            node.out_dims = Some(dims.clone());

                            node.output_max =
                                input_node.output_max * a_node.output_max * (in_dim as f32);

                            node.in_scale = input_node.out_scale;

                            node.out_scale = a_node.out_scale + input_node.out_scale;
                        }
                        FusedOp::Affine => {
                            let (input_node, weight_node, bias_node) =
                                (inputs[0], inputs[1], inputs[2]);

                            let in_dim = weight_node.out_dims.as_ref().unwrap()[1];
                            let out_dim = weight_node.out_dims.as_ref().unwrap()[0];
                            node.in_dims = Some(vec![in_dim]);
                            node.out_dims = Some(vec![out_dim]);

                            node.output_max =
                                input_node.output_max * weight_node.output_max * (in_dim as f32);

                            node.in_scale = input_node.out_scale;

                            assert_eq!(weight_node.out_scale, bias_node.out_scale);
                            node.out_scale = weight_node.out_scale + input_node.out_scale;
                        }
                        FusedOp::Add => {
                            node.output_max = (inputs
                                .iter()
                                .map(|input| input.output_max.ceil() as i32)
                                .max()
                                .unwrap() as f32)
                                * (inputs.len() as f32);
                            node.in_scale =
                                inputs.iter().map(|input| input.out_scale).max().unwrap();
                            node.out_scale = node.in_scale;
                        }
                        FusedOp::Sum => {
                            node.output_max = inputs
                                .iter()
                                .map(|input| {
                                    input.output_max
                                        * input.in_dims.clone().unwrap().iter().product::<usize>()
                                            as f32
                                })
                                .sum::<f32>();
                            node.in_scale =
                                inputs.iter().map(|input| input.out_scale).max().unwrap();
                            node.out_scale = node.in_scale;
                            node.out_dims = Some(vec![1]);
                        }
                        FusedOp::Sub => {
                            node.output_max = (inputs
                                .iter()
                                .map(|input| input.output_max.ceil() as i32)
                                .max()
                                .unwrap() as f32)
                                * (inputs.len() as f32);
                            node.in_scale =
                                inputs.iter().map(|input| input.out_scale).max().unwrap();
                            node.out_scale = node.in_scale;
                        }
                        FusedOp::Mult => {
                            node.output_max = f32::powf(
                                inputs
                                    .iter()
                                    .map(|input| input.output_max.ceil() as i32)
                                    .max()
                                    .unwrap() as f32,
                                inputs.len() as f32,
                            );
                            node.in_scale = input_node.out_scale;
                            node.out_scale =
                                inputs.iter().map(|input| input.out_scale).sum::<i32>();
                        }
                        FusedOp::Pow(_) => {
                            let mult = scale_to_multiplier(self.scale);
                            let pow = inputs[1].output_max / mult;
                            node.output_max = f32::powf(
                                inputs
                                    .iter()
                                    .map(|input| input.output_max.ceil() as i32)
                                    .max()
                                    .unwrap() as f32,
                                pow as f32,
                            );
                            node.in_scale = input_node.out_scale;
                            node.out_scale = node.in_scale * (pow as i32);

                            node.opkind = OpKind::Fused(FusedOp::Pow(pow as usize));
                        }
                    }
                    let scale_diff = node.in_scale - self.scale;
                    // output size
                    node.min_cols += node.out_dims.clone().unwrap()
                        [0..node.out_dims.clone().unwrap().len() - 1]
                        .iter()
                        .product::<usize>()
                        + 1;
                    if scale_diff > 0 {
                        // new op node

                        node.inter_idx = Some(inter_idx);
                        let mult = scale_to_multiplier(scale_diff);
                        node.opkind = OpKind::Rescaled(Box::new(OpKind::Fused(s)), mult as usize); // now the input will be scaled down to match
                        node.output_max /= mult;
                        node.out_scale -= scale_diff;
                        node.min_cols = max(
                            node.min_cols,
                            node.in_dims.as_ref().unwrap().iter().product(),
                        );
                    }
                }
                OpKind::Input => {}
                _ => node.inter_idx = None,
            };

            self.calc_inter_idx(&mut node, &mut inter_idx, order_len, node_idx);
            self.onnx_nodes[node_idx] = node;
        }

        Ok(())
    }

    pub fn calc_inter_idx(
        &self,
        node: &mut OnnxNode,
        inter_idx: &mut usize,
        num_nodes: usize,
        node_idx: usize,
    ) {
        match node.opkind {
            OpKind::Const => node.inter_idx = None,
            _ => node.inter_idx = Some(*inter_idx),
        }

        if node_idx < num_nodes - 1 {
            let mut next_node = &self.onnx_nodes[node_idx + 1];
            for n in self.onnx_nodes[node_idx + 1..].iter() {
                match n.opkind {
                    OpKind::Const => continue,
                    _ => {
                        next_node = n;
                        break;
                    }
                }
            }
            match (&node.opkind, &next_node.opkind) {
                (OpKind::Const, _) => {}
                (OpKind::Fused(_), OpKind::Fused(_)) => {}
                (_, _) => *inter_idx += 1,
            }
        }

        if node_idx > 0 {
            let mut prev_node = &self.onnx_nodes[node_idx - 1];
            for n in self.onnx_nodes[0..node_idx].iter().rev() {
                match n.opkind {
                    OpKind::Const => continue,
                    _ => {
                        prev_node = n;
                        break;
                    }
                }
            }
            match (&prev_node.opkind, &node.opkind) {
                (OpKind::Fused(_), OpKind::Rescaled(_, _)) => {
                    node.inter_idx = Some(*inter_idx);
                }
                _ => {}
            }
        }
    }

    pub fn idx_to_inter_idx(&self, indices: &[usize]) -> Vec<usize> {
        let mut result = vec![];
        for i in indices {
            if !self.onnx_nodes[*i].inter_idx.is_none() {
                result.push(self.onnx_nodes[*i].inter_idx.unwrap());
            }
        }
        result
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
        self.model.nodes().to_vec()
    }

    pub fn input_outlets(&self) -> Result<Vec<OutletId>> {
        Ok(self.model.input_outlets()?.to_vec())
    }

    pub fn output_outlets(&self) -> Result<Vec<OutletId>> {
        Ok(self.model.output_outlets()?.to_vec())
    }

    pub fn max_fixeds_width(&self) -> Result<usize> {
        self.max_advices_width() //todo, improve this computation
    }

    pub fn max_node_advices(&self) -> usize {
        self.onnx_nodes.iter().map(|n| n.min_cols).max().unwrap()
    }

    pub fn max_advices_width(&self) -> Result<usize> {
        let mut max: usize = 1;
        for node in &self.model.nodes {
            for shape in node_output_shapes(node)? {
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
}
