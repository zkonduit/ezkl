use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::circuit::eltwise::{DivideBy, EltwiseConfig, ReLu, Sigmoid};
use crate::circuit::fused::*;
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
use std::collections::{hash_map::Entry, HashMap};
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
#[derive(Clone, Debug, PartialEq)]
pub enum OpKind {
    Rescaled(Box<OpKind>, usize),
    ReLU(usize),
    Sigmoid(usize),
    Const,
    Input,
    Fused(FusedOp),
    Unknown(String),
}

impl OpKind {
    fn is_fused(&self) -> bool {
        matches!(self, OpKind::Fused(_))
    }

    fn is_const(&self) -> bool {
        matches!(self, OpKind::Const)
    }
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

#[derive(Clone, Default, Debug)]
pub enum NodeConfigTypes<F: FieldExt + TensorType> {
    Rescaled(EltwiseConfig<F, DivideBy<F>>, Box<NodeConfigTypes<F>>),
    ReLU(EltwiseConfig<F, ReLu<F>>, Vec<usize>),
    Sigmoid(EltwiseConfig<F, Sigmoid<F>>, Vec<usize>),
    Divide(EltwiseConfig<F, DivideBy<F>>),
    Fused(FusedConfig<F>, Vec<usize>),
    Const,
    Input,
    #[default]
    NotConfigured,
}

#[derive(Clone, Default, Debug)]
pub struct NodeConfig<F: FieldExt + TensorType> {
    config: NodeConfigTypes<F>,
    onnx_idx: Vec<usize>,
}

#[derive(Clone)]
pub struct OnnxModelConfig<F: FieldExt + TensorType> {
    configs: NodeGraph<NodeConfig<F>>,
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
    if !o.is_empty() {
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
    bucket: Option<usize>,
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
        let bucket = None;

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
            bucket,
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

#[derive(Clone, Default, Debug)]
pub struct NodeGraph<T: Clone>(HashMap<Option<usize>, HashMap<usize, T>>);

impl<T: Clone> NodeGraph<T> {
    pub fn new() -> Self {
        NodeGraph(HashMap::new())
    }

    fn insert(&mut self, idx: Option<usize>, node_idx: usize, config: T) {
        match self.0.entry(idx) {
            Entry::Vacant(e) => {
                e.insert(HashMap::from([(node_idx, config)]));
            }
            Entry::Occupied(mut e) => {
                e.get_mut().insert(node_idx, config);
            }
        }
    }
}

impl NodeGraph<OnnxNode> {
    pub fn flatten(&self) -> Vec<OnnxNode> {
        let a = self
            .0
            .clone()
            .into_values()
            .map(|d| d.into_values().collect())
            .collect::<Vec<Vec<OnnxNode>>>();
        let mut c: Vec<OnnxNode> = a
            .iter()
            .flatten()
            .collect::<Vec<&OnnxNode>>()
            .iter()
            .map(|e| (*e).clone())
            .collect();

        c.sort_by_key(|v| v.idx);
        c
    }

    pub fn filter(&self, idx: usize) -> OnnxNode {
        let a = self.flatten();
        let c = &a
            .iter()
            .filter(|i| i.idx == idx)
            .cloned()
            .collect::<Vec<OnnxNode>>()[0];
        c.clone()
    }
}

#[derive(Clone, Debug)]
pub enum Mode {
    Table,
    Mock,
    Prove,
    FullProve,
    Verify,
}

#[derive(Clone, Debug)]
pub struct OnnxModel {
    pub model: Graph<InferenceFact, Box<dyn InferenceOp>>, // The raw Tract data structure
    pub onnx_nodes: NodeGraph<OnnxNode>, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    pub bits: usize,
    pub scale: i32,
    pub mode: Mode,
}

impl OnnxModel {
    pub fn new(path: impl AsRef<Path>, scale: i32, bits: usize, mode: Mode) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();

        let onnx_nodes: HashMap<usize, OnnxNode> = model
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| (i, OnnxNode::new(n.clone(), scale, i)))
            .collect();

        let mut map = HashMap::new();
        map.insert(None, onnx_nodes);
        let mut om = OnnxModel {
            model,
            scale,
            onnx_nodes: NodeGraph(map),
            bits,
            mode,
        };

        om.forward_shape_and_quantize_pass().unwrap();

        debug!("{}", Table::new(om.onnx_nodes.flatten()).to_string());

        om
    }
    pub fn from_arg() -> Self {
        let args = Cli::parse();

        match args.command {
            Commands::Table { model } => {
                OnnxModel::new(model_path(model), args.scale, args.bits, Mode::Table)
            }
            Commands::Mock { data: _, model } => {
                OnnxModel::new(model_path(model), args.scale, args.bits, Mode::Mock)
            }
            Commands::Fullprove {
                data: _,
                model,
                pfsys: _,
            } => OnnxModel::new(model_path(model), args.scale, args.bits, Mode::FullProve),
            Commands::Prove {
                data: _,
                model,
                output: _,
                pfsys: _,
            } => OnnxModel::new(model_path(model), args.scale, args.bits, Mode::Prove),
            Commands::Verify {
                model,
                proof: _,
                pfsys: _,
            } => OnnxModel::new(model_path(model), args.scale, args.bits, Mode::Verify),
        }
    }

    pub fn configure<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> Result<OnnxModelConfig<F>> {
        info!("configuring model");
        let mut results = NodeGraph::default();

        for (b, block_nodes) in self.onnx_nodes.0.iter() {
            let non_fused_ops: HashMap<&usize, &OnnxNode> = block_nodes
                .iter()
                .filter(|(_, n)| !n.opkind.is_fused())
                .collect();

            if !non_fused_ops.is_empty() {
                for (i, n) in non_fused_ops.iter() {
                    results.insert(
                        *b,
                        **i,
                        NodeConfig {
                            config: self.configure_table(n, meta, advices.clone()),
                            onnx_idx: vec![**i],
                        },
                    );
                }
            }

            // preserves ordering
            let fused_ops: HashMap<&usize, &OnnxNode> = block_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_fused())
                .collect();
            if !fused_ops.is_empty() {
                results.insert(
                    *b,
                    **fused_ops.keys().max().unwrap(),
                    NodeConfig {
                        config: self.fuse_ops(
                            HashMap::from(fused_ops.clone()),
                            meta,
                            advices.clone(),
                        ),
                        onnx_idx: fused_ops.keys().map(|k| **k).sorted().collect_vec(),
                    },
                );
            }
        }

        let public_output: Column<Instance> = meta.instance_column();
        meta.enable_equality(public_output);

        Ok(OnnxModelConfig {
            configs: results,
            model: self.clone(),
            public_output,
        })
    }

    fn fuse_ops<F: FieldExt + TensorType>(
        &self,
        nodes: HashMap<&usize, &OnnxNode>,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        let input_nodes: HashMap<(&usize, FusedOp), Vec<OnnxNode>> = nodes
            .iter()
            .map(|(i, e)| {
                let op = match e.opkind {
                    OpKind::Fused(f) => f.clone(),
                    _ => panic!(),
                };
                (
                    (*i, op),
                    e.inputs
                        .iter()
                        .map(|i| self.onnx_nodes.filter(i.node))
                        .collect_vec(),
                )
            })
            .collect();

        let inputs_to_fused_layer: Vec<usize> = input_nodes
            .values()
            .flat_map(|e| {
                e.iter()
                    .filter(|i| !nodes.contains_key(&i.idx))
                    .map(|f| f.idx)
                    .collect_vec()
            })
            .collect_vec();

        let mut inter_counter = 0;
        let input_order: HashMap<(&usize, FusedOp), Vec<FusedInputType>> = input_nodes
            .iter()
            .sorted_by_key(|x| x.0 .0)
            .map(|(i, inputs)| {
                (
                    *i,
                    inputs
                        .iter()
                        .map(|i| {
                            if nodes.contains_key(&i.idx) {
                                inter_counter += 1;
                                FusedInputType::Inter(inter_counter - 1)
                            } else {
                                FusedInputType::Input(
                                    inputs_to_fused_layer
                                        .iter()
                                        .position(|&r| r == i.idx)
                                        .unwrap(),
                                )
                            }
                        })
                        .collect_vec(),
                )
            })
            .collect();

        let mut start = 0;
        let mut shapes: Vec<Vec<usize>> = inputs_to_fused_layer
            .iter()
            .map(|i| self.onnx_nodes.filter(*i).out_dims.unwrap())
            .collect();
        // output node
        shapes.push(
            self.onnx_nodes
                .filter(**nodes.keys().max().unwrap())
                .out_dims
                .unwrap(),
        );
        let variables: Vec<VarTensor> = shapes
            .iter()
            .map(|s| {
                let mut end = 1;
                if s.len() > 1 {
                    end = s[0..s.len() - 1].iter().product();
                }
                let a = advices.get_slice(&[start..start + end], s);
                start += end;
                a
            })
            .collect();

        let mut fused_nodes = vec![];
        input_order
            .iter()
            .sorted_by_key(|x| x.0 .0)
            .map(|((_, op), order)| {
                fused_nodes.push(FusedNode {
                    op: *op,
                    input_order: order.clone(),
                })
            })
            .collect_vec();

        NodeConfigTypes::Fused(
            FusedConfig::configure(
                meta,
                &variables[0..variables.len() - 1],
                variables.last().unwrap(),
                &fused_nodes,
            ),
            inputs_to_fused_layer,
        )
    }

    /// Infer the params, input, and output, and configure against the provided meta and Advice and Fixed columns.
    /// Note that we require the context of the Graph to complete this task.
    fn configure_table<F: FieldExt + TensorType>(
        &self,
        node: &OnnxNode,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        match &node.opkind {
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
                let inputs = node.inputs.iter().map(|e| e.node).collect();
                NodeConfigTypes::ReLU(conf, inputs)
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
                let inputs = node.inputs.iter().map(|e| e.node).collect();
                NodeConfigTypes::Sigmoid(conf, inputs)
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
                error!("For {:?} call fuse_fused_ops instead", s);
                panic!();
            }
            OpKind::Unknown(c) => {
                error!("{:?} not yet implemented", c);
                unimplemented!()
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
        inputs: &[ValTensor<F>],
    ) -> Result<ValTensor<F>> {
        info!("model layout");
        let mut results = HashMap::<usize, ValTensor<F>>::new();
        for i in inputs.iter().enumerate() {
            results.insert(i.0, i.1.clone());
        }
        for (bucket, node_config) in config.configs.0.iter().sorted_by_key(|x| x.0) {
            match bucket {
                // assert!(self.config_matches(&node.opkind, &node_config.config));
                Some(_b) => {
                    for (idx, c) in node_config.iter().sorted_by_key(|x| x.0) {
                        let mut display: String = "".to_string();
                        for (i, idx) in c.onnx_idx[0..].iter().enumerate() {
                            let node = &self.onnx_nodes.filter(*idx);
                            if i > 0 {
                                display.push_str(&format!(
                                    "| combined with node {} ({:?}) ",
                                    idx,
                                    node.node.op().name()
                                ));
                            } else {
                                display.push_str(&format!(
                                    "laying out node {} ({:?}) ",
                                    idx,
                                    node.node.op().name()
                                ));
                            }
                        }

                        info!("{}", display);

                        if let Some(vt) = self.layout_config(layouter, &mut results, c)? {
                            // we get the max as for fused nodes this corresponds to the node output
                            results.insert(*idx, vt);
                            //only use with mock prover
                            if matches!(self.mode, Mode::Mock) {
                                trace!("  output {:?}", results.get(&idx).unwrap().show());
                            }
                        }
                    }
                }
                None => {} // Some nodes don't produce tensor output, we skip these
            };
        }
        Ok(results
            .get(&(results.keys().max().unwrap()))
            .unwrap()
            .clone())
    }

    // Takes an input ValTensor; alternatively we could recursively layout all the predecessor tensors
    // (which may be more correct for some graphs).
    // Does not take parameters, instead looking them up in the network.
    // At the Source level, the input will be fed by the prover.
    fn layout_config<F: FieldExt + TensorType>(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &mut HashMap<usize, ValTensor<F>>,
        config: &NodeConfig<F>,
    ) -> Result<Option<ValTensor<F>>> {
        // The node kind and the config should be the same.
        let res = match config.config.clone() {
            NodeConfigTypes::Fused(mut ac, idx) => {
                let values: Vec<ValTensor<F>> = idx
                    .iter()
                    .map(|i| {
                        let node = &self.onnx_nodes.filter(*i);
                        match node.bucket {
                            None => {
                                let val = node
                                    .const_value
                                    .clone()
                                    .context("Tensor<i32> should already be loaded")
                                    .unwrap();
                                <Tensor<i32> as Into<Tensor<Value<F>>>>::into(val).into()
                            }
                            Some(_b) => inputs.get(&i).unwrap().clone(),
                        }
                    })
                    .collect_vec();

                Some(ac.layout(layouter, &values))
            }
            NodeConfigTypes::ReLU(rc, idx) => {
                assert_eq!(idx.len(), 1);
                // For activations and elementwise operations, the dimensions are sometimes only in one or the other of input and output.
                Some(rc.layout(layouter, &[inputs.get(&idx[0]).unwrap().clone()]))
            }
            NodeConfigTypes::Sigmoid(sc, idx) => {
                println!("idx {:?}", idx);
                assert_eq!(idx.len(), 1);

                Some(sc.layout(layouter, &[inputs.get(&idx[0]).unwrap().clone()]))
            }
            NodeConfigTypes::Input => None,
            c => {
                panic!("Not a configurable op {:?}", c)
            }
        };
        Ok(res)
    }

    /// Make a forward pass over the graph to determine tensor shapes and quantization strategy
    /// Mutates the nodes.
    pub fn forward_shape_and_quantize_pass(&mut self) -> Result<()> {
        info!("quantizing model activations");
        let order = self.eval_order()?;

        let mut nodes = NodeGraph(HashMap::<Option<usize>, HashMap<usize, OnnxNode>>::new());
        for node_idx in order.clone() {
            let mut node = self.onnx_nodes.filter(node_idx);
            let inputs: Vec<OnnxNode> = node
                .node
                .inputs
                .iter()
                .map(|i| nodes.filter(i.node))
                .collect();

            match node.opkind {
                OpKind::Sigmoid(_) => {
                    let input_node = &inputs[0];
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
                    let input_node = &inputs[0];
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
                    let input_node = &inputs[0];
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
                                (&inputs[0], &inputs[1], &inputs[2]);

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
                            let (a_node, b_node) = (&inputs[0], &inputs[1]);

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
                                (&inputs[0], &inputs[1], &inputs[2]);

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
                            node.inputs.pop();
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
                    // output size
                    node.min_cols += node.out_dims.clone().unwrap()
                        [0..node.out_dims.clone().unwrap().len() - 1]
                        .iter()
                        .product::<usize>()
                        + 1;
                }
                OpKind::Input => {}
                _ => {}
            };

            nodes.insert(node.bucket, node.idx, node.clone());
        }

        let bucketed_nodes = self.assign_execution_buckets(nodes, order);

        self.onnx_nodes = bucketed_nodes;

        Ok(())
    }

    pub fn assign_execution_buckets(
        &self,
        nodes: NodeGraph<OnnxNode>,
        order: Vec<usize>,
    ) -> NodeGraph<OnnxNode> {
        info!("assigning execution buckets to operations");

        let mut bucketed_nodes =
            NodeGraph(HashMap::<Option<usize>, HashMap<usize, OnnxNode>>::new());

        for node_idx in order.clone() {
            let mut node = nodes.filter(node_idx);

            let prev_bucket: Option<usize> = node
                .inputs
                .iter()
                .filter(|n| !bucketed_nodes.filter(n.node).opkind.is_const())
                .map(|n| match bucketed_nodes.filter(n.node).bucket {
                    Some(b) => b,
                    None => panic!(),
                })
                .max();

            match &node.opkind {
                OpKind::Input => node.bucket = Some(0),
                OpKind::Const => node.bucket = None,
                OpKind::Fused(_) => node.bucket = Some(prev_bucket.unwrap()),
                _ => node.bucket = Some(prev_bucket.unwrap() + 1),
            }
            bucketed_nodes.insert(node.bucket, node.idx, node.clone());
        }

        bucketed_nodes
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
        self.onnx_nodes
            .flatten()
            .iter()
            .map(|e| e.min_cols)
            .max()
            .unwrap()
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
