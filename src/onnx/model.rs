use super::utilities::{node_output_shapes, scale_to_multiplier, vector_to_quantized};
use crate::circuit::eltwise::{DivideBy, EltwiseConfig, ReLu, Sigmoid};
use crate::circuit::fused::*;
use crate::commands::{model_path, Cli, Commands};
use crate::tensor::ops::const_mult;
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
use std::collections::{hash_map::Entry, HashMap, HashSet};
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
/// Enum of the different kinds of operations `ezkl` can support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpKind {
    ReLU(usize),
    Sigmoid(usize),
    Div(usize),
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
            OpKind::ReLU(s) => write!(f, "relu w/ scaling: {}", s),
            OpKind::Div(s) => write!(f, "div  w/ scaling: {}", s),
            OpKind::Sigmoid(s) => write!(f, "sigmoid  w/ scaling: {}", s),
            OpKind::Const => write!(f, "const"),
            OpKind::Input => write!(f, "input"),
            OpKind::Fused(s) => write!(f, "{}", s),
            OpKind::Unknown(c) => write!(f, "? {}", c),
        }
    }
}

/// Enum of the different kinds of node configurations `ezkl` can support.
#[derive(Clone, Default, Debug)]
pub enum NodeConfigTypes<F: FieldExt + TensorType> {
    ReLU(EltwiseConfig<F, ReLu<F>>, Vec<usize>),
    Sigmoid(EltwiseConfig<F, Sigmoid<F>>, Vec<usize>),
    Divide(EltwiseConfig<F, DivideBy<F>>, Vec<usize>),
    Fused(FusedConfig<F>, Vec<usize>),
    Const,
    Input,
    #[default]
    NotConfigured,
}

/// A circuit configuration for a single node.
#[derive(Clone, Default, Debug)]
pub struct NodeConfig<F: FieldExt + TensorType> {
    config: NodeConfigTypes<F>,
    onnx_idx: Vec<usize>,
}

/// A circuit configuration for the entirety of a model loaded from an Onnx file.
#[derive(Clone)]
pub struct OnnxModelConfig<F: FieldExt + TensorType> {
    configs: HashMap<usize, NodeConfig<F>>,
    pub model: OnnxModel,
    pub public_outputs: Vec<Column<Instance>>,
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

/// A single operation in an [OnnxModel].
/// # Arguments:
/// * `node` - The raw Tract Node data structure.
/// * `opkind` - [OpKind] enum, i.e what operation this node represents.
/// * `output_max` - The inferred maximum value that can appear in the output tensor given previous quantization choices.
/// * `in_scale, out_scale` - The denominator in the fixed point representation. Tensors of differing scales should not be combined.
/// * `in_dims, out_dims` - The shape of the activations which enter and leave the node.
/// * `inputs` - The indices of other nodes that feed into this node.
/// * `const_value` - The constants potentially associated with this node.
/// * `idx` - The node's unique identifier.
/// * `bucket` - The execution bucket this node has been assigned to.
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
            "Div" => OpKind::Div(1),
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

/// Representation of an execution graph divided into execution 'buckets'.
#[derive(Clone, Default, Debug)]
pub struct NodeGraph(HashMap<Option<usize>, HashMap<usize, OnnxNode>>);

impl NodeGraph {
    pub fn new() -> Self {
        NodeGraph(HashMap::new())
    }

    fn insert(&mut self, idx: Option<usize>, node_idx: usize, config: OnnxNode) {
        match self.0.entry(idx) {
            Entry::Vacant(e) => {
                e.insert(HashMap::from([(node_idx, config)]));
            }
            Entry::Occupied(mut e) => {
                e.get_mut().insert(node_idx, config);
            }
        }
    }

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

/// Mode we're using the model in.
#[derive(Clone, Debug)]
pub enum Mode {
    Table,
    Mock,
    Prove,
    FullProve,
    Verify,
}

/// A struct for loading from an Onnx file and converting a computational graph to a circuit.
#[derive(Clone, Debug)]
pub struct OnnxModel {
    pub model: Graph<InferenceFact, Box<dyn InferenceOp>>, // The raw Tract data structure
    pub onnx_nodes: NodeGraph, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    pub bits: usize,
    pub scale: i32,
    pub mode: Mode,
}

impl OnnxModel {
    /// Creates an `OnnxModel` from a specified path to an Onnx file.
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `scale` - The denominator used for fixed point arithmetic (relevant for quantizing input data and model parameters).
    /// * `bits` - Number of bits to use.
    /// * `mode` -The [Mode] we're using the model in.
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
    /// Creates an `OnnxModel` based on CLI arguments
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

    /// Configures an `OnnxModel`. Does so one execution `bucket` at a time. Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// # Arguments
    ///
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the nodes loaded in `self.onnx_nodes`.
    pub fn configure<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> Result<OnnxModelConfig<F>> {
        info!("configuring model");
        let mut results = HashMap::new();

        for (_, bucket_nodes) in self.onnx_nodes.0.iter() {
            let non_fused_ops: HashMap<&usize, &OnnxNode> = bucket_nodes
                .iter()
                .filter(|(_, n)| !n.opkind.is_fused())
                .collect();

            if !non_fused_ops.is_empty() {
                for (i, n) in non_fused_ops.iter() {
                    results.insert(
                        **i,
                        NodeConfig {
                            config: self.configure_table(n, meta, advices.clone()),
                            onnx_idx: vec![**i],
                        },
                    );
                }
            }

            // preserves ordering
            let fused_ops: HashMap<&usize, &OnnxNode> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_fused())
                .collect();
            // preserves ordering
            if !fused_ops.is_empty() {
                results.insert(
                    **fused_ops.keys().max().unwrap(),
                    NodeConfig {
                        config: self.fuse_ops(&fused_ops, meta, advices.clone()),
                        onnx_idx: fused_ops.keys().map(|k| **k).sorted().collect_vec(),
                    },
                );
            }
        }

        let public_outputs = self
            .model
            .outputs
            .iter()
            .map(|_| {
                let l = meta.instance_column();
                meta.enable_equality(l);
                l
            })
            .collect_vec();

        Ok(OnnxModelConfig {
            configs: results,
            model: self.clone(),
            public_outputs,
        })
    }

    /// Configures a `HashMap` of 'fuseable' operations. These correspond to operations that are represented in
    /// the `circuit::fused` module. A single configuration is output, representing the amalgamation of these operations into
    /// a single Halo2 gate.
    /// # Arguments
    ///
    /// * `nodes` - A `HashMap` of (node index, [OnnxNode] pairs). The [OnnxNode] must represent a fuseable op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the passed `nodes`.
    fn fuse_ops<F: FieldExt + TensorType>(
        &self,
        nodes: &HashMap<&usize, &OnnxNode>,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        let input_nodes: HashMap<(&usize, &FusedOp), Vec<OnnxNode>> = nodes
            .iter()
            .map(|(i, e)| {
                (
                    (
                        *i,
                        match &e.opkind {
                            OpKind::Fused(f) => f,
                            _ => panic!(),
                        },
                    ),
                    e.inputs
                        .iter()
                        .map(|i| self.onnx_nodes.filter(i.node))
                        .collect_vec(),
                )
            })
            .collect();

        // This works because retain only keeps items for which the predicate returns true, and
        // insert only returns true if the item was not previously present in the set.
        // Since the vector is traversed in order, we end up keeping just the first occurrence of each item.
        let mut seen = HashSet::new();
        let mut start = 0;
        // impose an execution order here
        let inputs_to_layer: Vec<(usize, VarTensor)> = input_nodes
            .iter()
            .sorted_by_key(|x| x.0 .0)
            .flat_map(|x| {
                x.1.iter()
                    .filter(|i| !nodes.contains_key(&i.idx) && seen.insert(i.idx))
                    .map(|f| {
                        let s = f.out_dims.clone().unwrap();
                        let mut end = 1;
                        if s.len() > 1 {
                            end = s[0..s.len() - 1].iter().product();
                        }
                        let a = (f.idx, advices.get_slice(&[start..start + end], &s));
                        start += end;
                        a
                    })
                    .collect_vec()
            })
            .collect_vec();

        // output node
        let output_shape = self
            .onnx_nodes
            .filter(**nodes.keys().max().unwrap())
            .out_dims
            .unwrap();

        let mut end = 1;
        if output_shape.len() > 1 {
            end = output_shape[0..output_shape.len() - 1].iter().product();
        }
        let output = advices.get_slice(&[start..start + end], &output_shape);

        let mut inter_counter = 0;
        let fused_nodes: Vec<FusedNode> = input_nodes
            .iter()
            .sorted_by_key(|x| x.0 .0)
            .map(|(op, e)| {
                let order = e
                    .iter()
                    .map(|n| {
                        if !nodes.contains_key(&n.idx) {
                            FusedInputType::Input(
                                inputs_to_layer.iter().position(|r| r.0 == n.idx).unwrap(),
                            )
                        } else {
                            inter_counter += 1;
                            FusedInputType::Inter(inter_counter - 1)
                        }
                    })
                    .collect_vec();
                FusedNode {
                    op: op.1.clone(),
                    input_order: order,
                }
            })
            .collect_vec();

        let inputs = inputs_to_layer.iter();

        NodeConfigTypes::Fused(
            FusedConfig::configure(
                meta,
                &inputs.clone().map(|x| x.1.clone()).collect_vec(),
                &output,
                &fused_nodes,
            ),
            inputs.map(|x| x.0).collect_vec(),
        )
    }

    /// Configures a lookup table based operation. These correspond to operations that are represented in
    /// the `circuit::eltwise` module.
    /// # Arguments
    ///
    /// * `node` - The [OnnxNode] must represent a lookup based op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure the passed `node`.
    fn configure_table<F: FieldExt + TensorType>(
        &self,
        node: &OnnxNode,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        match &node.opkind {
            OpKind::Div(s) => {
                let dims = match &node.in_dims {
                    Some(v) => v,
                    None => {
                        error!("relu layer has no input shape");
                        panic!()
                    }
                };

                let length = dims.clone().into_iter().product();

                let conf: EltwiseConfig<F, DivideBy<F>> = EltwiseConfig::configure(
                    meta,
                    &[advices.get_slice(&[0..length], &[length])],
                    Some(&[self.bits, *s]),
                );
                let inputs = node.inputs.iter().map(|e| e.node).collect();
                NodeConfigTypes::Divide(conf, inputs)
            }
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
        }
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    ///
    /// * `config` - [OnnxModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    pub fn layout<F: FieldExt + TensorType>(
        &self,
        config: OnnxModelConfig<F>,
        layouter: &mut impl Layouter<F>,
        inputs: &[ValTensor<F>],
    ) -> Result<Vec<ValTensor<F>>> {
        info!("model layout");
        let mut results = HashMap::<usize, ValTensor<F>>::new();
        for i in inputs.iter().enumerate() {
            results.insert(i.0, i.1.clone());
        }
        for (idx, c) in config.configs.iter().sorted_by_key(|x| x.0) {
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
                        "------ laying out node {} ({:?}) ",
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
                    trace!("------------ output {:?}", results.get(idx).unwrap().show());
                }
            }
        }

        let output_nodes = self.model.outputs.iter();

        info!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().map(|o| o.node).collect_vec()
        );

        Ok(output_nodes
            .map(|o| results.get(&o.node).unwrap().clone())
            .collect_vec())
    }

    /// Assigns values to a single region, represented as a [NodeConfig].
    /// # Arguments
    ///
    /// * `config` - [NodeConfig] the signle region we will layout.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - `HashMap` of values to feed into the NodeConfig, can also include previous intermediate results, i.e the output of other nodes.
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
                        match node.opkind {
                            OpKind::Const => {
                                let val = node
                                    .const_value
                                    .clone()
                                    .context("Tensor<i32> should already be loaded")
                                    .unwrap();
                                <Tensor<i32> as Into<Tensor<Value<F>>>>::into(val).into()
                            }
                            _ => inputs.get(i).unwrap().clone(),
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
                assert_eq!(idx.len(), 1);
                Some(sc.layout(layouter, &[inputs.get(&idx[0]).unwrap().clone()]))
            }
            NodeConfigTypes::Divide(sc, idx) => {
                assert_eq!(idx.len(), 1);
                Some(sc.layout(layouter, &[inputs.get(&idx[0]).unwrap().clone()]))
            }
            NodeConfigTypes::Input => None,
            NodeConfigTypes::Const => None,
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

        let mut nodes = HashMap::<usize, OnnxNode>::new();
        for (_, node) in self
            .onnx_nodes
            .0
            .get_mut(&None)
            .unwrap()
            .iter_mut()
            .sorted_by_key(|x| x.0)
        {
            let inputs: Vec<OnnxNode> = node
                .node
                .inputs
                .iter_mut()
                .map(|i| nodes.get(&i.node).unwrap().clone())
                .collect();

            match &node.opkind {
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
                OpKind::Div(_) => {
                    let input_node = &inputs[0];
                    node.in_dims = input_node.out_dims.clone();
                    node.out_dims = input_node.out_dims.clone();

                    // rescale the divider
                    let mult = scale_to_multiplier(self.scale);
                    node.inputs.pop();
                    if inputs[1].out_dims.clone().unwrap() != [1] {
                        error!("ezkl currently only supports division by a constant");
                        unimplemented!()
                    }
                    let div = inputs[1].output_max / mult;

                    node.in_scale = input_node.out_scale;
                    node.out_scale = self.scale;
                    let scale_diff = node.in_scale - node.out_scale;
                    // We can also consider adjusting the scale of all inputs and the output in a more custom way.
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        node.opkind = OpKind::Div((div * mult) as usize); // now the input will be scaled down to match
                        node.output_max = input_node.output_max / (div * mult);
                    } else {
                        node.opkind = OpKind::Div(div as usize); // now the input will be scaled down to match
                        node.output_max = input_node.output_max / (div);
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

                            node.in_scale = input_node.out_scale;
                            node.out_scale = weight_node.out_scale + input_node.out_scale;
                            let scale_diff = node.out_scale - bias_node.out_scale;
                            let mut bias_node = nodes.get_mut(&node.node.inputs[2].node).unwrap();
                            bias_node = Self::scale_up_const_node(bias_node, scale_diff);

                            assert_eq!(
                                input_node.out_scale + weight_node.out_scale,
                                bias_node.out_scale
                            );

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
                        }
                        FusedOp::Matmul => {
                            let (a_node, b_node) = (&inputs[0], &inputs[1]);

                            let in_dim = a_node.out_dims.as_ref().unwrap()[1];
                            node.in_dims = Some(vec![in_dim]);

                            let a_dims = a_node.out_dims.as_ref().unwrap();
                            let b_dims = b_node.out_dims.as_ref().unwrap();
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

                            node.in_scale = input_node.out_scale;
                            node.out_scale = weight_node.out_scale + input_node.out_scale;
                            let scale_diff = node.out_scale - bias_node.out_scale;
                            let mut bias_node = nodes.get_mut(&node.node.inputs[2].node).unwrap();
                            bias_node = Self::scale_up_const_node(bias_node, scale_diff);

                            assert_eq!(
                                input_node.out_scale + weight_node.out_scale,
                                bias_node.out_scale
                            );

                            let in_dim = weight_node.out_dims.as_ref().unwrap()[1];
                            let out_dim = weight_node.out_dims.as_ref().unwrap()[0];
                            node.in_dims = Some(vec![in_dim]);
                            node.out_dims = Some(vec![out_dim]);

                            node.output_max =
                                input_node.output_max * weight_node.output_max * (in_dim as f32);
                        }
                        FusedOp::Add => {
                            Self::homogenize_input_scales(node, inputs.clone());

                            if let OpKind::Fused(FusedOp::Rescaled(_, mult)) = &node.opkind {
                                node.output_max = (inputs
                                    .iter()
                                    .enumerate()
                                    .map(|(idx, n)| {
                                        ((mult[idx].1 as f32) * (n.output_max.ceil())) as i32
                                    })
                                    .max()
                                    .unwrap()
                                    as f32)
                                    * (inputs.len() as f32);
                            } else {
                                error!("failed to homogenize input scalings for node {}", node.idx)
                            }

                            node.in_scale =
                                inputs.iter().map(|input| input.out_scale).max().unwrap();
                            node.out_scale = node.in_scale;
                        }
                        FusedOp::Sum => {
                            assert!(inputs.len() == 1);
                            node.output_max = inputs[0].output_max
                                * inputs[0].in_dims.clone().unwrap().iter().product::<usize>()
                                    as f32;
                            node.in_scale =
                                inputs.iter().map(|input| input.out_scale).max().unwrap();
                            node.out_scale = node.in_scale;
                            node.out_dims = Some(vec![1]);
                        }
                        FusedOp::Sub => {
                            Self::homogenize_input_scales(node, inputs.clone());
                            if let OpKind::Fused(FusedOp::Rescaled(_, mult)) = &node.opkind {
                                node.output_max = (inputs
                                    .iter()
                                    .enumerate()
                                    .map(|(idx, n)| {
                                        ((mult[idx].1 as f32) * (n.output_max.ceil())) as i32
                                    })
                                    .max()
                                    .unwrap()
                                    as f32)
                                    * (inputs.len() as f32);
                            } else {
                                error!("failed to homogenize input scalings for node {}", node.idx)
                            }
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
                            if inputs[1].out_dims.clone().unwrap() != [1] {
                                error!("ezkl currently only supports raising to the power by a constant");
                                unimplemented!()
                            }
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
                        FusedOp::Rescaled(_, _) => {
                            error!("operations should not already be rescaled at this stage")
                        }
                    }
                    // output size
                    node.min_cols += node.out_dims.clone().unwrap()
                        [0..node.out_dims.clone().unwrap().len() - 1]
                        .iter()
                        .product::<usize>()
                        + 1;
                }
                _ => {}
            };

            nodes.insert(node.idx, node.clone());
        }

        self.onnx_nodes = self.assign_execution_buckets(nodes)?;

        Ok(())
    }

    /// Ensures all inputs to a node have the same floating point denominator.
    pub fn homogenize_input_scales(node: &mut OnnxNode, inputs: Vec<OnnxNode>) {
        let mut multipliers = vec![1; inputs.len()];
        let out_scales = inputs.windows(1).map(|w| w[0].out_scale).collect_vec();
        if !out_scales.windows(2).all(|w| w[0] == w[1]) {
            let max_scale = out_scales.iter().max().unwrap();
            let _ = inputs
                .iter()
                .enumerate()
                .map(|(idx, input)| {
                    let scale_diff = max_scale - input.out_scale;
                    if scale_diff > 0 {
                        let mult = scale_to_multiplier(scale_diff);
                        multipliers[idx] = mult as usize;
                        info!(
                            "------ scaled op node input {:?}: {:?} -> {:?}",
                            input.idx,
                            input.out_scale,
                            input.out_scale + scale_diff
                        );
                    }
                })
                .collect_vec();
        }
        if let OpKind::Fused(c) = &node.opkind {
            node.opkind = OpKind::Fused(FusedOp::Rescaled(
                Box::new(c.clone()),
                (0..inputs.len()).zip(multipliers).collect_vec(),
            ))
        } else {
            error!("should not homegenize input scales for non fused ops.")
        }
    }

    /// Re-quantizes a constant value node to a new scale.
    pub fn scale_up_const_node(node: &mut OnnxNode, scale_diff: i32) -> &mut OnnxNode {
        assert!(matches!(node.opkind, OpKind::Const));
        if scale_diff > 0 {
            if let Some(val) = &node.const_value {
                let mult = scale_to_multiplier(scale_diff);
                node.const_value = Some(const_mult(val, mult as i32));
                info!(
                    "------ scaled const node {:?}: {:?} -> {:?}",
                    node.idx,
                    node.in_scale,
                    node.out_scale + scale_diff
                );
                node.output_max *= mult;
                node.out_scale += scale_diff;
            }
        }
        node
    }

    /// Iterates over OnnxNodes and assigns execution buckets to them.  Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// The logic for bucket assignment is thus: we assign all data intake nodes to the 0 bucket.
    /// We iterate over each node in turn. If the node is a fuseable op, assign to it the maximum bucket of it's inputs.
    /// If the node is a lookup table, assign to it the maximum bucket of it's inputs incremented by 1.
    /// # Arguments
    ///
    /// * `nodes` - `HashMap` of (node index, [OnnxNode]) pairs.
    pub fn assign_execution_buckets(
        &mut self,
        mut nodes: HashMap<usize, OnnxNode>,
    ) -> Result<NodeGraph> {
        info!("assigning configuration buckets to operations");

        let mut bucketed_nodes =
            NodeGraph(HashMap::<Option<usize>, HashMap<usize, OnnxNode>>::new());

        for (_, node) in nodes.iter_mut().sorted_by_key(|x| x.0) {
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

        Ok(bucketed_nodes)
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

    pub fn get_output_scales(&self) -> Vec<i32> {
        let output_nodes = self.model.outputs.iter();
        output_nodes
            .map(|o| self.onnx_nodes.filter(o.node).out_scale)
            .collect_vec()
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
