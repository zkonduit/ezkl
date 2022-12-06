use super::node::*;
use super::utilities::scale_to_multiplier;
use crate::abort;
use crate::circuit::eltwise::{DivideBy, EltwiseConfig, EltwiseTable, ReLu, Sigmoid};
use crate::circuit::fused::*;
use crate::circuit::range::*;
use crate::commands::{model_path, Cli, Commands};
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor, VarTensor};
use anyhow::{Context, Result};
use clap::Parser;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use log::{debug, error, info, trace};
use serde::Deserialize;
use std::cmp::max;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::{cell::RefCell, rc::Rc};
use tabled::Table;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, Node as OnnxNode, OutletId};
use tract_onnx::tract_hir::internal::InferenceOp;

/// Mode we're using the model in.
#[derive(Clone, Debug)]
pub enum Mode {
    Table,
    Mock,
    Prove,
    FullProve,
    Verify,
}

#[derive(Clone, Debug, Deserialize)]
pub enum Visibility {
    Private,
    Public,
}
impl Visibility {
    pub fn is_public(&self) -> bool {
        matches!(&self, Visibility::Public)
    }
}
impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Visibility::Private => write!(f, "private"),
            Visibility::Public => write!(f, "public"),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelVisibility {
    pub input: Visibility,
    pub params: Visibility,
    pub output: Visibility,
}
impl std::fmt::Display for ModelVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(inputs: {}, params: {}, outputs: {})",
            self.input, self.params, self.output
        )
    }
}

impl ModelVisibility {
    pub fn from_args() -> Self {
        let args = Cli::parse();

        let input_vis = if args.public_inputs {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let params_vis = if args.public_params {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let output_vis = if args.public_outputs {
            Visibility::Public
        } else {
            Visibility::Private
        };
        if !output_vis.is_public() & !params_vis.is_public() & !input_vis.is_public() {
            abort!("at least one set of variables should be public");
        }
        Self {
            input: input_vis,
            params: params_vis,
            output: output_vis,
        }
    }
}

enum TableTypes<F: FieldExt + TensorType> {
    ReLu(Rc<RefCell<EltwiseTable<F, ReLu<F>>>>),
    DivideBy(Rc<RefCell<EltwiseTable<F, DivideBy<F>>>>),
    Sigmoid(Rc<RefCell<EltwiseTable<F, Sigmoid<F>>>>),
}
impl<F: FieldExt + TensorType> TableTypes<F> {
    fn get_relu(&self) -> Rc<RefCell<EltwiseTable<F, ReLu<F>>>> {
        match self {
            TableTypes::ReLu(inner) => inner.clone(),
            _ => {
                abort!("fetching wrong table type");
            }
        }
    }
    fn get_div(&self) -> Rc<RefCell<EltwiseTable<F, DivideBy<F>>>> {
        match self {
            TableTypes::DivideBy(inner) => inner.clone(),
            _ => {
                abort!("fetching wrong table type");
            }
        }
    }
    fn get_sig(&self) -> Rc<RefCell<EltwiseTable<F, Sigmoid<F>>>> {
        match self {
            TableTypes::Sigmoid(inner) => inner.clone(),
            _ => {
                abort!("fetching wrong table type");
            }
        }
    }
}

#[derive(Clone)]
pub struct ModelVars<F: FieldExt + TensorType> {
    pub advices: Vec<VarTensor>,
    pub fixed: Vec<VarTensor>,
    pub instances: Vec<ValTensor<F>>,
}
/// A wrapper for holding all columns that will be assigned to by a model.
impl<F: FieldExt + TensorType> ModelVars<F> {
    pub fn new(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        advice_dims: (usize, usize),
        fixed_dims: (usize, usize),
        instance_dims: (usize, usize),
    ) -> Self {
        let advices = (0..advice_dims.0)
            .map(|_| {
                VarTensor::new_advice(
                    cs,
                    logrows as usize,
                    advice_dims.1,
                    vec![advice_dims.1],
                    true,
                )
            })
            .collect_vec();
        // todo init fixed
        let fixed = (0..fixed_dims.0)
            .map(|_| {
                VarTensor::new_fixed(cs, logrows as usize, fixed_dims.1, vec![fixed_dims.1], true)
            })
            .collect_vec();
        // todo init fixed
        let instances = (0..instance_dims.0)
            .map(|_| ValTensor::new_instance(cs, vec![instance_dims.1], true))
            .collect_vec();
        ModelVars {
            advices,
            fixed,
            instances,
        }
    }
}

/// A circuit configuration for the entirety of a model loaded from an Onnx file.
#[derive(Clone)]
pub struct ModelConfig<F: FieldExt + TensorType> {
    configs: BTreeMap<usize, NodeConfig<F>>,
    pub model: Model,
    pub public_outputs: Vec<RangeCheckConfig<F>>,
    pub vars: ModelVars<F>,
}

/// A struct for loading from an Onnx file and converting a computational graph to a circuit.
#[derive(Clone, Debug)]
pub struct Model {
    pub model: Graph<InferenceFact, Box<dyn InferenceOp>>, // The raw Tract data structure
    pub nodes: NodeGraph, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    pub bits: usize,
    pub logrows: u32,
    pub scale: i32,
    pub tolerance: usize,
    pub mode: Mode,
    pub visibility: ModelVisibility,
}

impl Model {
    /// Creates an `Model` from a specified path to an Onnx file.
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `scale` - The denominator used for fixed point arithmetic (relevant for quantizing input data and model parameters).
    /// * `bits` - Number of bits to use.
    /// * `mode` -The [Mode] we're using the model in.
    pub fn new(
        path: impl AsRef<Path>,
        scale: i32,
        bits: usize,
        logrows: u32,
        tolerance: usize,
        mode: Mode,
        visibility: ModelVisibility,
    ) -> Self {
        let model = tract_onnx::onnx().model_for_path(path).unwrap();
        info!("visibility: {}", visibility);

        let mut nodes = BTreeMap::<usize, Node>::new();
        let _ = model
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let n = Node::new(n.clone(), &mut nodes, scale, i);
                nodes.insert(i, n);
            })
            .collect_vec();
        let om = Model {
            model: model.clone(),
            scale,
            tolerance,
            nodes: Self::assign_execution_buckets(nodes)
                .expect("failed to assign execution buckets"),
            bits,
            logrows,
            mode,
            visibility,
        };

        debug!("{}", Table::new(om.nodes.flatten()).to_string());

        om
    }
    /// Creates a `Model` based on CLI arguments
    pub fn from_arg() -> Self {
        let args = Cli::parse();
        let visibility = ModelVisibility::from_args();
        match args.command {
            Commands::Table { model } => Model::new(
                model_path(model),
                args.scale,
                args.bits,
                args.logrows,
                args.tolerance,
                Mode::Table,
                visibility,
            ),
            Commands::Mock { data: _, model } => Model::new(
                model_path(model),
                args.scale,
                args.bits,
                args.logrows,
                args.tolerance,
                Mode::Mock,
                visibility,
            ),
            Commands::Fullprove {
                data: _,
                model,
                pfsys: _,
            } => Model::new(
                model_path(model),
                args.scale,
                args.bits,
                args.logrows,
                args.tolerance,
                Mode::FullProve,
                visibility,
            ),
            Commands::Prove {
                data: _,
                model,
                output: _,
                pfsys: _,
            } => Model::new(
                model_path(model),
                args.scale,
                args.bits,
                args.logrows,
                args.tolerance,
                Mode::Prove,
                visibility,
            ),
            Commands::Verify {
                model,
                proof: _,
                pfsys: _,
            } => Model::new(
                model_path(model),
                args.scale,
                args.bits,
                args.logrows,
                args.tolerance,
                Mode::Verify,
                visibility,
            ),
        }
    }

    /// Configures an `Model`. Does so one execution `bucket` at a time. Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// # Arguments
    ///
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the nodes loaded in `self.nodes`.
    pub fn configure<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Result<ModelConfig<F>> {
        info!("configuring model");
        let mut results = BTreeMap::new();
        let mut tables = BTreeMap::new();

        for (bucket, bucket_nodes) in self.nodes.0.iter() {
            trace!("configuring bucket: {:?}", bucket);
            let non_fused_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| !n.opkind.is_fused())
                .collect();

            if !non_fused_ops.is_empty() {
                for (i, n) in non_fused_ops.iter() {
                    let config = self.configure_table(n, meta, vars, &mut tables);
                    results.insert(
                        **i,
                        NodeConfig {
                            config,
                            onnx_idx: vec![**i],
                        },
                    );
                }
            }

            // preserves ordering
            let fused_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_fused())
                .collect();
            // preserves ordering
            if !fused_ops.is_empty() {
                let config = self.fuse_ops(&fused_ops, meta, vars);
                results.insert(
                    **fused_ops.keys().max().unwrap(),
                    NodeConfig {
                        config,
                        onnx_idx: fused_ops.keys().map(|k| **k).sorted().collect_vec(),
                    },
                );
            }
        }

        let mut public_outputs = vec![];
        if !self.visibility.output.is_public() {
            public_outputs = self.range_check_outputs(meta, vars)
        };

        Ok(ModelConfig {
            configs: results,
            model: self.clone(),
            public_outputs,
            vars: vars.clone(),
        })
    }

    fn range_check_outputs<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Vec<RangeCheckConfig<F>> {
        let mut configs = vec![];
        let output_nodes = self.model.outputs.clone();
        let output_shapes = output_nodes
            .iter()
            .map(|o| self.nodes.filter(o.node).out_dims)
            .collect_vec();

        info!("output_shapes {:?}", output_shapes);

        for s in &output_shapes {
            let input = vars.advices[0].reshape(s);
            let output = vars.advices[1].reshape(s);

            configs.push(RangeCheckConfig::configure(
                meta,
                &input,
                &output,
                self.tolerance,
            ));
        }
        configs
    }

    /// Configures a `BTreeMap` of 'fuseable' operations. These correspond to operations that are represented in
    /// the `circuit::fused` module. A single configuration is output, representing the amalgamation of these operations into
    /// a single Halo2 gate.
    /// # Arguments
    ///
    /// * `nodes` - A `BTreeMap` of (node index, [Node] pairs). The [Node] must represent a fuseable op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the passed `nodes`.
    fn fuse_ops<F: FieldExt + TensorType>(
        &self,
        nodes: &BTreeMap<&usize, &Node>,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> NodeConfigTypes<F> {
        let input_nodes: BTreeMap<(&usize, &FusedOp), Vec<Node>> = nodes
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
                        .map(|i| self.nodes.filter(i.node))
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
            .flat_map(|x| {
                x.1.iter()
                    .filter(|i| !nodes.contains_key(&i.idx) && seen.insert(i.idx))
                    .map(|f| {
                        let s = f.out_dims.clone();
                        let a = if f.opkind.is_const() && self.visibility.params.is_public() {
                            (f.idx, vars.fixed[start].reshape(&s))
                        } else {
                            (f.idx, vars.advices[start].reshape(&s))
                        };
                        start += 1;
                        a
                    })
                    .collect_vec()
            })
            .collect_vec();

        let output_shape = self.nodes.filter(**nodes.keys().max().unwrap()).out_dims;
        // output node
        let output = &vars.advices[start].reshape(&output_shape);

        let mut inter_counter = 0;
        let fused_nodes: Vec<FusedNode> = input_nodes
            .iter()
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
                output,
                &fused_nodes,
            ),
            inputs.map(|x| x.0).collect_vec(),
        )
    }

    /// Configures a lookup table based operation. These correspond to operations that are represented in
    /// the `circuit::eltwise` module.
    /// # Arguments
    ///
    /// * `node` - The [Node] must represent a lookup based op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure the passed `node`.
    fn configure_table<F: FieldExt + TensorType>(
        &self,
        node: &Node,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
        tables: &mut BTreeMap<OpKind, TableTypes<F>>,
    ) -> NodeConfigTypes<F> {
        let input_len = node.in_dims[0].iter().product();
        let input = &vars.advices[0].reshape(&[input_len]);
        let output = &vars.advices[1].reshape(&[input_len]);
        let node_inputs = node.inputs.iter().map(|e| e.node).collect();

        match &node.opkind {
            OpKind::Div(s) => {
                if tables.contains_key(&node.opkind) {
                    let table = tables.get(&node.opkind).unwrap();
                    let conf: EltwiseConfig<F, DivideBy<F>> =
                        EltwiseConfig::configure_with_table(meta, input, output, table.get_div());
                    NodeConfigTypes::Divide(conf, node_inputs)
                } else {
                    let conf: EltwiseConfig<F, DivideBy<F>> =
                        EltwiseConfig::configure(meta, input, output, Some(&[self.bits, *s]));
                    tables.insert(
                        node.opkind.clone(),
                        TableTypes::DivideBy(conf.table.clone()),
                    );
                    NodeConfigTypes::Divide(conf, node_inputs)
                }
            }
            OpKind::ReLU(s) => {
                if tables.contains_key(&node.opkind) {
                    let table = tables.get(&node.opkind).unwrap();
                    let conf: EltwiseConfig<F, ReLu<F>> =
                        EltwiseConfig::configure_with_table(meta, input, output, table.get_relu());
                    NodeConfigTypes::ReLU(conf, node_inputs)
                } else {
                    let conf: EltwiseConfig<F, ReLu<F>> =
                        EltwiseConfig::configure(meta, input, output, Some(&[self.bits, *s]));
                    tables.insert(node.opkind.clone(), TableTypes::ReLu(conf.table.clone()));
                    NodeConfigTypes::ReLU(conf, node_inputs)
                }
            }
            OpKind::Sigmoid(s) => {
                if tables.contains_key(&node.opkind) {
                    let table = tables.get(&node.opkind).unwrap();
                    let conf: EltwiseConfig<F, Sigmoid<F>> =
                        EltwiseConfig::configure_with_table(meta, input, output, table.get_sig());
                    NodeConfigTypes::Sigmoid(conf, node_inputs)
                } else {
                    let conf: EltwiseConfig<F, Sigmoid<F>> = EltwiseConfig::configure(
                        meta,
                        input,
                        output,
                        Some(&[self.bits, *s, scale_to_multiplier(self.scale) as usize]),
                    );
                    tables.insert(node.opkind.clone(), TableTypes::Sigmoid(conf.table.clone()));
                    NodeConfigTypes::Sigmoid(conf, node_inputs)
                }
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
            _ => panic!(),
        }
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    ///
    /// * `config` - [ModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    pub fn layout<F: FieldExt + TensorType>(
        &self,
        config: ModelConfig<F>,
        layouter: &mut impl Layouter<F>,
        inputs: &[ValTensor<F>],
        vars: &ModelVars<F>,
    ) -> Result<()> {
        info!("model layout");
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();
        for i in inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(i.0, vars.instances[i.0].clone());
            } else {
                results.insert(i.0, i.1.clone());
            }
        }
        for (idx, c) in config.configs.iter() {
            let mut display: String = "".to_string();
            for (i, idx) in c.onnx_idx[0..].iter().enumerate() {
                let node = &self.nodes.filter(*idx);
                if i > 0 {
                    display.push_str(&format!(
                        "| combined with node {} ({:?}) ",
                        idx, node.opkind
                    ));
                } else {
                    display.push_str(&format!(
                        "------ laying out node {} ({:?}) ",
                        idx, node.opkind
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
        let outputs = output_nodes
            .map(|o| results.get(&o.node).unwrap().clone())
            .collect_vec();
        let _ = config
            .public_outputs
            .iter()
            .zip(outputs)
            .enumerate()
            .map(|(i, (range_check, output))| {
                let mut offset = 0;
                if self.visibility.input.is_public() {
                    offset += inputs.len();
                };
                range_check.layout(
                    layouter.namespace(|| "range check outputs"),
                    output,
                    vars.instances[offset + i].clone(),
                )
            })
            .collect_vec();
        info!("computing...");
        Ok(())
    }

    /// Assigns values to a single region, represented as a [NodeConfig].
    /// # Arguments
    ///
    /// * `config` - [NodeConfig] the single region we will layout.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - `BTreeMap` of values to feed into the NodeConfig, can also include previous intermediate results, i.e the output of other nodes.
    fn layout_config<F: FieldExt + TensorType>(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &mut BTreeMap<usize, ValTensor<F>>,
        config: &NodeConfig<F>,
    ) -> Result<Option<ValTensor<F>>> {
        // The node kind and the config should be the same.
        let res = match config.config.clone() {
            NodeConfigTypes::Fused(mut ac, idx) => {
                let values: Vec<ValTensor<F>> = idx
                    .iter()
                    .map(|i| {
                        let node = &self.nodes.filter(*i);
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
                Some(rc.layout(layouter, inputs.get(&idx[0]).unwrap().clone()))
            }
            NodeConfigTypes::Sigmoid(sc, idx) => {
                assert_eq!(idx.len(), 1);
                Some(sc.layout(layouter, inputs.get(&idx[0]).unwrap().clone()))
            }
            NodeConfigTypes::Divide(dc, idx) => {
                assert_eq!(idx.len(), 1);
                Some(dc.layout(layouter, inputs.get(&idx[0]).unwrap().clone()))
            }
            NodeConfigTypes::Input => None,
            NodeConfigTypes::Const => None,
            c => {
                panic!("Not a configurable op {:?}", c)
            }
        };
        Ok(res)
    }

    /// Iterates over Nodes and assigns execution buckets to them.  Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// The logic for bucket assignment is thus: we assign all data intake nodes to the 0 bucket.
    /// We iterate over each node in turn. If the node is a fuseable op, assign to it the maximum bucket of it's inputs.
    /// If the node is a lookup table, assign to it the maximum bucket of it's inputs incremented by 1.
    /// # Arguments
    ///
    /// * `nodes` - `BTreeMap` of (node index, [Node]) pairs.
    pub fn assign_execution_buckets(mut nodes: BTreeMap<usize, Node>) -> Result<NodeGraph> {
        info!("assigning configuration buckets to operations");

        let mut bucketed_nodes = NodeGraph(BTreeMap::<Option<usize>, BTreeMap<usize, Node>>::new());

        for (_, node) in nodes.iter_mut() {
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
    pub fn nodes(&self) -> Vec<OnnxNode<InferenceFact, Box<dyn InferenceOp>>> {
        self.model.nodes().to_vec()
    }

    pub fn input_outlets(&self) -> Result<Vec<OutletId>> {
        Ok(self.model.input_outlets()?.to_vec())
    }

    pub fn output_outlets(&self) -> Result<Vec<OutletId>> {
        Ok(self.model.output_outlets()?.to_vec())
    }

    pub fn num_inputs(&self) -> usize {
        let input_nodes = self.model.inputs.iter();
        input_nodes.len()
    }

    pub fn num_outputs(&self) -> usize {
        let output_nodes = self.model.outputs.iter();
        output_nodes.len()
    }

    pub fn get_output_scales(&self) -> Vec<i32> {
        let output_nodes = self.model.outputs.iter();
        output_nodes
            .map(|o| self.nodes.filter(o.node).out_scale)
            .collect_vec()
    }

    pub fn max_node_size(&self) -> usize {
        max(
            self.nodes
                .flatten()
                .iter()
                .map(|e| {
                    e.in_dims
                        .iter()
                        .map(|dims| dims.iter().product::<usize>())
                        .max()
                        .unwrap()
                })
                .max()
                .unwrap(),
            self.nodes
                .flatten()
                .iter()
                .map(|e| e.out_dims.iter().product())
                .max()
                .unwrap(),
        )
    }

    pub fn max_node_vars(&self) -> usize {
        let mut maximum_number_inputs = 0;
        for (_, bucket_nodes) in self.nodes.0.iter() {
            let non_fused_ops = bucket_nodes
                .iter()
                .filter(|(_, n)| !n.opkind.is_fused())
                .map(|(_, n)| n.inputs.len())
                .max()
                .unwrap_or(0);

            maximum_number_inputs = max(maximum_number_inputs, non_fused_ops);

            let fused_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_fused())
                .collect();

            let fused_inputs = fused_ops
                .iter()
                .flat_map(|(_, n)| n.inputs.iter().map(|o| o.node).collect_vec())
                // here we remove intermediary calculation / nodes within the layer
                .filter(|id| !fused_ops.contains_key(id))
                .unique()
                .collect_vec();

            maximum_number_inputs = max(maximum_number_inputs, fused_inputs.len());
        }
        // add 1 for layer output
        maximum_number_inputs + 1
    }
}
