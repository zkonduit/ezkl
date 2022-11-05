use super::node::*;
use super::utilities::{node_output_shapes, scale_to_multiplier};
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
use log::{debug, error, info, trace};

use std::collections::{btree_map::Entry, BTreeMap, HashSet};

use std::path::Path;
use tabled::{Table};
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, Node, OutletId};
use tract_onnx::tract_hir::{
    internal::InferenceOp,
};

/// A circuit configuration for the entirety of a model loaded from an Onnx file.
#[derive(Clone)]
pub struct OnnxModelConfig<F: FieldExt + TensorType> {
    configs: BTreeMap<usize, NodeConfig<F>>,
    pub model: OnnxModel,
    pub public_outputs: Vec<Column<Instance>>,
}

/// Representation of an execution graph divided into execution 'buckets'.
#[derive(Clone, Default, Debug)]
pub struct NodeGraph(BTreeMap<Option<usize>, BTreeMap<usize, OnnxNode>>);

impl NodeGraph {
    pub fn new() -> Self {
        NodeGraph(BTreeMap::new())
    }

    fn insert(&mut self, idx: Option<usize>, node_idx: usize, config: OnnxNode) {
        match self.0.entry(idx) {
            Entry::Vacant(e) => {
                e.insert(BTreeMap::from([(node_idx, config)]));
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

        let mut onnx_nodes = BTreeMap::<usize, OnnxNode>::new();
        let _ = model
            .nodes()
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let n = OnnxNode::new(n.clone(), &mut onnx_nodes, scale, i);
                onnx_nodes.insert(i, n);
            })
            .collect_vec();
        debug!("{:?}", onnx_nodes);
        let om = OnnxModel {
            model: model.clone(),
            scale,
            onnx_nodes: Self::assign_execution_buckets(onnx_nodes)
                .expect("failed to assign execution buckets"),
            bits,
            mode,
        };

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
        let mut results = BTreeMap::new();

        for (_, bucket_nodes) in self.onnx_nodes.0.iter() {
            let non_fused_ops: BTreeMap<&usize, &OnnxNode> = bucket_nodes
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
            let fused_ops: BTreeMap<&usize, &OnnxNode> = bucket_nodes
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

    /// Configures a `BTreeMap` of 'fuseable' operations. These correspond to operations that are represented in
    /// the `circuit::fused` module. A single configuration is output, representing the amalgamation of these operations into
    /// a single Halo2 gate.
    /// # Arguments
    ///
    /// * `nodes` - A `BTreeMap` of (node index, [OnnxNode] pairs). The [OnnxNode] must represent a fuseable op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `advices` - A `VarTensor` holding columns of advices. Must be sufficiently large to configure all the passed `nodes`.
    fn fuse_ops<F: FieldExt + TensorType>(
        &self,
        nodes: &BTreeMap<&usize, &OnnxNode>,
        meta: &mut ConstraintSystem<F>,
        advices: VarTensor,
    ) -> NodeConfigTypes<F> {
        let input_nodes: BTreeMap<(&usize, &FusedOp), Vec<OnnxNode>> = nodes
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
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();
        for i in inputs.iter().enumerate() {
            results.insert(i.0, i.1.clone());
        }
        for (idx, c) in config.configs.iter() {
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

    /// Iterates over OnnxNodes and assigns execution buckets to them.  Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// The logic for bucket assignment is thus: we assign all data intake nodes to the 0 bucket.
    /// We iterate over each node in turn. If the node is a fuseable op, assign to it the maximum bucket of it's inputs.
    /// If the node is a lookup table, assign to it the maximum bucket of it's inputs incremented by 1.
    /// # Arguments
    ///
    /// * `nodes` - `BTreeMap` of (node index, [OnnxNode]) pairs.
    pub fn assign_execution_buckets(mut nodes: BTreeMap<usize, OnnxNode>) -> Result<NodeGraph> {
        info!("assigning configuration buckets to operations");

        let mut bucketed_nodes =
            NodeGraph(BTreeMap::<Option<usize>, BTreeMap<usize, OnnxNode>>::new());

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
