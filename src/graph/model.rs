use super::node::*;
use super::vars::*;
use super::GraphError;
use crate::circuit::lookup::Config as LookupConfig;
use crate::circuit::lookup::Op as LookupOp;
use crate::circuit::lookup::Table as LookupTable;
use crate::circuit::polynomial::Config as PolyConfig;
use crate::circuit::polynomial::InputType as PolyInputType;
use crate::circuit::polynomial::Node as PolyNode;
use crate::circuit::polynomial::Op as PolyOp;

// use crate::circuit::polynomial::InputType as PolyInputType;

use crate::circuit::range::*;
use crate::commands::RunArgs;
use crate::commands::{Cli, Commands};
use crate::graph::scale_to_multiplier;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor, VarTensor};
//use clap::Parser;
use anyhow::{Context, Error as AnyError};
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use log::{debug, info, trace};
use std::cell::RefCell;
use std::cmp::max;
use std::cmp::min;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::path::Path;
use std::rc::Rc;
use tabled::Table;
use tract_onnx;
use tract_onnx::prelude::{Framework, Graph, InferenceFact, Node as OnnxNode, OutletId};
use tract_onnx::tract_hir::internal::InferenceOp;
/// Mode we're using the model in.
#[derive(Clone, Debug)]
pub enum Mode {
    /// Initialize the model and display the operations table / graph
    Table,
    /// Initialize the model and generate a mock proof
    Mock,
    /// Initialize the model and generate a proof
    Prove,
    /// Initialize the model, generate a proof, and verify
    FullProve,
    /// Initialize the model and verify an already generated proof
    Verify,
}

/// A circuit configuration for the entirety of a model loaded from an Onnx file.
#[derive(Clone, Debug)]
pub struct ModelConfig<F: FieldExt + TensorType> {
    configs: BTreeMap<usize, NodeConfig<F>>,
    /// The model struct
    pub model: Model,
    /// (optional) range checked outputs of the model graph
    pub public_outputs: Vec<RangeCheckConfig<F>>,
    /// A wrapper for holding all columns that will be assigned to by the model
    pub vars: ModelVars<F>,
}

/// A struct for loading from an Onnx file and converting a computational graph to a circuit.
#[derive(Clone, Debug)]
pub struct Model {
    /// The raw tract [Graph] data structure.
    pub model: Graph<InferenceFact, Box<dyn InferenceOp>>,
    /// Graph of nodes we are loading from Onnx.
    pub nodes: NodeGraph, // Wrapped nodes with additional methods and data (e.g. inferred shape, quantization)
    /// The [RunArgs] being used
    pub run_args: RunArgs,
    /// The [Mode] we're using the model in.
    pub mode: Mode,
    /// Defines which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    pub visibility: VarVisibility,
}

impl Model {
    /// Creates an `Model` from a specified path to an Onnx file.
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `run_args` - [RunArgs]
    /// * `mode` - The [Mode] we're using the model in.
    /// * `visibility` - Which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    pub fn new(
        path: impl AsRef<Path>,
        run_args: RunArgs,
        mode: Mode,
        visibility: VarVisibility,
    ) -> Result<Self, Box<dyn Error>> {
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|_| GraphError::ModelLoad)?;
        info!("visibility: {}", visibility);

        let mut nodes = BTreeMap::<usize, Node>::new();
        for (i, n) in model.nodes.iter().enumerate() {
            let n = Node::new(n.clone(), &mut nodes, run_args.scale, i)?;
            nodes.insert(i, n);
        }
        let om = Model {
            model: model.clone(),
            run_args,
            nodes: Self::assign_execution_buckets(nodes)?,
            mode,
            visibility,
        };

        debug!("{}", Table::new(om.nodes.flatten()).to_string());

        Ok(om)
    }

    /// Runs a dummy forward pass on sample data !
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn forward(
        model_path: impl AsRef<Path>,
        model_inputs: &[Tensor<i32>],
        run_args: RunArgs,
    ) -> Result<Vec<Tensor<f32>>, Box<dyn Error>> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .map_err(|_| GraphError::ModelLoad)?;
        info!("running forward pass");

        let mut nodes = BTreeMap::<usize, Node>::new();
        for (i, n) in model.nodes.iter().enumerate() {
            let n = Node::new(n.clone(), &mut nodes, run_args.scale, i)?;
            nodes.insert(i, n);
        }

        debug!("{}", Table::new(nodes.clone()).to_string());

        let mut results: BTreeMap<&usize, Tensor<i32>> = BTreeMap::new();
        for (i, n) in nodes.iter() {
            let mut inputs = vec![];
            for i in n.inputs.iter() {
                match results.get(&i.node) {
                    Some(value) => inputs.push(value.clone()),
                    None => return Err(Box::new(GraphError::MissingNode(i.node))),
                }
            }
            match &n.opkind {
                OpKind::Lookup(op) => {
                    assert_eq!(inputs.len(), 1);
                    results.insert(i, op.f(inputs[0].clone()));
                }
                OpKind::Poly(op) => {
                    results.insert(i, op.f(inputs)?);
                }
                OpKind::Input => {
                    let mut t = model_inputs[*i].clone();
                    t.reshape(&n.out_dims);
                    results.insert(i, t);
                }
                OpKind::Const => {
                    results.insert(i, n.const_value.as_ref().unwrap().clone());
                }
                _ => {
                    panic!("unsupported op")
                }
            }
        }

        let output_nodes = model.outputs.iter();
        info!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().map(|o| o.node).collect_vec()
        );
        let outputs = output_nodes
            .map(|o| {
                let n = nodes.get(&o.node).unwrap();
                let scale = scale_to_multiplier(n.out_scale);
                results
                    .get(&o.node)
                    .unwrap()
                    .clone()
                    .map(|x| (x as f32) / scale)
            })
            .collect_vec();

        Ok(outputs)
    }

    /// Creates a `Model` from parsed CLI arguments
    pub fn from_ezkl_conf(cli: Cli) -> Result<Self, Box<dyn Error>> {
        let visibility = VarVisibility::from_args(cli.args.clone())?;
        match cli.command {
            Commands::Table { model } | Commands::Mock { model, .. } => {
                Model::new(model, cli.args, Mode::Table, visibility)
            }
            Commands::CreateEVMVerifier { model, .. }
            | Commands::Prove { model, .. }
            | Commands::Verify { model, .. }
            | Commands::Aggregate { model, .. } => {
                Model::new(model, cli.args, Mode::Table, visibility)
            }
            #[cfg(feature = "render")]
            Commands::RenderCircuit { model, .. } => {
                Model::new(model, cli.args, Mode::Table, visibility)
            }
            _ => panic!(),
        }
    }

    /// Creates a `Model` based on CLI arguments
    pub fn from_arg() -> Result<Self, Box<dyn Error>> {
        let args = Cli::create()?;
        Self::from_ezkl_conf(args)
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
    ) -> Result<ModelConfig<F>, Box<dyn Error>> {
        info!("configuring model");
        let mut results = BTreeMap::new();
        let mut tables = BTreeMap::new();

        for (bucket, bucket_nodes) in self.nodes.0.iter() {
            trace!("configuring bucket: {:?}", bucket);
            let non_op_nodes: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_const() || n.opkind.is_input())
                .collect();
            if !non_op_nodes.is_empty() {
                for (i, node) in non_op_nodes {
                    let config = self.conf_non_op_node(node)?;
                    results.insert(*i, config);
                }
            }

            let lookup_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_lookup())
                .collect();

            if !lookup_ops.is_empty() {
                for (i, node) in lookup_ops {
                    let config = self.conf_table(node, meta, vars, &mut tables)?;
                    results.insert(*i, config);
                }
            }

            // preserves ordering
            let poly_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_poly())
                .collect();
            // preserves ordering
            if !poly_ops.is_empty() {
                let config = self.conf_poly_ops(&poly_ops, meta, vars)?;
                results.insert(**poly_ops.keys().max().unwrap(), config);

                let mut display: String = "Poly nodes: ".to_string();
                for idx in poly_ops.keys().map(|k| **k).sorted() {
                    let node = &self.nodes.filter(idx);
                    display.push_str(&format!("| {} ({:?}) | ", idx, node.opkind));
                }
                trace!("{}", display);
            }
        }

        let mut public_outputs = vec![];
        if self.visibility.output.is_public() {
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
                self.run_args.tolerance,
            ));
        }
        configs
    }
    /// Configures non op related nodes (eg. representing an input or const value)
    pub fn conf_non_op_node<F: FieldExt + TensorType>(
        &self,
        node: &Node,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        match &node.opkind {
            OpKind::Const => {
                // Typically parameters for one or more layers.
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(NodeConfig::Const)
            }
            OpKind::Input => {
                // This is the input to the model (e.g. the image).
                // Currently this is handled in the consuming node(s), but will be moved here.
                Ok(NodeConfig::Input)
            }
            OpKind::Unknown(_c) => {
                unimplemented!()
            }
            c => Err(Box::new(GraphError::WrongMethod(node.idx, c.clone()))),
        }
    }

    /// Configures a [BTreeMap] of operations that can be constrained using polynomials. These correspond to operations that are represented in
    /// the `circuit::polynomial` module. A single configuration is output, representing the amalgamation of these operations into
    /// a single Halo2 gate.
    /// # Arguments
    ///
    /// * `nodes` - A [BTreeMap] of (node index, [Node] pairs). The [Node] must represent a polynomial op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `vars` - [ModelVars] for the model.
    fn conf_poly_ops<F: FieldExt + TensorType>(
        &self,
        nodes: &BTreeMap<&usize, &Node>,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        let mut input_nodes: BTreeMap<(&usize, &PolyOp), Vec<Node>> = BTreeMap::new();

        for (i, e) in nodes.iter() {
            let key = (
                *i,
                match &e.opkind {
                    OpKind::Poly(f) => f,
                    _ => {
                        return Err(Box::new(GraphError::WrongMethod(e.idx, e.opkind.clone())));
                    }
                },
            );
            let value = e
                .inputs
                .iter()
                .map(|i| self.nodes.filter(i.node))
                .collect_vec();
            input_nodes.insert(key, value);
        }

        // This works because retain only keeps items for which the predicate returns true, and
        // insert only returns true if the item was not previously present in the set.
        // Since the vector is traversed in order, we end up keeping just the first occurrence of each item.
        let mut seen = HashSet::new();
        let mut advice_idx = 0;
        let mut fixed_idx = 0;
        // impose an execution order here
        let inputs_to_layer: Vec<(usize, VarTensor)> = input_nodes
            .iter()
            .flat_map(|x| {
                x.1.iter()
                    .filter(|i| !nodes.contains_key(&i.idx) && seen.insert(i.idx))
                    .map(|f| {
                        let s = f.out_dims.clone();
                        if f.opkind.is_const() && self.visibility.params.is_public() {
                            let vars = (f.idx, vars.fixed[fixed_idx].reshape(&s));
                            fixed_idx += 1;
                            vars
                        } else {
                            let vars = (f.idx, vars.advices[advice_idx].reshape(&s));
                            advice_idx += 1;
                            vars
                        }
                    })
                    .collect_vec()
            })
            .collect_vec();

        let output_shape = self.nodes.filter(**nodes.keys().max().unwrap()).out_dims;
        // output node
        let output = &vars.advices[advice_idx].reshape(&output_shape);

        let mut inter_counter = 0;
        let fused_nodes: Vec<PolyNode> = input_nodes
            .iter()
            .map(|(op, e)| {
                let order = e
                    .iter()
                    .map(|n| {
                        if !nodes.contains_key(&n.idx) {
                            PolyInputType::Input(
                                inputs_to_layer.iter().position(|r| r.0 == n.idx).unwrap(),
                            )
                        } else {
                            inter_counter += 1;
                            PolyInputType::Inter(inter_counter - 1)
                        }
                    })
                    .collect_vec();
                PolyNode {
                    op: op.1.clone(),
                    input_order: order,
                }
            })
            .collect_vec();

        let inputs = inputs_to_layer.iter();

        let config = NodeConfig::Poly(
            PolyConfig::configure(
                meta,
                &inputs.clone().map(|x| x.1.clone()).collect_vec(),
                output,
                &fused_nodes,
            ),
            inputs.map(|x| x.0).collect_vec(),
        );
        Ok(config)
    }

    /// Configures a lookup table based operation. These correspond to operations that are represented in
    /// the `circuit::eltwise` module.
    /// # Arguments
    ///
    /// * `node` - The [Node] must represent a lookup based op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `vars` - [ModelVars] for the model.
    fn conf_table<F: FieldExt + TensorType>(
        &self,
        node: &Node,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
        tables: &mut BTreeMap<Vec<LookupOp>, Rc<RefCell<LookupTable<F>>>>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        let input_len = node.in_dims[0].iter().product();
        let input = &vars.advices[0].reshape(&[input_len]);
        let output = &vars.advices[1].reshape(&[input_len]);
        let node_inputs = node.inputs.iter().map(|e| e.node).collect();

        let op = match &node.opkind {
            OpKind::Lookup(l) => l,
            c => {
                return Err(Box::new(GraphError::WrongMethod(node.idx, c.clone())));
            }
        };

        let config =
            if let std::collections::btree_map::Entry::Vacant(e) = tables.entry(vec![op.clone()]) {
                let conf: LookupConfig<F> =
                    LookupConfig::configure(meta, input, output, self.run_args.bits, &[op.clone()]);
                e.insert(conf.table.clone());
                NodeConfig::Lookup(conf, node_inputs)
            } else {
                let table = tables.get(&vec![op.clone()]).unwrap();
                let conf: LookupConfig<F> =
                    LookupConfig::configure_with_table(meta, input, output, table.clone());
                NodeConfig::Lookup(conf, node_inputs)
            };
        Ok(config)
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
    ) -> Result<(), Box<dyn Error>> {
        info!("model layout");
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();
        for i in inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(i.0, vars.instances[i.0].clone());
            } else {
                results.insert(i.0, i.1.clone());
            }
        }
        for (idx, config) in config.configs.iter() {
            if let Some(vt) = self.layout_config(layouter, &mut results, config)? {
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
    /// * `inputs` - [BTreeMap] of values to feed into the [NodeConfig], can also include previous intermediate results, i.e the output of other nodes.
    fn layout_config<F: FieldExt + TensorType>(
        &self,
        layouter: &mut impl Layouter<F>,
        inputs: &mut BTreeMap<usize, ValTensor<F>>,
        config: &NodeConfig<F>,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        // The node kind and the config should be the same.
        let res = match config.clone() {
            NodeConfig::Poly(mut ac, idx) => {
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

                Some(ac.layout(layouter, &values)?)
            }
            NodeConfig::Lookup(rc, idx) => {
                if idx.len() != 1 {
                    return Err(Box::new(GraphError::InvalidLookupInputs));
                }
                // For activations and elementwise operations, the dimensions are sometimes only in one or the other of input and output.
                Some(rc.layout(layouter, inputs.get(&idx[0]).unwrap())?)
            }
            NodeConfig::Input => None,
            NodeConfig::Const => None,
            _ => {
                return Err(Box::new(GraphError::UnsupportedOp));
            }
        };
        Ok(res)
    }

    /// Iterates over Nodes and assigns execution buckets to them.  Each bucket holds either:
    /// a) independent lookup operations (i.e operations that don't feed into one another so can be processed in parallel).
    /// b) operations that can be fused together, i.e the output of one op might feed into another.
    /// The logic for bucket assignment is thus: we assign all data intake nodes to the 0 bucket.
    /// We iterate over each node in turn. If the node is a polynomial op, assign to it the maximum bucket of it's inputs.
    /// If the node is a lookup table, assign to it the maximum bucket of it's inputs incremented by 1.
    /// # Arguments
    ///
    /// * `nodes` - [BTreeMap] of (node index, [Node]) pairs.
    pub fn assign_execution_buckets(
        mut nodes: BTreeMap<usize, Node>,
    ) -> Result<NodeGraph, GraphError> {
        info!("assigning configuration buckets to operations");

        let mut bucketed_nodes = NodeGraph(BTreeMap::<Option<usize>, BTreeMap<usize, Node>>::new());

        for (_, node) in nodes.iter_mut() {
            let mut prev_buckets = vec![];
            for n in node
                .inputs
                .iter()
                .filter(|n| !bucketed_nodes.filter(n.node).opkind.is_const())
            {
                match bucketed_nodes.filter(n.node).bucket {
                    Some(b) => prev_buckets.push(b),
                    None => {
                        return Err(GraphError::MissingNode(n.node));
                    }
                }
            }
            let prev_bucket: Option<&usize> = prev_buckets.iter().max();

            match &node.opkind {
                OpKind::Input => node.bucket = Some(0),
                OpKind::Const => node.bucket = None,
                OpKind::Poly(_) => node.bucket = Some(*prev_bucket.unwrap()),
                OpKind::Lookup(_) => node.bucket = Some(prev_bucket.unwrap() + 1),
                op => {
                    return Err(GraphError::WrongMethod(node.idx, op.clone()));
                }
            }
            bucketed_nodes.insert(node.bucket, node.idx, node.clone());
        }

        Ok(bucketed_nodes)
    }

    /// Get a linear extension of the model (an evaluation order), for example to feed to circuit construction.
    /// Note that this order is not stable over multiple reloads of the model.  For example, it will freely
    /// interchange the order of evaluation of fixed parameters.   For example weight could have id 1 on one load,
    /// and bias id 2, and vice versa on the next load of the same file. The ids are also not stable.
    pub fn eval_order(&self) -> Result<Vec<usize>, AnyError> {
        self.model.eval_order()
    }

    /// Note that this order is not stable.
    pub fn nodes(&self) -> Vec<OnnxNode<InferenceFact, Box<dyn InferenceOp>>> {
        self.model.nodes().to_vec()
    }

    /// Returns the ID of the computational graph's inputs
    pub fn input_outlets(&self) -> Result<Vec<OutletId>, Box<dyn Error>> {
        Ok(self.model.input_outlets()?.to_vec())
    }

    /// Returns the ID of the computational graph's outputs
    pub fn output_outlets(&self) -> Result<Vec<OutletId>, Box<dyn Error>> {
        Ok(self.model.output_outlets()?.to_vec())
    }

    /// Returns the number of the computational graph's inputs
    pub fn num_inputs(&self) -> usize {
        let input_nodes = self.model.inputs.iter();
        input_nodes.len()
    }

    ///  Returns shapes of the computational graph's inputs
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.model
            .inputs
            .iter()
            .map(|o| self.nodes.filter(o.node).out_dims)
            .collect_vec()
    }

    /// Returns the number of the computational graph's outputs
    pub fn num_outputs(&self) -> usize {
        let output_nodes = self.model.outputs.iter();
        output_nodes.len()
    }

    /// Returns shapes of the computational graph's outputs
    pub fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.model
            .outputs
            .iter()
            .map(|o| self.nodes.filter(o.node).out_dims)
            .collect_vec()
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<i32> {
        let output_nodes = self.model.outputs.iter();
        output_nodes
            .map(|o| self.nodes.filter(o.node).out_scale)
            .collect_vec()
    }

    /// Max parameter sizes (i.e trainable weights) across the computational graph
    pub fn max_params_poly(&self) -> Vec<usize> {
        let mut maximum_sizes = vec![];
        for (_, bucket_nodes) in self.nodes.0.iter() {
            let fused_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_poly())
                .collect();

            let params = fused_ops
                .iter()
                .flat_map(|(_, n)| n.inputs.iter().map(|o| o.node).collect_vec())
                // here we remove intermediary calculation / nodes within the layer
                .filter(|id| {
                    !fused_ops.contains_key(id) && self.nodes.filter(*id).opkind.is_const()
                })
                .unique()
                .collect_vec();

            for (i, id) in params.iter().enumerate() {
                let param_size = self.nodes.filter(*id).out_dims.iter().product();
                if i >= maximum_sizes.len() {
                    // we've already ascertained this is a param node so out_dims = parameter shape
                    maximum_sizes.push(param_size)
                } else {
                    maximum_sizes[i] = max(maximum_sizes[i], param_size);
                }
            }
        }
        // add 1 for layer output
        maximum_sizes
    }

    /// Maximum number of input variables in fused layers
    pub fn max_vars_and_params_poly(&self) -> Vec<usize> {
        let mut maximum_sizes = vec![];
        for (_, bucket_nodes) in self.nodes.0.iter() {
            let poly_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_poly())
                .collect();

            let inputs = poly_ops
                .iter()
                .flat_map(|(_, n)| n.inputs.iter().map(|o| o.node).collect_vec())
                // here we remove intermediary calculation / nodes within the layer
                .filter(|id| !poly_ops.contains_key(id))
                .unique()
                .collect_vec();

            for (i, id) in inputs.iter().enumerate() {
                let input_size = self.nodes.filter(*id).out_dims.iter().product();
                if i >= maximum_sizes.len() {
                    // we've already ascertained this is the input node so out_dims = input shape
                    maximum_sizes.push(input_size)
                } else {
                    maximum_sizes[i] = max(maximum_sizes[i], input_size);
                }
            }

            // handle output variables
            let max_id = poly_ops.keys().max();
            match max_id {
                Some(m) => {
                    let output_size = self.nodes.filter(**m).out_dims.iter().product();
                    if inputs.len() == maximum_sizes.len() {
                        maximum_sizes.push(output_size)
                    } else {
                        let output_idx = inputs.len();
                        // set last entry to be the output column
                        maximum_sizes[output_idx] = max(maximum_sizes[output_idx], output_size);
                    }
                }
                // None if the bucket is empty
                None => {}
            }
        }
        // add 1 for layer output
        maximum_sizes
    }

    /// Maximum of non params variable sizes in fused layers
    pub fn max_vars_poly(&self) -> Vec<usize> {
        let mut maximum_sizes = vec![];
        for (_, bucket_nodes) in self.nodes.0.iter() {
            let fused_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_poly())
                .collect();

            let inputs = fused_ops
                .iter()
                .flat_map(|(_, n)| n.inputs.iter().map(|o| o.node).collect_vec())
                // here we remove intermediary calculation / nodes within the layer
                .filter(|id| {
                    !fused_ops.contains_key(id) && !self.nodes.filter(*id).opkind.is_const()
                })
                .unique()
                .collect_vec();

            for (i, id) in inputs.iter().enumerate() {
                let input_size = self.nodes.filter(*id).out_dims.iter().product();
                if i >= maximum_sizes.len() {
                    // we've already ascertained this is the input node so out_dims = input shape
                    maximum_sizes.push(input_size)
                } else {
                    maximum_sizes[i] = max(maximum_sizes[i], input_size);
                }
            }

            // handle output variables
            let max_id = fused_ops.keys().max();
            match max_id {
                Some(m) => {
                    let output_size = self.nodes.filter(**m).out_dims.iter().product();
                    if inputs.len() == maximum_sizes.len() {
                        maximum_sizes.push(output_size)
                    } else {
                        let output_idx = inputs.len();
                        // set last entry to be the output column
                        maximum_sizes[output_idx] = max(maximum_sizes[output_idx], output_size);
                    }
                }
                // None if the bucket is empty
                None => {}
            }
        }
        // add 1 for layer output
        maximum_sizes
    }

    /// Maximum variable sizes in non-fused layers
    pub fn max_vars_lookup(&self) -> Vec<usize> {
        let mut maximum_sizes = vec![];
        for (_, bucket_nodes) in self.nodes.0.iter() {
            let non_fused_ops: BTreeMap<&usize, &Node> = bucket_nodes
                .iter()
                .filter(|(_, n)| n.opkind.is_lookup())
                .collect();

            for (_, n) in non_fused_ops {
                for (j, dims) in n.in_dims.iter().enumerate() {
                    let input_size = dims.iter().product();
                    if j >= maximum_sizes.len() {
                        maximum_sizes.push(input_size)
                    } else {
                        maximum_sizes[j] = max(maximum_sizes[j], input_size);
                    }
                }
                // handle output variables
                let output_size = n.out_dims.iter().product();
                if (n.in_dims.len()) == maximum_sizes.len() {
                    maximum_sizes.push(output_size)
                } else {
                    let output_idx = n.in_dims.len();
                    // set last entry to be the output column
                    maximum_sizes[output_idx] = max(maximum_sizes[output_idx], output_size);
                }
            }
        }
        maximum_sizes
    }

    /// Number of instances used by the circuit
    pub fn instance_shapes(&self) -> Vec<Vec<usize>> {
        // for now the number of instances corresponds to the number of graph / model outputs
        let mut instance_shapes = vec![];
        if self.visibility.input.is_public() {
            instance_shapes.extend(self.input_shapes());
        }
        if self.visibility.output.is_public() {
            instance_shapes.extend(self.output_shapes());
        }
        instance_shapes
    }

    /// Number of advice used by the circuit
    pub fn advice_shapes(&self) -> Vec<usize> {
        // max sizes in lookup
        let max_lookup_sizes = self.max_vars_lookup();
        let max_poly_sizes = if self.visibility.params.is_public() {
            // max sizes for poly inputs
            self.max_vars_poly()
        } else {
            // max sizes for poly inputs + params
            self.max_vars_and_params_poly()
        };

        let mut advice_shapes = if max_poly_sizes.len() >= max_lookup_sizes.len() {
            max_poly_sizes.clone()
        } else {
            max_lookup_sizes.clone()
        };

        for i in 0..min(max_poly_sizes.len(), max_lookup_sizes.len()) {
            advice_shapes[i] = max(max_poly_sizes[i], max_lookup_sizes[i]);
        }
        advice_shapes
    }

    /// Maximum sizes of fixed columns (and their sizes) used by the circuit
    pub fn fixed_shapes(&self) -> Vec<usize> {
        let mut fixed_shapes = vec![];
        if self.visibility.params.is_public() {
            fixed_shapes = self.max_params_poly();
        }
        fixed_shapes
    }
}
