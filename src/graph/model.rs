use super::node::*;
use super::vars::*;
use super::GraphError;
use crate::circuit::base::BaseConfig as PolyConfig;
use crate::circuit::base::CheckMode;
use crate::circuit::base::Op as PolyOp;
use crate::circuit::lookup::Config as LookupConfig;
use crate::circuit::lookup::Op as LookupOp;
use crate::circuit::lookup::Table as LookupTable;
use crate::commands::RunArgs;
use crate::commands::{Cli, Commands};
use crate::graph::scale_to_multiplier;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor, VarTensor};
use anyhow::Context;
//use clap::Parser;
use anyhow::Error as AnyError;
use core::panic;
use halo2_proofs::circuit::Region;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use log::error;
use log::{debug, info, trace};
use std::cell::RefCell;
use std::cmp::max;
use std::collections::BTreeMap;
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
    pub range_checks: Vec<Rc<RefCell<PolyConfig<F>>>>,
    /// (optional) packed outputs of the model graph
    pub packed_outputs: Vec<Rc<RefCell<PolyConfig<F>>>>,
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
            nodes,
            mode,
            visibility,
        };

        debug!("{}", Table::new(om.nodes.iter()).to_string());

        Ok(om)
    }

    /// Runs a dummy forward pass on sample data !
    /// # Arguments
    ///
    /// * `path` - A path to an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn forward(
        model_path: impl AsRef<Path>,
        model_inputs: &[Tensor<i128>],
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

        let mut results: BTreeMap<&usize, Tensor<i128>> = BTreeMap::new();
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
                Model::new(model, cli.args, Mode::Mock, visibility)
            }
            Commands::Prove { model, .. }
            | Commands::Verify { model, .. }
            | Commands::Aggregate { model, .. } => {
                Model::new(model, cli.args, Mode::Prove, visibility)
            }
            #[cfg(not(target_arch = "wasm32"))]
            Commands::CreateEVMVerifier { model, .. } => {
                Model::new(model, cli.args, Mode::Prove, visibility)
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
        let mut base_gates = BTreeMap::new();

        let non_op_nodes: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_const() || n.opkind.is_input())
            .collect();
        if !non_op_nodes.is_empty() {
            for (i, node) in non_op_nodes {
                let config = self.conf_non_op_node(node)?;
                results.insert(*i, config);
            }
        }

        let lookup_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_lookup())
            .collect();

        if !lookup_ops.is_empty() {
            for (i, node) in lookup_ops {
                let config = if !self.run_args.single_lookup {
                    // assume a single input
                    let input_len = node.in_dims[0].iter().product();
                    self.conf_lookup(node, input_len, meta, vars, &mut tables)?
                } else {
                    self.reuse_lookup_conf(*i, node, &results, meta, vars, &mut tables)?
                };
                results.insert(*i, config);
            }
        }

        // preserves ordering
        let poly_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_poly())
            .collect();
        // preserves ordering
        if !poly_ops.is_empty() {
            for (i, node) in poly_ops {
                let config = self.conf_poly_ops(node, meta, vars, &mut base_gates)?;
                results.insert(*i, config);

                let mut display: String = "Poly nodes: ".to_string();
                display.push_str(&format!("| {} ({:?}) | ", i, node.opkind));

                trace!("{}", display);
            }
        }

        let mut range_checks = vec![];
        let mut packed_outputs = vec![];
        if self.run_args.pack_base > 1 {
            info!("packing outputs...");
            packed_outputs = self.output_ops(meta, vars, &mut base_gates);
        }
        if self.visibility.output.is_public() {
            range_checks = self.output_ops(meta, vars, &mut base_gates);
        };

        Ok(ModelConfig {
            configs: results,
            model: self.clone(),
            range_checks,
            packed_outputs,
            vars: vars.clone(),
        })
    }

    fn output_ops<F: FieldExt + TensorType>(
        &self,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
        base_gates: &mut BTreeMap<bool, Rc<RefCell<PolyConfig<F>>>>,
    ) -> Vec<Rc<RefCell<PolyConfig<F>>>> {
        let mut configs = vec![];

        for _ in self.output_shapes() {
            let config = match base_gates.get(&false) {
                Some(config) => config.clone(),
                None => {
                    let config = Rc::new(RefCell::new(PolyConfig::<F>::configure(
                        meta,
                        &[vars.advices[0].clone(), vars.advices[1].clone()],
                        &vars.advices[2],
                        CheckMode::SAFE,
                        self.run_args.tolerance.try_into().unwrap(),
                    )));
                    base_gates.insert(false, config.clone());
                    config
                }
            };
            configs.push(config);
        }

        configs
    }

    fn reuse_lookup_conf<F: FieldExt + TensorType>(
        &self,
        i: usize,
        node: &Node,
        prev_configs: &BTreeMap<usize, NodeConfig<F>>,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
        tables: &mut BTreeMap<Vec<LookupOp>, Rc<RefCell<LookupTable<F>>>>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        match &node.opkind {
            OpKind::Lookup(op) => {
                let mut conf = None;
                // iterate in reverse order so we get the last relevant op
                for (_, prev_config) in prev_configs.iter().rev() {
                    if let NodeConfig::Lookup { config, .. } = prev_config {
                        // check if there's a config for the same op
                        if config.borrow().table.borrow().nonlinearities == vec![op.clone()] {
                            conf = Some(NodeConfig::Lookup {
                                config: config.clone(),
                                inputs: node.inputs.iter().map(|e| e.node).collect(),
                            });

                            break;
                        }
                    }
                }
                let conf = match conf {
                    None => {
                        let input_len = self.num_vars_lookup_op(op)[0];
                        self.conf_lookup(node, input_len, meta, vars, tables)?
                    }
                    Some(c) => c,
                };
                Ok(conf)
            }
            // should never reach here
            _ => Err(Box::new(GraphError::OpMismatch(i, node.opkind.clone()))),
        }
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
        node: &Node,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
        base_gates: &mut BTreeMap<bool, Rc<RefCell<PolyConfig<F>>>>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
        let input_nodes = node
            .inputs
            .iter()
            .map(|i| self.nodes.get(&i.node).unwrap())
            .collect_vec();

        let input_idx = input_nodes.iter().map(|f| f.idx).collect_vec();

        let fixed_flag = !input_nodes
            .iter()
            .filter(|f| f.opkind.is_const() && self.visibility.params.is_public())
            .collect_vec()
            .is_empty();

        let config = match base_gates.get(&fixed_flag) {
            Some(config) => {
                trace!("reusing base gate config");
                config.clone()
            }
            None => {
                let inputs: [VarTensor; 2] = if fixed_flag {
                    [vars.fixed[0].clone(), vars.advices[1].clone()]
                } else {
                    [vars.advices[0].clone(), vars.advices[1].clone()]
                };
                // output node
                let output_shape = &node.out_dims;
                let output = &vars.advices[2].reshape(output_shape);
                let config = Rc::new(RefCell::new(PolyConfig::configure(
                    meta,
                    inputs.into_iter().collect_vec()[..].try_into()?,
                    output,
                    CheckMode::SAFE,
                    self.run_args.tolerance.try_into().unwrap(),
                )));
                base_gates.insert(fixed_flag, config.clone());
                config
            }
        };

        if let OpKind::Poly(op) = &node.opkind {
            let config = NodeConfig::Poly {
                config,
                inputs: input_idx,
                op: op.clone(),
            };
            Ok(config)
        } else {
            panic!()
        }
    }

    /// Configures a lookup table based operation. These correspond to operations that are represented in
    /// the `circuit::eltwise` module.
    /// # Arguments
    ///
    /// * `node` - The [Node] must represent a lookup based op.
    /// * `meta` - Halo2 ConstraintSystem.
    /// * `vars` - [ModelVars] for the model.
    fn conf_lookup<F: FieldExt + TensorType>(
        &self,
        node: &Node,
        input_len: usize,
        meta: &mut ConstraintSystem<F>,
        vars: &mut ModelVars<F>,
        tables: &mut BTreeMap<Vec<LookupOp>, Rc<RefCell<LookupTable<F>>>>,
    ) -> Result<NodeConfig<F>, Box<dyn Error>> {
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
                let config: LookupConfig<F> =
                    LookupConfig::configure(meta, input, output, self.run_args.bits, &[op.clone()]);
                e.insert(config.table.clone());
                NodeConfig::Lookup {
                    config: Rc::new(RefCell::new(config)),
                    inputs: node_inputs,
                }
            } else {
                let table = tables.get(&vec![op.clone()]).unwrap();
                let config: LookupConfig<F> =
                    LookupConfig::configure_with_table(meta, input, output, table.clone());
                NodeConfig::Lookup {
                    config: Rc::new(RefCell::new(config)),
                    inputs: node_inputs,
                }
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
        mut config: ModelConfig<F>,
        layouter: &mut impl Layouter<F>,
        inputs: &[ValTensor<F>],
        vars: &ModelVars<F>,
    ) -> Result<(), Box<dyn Error>> {
        info!("model layout");
        let mut results = BTreeMap::<usize, ValTensor<F>>::new();
        for (i, input_value) in inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(i, vars.instances[i].clone());
            } else {
                results.insert(i, input_value.clone());
            }
        }

        let mut offset: usize = 0;

        // layout any lookup tables
        let _: Vec<()> = config
            .configs
            .iter()
            .map(|(_, c)| match c {
                NodeConfig::Lookup { config, .. } => config.borrow_mut().layout_table(layouter),
                _ => Ok(()),
            })
            .collect::<Result<Vec<()>, _>>()?;

        layouter.assign_region(
            || "model",
            |mut region| {
                for (idx, config) in config.configs.iter() {
                    if let Some(vt) = self
                        .layout_config(&mut region, &mut results, config, &mut offset)
                        .map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?
                    {
                        // we get the max as for fused nodes this corresponds to the node output
                        results.insert(*idx, vt);
                        //only use with mock prover
                        if matches!(self.mode, Mode::Mock) {
                            trace!(
                                "------------ output node {:?}: {:?}",
                                idx,
                                results.get(idx).unwrap().show()
                            );
                        }
                    }
                }

                let output_nodes = self.model.outputs.iter();
                info!(
                    "model outputs are nodes: {:?}",
                    output_nodes.clone().map(|o| o.node).collect_vec()
                );
                let mut outputs = output_nodes
                    .map(|o| results.get(&o.node).unwrap().clone())
                    .collect_vec();

                // pack outputs if need be
                for (i, packed_output) in config.packed_outputs.iter_mut().enumerate() {
                    info!("packing outputs...");
                    outputs[i] = packed_output
                        .borrow_mut()
                        .layout(
                            &mut region,
                            &outputs[i..i + 1],
                            &mut offset,
                            PolyOp::Pack(self.run_args.pack_base, self.run_args.scale),
                        )
                        .map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?;
                    // only use with mock prover
                    if matches!(self.mode, Mode::Mock) {
                        trace!("------------ packed output {:?}", outputs[i].show());
                    }
                }

                let _ = config
                    .range_checks
                    .iter()
                    .zip(outputs)
                    .enumerate()
                    .map(|(i, (range_check, output))| {
                        let mut offset = 0;
                        if self.visibility.input.is_public() {
                            offset += inputs.len();
                        };
                        range_check.borrow_mut().layout(
                            &mut region,
                            &[
                                output,
                                vars.instances[offset + i].clone(),
                                vars.instances[offset + i].clone(),
                            ],
                            &mut offset,
                            PolyOp::RangeCheck(self.run_args.tolerance as i32),
                        )
                    })
                    .collect_vec();
                Ok(())
            },
        )?;
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
        region: &mut Region<F>,
        inputs: &mut BTreeMap<usize, ValTensor<F>>,
        config: &NodeConfig<F>,
        offset: &mut usize,
    ) -> Result<Option<ValTensor<F>>, Box<dyn Error>> {
        // The node kind and the config should be the same.
        let res = match config.clone() {
            NodeConfig::Poly {
                config,
                inputs: idx,
                op,
            } => {
                let values: Vec<ValTensor<F>> = idx
                    .iter()
                    .map(|i| {
                        let node = &self.nodes.get(i).unwrap();
                        match node.opkind {
                            OpKind::Const => {
                                let val = node
                                    .const_value
                                    .clone()
                                    .context("Tensor<i128> should already be loaded")
                                    .unwrap();
                                <Tensor<i128> as Into<Tensor<Value<F>>>>::into(val).into()
                            }
                            _ => inputs.get(i).unwrap().clone(),
                        }
                    })
                    .collect_vec();

                let res = config
                    .borrow_mut()
                    .layout(region, &values, offset, op.clone())?;

                Some(res)
            }
            NodeConfig::Lookup {
                config,
                inputs: idx,
            } => {
                if idx.len() != 1 {
                    return Err(Box::new(GraphError::InvalidLookupInputs));
                }

                let res =
                    config
                        .borrow_mut()
                        .layout(region, inputs.get(&idx[0]).unwrap(), offset)?;

                // For activations and elementwise operations, the dimensions are sometimes only in one or the other of input and output.
                Some(res)
            }
            NodeConfig::Input => None,
            NodeConfig::Const => None,
            _ => {
                return Err(Box::new(GraphError::UnsupportedOp));
            }
        };
        Ok(res)
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
            .map(|o| self.nodes.get(&o.node).unwrap().out_dims.clone())
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
            .map(|o| self.nodes.get(&o.node).unwrap().out_dims.clone())
            .collect_vec()
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<u32> {
        let output_nodes = self.model.outputs.iter();
        output_nodes
            .map(|o| self.nodes.get(&o.node).unwrap().out_scale)
            .collect_vec()
    }

    /// Total number of variables in lookup layers
    pub fn num_vars_lookup_op(&self, lookup_op: &LookupOp) -> Vec<usize> {
        let mut count = BTreeMap::<LookupOp, (usize, usize)>::new();
        for (_, n) in self.nodes.iter() {
            if n.opkind == OpKind::Lookup(lookup_op.clone()) {
                match &n.opkind {
                    OpKind::Lookup(op) => {
                        let elem = count.get_mut(op);
                        // handle output variables
                        let output_size: usize = n.out_dims.iter().product();
                        let input_size = output_size;
                        match elem {
                            None => {
                                count.insert(op.clone(), (input_size, output_size));
                            }
                            Some(m) => {
                                m.0 += input_size;
                                m.1 += output_size;
                            }
                        }
                    }
                    // should never reach here
                    _ => panic!(),
                }
            }
        }
        // now get the max across all ops
        let (mut num_inputs, mut num_outputs) = (0, 0);
        for (_, v) in count.iter() {
            num_inputs = max(num_inputs, v.0);
            num_outputs = max(num_outputs, v.1);
        }
        vec![num_inputs, num_outputs]
    }

    /// Maximum number of input variables
    pub fn max_input_var_len(&self) -> usize {
        let mut maximum_var_len = 0;

        let poly_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_poly())
            .collect();

        let _: Vec<_> = poly_ops
            .iter()
            .map(|(_, n)| match &n.opkind {
                OpKind::Poly(p) => {
                    let in_dims = n
                        .inputs
                        .iter()
                        .map(|i| self.nodes.get(&i.node).unwrap().out_dims.clone());
                    let layout_shape = p.circuit_shapes(in_dims.collect_vec());
                    maximum_var_len += layout_shape.last().unwrap();
                }
                _ => panic!(),
            })
            .collect();

        let lookup_ops: BTreeMap<&usize, &Node> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.opkind.is_lookup())
            .collect();

        // single lookup
        let input_size: usize = lookup_ops
            .iter()
            .map(|(_, n)| (*n.in_dims[0]).into_iter().product::<usize>())
            .sum();
        maximum_var_len += input_size;

        let output_lens: usize = self
            .output_shapes()
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>();

        let input_lens: usize = self
            .input_shapes()
            .iter()
            .map(|s| s.iter().product::<usize>())
            .sum::<usize>();

        if self.run_args.pack_base > 1 {
            maximum_var_len += output_lens;
        }
        if matches!(self.visibility.output, Visibility::Public) {
            maximum_var_len += output_lens;
        }
        if matches!(self.visibility.output, Visibility::Public) {
            maximum_var_len += input_lens;
        }

        maximum_var_len
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
}
