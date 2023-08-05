use super::extract_const_quantized_values;
use super::node::*;
use super::scale_to_multiplier;
use super::vars::*;
use super::GraphError;
use super::GraphSettings;
use crate::circuit::hybrid::HybridOp;
use crate::circuit::region::RegionCtx;
use crate::circuit::Input;
use crate::circuit::Unknown;
use crate::fieldutils::felt_to_i128;
use crate::{
    circuit::{lookup::LookupOp, BaseConfig as PolyConfig, CheckMode, Op},
    commands::RunArgs,
    tensor::{Tensor, ValTensor},
};
use halo2curves::bn256::Fr as Fp;

use colored::Colorize;
use serde::Deserialize;
use serde::Serialize;
use tract_onnx::prelude::{
    DatumExt, Graph, InferenceFact, InferenceModelExt, SymbolValues, TypedFact, TypedOp,
};
use tract_onnx::tract_hir::ops::scan::Scan;

use core::panic;
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use log::error;
use log::{debug, info, trace};
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use tabled::Table;
use tract_onnx;
use tract_onnx::prelude::Framework;

/// The result of a forward pass.
#[derive(Clone, Debug)]
pub struct ForwardResult {
    /// The outputs of the forward pass.
    pub outputs: Vec<Tensor<Fp>>,
    /// The maximum value of any input to a lookup operation.
    pub max_lookup_inputs: i128,
}

/// A circuit configuration for the entirety of a model loaded from an Onnx file.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// The base configuration for the circuit
    pub base: PolyConfig<Fp>,
    /// A wrapper for holding all columns that will be assigned to by the model
    pub vars: ModelVars<Fp>,
}

/// Representation of execution graph
pub type NodeGraph = BTreeMap<usize, NodeType>;

/// A struct for loading from an Onnx file and converting a computational graph to a circuit.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Model {
    /// input indices
    pub graph: ParsedNodes,
    /// Defines which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    pub visibility: VarVisibility,
}

/// Enables model as subnode of other models
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    /// A node in the model
    Node(Node),
    /// A submodel
    SubGraph {
        /// The subgraph
        model: Model,
        /// The subgraph's inputs
        inputs: Vec<usize>,
        /// the subgraph's idx within the parent graph
        idx: usize,
    },
}

impl NodeType {
    /// Returns the indices of the node's inputs.
    pub fn inputs(&self) -> Vec<usize> {
        match self {
            NodeType::Node(n) => n.inputs.clone(),
            NodeType::SubGraph { inputs, .. } => inputs.clone(),
        }
    }

    /// Returns the dimensions of the node's output.
    pub fn out_dims(&self) -> Vec<Vec<usize>> {
        match self {
            NodeType::Node(n) => vec![n.out_dims.clone()],
            NodeType::SubGraph { model, .. } => model.graph.output_shapes(),
        }
    }
    /// Returns the lookups required by a graph
    pub fn required_lookups(&self) -> Vec<LookupOp> {
        match self {
            NodeType::Node(n) => n.opkind.required_lookups(),
            NodeType::SubGraph { model, .. } => model.required_lookups(),
        }
    }
    /// Returns the scales of the node's output.
    pub fn out_scales(&self) -> Vec<u32> {
        match self {
            NodeType::Node(n) => vec![n.out_scale],
            NodeType::SubGraph { model, .. } => model.graph.get_output_scales(),
        }
    }

    /// Returns a string representation of the operation.
    pub fn as_str(&self) -> String {
        match self {
            NodeType::Node(n) => n.opkind.as_string(),
            NodeType::SubGraph { .. } => "SUBGRAPH".into(),
        }
    }

    /// Returns true if the operation is an input.
    pub fn is_input(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_input(),
            NodeType::SubGraph { .. } => false,
        }
    }
    /// Returns the node's unique identifier.
    pub fn idx(&self) -> usize {
        match self {
            NodeType::Node(n) => n.idx,
            NodeType::SubGraph { idx, .. } => *idx,
        }
    }

    /// decrement const num times used
    pub fn decrement_const(&mut self) {
        match self {
            NodeType::Node(n) => {
                if let Some(c) = n
                    .opkind
                    .as_any()
                    .downcast_ref::<crate::circuit::Constant<Fp>>()
                {
                    if c.num_uses > 0 {
                        n.opkind = SupportedOp::Constant(crate::circuit::Constant {
                            num_uses: c.num_uses - 1,
                            ..c.clone()
                        });
                    }
                }
            }
            NodeType::SubGraph { .. } => log::warn!("Cannot decrement const of subgraph"),
        }
    }

    /// bunp scale of node
    pub fn bump_scale(&mut self, scale: u32) {
        match self {
            NodeType::Node(n) => n.out_scale = scale,
            NodeType::SubGraph { .. } => log::warn!("Cannot bump scale of subgraph"),
        }
    }

    /// Replace the operation kind of the node.
    pub fn replace_opkind(&mut self, opkind: SupportedOp) {
        match self {
            NodeType::Node(n) => n.opkind = opkind,
            NodeType::SubGraph { .. } => log::warn!("Cannot replace opkind of subgraph"),
        }
    }

    /// Returns the operation kind of the node (if any).
    pub fn opkind(&self) -> Box<dyn Op<Fp>> {
        match self {
            NodeType::Node(n) => n.opkind.clone_dyn(),
            NodeType::SubGraph { .. } => Unknown.clone_dyn(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// A set of EZKL nodes that represent a computational graph.
pub struct ParsedNodes {
    nodes: BTreeMap<usize, NodeType>,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

impl ParsedNodes {
    /// Returns the number of the computational graph's inputs
    pub fn num_inputs(&self) -> usize {
        let input_nodes = self.inputs.iter();
        input_nodes.len()
    }

    ///  Returns shapes of the computational graph's inputs
    pub fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.inputs
            .iter()
            .flat_map(|o| self.nodes.get(o).unwrap().out_dims())
            .collect_vec()
    }

    /// Returns the number of the computational graph's outputs
    pub fn num_outputs(&self) -> usize {
        let output_nodes = self.outputs.iter();
        output_nodes.len()
    }

    /// Returns shapes of the computational graph's outputs
    pub fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.outputs
            .iter()
            .flat_map(|o| self.nodes.get(o).unwrap().out_dims())
            .collect_vec()
    }

    /// Returns the fixed point scale of the computational graph's inputs
    pub fn get_input_scales(&self) -> Vec<u32> {
        let input_nodes = self.inputs.iter();
        input_nodes
            .flat_map(|o| self.nodes.get(o).unwrap().out_scales())
            .collect_vec()
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Vec<u32> {
        let output_nodes = self.outputs.iter();
        output_nodes
            .flat_map(|o| self.nodes.get(o).unwrap().out_scales())
            .collect_vec()
    }
}

impl Model {
    fn required_lookups(&self) -> Vec<LookupOp> {
        self.graph
            .nodes
            .values()
            .flat_map(|n| n.required_lookups())
            .collect_vec()
    }

    /// Creates a `Model` from a specified path to an Onnx file.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `run_args` - [RunArgs]
    pub fn new(reader: &mut dyn std::io::Read, run_args: RunArgs) -> Result<Self, Box<dyn Error>> {
        let visibility = VarVisibility::from_args(run_args)?;

        let graph = Self::load_onnx_model(reader, &run_args, &visibility)?;

        let om = Model { graph, visibility };

        debug!("\n {}", om.table_nodes());

        Ok(om)
    }

    ///
    pub fn save(&self, path: PathBuf) -> Result<(), Box<dyn Error>> {
        let f = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(f);
        bincode::serialize_into(writer, &self)?;
        Ok(())
    }

    ///
    pub fn load(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        // read bytes from file
        let mut f = std::fs::File::open(&path).expect("no file found");
        let metadata = fs::metadata(&path).expect("unable to read metadata");
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer).expect("buffer overflow");
        let result = bincode::deserialize(&buffer)?;
        Ok(result)
    }

    /// Generate model parameters for the circuit
    pub fn gen_params(
        &self,
        run_args: RunArgs,
        check_mode: CheckMode,
    ) -> Result<GraphSettings, Box<dyn Error>> {
        let instance_shapes = self.instance_shapes();
        // this is the total number of variables we will need to allocate
        // for the circuit

        let num_constraints = if let Some(num_constraints) = run_args.allocated_constraints {
            num_constraints
        } else {
            self.dummy_layout(&run_args, &self.graph.input_shapes())
                .unwrap()
        };

        // Then number of columns in the circuits
        info!(
            "{} {} {}",
            "model generates".blue(),
            num_constraints.to_string().blue(),
            "constraints (excluding modules)".blue()
        );

        // extract the requisite lookup ops from the model
        let mut lookup_ops: Vec<LookupOp> = self.required_lookups();

        // if we're using percentage tolerance, we need to add the necessary range check ops for it.

        if run_args.tolerance.val > 0.0 {
            for scale in self.graph.get_output_scales() {
                let mut tolerance = run_args.tolerance;
                tolerance.scales = (
                    scale_to_multiplier(scale) as usize,
                    scale_to_multiplier(run_args.scale) as usize,
                );
                let opkind: Box<dyn Op<Fp>> = Box::new(HybridOp::RangeCheck(tolerance));
                lookup_ops.extend(opkind.required_lookups());
            }
        }

        let set: HashSet<_> = lookup_ops.drain(..).collect(); // dedup
        lookup_ops.extend(set.into_iter().sorted());

        let batch_size = self.graph.input_shapes()[0][0];
        assert!(self.graph.input_shapes().iter().all(|x| x[0] == batch_size));

        Ok(GraphSettings {
            run_args,
            model_instance_shapes: instance_shapes,
            module_sizes: crate::graph::modules::ModuleSizes::default(),
            num_constraints,
            required_lookups: lookup_ops,
            model_output_scales: self.graph.get_output_scales(),
            check_mode,
            version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Runs a forward pass on sample data !
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `model_inputs` - A vector of [Tensor]s to use as inputs to the model.
    /// * `run_args` - [RunArgs]
    pub fn forward(&self, model_inputs: &[Tensor<Fp>]) -> Result<ForwardResult, Box<dyn Error>> {
        let mut results: BTreeMap<&usize, Tensor<Fp>> = BTreeMap::new();
        let mut max_lookup_inputs = 0;
        let mut input_idx = 0;
        for (idx, n) in self.graph.nodes.iter() {
            let mut inputs = vec![];
            if n.is_input() {
                let mut t = model_inputs[input_idx].clone();
                input_idx += 1;
                t.reshape(&n.out_dims()[0]);
                inputs.push(t);
            } else {
                debug!("executing {}: {}", idx, n.as_str());
                trace!("dims: {:?}", n.out_dims());
                for i in n.inputs().iter() {
                    match results.get(&i) {
                        Some(value) => inputs.push(value.clone()),
                        None => return Err(Box::new(GraphError::MissingNode(*i))),
                    }
                }
            };

            if !n.required_lookups().is_empty() {
                let mut max = 0;
                for i in &inputs {
                    max = max.max(i.iter().map(|x| felt_to_i128(*x).abs()).max().unwrap());
                }
                max_lookup_inputs = max_lookup_inputs.max(max);
            }

            match n {
                NodeType::Node(n) => {
                    let res = Op::<Fp>::f(&n.opkind, &inputs)?;
                    // see if any of the intermediate lookup calcs are the max
                    if !res.intermediate_lookups.is_empty() {
                        let mut max = 0;
                        for i in &res.intermediate_lookups {
                            max = max.max(i.iter().map(|x| x.abs()).max().unwrap());
                        }
                        max_lookup_inputs = max_lookup_inputs.max(max);
                    }

                    results.insert(idx, res.output);
                }
                NodeType::SubGraph { model, .. } => {
                    let res = model.forward(&inputs)?;
                    // recursively get the max lookup inputs for subgraphs
                    max_lookup_inputs = max_lookup_inputs.max(res.max_lookup_inputs);

                    let mut res = res.outputs.last().unwrap().clone();
                    res.flatten();
                    results.insert(idx, res);
                }
            }
        }

        let output_nodes = self.graph.outputs.iter();
        debug!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().collect_vec()
        );
        let outputs = output_nodes
            .map(|o| results.get(&o).unwrap().clone().map(|x| x))
            .collect_vec();

        let res = ForwardResult {
            outputs,
            max_lookup_inputs,
        };

        Ok(res)
    }

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `scale` - The scale to use for quantization.
    /// * `public_params` - Whether to make the params public.
    fn load_onnx_model(
        reader: &mut dyn std::io::Read,
        run_args: &RunArgs,
        visibility: &VarVisibility,
    ) -> Result<ParsedNodes, Box<dyn Error>> {
        let start_time = instant::Instant::now();

        let mut model = tract_onnx::onnx().model_for_read(reader).map_err(|e| {
            error!("Error loading model: {}", e);
            GraphError::ModelLoad
        })?;

        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node(id.node);

            let mut dims = vec![];
            let extracted_dims: Vec<usize> = input.outputs[0]
                .fact
                .shape
                .dims()
                .filter_map(tract_onnx::tract_hir::internal::Factoid::concretize)
                .map(|x| match x.to_i64() {
                    Ok(x) => x as usize,
                    Err(_e) => {
                        if x.to_string() == "batch_size" {
                            run_args.batch_size
                        } else {
                            panic!("Unknown dimension {}: {:?}", x.to_string(), x)
                        }
                    }
                })
                .collect();

            dims.extend(extracted_dims);

            model.set_input_fact(i, f32::fact(dims).into())?;
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default()).unwrap();
        }
        // Note: do not optimize the model, as the layout will depend on underlying hardware
        let model = model.into_typed()?.into_decluttered()?;
        let batch_size_sym = model.symbol_table.sym("batch_size");
        let seq_len_sym = model.symbol_table.sym("sequence_length");
        let model = model
            .concretize_dims(
                &SymbolValues::default().with(&batch_size_sym, run_args.batch_size as i64),
            )?
            .concretize_dims(&SymbolValues::default().with(&seq_len_sym, 1))?;

        info!("set batch size to {}", run_args.batch_size);

        let nodes = Self::nodes_from_graph(
            &model,
            run_args,
            visibility,
            model.inputs.iter().map(|_| run_args.scale).collect(),
        )?;

        debug!("\n {}", model);

        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| o.node).collect(),
        };

        let duration = start_time.elapsed();
        trace!("model loading took: {:?}", duration);

        Ok(parsed_nodes)
    }

    /// Formats nodes (including subgraphs) into tables !
    pub fn table_nodes(&self) -> String {
        let mut node_accumulator = vec![];
        let mut string = String::new();
        for (idx, node) in &self.graph.nodes {
            match node {
                NodeType::Node(n) => {
                    node_accumulator.push(n);
                }
                NodeType::SubGraph { model, inputs, .. } => {
                    let mut table = Table::new(node_accumulator.iter());
                    table.with(tabled::settings::Style::modern());
                    table.with(tabled::settings::Shadow::new(1));
                    table.with(
                        tabled::settings::style::BorderColor::default()
                            .top(tabled::settings::Color::BG_YELLOW),
                    );
                    string = format!("{} \n\n  MAIN GRAPH \n\n{}", string, table);
                    node_accumulator = vec![];
                    string = format!(
                        "{}\n\n SUBGRAPH AT IDX {} WITH INPUTS {:?}\n{}",
                        string,
                        idx,
                        inputs,
                        model.table_nodes(),
                    );
                }
            }
        }

        let mut table = Table::new(node_accumulator.iter());
        table.with(tabled::settings::Style::modern());
        format!("{} \n{}", string, table)
    }

    /// Creates ezkl nodes from a tract graph
    /// # Arguments
    /// * `graph` - A tract graph.
    /// * `run_args` - [RunArgs]
    /// * `visibility` - Which inputs to the model are public and private (params, inputs, outputs) using [VarVisibility].
    /// * `input_scales` - The scales of the model's inputs.
    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        run_args: &RunArgs,
        visibility: &VarVisibility,
        input_scales: Vec<u32>,
    ) -> Result<BTreeMap<usize, NodeType>, Box<dyn Error>> {
        let mut nodes = BTreeMap::<usize, NodeType>::new();
        let mut input_idx = 0;
        for (i, n) in graph.nodes.iter().enumerate() {
            // Extract the slope layer hyperparams
            match n.op().downcast_ref::<Scan>() {
                Some(b) => {
                    let model = b.body.clone();
                    let input_scales = n
                        .inputs
                        .iter()
                        .map(|i| nodes.get(&i.node).unwrap().out_scales()[0])
                        .collect_vec();
                    let subgraph_nodes =
                        Self::nodes_from_graph(&model, run_args, visibility, input_scales)?;

                    let subgraph = ParsedNodes {
                        nodes: subgraph_nodes,
                        inputs: model.inputs.iter().map(|o| o.node).collect(),
                        outputs: model.outputs.iter().map(|o| o.node).collect(),
                    };

                    let om = Model {
                        graph: subgraph,
                        visibility: visibility.clone(),
                    };
                    nodes.insert(
                        i,
                        NodeType::SubGraph {
                            model: om,
                            inputs: n.inputs.iter().map(|i| i.node).collect_vec(),

                            idx: i,
                        },
                    );
                }
                None => {
                    let mut n = Node::new(
                        n.clone(),
                        &mut nodes,
                        run_args.scale,
                        run_args.param_visibility,
                        i,
                    )?;
                    if n.opkind.is_input() {
                        n.opkind = SupportedOp::Input(Input {
                            scale: input_scales[input_idx],
                        });
                        n.out_scale = n.opkind.out_scale(vec![], 0);
                        input_idx += 1
                    }
                    nodes.insert(i, NodeType::Node(n));
                }
            }
        }

        fn clean_useless_consts(nodes: &mut BTreeMap<usize, NodeType>) {
            // remove all nodes that are consts with 0 uses now
            nodes.retain(|_, n| match n {
                NodeType::Node(n) => {
                    if let Some(c) = n
                        .opkind
                        .as_any()
                        .downcast_ref::<crate::circuit::Constant<Fp>>()
                    {
                        c.num_uses > 0
                    } else {
                        true
                    }
                }
                NodeType::SubGraph { model, .. } => {
                    clean_useless_consts(&mut model.graph.nodes);
                    true
                }
            });
        }

        clean_useless_consts(&mut nodes);

        Ok(nodes)
    }

    /// Creates a `Model` from parsed run_args
    /// # Arguments
    /// * `params` - A [GraphSettings] struct holding parsed CLI arguments.
    pub fn from_run_args(
        run_args: &RunArgs,
        model: &std::path::PathBuf,
    ) -> Result<Self, Box<dyn Error>> {
        Model::new(&mut std::fs::File::open(model)?, *run_args)
    }

    /// Configures a model for the circuit
    /// # Arguments
    /// * `meta` - The constraint system.
    /// * `vars` - The variables for the circuit.
    /// * `run_args` - [RunArgs]
    /// * `required_lookups` - The required lookup operations for the circuit.
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        vars: &mut ModelVars<Fp>,
        num_bits: usize,
        required_lookups: Vec<LookupOp>,
        check_mode: CheckMode,
    ) -> Result<PolyConfig<Fp>, Box<dyn Error>> {
        info!("configuring model");

        let mut base_gate = PolyConfig::configure(
            meta,
            vars.advices[0..2].try_into()?,
            &vars.advices[2],
            check_mode,
        );
        // set scale for HybridOp::RangeCheck and call self.conf_lookup on that op for percentage tolerance case
        let input = &vars.advices[0];
        let output = &vars.advices[1];
        for op in required_lookups {
            base_gate.configure_lookup(meta, input, output, num_bits, &op)?;
        }

        Ok(base_gate)
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    /// * `config` - [ModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    /// * `vars` - The variables for the circuit.
    pub fn layout(
        &self,
        mut config: ModelConfig,
        layouter: &mut impl Layouter<Fp>,
        run_args: &RunArgs,
        inputs: &[ValTensor<Fp>],
        vars: &ModelVars<Fp>,
    ) -> Result<Vec<ValTensor<Fp>>, Box<dyn Error>> {
        info!("model layout...");

        let start_time = instant::Instant::now();

        // reshape inputs
        let inputs: Vec<ValTensor<Fp>> = inputs
            .iter()
            .zip(self.graph.input_shapes().iter())
            .map(|(input, shape)| {
                let mut t = input.clone();
                t.reshape(shape).unwrap();
                t
            })
            .collect_vec();

        let mut results = BTreeMap::<usize, ValTensor<Fp>>::new();

        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(*input_idx, vars.instances[i].clone());
            } else {
                results.insert(*input_idx, inputs[i].clone());
            }
        }

        config.base.layout_tables(layouter)?;

        let outputs = layouter.assign_region(
            || "model",
            |region| {
                let mut thread_safe_region = RegionCtx::new(region, 0);

                let outputs = self
                    .layout_nodes(&mut config, &mut thread_safe_region, &mut results)
                    .map_err(|e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    })?;

                if run_args.output_visibility == Visibility::Public {
                    let output_scales = self.graph.get_output_scales();
                    let global_scale = scale_to_multiplier(run_args.scale) as usize;
                    let _ = outputs
                        .iter()
                        .enumerate()
                        .map(|(i, output)| {
                            let mut tolerance = run_args.tolerance;
                            tolerance.scales =
                                (scale_to_multiplier(output_scales[i]) as usize, global_scale);

                            let mut instance_offset = 0;
                            if self.visibility.input.is_public() {
                                instance_offset += inputs.len();
                            };
                            config.base.layout(
                                &mut thread_safe_region,
                                &[output.clone(), vars.instances[instance_offset + i].clone()],
                                Box::new(HybridOp::RangeCheck(tolerance)),
                            )
                        })
                        .collect_vec();
                }
                info!("computing...");
                Ok(outputs)
            },
        )?;

        let duration = start_time.elapsed();
        trace!("model layout took: {:?}", duration);

        Ok(outputs)
    }

    fn layout_nodes(
        &self,
        config: &mut ModelConfig,
        region: &mut RegionCtx<Fp>,
        results: &mut BTreeMap<usize, ValTensor<Fp>>,
    ) -> Result<Vec<ValTensor<Fp>>, Box<dyn Error>> {
        let inputs = self.graph.inputs.clone();
        // index over results to get original inputs
        let orig_inputs: BTreeMap<usize, _> = results
            .clone()
            .into_iter()
            .filter(|(idx, _)| inputs.contains(idx))
            .collect();

        for (idx, node) in self.graph.nodes.iter() {
            let values: Vec<ValTensor<Fp>> = if !node.is_input() {
                node.inputs()
                    .iter()
                    .map(|i| results.get(i).unwrap().clone())
                    .collect_vec()
            } else {
                // we re-assign inputs
                vec![results.get(idx).unwrap().clone()]
            };

            debug!(
                "laying out {}: {}, offset:{}",
                idx,
                node.as_str(),
                region.offset()
            );
            trace!("dims: {:?}", node.out_dims());
            trace!(
                "input_dims {:?}",
                values.iter().map(|v| v.dims()).collect_vec()
            );

            match node {
                NodeType::Node(n) => {
                    let res = config
                        .base
                        .layout(region, &values, n.opkind.clone_dyn())
                        .map_err(|e| {
                            error!("{}", e);
                            halo2_proofs::plonk::Error::Synthesis
                        })?;

                    if let Some(vt) = res {
                        // we get the max as for fused nodes this corresponds to the node output
                        results.insert(*idx, vt);
                        //only use with mock prover
                        trace!(
                            "------------ output node {:?}: {:?}",
                            idx,
                            results.get(idx).unwrap().show()
                        );
                    }
                }
                NodeType::SubGraph { model, .. } => {
                    let res = model.layout_nodes(config, region, results)?;
                    let mut res = res.last().unwrap().clone();
                    res.flatten();
                    results.insert(*idx, res);
                }
            }
        }

        // we do this so we can support multiple passes of the same model and have deterministic results (Non-assigned inputs etc... etc...)
        results.extend(orig_inputs);

        let output_nodes = self.graph.outputs.iter();
        debug!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().collect_vec()
        );
        let outputs = output_nodes
            .map(|o| results.get(o).unwrap().clone())
            .collect_vec();

        Ok(outputs)
    }

    /// Assigns dummy values to the regions created when calling `configure`.
    /// # Arguments
    /// * `input_shapes` - The shapes of the inputs to the model.
    pub fn dummy_layout(
        &self,
        run_args: &RunArgs,
        input_shapes: &[Vec<usize>],
    ) -> Result<usize, Box<dyn Error>> {
        info!("calculating num of constraints using dummy model layout...");

        let start_time = instant::Instant::now();

        let mut results = BTreeMap::<usize, ValTensor<Fp>>::new();

        let inputs: Vec<ValTensor<Fp>> = input_shapes
            .iter()
            .map(|shape| {
                let mut t: Tensor<Value<Fp>> =
                    Tensor::from(vec![Value::<Fp>::unknown(); shape.iter().product()].into_iter());
                t.reshape(shape);
                t.into()
            })
            .collect_vec();

        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            results.insert(*input_idx, inputs[i].clone());
        }

        let mut dummy_config = PolyConfig::dummy(run_args.logrows as usize);
        let mut model_config = ModelConfig {
            base: dummy_config.clone(),
            vars: ModelVars::new_dummy(),
        };

        let mut region = RegionCtx::new_dummy(0);

        let outputs = self.layout_nodes(&mut model_config, &mut region, &mut results)?;

        if run_args.output_visibility == Visibility::Public {
            let _ = outputs
                .into_iter()
                .map(|output| {
                    dummy_config
                        .layout(
                            &mut region,
                            &[output.clone(), output],
                            Box::new(HybridOp::RangeCheck(run_args.tolerance)),
                        )
                        .unwrap()
                })
                .collect_vec();
        }

        let duration = start_time.elapsed();
        trace!("dummy model layout took: {:?}", duration);

        Ok(region.offset())
    }

    /// Retrieves all constants from the model.
    pub fn get_all_params(&self) -> Vec<Tensor<Fp>> {
        let mut params = vec![];
        for node in self.graph.nodes.values() {
            match node {
                NodeType::Node(n) => {
                    let boxed_op = n.opkind.clone_dyn();
                    if let Some(constant) = extract_const_quantized_values(boxed_op.clone()) {
                        params.push(constant);
                    }
                }
                NodeType::SubGraph { model, .. } => {
                    params.extend(model.get_all_params());
                }
            }
        }
        params
    }

    /// Shapes of the computational graph's constants
    pub fn const_shapes(&self) -> Vec<Vec<usize>> {
        let mut const_shapes = vec![];
        for node in self.graph.nodes.values() {
            match node {
                NodeType::Node(n) => {
                    let boxed_op = n.opkind.clone_dyn();
                    if let Some(constant) = extract_const_quantized_values(boxed_op) {
                        const_shapes.push(constant.dims().to_vec());
                    };
                }
                NodeType::SubGraph { model, .. } => {
                    const_shapes.extend(model.const_shapes());
                }
            }
        }
        const_shapes
    }

    /// Replaces all constants in the model with the provided values (in order of indexing)
    pub fn replace_consts(&mut self, consts: Vec<ValTensor<Fp>>) {
        let mut const_idx = 0;
        for node in self.graph.nodes.values_mut() {
            match node {
                NodeType::Node(n) => {
                    let boxed_op = n.opkind.clone_dyn();
                    if let Some(constant) = boxed_op
                        .as_any()
                        .downcast_ref::<crate::circuit::ops::Constant<Fp>>()
                    {
                        let mut op = crate::circuit::Constant::new(
                            constant.quantized_values.clone(),
                            constant.raw_values.clone(),
                        );
                        op.pre_assign(consts[const_idx].clone());
                        n.opkind = SupportedOp::Constant(op);

                        const_idx += 1;
                    }
                }
                NodeType::SubGraph { model, .. } => {
                    model.replace_consts(consts.clone());
                }
            }
        }
    }

    /// Shapes of the computational graph's public inputs (if any)
    pub fn instance_shapes(&self) -> Vec<Vec<usize>> {
        let mut instance_shapes = vec![];
        if self.visibility.input.is_public() {
            instance_shapes.extend(self.graph.input_shapes());
        }
        if self.visibility.output.is_public() {
            instance_shapes.extend(self.graph.output_shapes());
        }
        instance_shapes
    }
}
