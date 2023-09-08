use super::extract_const_quantized_values;
use super::node::*;
use super::scale_to_multiplier;
use super::vars::*;
use super::GraphError;
use super::GraphSettings;
use crate::circuit::hybrid::HybridOp;
use crate::circuit::region::RegionCtx;
#[cfg(not(target_arch = "wasm32"))]
use crate::circuit::Input;
use crate::circuit::Unknown;
use crate::fieldutils::felt_to_i128;
use crate::{
    circuit::{lookup::LookupOp, BaseConfig as PolyConfig, CheckMode, Op},
    tensor::{Tensor, ValTensor},
    RunArgs,
};
use halo2curves::bn256::Fr as Fp;

#[cfg(not(target_arch = "wasm32"))]
use colored::Colorize;
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::{
    Framework, Graph, InferenceFact, InferenceModelExt, SymbolValues, TypedFact, TypedOp,
};
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_hir::ops::scan::Scan;

use log::error;
use log::{debug, info, trace};
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use tabled::Table;
use unzip_n::unzip_n;

unzip_n!(pub 3);

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

///
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OutputMapping {
    ///
    Single {
        ///
        outlet: usize,
        ///
        is_state: bool,
    },
    ///
    Stacked {
        ///
        outlet: usize,
        ///
        axis: usize,
        ///
        is_state: bool,
    },
}

impl OutputMapping {
    ///
    pub fn is_state(&self) -> bool {
        match self {
            OutputMapping::Single { is_state, .. } => *is_state,
            OutputMapping::Stacked { is_state, .. } => *is_state,
        }
    }

    ///
    pub fn outlet(&self) -> usize {
        match self {
            OutputMapping::Single { outlet, .. } => *outlet,
            OutputMapping::Stacked { outlet, .. } => *outlet,
        }
    }
}

///
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InputMapping {
    ///
    Full,
    ///
    State,
    ///
    Stacked {
        ///
        axis: usize,
        ///
        chunk: usize,
    },
}

fn number_of_iterations(mappings: &[InputMapping], dims: Vec<&[usize]>) -> usize {
    let mut number_of_iterations =
        dims.iter()
            .zip(mappings)
            .filter_map(|(dims, mapping)| match mapping {
                InputMapping::Stacked { axis, chunk } => Some(
                    // number of iterations given the dim size along the axis
                    // and the chunk size
                    (dims[*axis] + chunk - 1) / chunk,
                ),
                _ => None,
            });
    // assert all collected number of iterations are equal
    assert!(number_of_iterations.clone().all_equal());

    number_of_iterations.next().unwrap_or(1)
}

fn input_state_idx(input_mappings: &[InputMapping]) -> Vec<usize> {
    input_mappings
        .iter()
        .enumerate()
        .filter(|(_, r)| matches!(r, InputMapping::State))
        .map(|(index, _)| index)
        .collect::<Vec<_>>()
}

fn output_state_idx(output_mappings: &[Vec<OutputMapping>]) -> Vec<usize> {
    output_mappings
        .iter()
        .flatten()
        .filter_map(|x| if x.is_state() { Some(x.outlet()) } else { None })
        .collect::<Vec<_>>()
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
        inputs: Vec<Outlet>,
        /// the subgraph's idx within the parent graph
        idx: usize,
        /// output mappings
        output_mappings: Vec<Vec<OutputMapping>>,
        /// input mappings
        input_mappings: Vec<InputMapping>,
        ///
        out_dims: Vec<Vec<usize>>,
        ///
        out_scales: Vec<u32>,
    },
}

impl NodeType {
    /// Returns the indices of the node's inputs.
    pub fn inputs(&self) -> Vec<Outlet> {
        match self {
            NodeType::Node(n) => n.inputs.clone(),
            NodeType::SubGraph { inputs, .. } => inputs.clone(),
        }
    }

    /// Returns the dimensions of the node's output.
    pub fn out_dims(&self) -> Vec<Vec<usize>> {
        match self {
            NodeType::Node(n) => vec![n.out_dims.clone()],
            NodeType::SubGraph { out_dims, .. } => out_dims.clone(),
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
            NodeType::SubGraph { out_scales, .. } => out_scales.clone(),
        }
    }

    /// Returns a string representation of the operation.
    pub fn as_str(&self) -> String {
        match self {
            NodeType::Node(n) => n.opkind.as_string(),
            NodeType::SubGraph { .. } => "SUBGRAPH".into(),
        }
    }

    /// Returns true if the operation is a rebase
    pub fn is_rebase(&self) -> bool {
        match self {
            NodeType::Node(n) => matches!(n.opkind, SupportedOp::RebaseScale { .. }),
            NodeType::SubGraph { .. } => false,
        }
    }

    /// Returns true if the operation is an input.
    pub fn is_input(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_input(),
            NodeType::SubGraph { .. } => false,
        }
    }
    /// Returns true if the operation is a const.
    pub fn is_constant(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_constant(),
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
            NodeType::Node(n) => match &n.opkind {
                SupportedOp::Constant(c) => {
                    n.opkind = SupportedOp::Constant(crate::circuit::Constant {
                        num_uses: c.num_uses - 1,
                        ..c.clone()
                    })
                }
                _ => log::warn!("Cannot decrement const of non-const node"),
            },
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
    pub fn opkind(&self) -> SupportedOp {
        match self {
            NodeType::Node(n) => n.opkind.clone(),
            NodeType::SubGraph { .. } => SupportedOp::Unknown(Unknown),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// A set of EZKL nodes that represent a computational graph.
pub struct ParsedNodes {
    /// The nodes in the graph.
    pub nodes: BTreeMap<usize, NodeType>,
    inputs: Vec<usize>,
    outputs: Vec<Outlet>,
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
            .map(|(idx, outlet)| self.nodes.get(idx).unwrap().out_dims()[*outlet].clone())
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
            .map(|(idx, outlet)| self.nodes.get(idx).unwrap().out_scales()[*outlet])
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> Result<Self, Box<dyn Error>> {
        let visibility = VarVisibility::from_args(run_args)?;

        let graph = Self::load_onnx_model(reader, run_args, &visibility)?;

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
        let mut f = std::fs::File::open(&path)
            .unwrap_or_else(|_| panic!("failed to load model at {}", path.display()));
        let metadata = fs::metadata(&path).expect("unable to read metadata");
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer).expect("buffer overflow");
        let result = bincode::deserialize(&buffer)?;
        Ok(result)
    }

    /// Generate model parameters for the circuit
    pub fn gen_params(
        &self,
        run_args: &RunArgs,
        check_mode: CheckMode,
    ) -> Result<GraphSettings, Box<dyn Error>> {
        let instance_shapes = self.instance_shapes();
        #[cfg(not(target_arch = "wasm32"))]
        info!(
            "{} {} {}",
            "model has".blue(),
            instance_shapes.len().to_string().blue(),
            "instances".blue()
        );
        // this is the total number of variables we will need to allocate
        // for the circuit
        let (num_constraints, total_const_size) = self
            .dummy_layout(run_args, &self.graph.input_shapes())
            .unwrap();

        // Then number of columns in the circuits
        #[cfg(not(target_arch = "wasm32"))]
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
                tolerance.scale = scale_to_multiplier(scale).into();
                let opkind: Box<dyn Op<Fp>> = Box::new(HybridOp::RangeCheck(tolerance));
                lookup_ops.extend(opkind.required_lookups());
            }
        }

        let set: HashSet<_> = lookup_ops.drain(..).collect(); // dedup
        lookup_ops.extend(set.into_iter().sorted());

        Ok(GraphSettings {
            run_args: run_args.clone(),
            model_instance_shapes: instance_shapes,
            module_sizes: crate::graph::modules::ModuleSizes::default(),
            num_constraints,
            required_lookups: lookup_ops,
            model_output_scales: self.graph.get_output_scales(),
            total_const_size,
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
        let mut results: BTreeMap<&usize, Vec<Tensor<Fp>>> = BTreeMap::new();
        let mut max_lookup_inputs = 0;
        let mut input_idx = 0;
        for (idx, n) in self.graph.nodes.iter() {
            let mut inputs = vec![];
            if n.is_input() {
                let t = model_inputs[input_idx].clone();
                input_idx += 1;
                inputs.push(t);
            } else {
                for (idx, outlet) in n.inputs().iter() {
                    match results.get(&idx) {
                        Some(value) => inputs.push(value[*outlet].clone()),
                        None => return Err(Box::new(GraphError::MissingNode(*idx))),
                    }
                }
            };

            debug!("executing {}: {}", idx, n.as_str());
            debug!("dims: {:?}", n.out_dims());
            debug!(
                "input_dims: {:?}",
                inputs.iter().map(|x| x.dims()).collect::<Vec<_>>()
            );

            if !n.required_lookups().is_empty() {
                let mut max = 0;
                for i in &inputs {
                    max = max.max(i.iter().map(|x| felt_to_i128(*x).abs()).max().unwrap());
                }
                max_lookup_inputs = max_lookup_inputs.max(max);
            }

            match n {
                NodeType::Node(n) => {
                    // execute the op
                    let start = instant::Instant::now();
                    let res = Op::<Fp>::f(&n.opkind, &inputs)?;
                    let elapsed = start.elapsed();
                    trace!("op took: {:?}", elapsed);
                    // see if any of the intermediate lookup calcs are the max
                    if !res.intermediate_lookups.is_empty() {
                        let mut max = 0;
                        for i in &res.intermediate_lookups {
                            max = max.max(i.iter().map(|x| x.abs()).max().unwrap());
                        }
                        max_lookup_inputs = max_lookup_inputs.max(max);
                    }
                    debug!(
                        "------------ output node {}: {:?}",
                        idx,
                        res.output.map(crate::fieldutils::felt_to_i32).show()
                    );
                    results.insert(idx, vec![res.output]);
                }
                NodeType::SubGraph {
                    model,
                    output_mappings,
                    input_mappings,
                    inputs: input_tuple,
                    ..
                } => {
                    let mut orig_inputs = inputs.clone();
                    let mut input_mappings = input_mappings.clone();
                    (input_mappings, inputs, orig_inputs) = input_mappings
                        .iter()
                        .zip(inputs)
                        .zip(orig_inputs)
                        .zip(model.graph.inputs.iter())
                        .sorted_by_key(|(_, source)| *source)
                        .map(|(((x, y), z), _)| (x.clone(), y.clone(), z.clone()))
                        .unzip_n_vec();

                    let input_dims = inputs.iter().map(|inp| inp.dims());
                    let num_iter = number_of_iterations(&input_mappings, input_dims.collect());

                    debug!(
                        "{} iteration(s) in a subgraph with inputs {:?} and sources {:?}",
                        num_iter, input_tuple, model.graph.inputs
                    );

                    debug!("input_mappings: {:?}", input_mappings);

                    let mut full_results: Vec<Tensor<Fp>> = vec![];

                    for i in 0..num_iter {
                        // replace the Stacked input with the current chunk iter
                        for ((mapping, inp), og_input) in
                            input_mappings.iter().zip(&mut inputs).zip(&orig_inputs)
                        {
                            match mapping {
                                InputMapping::Stacked { axis, chunk } => {
                                    let start = i * chunk;
                                    let end = (i + 1) * chunk;
                                    let t =
                                        crate::tensor::ops::slice(og_input, axis, &start, &end)?;
                                    *inp = t;
                                }
                                _ => {}
                            }
                        }

                        let res = model.forward(&inputs)?;
                        // recursively get the max lookup inputs for subgraphs
                        max_lookup_inputs = max_lookup_inputs.max(res.max_lookup_inputs);

                        let mut outlets = BTreeMap::new();
                        for (mappings, outlet_res) in output_mappings.iter().zip(res.outputs) {
                            for mapping in mappings {
                                match mapping {
                                    OutputMapping::Single { outlet, .. } => {
                                        outlets.insert(outlet, outlet_res.clone());
                                    }
                                    OutputMapping::Stacked { outlet, axis, .. } => {
                                        if !full_results.is_empty() {
                                            let stacked_res = crate::tensor::ops::concat(
                                                &[
                                                    full_results[*outlet].clone(),
                                                    outlet_res.clone(),
                                                ],
                                                *axis,
                                            )?;

                                            outlets.insert(outlet, stacked_res);
                                        } else {
                                            outlets.insert(outlet, outlet_res.clone());
                                        }
                                    }
                                }
                            }
                        }

                        full_results = outlets.into_values().collect_vec();

                        let output_states = output_state_idx(output_mappings);
                        let input_states = input_state_idx(&input_mappings);

                        assert_eq!(input_states.len(), output_states.len());

                        for (input_idx, output_idx) in input_states.iter().zip(output_states) {
                            inputs[*input_idx] = full_results[output_idx].clone();
                        }
                    }

                    trace!(
                        "------------ output subgraph node {}: {:?}",
                        idx,
                        full_results
                            .iter()
                            .map(|x|
                            // convert to tensor i32
                            x.map(crate::fieldutils::felt_to_i32).show())
                            .collect_vec()
                    );

                    results.insert(idx, full_results);
                }
            }
        }

        let output_nodes = self.graph.outputs.iter();
        debug!(
            "model outputs are nodes: {:?}",
            output_nodes.clone().collect_vec()
        );
        let outputs = output_nodes
            .map(|(idx, outlet)| results.get(&idx).unwrap()[*outlet].clone())
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
    #[cfg(not(target_arch = "wasm32"))]
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

        let variables: std::collections::HashMap<String, usize> =
            std::collections::HashMap::from_iter(run_args.variables.clone());

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
                    Err(_e) => *variables.get(&x.to_string()).unwrap_or_else(|| {
                        panic!(
                            "Unknown dimension {}: {:?} in model inputs",
                            x.to_string(),
                            x
                        )
                    }),
                })
                .collect();

            dims.extend(extracted_dims);

            let fact = model.node(id.node).outputs[0].fact.clone();
            let fact = fact.with_shape(dims);

            model.set_input_fact(i, fact)?;
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default()).unwrap();
        }
        // Note: do not optimize the model, as the layout will depend on underlying hardware
        let mut model = model.into_typed()?.into_decluttered()?;
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbol_table.sym(symbol);
            model = model.concretize_dims(&SymbolValues::default().with(&symbol, *value as i64))?;
            info!("set {} to {}", symbol, value);
        }

        let scales = VarScales::from_args(run_args)?;
        let nodes = Self::nodes_from_graph(&model, run_args, &scales, visibility, None)?;

        debug!("\n {}", model);

        let parsed_nodes = ParsedNodes {
            nodes,
            inputs: model.inputs.iter().map(|o| o.node).collect(),
            outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
        };

        let duration = start_time.elapsed();
        trace!("model loading took: {:?}", duration);

        Ok(parsed_nodes)
    }

    /// Formats nodes (including subgraphs) into tables !
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn nodes_from_graph(
        graph: &Graph<TypedFact, Box<dyn TypedOp>>,
        run_args: &RunArgs,
        scales: &VarScales,
        visibility: &VarVisibility,
        override_input_scales: Option<Vec<u32>>,
    ) -> Result<BTreeMap<usize, NodeType>, Box<dyn Error>> {
        use crate::graph::node_output_shapes;

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
                    let subgraph_nodes = Self::nodes_from_graph(
                        &model,
                        run_args,
                        scales,
                        visibility,
                        Some(input_scales.clone()),
                    )?;

                    let subgraph = ParsedNodes {
                        nodes: subgraph_nodes,
                        inputs: model.inputs.iter().map(|o| o.node).collect(),
                        outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
                    };

                    let mut om = Model {
                        graph: subgraph,
                        visibility: visibility.clone(),
                    };

                    let mut output_mappings = vec![];
                    let mut output_scales = BTreeMap::new();

                    for (i, mapping) in b.output_mapping.iter().enumerate() {
                        let mut mappings = vec![];
                        if let Some(outlet) = mapping.last_value_slot {
                            mappings.push(OutputMapping::Single {
                                outlet,
                                is_state: mapping.state,
                            });
                            output_scales.insert(outlet, om.graph.get_output_scales()[i]);
                        }
                        if let Some(last) = mapping.scan {
                            mappings.push(OutputMapping::Stacked {
                                outlet: last.0,
                                axis: last.1.axis,
                                is_state: false,
                            });
                            output_scales.insert(last.0, om.graph.get_output_scales()[i]);
                        }
                        output_mappings.push(mappings);
                    }

                    let mut out_scales = output_scales.into_values().collect_vec();
                    let output_state_idx = output_state_idx(&output_mappings);

                    let mut input_mappings = vec![];
                    for mapping in &b.input_mapping {
                        match mapping {
                            tract_onnx::tract_hir::ops::scan::InputMapping::Scan(info) => {
                                input_mappings.push(InputMapping::Stacked {
                                    axis: info.axis,
                                    chunk: info.chunk as usize,
                                });
                            }
                            tract_onnx::tract_hir::ops::scan::InputMapping::State => {
                                input_mappings.push(InputMapping::State);
                            }
                            tract_onnx::tract_hir::ops::scan::InputMapping::Full => {
                                input_mappings.push(InputMapping::Full);
                            }
                        }
                    }

                    let input_state_idx = input_state_idx(&input_mappings);

                    // if input_state_idx and output_state_idx have mismatched scales we need to rebase the scale of the output node
                    for (input_idx, output_idx) in input_state_idx.iter().zip(output_state_idx) {
                        let output_scale = out_scales[output_idx];
                        let input_scale = input_scales[*input_idx];
                        let scale_diff = output_scale as i32 - input_scale as i32;
                        if scale_diff != 0 {
                            // output mappings is a vec of vec. we need to find the outer index of the output node we want to rebase.
                            let mut traversed_len = 0;
                            for (outer_idx, mappings) in output_mappings.iter().enumerate() {
                                let mapping_len = mappings.len();
                                if traversed_len + mapping_len > output_idx {
                                    let output_node_idx = b.body.outputs[outer_idx].node;
                                    let node = om.graph.nodes.get_mut(&output_node_idx).unwrap();
                                    let rebased_node = if scale_diff > 0 {
                                        RebaseScale::rebase(
                                            node.opkind(),
                                            input_scale,
                                            output_scale,
                                            1,
                                        )
                                    } else {
                                        RebaseScale::rebase_up(
                                            node.opkind(),
                                            input_scale,
                                            output_scale,
                                        )
                                    };
                                    out_scales[output_idx] = input_scale;
                                    node.replace_opkind(rebased_node);
                                    break;
                                }
                                traversed_len += mapping_len;
                            }
                        }
                    }

                    let out_dims = node_output_shapes(n)?
                        .iter()
                        .map(|shape| shape.as_ref().unwrap().clone())
                        .collect_vec();

                    nodes.insert(
                        i,
                        NodeType::SubGraph {
                            model: om,
                            inputs: n.inputs.iter().map(|i| (i.node, i.slot)).collect_vec(),
                            idx: i,
                            output_mappings,
                            input_mappings,
                            out_dims,
                            out_scales,
                        },
                    );
                }
                None => {
                    let mut n =
                        Node::new(n.clone(), &mut nodes, scales, run_args.param_visibility, i)?;
                    if override_input_scales.is_some() {
                        if let Some(inp) = n.opkind.get_input() {
                            let scale = override_input_scales.as_ref().unwrap()[input_idx];
                            n.opkind = SupportedOp::Input(Input {
                                scale,
                                datum_type: inp.datum_type,
                            });
                            input_idx += 1;
                            n.out_scale = scale;
                        }
                    }
                    nodes.insert(i, NodeType::Node(n));
                }
            }
        }
        Self::empty_raw_const_value(&mut nodes);

        Ok(nodes)
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Removes all nodes that are consts with 0 uses
    fn empty_raw_const_value(nodes: &mut BTreeMap<usize, NodeType>) {
        // remove all nodes that are consts with 0 uses now
        nodes.retain(|_, n| match n {
            NodeType::Node(n) => match &mut n.opkind {
                SupportedOp::Constant(c) => {
                    c.empty_raw_value();
                    c.num_uses > 0
                }
                _ => true,
            },
            NodeType::SubGraph { model, .. } => {
                Self::empty_raw_const_value(&mut model.graph.nodes);
                true
            }
        });
    }

    /// Creates a `Model` from parsed run_args
    /// # Arguments
    /// * `params` - A [GraphSettings] struct holding parsed CLI arguments.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_run_args(
        run_args: &RunArgs,
        model: &std::path::Path,
    ) -> Result<Self, Box<dyn Error>> {
        Model::new(
            &mut std::fs::File::open(model)
                .map_err(|_| format!("failed to load model at {}", model.display()))?,
            run_args,
        )
    }

    /// Configures a model for the circuit
    /// # Arguments
    /// * `meta` - The constraint system.
    /// * `vars` - The variables for the circuit.
    /// * `run_args` - [RunArgs]
    /// * `required_lookups` - The required lookup operations for the circuit.
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        vars: &ModelVars<Fp>,
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

        let mut results = BTreeMap::<usize, Vec<ValTensor<Fp>>>::new();

        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                results.insert(*input_idx, vec![vars.instances[i].clone()]);
            } else {
                results.insert(*input_idx, vec![inputs[i].clone()]);
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
                    let _ = outputs
                        .iter()
                        .enumerate()
                        .map(|(i, output)| {
                            let mut tolerance = run_args.tolerance;
                            tolerance.scale = scale_to_multiplier(output_scales[i]).into();

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
                info!("model has {} assigned rows", thread_safe_region.offset());
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
        results: &mut BTreeMap<usize, Vec<ValTensor<Fp>>>,
    ) -> Result<Vec<ValTensor<Fp>>, Box<dyn Error>> {
        // index over results to get original inputs
        let orig_inputs: BTreeMap<usize, _> = results
            .clone()
            .into_iter()
            .filter(|(idx, _)| self.graph.inputs.contains(idx))
            .collect();

        for (idx, node) in self.graph.nodes.iter() {
            let mut values: Vec<ValTensor<Fp>> = if !node.is_input() {
                node.inputs()
                    .iter()
                    .map(|(idx, outlet)| results.get(idx).unwrap()[*outlet].clone())
                    .collect_vec()
            } else {
                // we re-assign inputs, always from the 0 outlet
                vec![results.get(idx).unwrap()[0].clone()]
            };

            debug!(
                "laying out {}: {}, offset:{}",
                idx,
                node.as_str(),
                region.offset()
            );
            debug!("dims: {:?}", node.out_dims());
            debug!(
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
                        results.insert(*idx, vec![vt.clone()]);
                        //only use with mock prover
                        debug!("------------ output node {:?}: {:?}", idx, vt.show());
                    }
                }
                NodeType::SubGraph {
                    model,
                    inputs,
                    output_mappings,
                    input_mappings,
                    ..
                } => {
                    let mut original_values = values.clone();
                    let mut input_mappings = input_mappings.clone();
                    (input_mappings, values, original_values) = input_mappings
                        .iter()
                        .zip(values)
                        .zip(original_values)
                        .zip(model.graph.inputs.iter())
                        .sorted_by_key(|(_, source)| *source)
                        .map(|(((x, y), z), _)| (x.clone(), y.clone(), z.clone()))
                        .unzip_n_vec();

                    let input_dims = values.iter().map(|inp| inp.dims());
                    let num_iter = number_of_iterations(&input_mappings, input_dims.collect());

                    debug!(
                        "{} iteration(s) in a subgraph with inputs {:?} and sources {:?}",
                        num_iter, inputs, model.graph.inputs
                    );

                    let mut full_results: Vec<ValTensor<Fp>> = vec![];

                    for i in 0..num_iter {
                        debug!(" -------------- subgraph iteration: {}", i);
                        // replace the Stacked input with the current chunk iter
                        for ((mapping, inp), og_inp) in
                            input_mappings.iter().zip(&mut values).zip(&original_values)
                        {
                            match mapping {
                                InputMapping::Stacked { axis, chunk } => {
                                    let start = i * chunk;
                                    let end = (i + 1) * chunk;
                                    let mut sliced_input = og_inp.clone();
                                    sliced_input.slice(axis, &start, &end)?;
                                    *inp = sliced_input;
                                }
                                _ => {}
                            }
                        }

                        let mut subgraph_results = BTreeMap::from_iter(
                            model
                                .graph
                                .inputs
                                .clone()
                                .into_iter()
                                .sorted()
                                .zip(values.clone().into_iter().map(|v| vec![v])),
                        );

                        let res = model.layout_nodes(config, region, &mut subgraph_results)?;

                        let mut outlets = BTreeMap::new();

                        for (mappings, outlet_res) in output_mappings.iter().zip(res) {
                            for mapping in mappings {
                                match mapping {
                                    OutputMapping::Single { outlet, .. } => {
                                        outlets.insert(outlet, outlet_res.clone());
                                    }
                                    OutputMapping::Stacked { outlet, axis, .. } => {
                                        if !full_results.is_empty() {
                                            let stacked_res = full_results[*outlet]
                                                .clone()
                                                .concat_axis(outlet_res.clone(), axis)?;

                                            outlets.insert(outlet, stacked_res);
                                        } else {
                                            outlets.insert(outlet, outlet_res.clone());
                                        }
                                    }
                                }
                            }
                        }

                        full_results = outlets.into_values().collect_vec();

                        let output_states = output_state_idx(output_mappings);
                        let input_states = input_state_idx(&input_mappings);

                        assert_eq!(input_states.len(), output_states.len());

                        for (input_idx, output_idx) in input_states.iter().zip(output_states) {
                            values[*input_idx] = full_results[output_idx].clone();
                        }
                    }

                    //only use with mock prover
                    trace!(
                        "------------ output subgraph node {:?}: {:?}",
                        idx,
                        full_results.iter().map(|x| x.show()).collect_vec()
                    );

                    results.insert(*idx, full_results);
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
            .map(|(idx, outlet)| results.get(idx).unwrap()[*outlet].clone())
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
    ) -> Result<(usize, usize), Box<dyn Error>> {
        info!("calculating num of constraints using dummy model layout...");

        let start_time = instant::Instant::now();

        let mut results = BTreeMap::<usize, Vec<ValTensor<Fp>>>::new();

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
            results.insert(*input_idx, vec![inputs[i].clone()]);
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

        Ok((region.offset(), region.total_constants()))
    }

    /// Retrieves all constants from the model.
    pub fn get_all_params(&self) -> Vec<Tensor<Fp>> {
        let mut params = vec![];
        for node in self.graph.nodes.values() {
            match node {
                NodeType::Node(_) => {
                    if let Some(constant) = extract_const_quantized_values(node.opkind()) {
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
                NodeType::Node(_) => {
                    if let Some(constant) = extract_const_quantized_values(node.opkind()) {
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

    /// Replaces all constants in the model with the provided values (in order of indexing), returns the number of consts
    pub fn replace_consts(&mut self, consts: &[ValTensor<Fp>]) -> usize {
        let mut const_idx = 0;
        for node in self.graph.nodes.values_mut() {
            match node {
                NodeType::Node(n) => match &n.opkind {
                    SupportedOp::Constant(c) => {
                        let mut op = crate::circuit::Constant::new(
                            c.quantized_values.clone(),
                            c.raw_values.clone(),
                        );
                        op.pre_assign(consts[const_idx].clone());
                        n.opkind = SupportedOp::Constant(op);

                        const_idx += 1;
                    }
                    _ => {}
                },
                NodeType::SubGraph { model, .. } => {
                    let total_consts = model.replace_consts(&consts[const_idx..]);
                    const_idx += total_consts;
                }
            }
        }
        const_idx
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
