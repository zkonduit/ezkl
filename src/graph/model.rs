use super::errors::GraphError;
use super::extract_const_quantized_values;
use super::node::*;
use super::scale_to_multiplier;
use super::vars::*;
use super::GraphSettings;
use crate::circuit::hybrid::HybridOp;
use crate::circuit::region::ConstantsMap;
use crate::circuit::region::RegionCtx;
use crate::circuit::table::Range;
use crate::circuit::Input;
use crate::circuit::InputType;
use crate::circuit::Unknown;
use crate::tensor::ValType;
use crate::{
    circuit::{lookup::LookupOp, BaseConfig as PolyConfig, CheckMode, Op},
    tensor::{Tensor, ValTensor},
    RunArgs,
};
use halo2curves::bn256::Fr as Fp;

#[cfg(not(target_arch = "wasm32"))]
use super::input::GraphData;
#[cfg(not(target_arch = "wasm32"))]
use colored::Colorize;
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::ConstraintSystem,
};
use halo2curves::ff::Field;
use itertools::Itertools;
use log::error;
use log::{debug, info, trace};
use serde::Deserialize;
use serde::Serialize;
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use tabled::Table;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::prelude::{
    Framework, Graph, InferenceFact, InferenceModelExt, SymbolValues, TypedFact, TypedOp,
};
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_core::internal::DatumType;
#[cfg(not(target_arch = "wasm32"))]
use tract_onnx::tract_hir::ops::scan::Scan;
use unzip_n::unzip_n;

unzip_n!(pub 3);

#[cfg(not(target_arch = "wasm32"))]
type TractResult = (Graph<TypedFact, Box<dyn TypedOp>>, SymbolValues);
/// The result of a forward pass.
#[derive(Clone, Debug)]
pub struct ForwardResult {
    /// The outputs of the forward pass.
    pub outputs: Vec<Tensor<Fp>>,
    /// The maximum value of any input to a lookup operation.
    pub max_lookup_inputs: i64,
    /// The minimum value of any input to a lookup operation.
    pub min_lookup_inputs: i64,
    /// The max range check size
    pub max_range_size: i64,
}

impl From<DummyPassRes> for ForwardResult {
    fn from(res: DummyPassRes) -> Self {
        Self {
            outputs: res.outputs,
            max_lookup_inputs: res.max_lookup_inputs,
            min_lookup_inputs: res.min_lookup_inputs,
            max_range_size: res.max_range_size,
        }
    }
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
pub struct DummyPassRes {
    /// number of rows use
    pub num_rows: usize,
    /// num dynamic lookups
    pub num_dynamic_lookups: usize,
    /// dynamic lookup col size
    pub dynamic_lookup_col_coord: usize,
    /// num shuffles
    pub num_shuffles: usize,
    /// shuffle
    pub shuffle_col_coord: usize,
    /// linear coordinate
    pub linear_coord: usize,
    /// total const size
    pub total_const_size: usize,
    /// lookup ops
    pub lookup_ops: HashSet<LookupOp>,
    /// range checks
    pub range_checks: HashSet<Range>,
    /// max lookup inputs
    pub max_lookup_inputs: i64,
    /// min lookup inputs
    pub min_lookup_inputs: i64,
    /// min range check
    pub max_range_size: i64,
    /// outputs
    pub outputs: Vec<Tensor<Fp>>,
}

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
        out_scales: Vec<crate::Scale>,
    },
}

impl NodeType {
    ///
    pub fn is_lookup(&self) -> bool {
        match self {
            NodeType::Node(n) => n.opkind.is_lookup(),
            NodeType::SubGraph { .. } => false,
        }
    }
    ///
    pub fn num_uses(&self) -> usize {
        match self {
            NodeType::Node(n) => n.num_uses,
            NodeType::SubGraph { .. } => 0,
        }
    }

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

    /// Returns the scales of the node's output.
    pub fn out_scales(&self) -> Vec<crate::Scale> {
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
    pub fn decrement_use(&mut self) {
        match self {
            NodeType::Node(n) => n.num_uses -= 1,
            NodeType::SubGraph { .. } => log::warn!("Cannot decrement const of subgraph"),
        }
    }

    /// bunp scale of node
    pub fn bump_scale(&mut self, scale: crate::Scale) {
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

    /// Input types
    pub fn get_input_types(&self) -> Result<Vec<InputType>, GraphError> {
        self.inputs
            .iter()
            .map(|o| {
                match self
                    .nodes
                    .get(o)
                    .ok_or(GraphError::MissingNode(*o))?
                    .opkind()
                {
                    SupportedOp::Input(Input { datum_type, .. }) => Ok(datum_type.clone()),
                    _ => Err(GraphError::InvalidInputTypes),
                }
            })
            .collect::<Result<Vec<_>, _>>()
    }

    ///  Returns shapes of the computational graph's inputs
    pub fn input_shapes(&self) -> Result<Vec<Vec<usize>>, GraphError> {
        let mut inputs = vec![];

        for input in self.inputs.iter() {
            let node = self
                .nodes
                .get(input)
                .ok_or(GraphError::MissingNode(*input))?;
            let input_dims = node.out_dims();
            let input_dim = input_dims.first().ok_or(GraphError::MissingNode(*input))?;
            inputs.push(input_dim.clone());
        }

        Ok(inputs)
    }

    /// Returns the number of the computational graph's outputs
    pub fn num_outputs(&self) -> usize {
        let output_nodes = self.outputs.iter();
        output_nodes.len()
    }

    /// Returns shapes of the computational graph's outputs
    pub fn output_shapes(&self) -> Result<Vec<Vec<usize>>, GraphError> {
        let mut outputs = vec![];

        for output in self.outputs.iter() {
            let (idx, outlet) = output;
            let node = self.nodes.get(idx).ok_or(GraphError::MissingNode(*idx))?;
            let out_dims = node.out_dims();
            let out_dim = out_dims
                .get(*outlet)
                .ok_or(GraphError::MissingNode(*outlet))?;
            outputs.push(out_dim.clone());
        }

        Ok(outputs)
    }

    /// Returns the fixed point scale of the computational graph's inputs
    pub fn get_input_scales(&self) -> Vec<crate::Scale> {
        let input_nodes = self.inputs.iter();
        input_nodes
            .flat_map(|idx| {
                self.nodes
                    .get(idx)
                    .ok_or(GraphError::MissingNode(*idx))
                    .map(|n| n.out_scales())
                    .unwrap_or_default()
            })
            .collect()
    }

    /// Returns the fixed point scale of the computational graph's outputs
    pub fn get_output_scales(&self) -> Result<Vec<crate::Scale>, GraphError> {
        let output_nodes = self.outputs.iter();
        output_nodes
            .map(|(idx, outlet)| {
                Ok(self
                    .nodes
                    .get(idx)
                    .ok_or(GraphError::MissingNode(*idx))?
                    .out_scales()[*outlet])
            })
            .collect::<Result<Vec<_>, GraphError>>()
    }
}

impl Model {
    /// Creates a `Model` from a specified path to an Onnx file.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `run_args` - [RunArgs]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(reader: &mut dyn std::io::Read, run_args: &RunArgs) -> Result<Self, GraphError> {
        let visibility = VarVisibility::from_args(run_args)?;

        let graph = Self::load_onnx_model(reader, run_args, &visibility)?;

        let om = Model { graph, visibility };

        debug!("\n {}", om.table_nodes());

        Ok(om)
    }

    ///
    pub fn save(&self, path: PathBuf) -> Result<(), GraphError> {
        let f = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(f);
        bincode::serialize_into(writer, &self)?;
        Ok(())
    }

    ///
    pub fn load(path: PathBuf) -> Result<Self, GraphError> {
        // read bytes from file
        let mut f = std::fs::File::open(&path)?;
        let metadata = fs::metadata(&path)?;
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer)?;
        let result = bincode::deserialize(&buffer)?;
        Ok(result)
    }

    /// Generate model parameters for the circuit
    pub fn gen_params(
        &self,
        run_args: &RunArgs,
        check_mode: CheckMode,
    ) -> Result<GraphSettings, GraphError> {
        let instance_shapes = self.instance_shapes()?;
        #[cfg(not(target_arch = "wasm32"))]
        debug!(
            "{} {} {}",
            "model has".blue(),
            instance_shapes.len().to_string().blue(),
            "instances".blue()
        );

        let inputs: Vec<ValTensor<Fp>> = self
            .graph
            .input_shapes()?
            .iter()
            .map(|shape| {
                let len = shape.iter().product();
                let mut t: ValTensor<Fp> = (0..len)
                    .map(|_| {
                        if !self.visibility.input.is_fixed() {
                            ValType::Value(Value::<Fp>::unknown())
                        } else {
                            ValType::Constant(Fp::random(&mut rand::thread_rng()))
                        }
                    })
                    .collect::<Vec<_>>()
                    .into();

                t.reshape(shape)?;
                Ok(t)
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        let res = self.dummy_layout(run_args, &inputs, false, false)?;

        // if we're using percentage tolerance, we need to add the necessary range check ops for it.

        Ok(GraphSettings {
            run_args: run_args.clone(),
            model_instance_shapes: instance_shapes,
            module_sizes: crate::graph::modules::ModuleSizes::default(),
            num_rows: res.num_rows,
            total_assignments: res.linear_coord,
            required_lookups: res.lookup_ops.into_iter().collect(),
            required_range_checks: res.range_checks.into_iter().collect(),
            model_output_scales: self.graph.get_output_scales()?,
            model_input_scales: self.graph.get_input_scales(),
            num_dynamic_lookups: res.num_dynamic_lookups,
            total_dynamic_col_size: res.dynamic_lookup_col_coord,
            num_shuffles: res.num_shuffles,
            total_shuffle_col_size: res.shuffle_col_coord,
            total_const_size: res.total_const_size,
            check_mode,
            version: env!("CARGO_PKG_VERSION").to_string(),
            num_blinding_factors: None,
            // unix time timestamp
            #[cfg(not(target_arch = "wasm32"))]
            timestamp: Some(
                instant::SystemTime::now()
                    .duration_since(instant::SystemTime::UNIX_EPOCH)?
                    .as_millis(),
            ),
            #[cfg(target_arch = "wasm32")]
            timestamp: None,
        })
    }

    /// Runs a forward pass on sample data !
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `model_inputs` - A vector of [Tensor]s to use as inputs to the model.
    /// * `run_args` - [RunArgs]
    pub fn forward(
        &self,
        model_inputs: &[Tensor<Fp>],
        run_args: &RunArgs,
        witness_gen: bool,
        check_lookup: bool,
    ) -> Result<ForwardResult, GraphError> {
        let valtensor_inputs: Vec<ValTensor<Fp>> = model_inputs
            .iter()
            .map(|x| x.map(|elem| ValType::Value(Value::known(elem))).into())
            .collect();
        let res = self.dummy_layout(run_args, &valtensor_inputs, witness_gen, check_lookup)?;
        Ok(res.into())
    }

    /// Loads an Onnx model from a specified path.
    /// # Arguments
    /// * `reader` - A reader for an Onnx file.
    /// * `scale` - The scale to use for quantization.
    /// * `public_params` - Whether to make the params public.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_onnx_using_tract(
        reader: &mut dyn std::io::Read,
        run_args: &RunArgs,
    ) -> Result<TractResult, GraphError> {
        use tract_onnx::{
            tract_core::internal::IntoArcTensor, tract_hir::internal::GenericFactoid,
        };

        let mut model = tract_onnx::onnx().model_for_read(reader)?;

        let variables: std::collections::HashMap<String, usize> =
            std::collections::HashMap::from_iter(run_args.variables.clone());

        for (i, id) in model.clone().inputs.iter().enumerate() {
            let input = model.node_mut(id.node);
            let mut fact: InferenceFact = input.outputs[0].fact.clone();

            for (i, x) in fact.clone().shape.dims().enumerate() {
                if matches!(x, GenericFactoid::Any) {
                    let batch_size = match variables.get("batch_size") {
                        Some(x) => x,
                        None => return Err(GraphError::MissingBatchSize),
                    };
                    fact.shape
                        .set_dim(i, tract_onnx::prelude::TDim::Val(*batch_size as i64));
                }
            }

            model.set_input_fact(i, fact)?;
        }

        for (i, _) in model.clone().outputs.iter().enumerate() {
            model.set_output_fact(i, InferenceFact::default())?;
        }

        let mut symbol_values = SymbolValues::default();
        for (symbol, value) in run_args.variables.iter() {
            let symbol = model.symbol_table.sym(symbol);
            symbol_values = symbol_values.with(&symbol, *value as i64);
            debug!("set {} to {}", symbol, value);
        }

        // Note: do not optimize the model, as the layout will depend on underlying hardware
        let mut typed_model = model
            .into_typed()?
            .concretize_dims(&symbol_values)?
            .into_decluttered()?;

        // concretize constants
        for node in typed_model.eval_order()? {
            let node = typed_model.node_mut(node);
            if let Some(op) = node.op_as_mut::<tract_onnx::tract_core::ops::konst::Const>() {
                if op.0.datum_type() == DatumType::TDim {
                    // get inner value to Arc<Tensor>
                    let mut constant = op.0.as_ref().clone();
                    // Generally a shape or hyperparam
                    constant
                        .as_slice_mut::<tract_onnx::prelude::TDim>()?
                        .iter_mut()
                        .for_each(|x| *x = x.eval(&symbol_values));

                    op.0 = constant.into_arc_tensor();
                }
            }
        }

        Ok((typed_model, symbol_values))
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
    ) -> Result<ParsedNodes, GraphError> {
        let start_time = instant::Instant::now();

        let (model, symbol_values) = Self::load_onnx_using_tract(reader, run_args)?;

        let scales = VarScales::from_args(run_args);
        let nodes = Self::nodes_from_graph(
            &model,
            run_args,
            &scales,
            visibility,
            &symbol_values,
            None,
            None,
        )?;

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
        symbol_values: &SymbolValues,
        override_input_scales: Option<Vec<crate::Scale>>,
        override_output_scales: Option<HashMap<usize, crate::Scale>>,
    ) -> Result<BTreeMap<usize, NodeType>, GraphError> {
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
                        .map(|i| {
                            Ok(nodes
                                .get(&i.node)
                                .ok_or(GraphError::MissingNode(i.node))?
                                .out_scales()[0])
                        })
                        .collect::<Result<Vec<_>, GraphError>>()?;

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

                    let mut output_mappings = vec![];
                    for (i, mapping) in b.output_mapping.iter().enumerate() {
                        let mut mappings = vec![];
                        if let Some(outlet) = mapping.last_value_slot {
                            mappings.push(OutputMapping::Single {
                                outlet,
                                is_state: mapping.state,
                            });
                        } else if mapping.state {
                            mappings.push(OutputMapping::Single {
                                outlet: i,
                                is_state: mapping.state,
                            });
                        }
                        if let Some(last) = mapping.scan {
                            mappings.push(OutputMapping::Stacked {
                                outlet: last.0,
                                axis: last.1.axis,
                                is_state: false,
                            });
                        }

                        output_mappings.push(mappings);
                    }

                    let output_state_idx = output_state_idx(&output_mappings);

                    let mut output_scale_override = HashMap::new();
                    // if input_state_idx and output_state_idx have mismatched scales we need to rebase the scale of the output node
                    for (input_idx, output_idx) in input_state_idx.iter().zip(output_state_idx) {
                        let input_scale = input_scales[*input_idx];
                        // output mappings is a vec of vec. we need to find the outer index of the output node we want to rebase.
                        let mut traversed_len = 0;
                        for (outer_idx, mappings) in output_mappings.iter().enumerate() {
                            let mapping_len = mappings.len();
                            if traversed_len + mapping_len > output_idx {
                                let output_node_idx = b.body.outputs[outer_idx].node;
                                output_scale_override.insert(output_node_idx, input_scale);
                            }
                            traversed_len += mapping_len;
                        }
                    }

                    let subgraph_nodes = Self::nodes_from_graph(
                        &model,
                        run_args,
                        scales,
                        visibility,
                        symbol_values,
                        Some(input_scales.clone()),
                        Some(output_scale_override),
                    )?;

                    let subgraph = ParsedNodes {
                        nodes: subgraph_nodes,
                        inputs: model.inputs.iter().map(|o| o.node).collect(),
                        outputs: model.outputs.iter().map(|o| (o.node, o.slot)).collect(),
                    };

                    let om = Model {
                        graph: subgraph,
                        visibility: visibility.clone(),
                    };

                    let out_dims = node_output_shapes(n, symbol_values)?;

                    let mut output_scales = BTreeMap::new();

                    for (i, _mapping) in b.output_mapping.iter().enumerate() {
                        for mapping in b.output_mapping.iter() {
                            if let Some(outlet) = mapping.last_value_slot {
                                output_scales.insert(outlet, om.graph.get_output_scales()?[i]);
                            }
                            if let Some(last) = mapping.scan {
                                output_scales.insert(last.0, om.graph.get_output_scales()?[i]);
                            }
                        }
                    }

                    let out_scales = output_scales.into_values().collect_vec();

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
                    let mut n = Node::new(
                        n.clone(),
                        &mut nodes,
                        scales,
                        &run_args.param_visibility,
                        i,
                        symbol_values,
                        run_args.div_rebasing,
                        run_args.rebase_frac_zero_constants,
                    )?;
                    if let Some(ref scales) = override_input_scales {
                        if let Some(inp) = n.opkind.get_input() {
                            let scale = scales[input_idx];
                            n.opkind = SupportedOp::Input(Input {
                                scale,
                                datum_type: inp.datum_type,
                            });
                            input_idx += 1;
                            n.out_scale = scale;
                        }
                    }
                    if let Some(ref scales) = override_output_scales {
                        if scales.contains_key(&i) {
                            let scale_diff = n.out_scale - scales[&i];
                            n.opkind = if scale_diff > 0 {
                                RebaseScale::rebase(
                                    n.opkind,
                                    scales[&i],
                                    n.out_scale,
                                    1,
                                    run_args.div_rebasing,
                                )
                            } else {
                                RebaseScale::rebase_up(
                                    n.opkind,
                                    scales[&i],
                                    n.out_scale,
                                    run_args.div_rebasing,
                                )
                            };
                            n.out_scale = scales[&i];
                        }
                    }
                    nodes.insert(i, NodeType::Node(n));
                }
            }
        }
        Self::remove_unused_nodes(&mut nodes);

        Ok(nodes)
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Removes all nodes that are consts with 0 uses
    fn remove_unused_nodes(nodes: &mut BTreeMap<usize, NodeType>) {
        // remove all nodes that are consts with 0 uses now
        nodes.retain(|_, n| match n {
            NodeType::Node(n) => match &mut n.opkind {
                SupportedOp::Constant(c) => {
                    c.empty_raw_value();
                    n.num_uses > 0
                }
                _ => n.num_uses > 0,
            },
            NodeType::SubGraph { model, .. } => {
                Self::remove_unused_nodes(&mut model.graph.nodes);
                true
            }
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Run tract onnx model on sample data !
    pub fn run_onnx_predictions(
        run_args: &RunArgs,
        model_path: &std::path::Path,
        data_chunks: &[GraphData],
        input_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Vec<Tensor<f32>>>, GraphError> {
        use tract_onnx::tract_core::internal::IntoArcTensor;

        let (model, _) =
            Model::load_onnx_using_tract(&mut std::fs::File::open(model_path)?, run_args)?;

        let datum_types: Vec<DatumType> = model
            .input_outlets()?
            .iter()
            .map(|o| model.node(o.node).outputs[o.slot].fact.datum_type)
            .collect();

        let runnable_model = model.into_runnable()?;
        let mut outputs = vec![];
        for chunk in data_chunks {
            let result = runnable_model.run(chunk.to_tract_data(&input_shapes, &datum_types)?)?;
            outputs.push(
                result
                    .into_iter()
                    .map(|t| {
                        crate::graph::utilities::extract_tensor_value(t.into_arc_tensor()).unwrap()
                    })
                    .collect(),
            );
        }
        Ok(outputs)
    }

    /// Creates a `Model` from parsed run_args
    /// # Arguments
    /// * `params` - A [GraphSettings] struct holding parsed CLI arguments.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_run_args(run_args: &RunArgs, model: &std::path::Path) -> Result<Self, GraphError> {
        Model::new(&mut std::fs::File::open(model)?, run_args)
    }

    /// Configures a model for the circuit
    /// # Arguments
    /// * `meta` - The constraint system.
    /// * `vars` - The variables for the circuit.
    /// * `settings` - [GraphSettings]
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        vars: &ModelVars<Fp>,
        settings: &GraphSettings,
    ) -> Result<PolyConfig<Fp>, GraphError> {
        debug!("configuring model");

        let lookup_range = settings.run_args.lookup_range;
        let logrows = settings.run_args.logrows as usize;
        let required_lookups = settings.required_lookups.clone();
        let required_range_checks = settings.required_range_checks.clone();

        let mut base_gate = PolyConfig::configure(
            meta,
            vars.advices[0..2].try_into()?,
            &vars.advices[2],
            settings.check_mode,
        );
        // set scale for HybridOp::RangeCheck and call self.conf_lookup on that op for percentage tolerance case
        let input = &vars.advices[0];
        let output = &vars.advices[2];
        let index = &vars.advices[1];
        for op in required_lookups {
            base_gate.configure_lookup(meta, input, output, index, lookup_range, logrows, &op)?;
        }

        for range in required_range_checks {
            base_gate.configure_range_check(meta, input, index, range, logrows)?;
        }

        if settings.requires_dynamic_lookup() {
            base_gate.configure_dynamic_lookup(
                meta,
                vars.advices[0..3].try_into()?,
                vars.advices[3..6].try_into()?,
            )?;
        }

        if settings.requires_shuffle() {
            base_gate.configure_shuffles(
                meta,
                vars.advices[0..2].try_into()?,
                vars.advices[3..5].try_into()?,
            )?;
        }

        Ok(base_gate)
    }

    /// Assigns values to the regions created when calling `configure`.
    /// # Arguments
    /// * `config` - [ModelConfig] holding all node configs.
    /// * `layouter` - Halo2 Layouter.
    /// * `inputs` - The values to feed into the circuit.
    /// * `vars` - The variables for the circuit.
    /// * `witnessed_outputs` - The values to compare against.
    /// * `constants` - The constants for the circuit.
    pub fn layout(
        &self,
        mut config: ModelConfig,
        layouter: &mut impl Layouter<Fp>,
        run_args: &RunArgs,
        inputs: &[ValTensor<Fp>],
        vars: &mut ModelVars<Fp>,
        witnessed_outputs: &[ValTensor<Fp>],
        constants: &mut ConstantsMap<Fp>,
    ) -> Result<Vec<ValTensor<Fp>>, GraphError> {
        info!("model layout...");

        let start_time = instant::Instant::now();

        let mut results = BTreeMap::<usize, Vec<ValTensor<Fp>>>::new();

        let input_shapes = self.graph.input_shapes()?;
        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            if self.visibility.input.is_public() {
                let instance = vars
                    .instance
                    .as_ref()
                    .ok_or(GraphError::MissingInstances)?
                    .clone();
                results.insert(*input_idx, vec![instance]);
                vars.increment_instance_idx();
            } else {
                let mut input = inputs[i].clone();
                input.reshape(&input_shapes[i])?;
                results.insert(*input_idx, vec![input]);
            }
        }

        let instance_idx = vars.get_instance_idx();

        config.base.layout_tables(layouter)?;
        config.base.layout_range_checks(layouter)?;

        let original_constants = constants.clone();

        let outputs = layouter.assign_region(
            || "model",
            |region| {
                let mut thread_safe_region = RegionCtx::new_with_constants(
                    region,
                    0,
                    run_args.num_inner_cols,
                    original_constants.clone(),
                );
                // we need to do this as this loop is called multiple times
                vars.set_instance_idx(instance_idx);

                let outputs = self
                    .layout_nodes(&mut config, &mut thread_safe_region, &mut results)
                    .map_err(|e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    })?;

                if run_args.output_visibility.is_public() || run_args.output_visibility.is_fixed() {
                    let output_scales = self.graph.get_output_scales().map_err(|e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    })?;
                    let res = outputs
                        .iter()
                        .enumerate()
                        .map(|(i, output)| {
                            let mut tolerance = run_args.tolerance;
                            tolerance.scale = scale_to_multiplier(output_scales[i]).into();

                            let comparators = if run_args.output_visibility == Visibility::Public {
                                let res = vars
                                    .instance
                                    .as_ref()
                                    .ok_or(GraphError::MissingInstances)?
                                    .clone();
                                vars.increment_instance_idx();
                                res
                            } else {
                                // if witnessed_outputs is of len less than i  error
                                if witnessed_outputs.len() <= i {
                                    return Err(GraphError::InsufficientWitnessValues);
                                }
                                witnessed_outputs[i].clone()
                            };

                            config
                                .base
                                .layout(
                                    &mut thread_safe_region,
                                    &[output.clone(), comparators],
                                    Box::new(HybridOp::RangeCheck(tolerance)),
                                )
                                .map_err(|e| e.into())
                        })
                        .collect::<Result<Vec<_>, GraphError>>();
                    res.map_err(|e| {
                        error!("{}", e);
                        halo2_proofs::plonk::Error::Synthesis
                    })?;
                }
                // Then number of columns in the circuits
                #[cfg(not(target_arch = "wasm32"))]
                thread_safe_region.debug_report();

                *constants = thread_safe_region.assigned_constants().clone();

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
    ) -> Result<Vec<ValTensor<Fp>>, GraphError> {
        // index over results to get original inputs
        let orig_inputs: BTreeMap<usize, _> = results
            .clone()
            .into_iter()
            .filter(|(idx, _)| self.graph.inputs.contains(idx))
            .collect();

        for (idx, node) in self.graph.nodes.iter() {
            debug!("laying out {}: {}", idx, node.as_str(),);
            // Then number of columns in the circuits
            #[cfg(not(target_arch = "wasm32"))]
            region.debug_report();
            debug!("input indices: {:?}", node.inputs());
            debug!("output scales: {:?}", node.out_scales());
            debug!(
                "input scales: {:?}",
                node.inputs()
                    .iter()
                    .map(|(idx, outlet)| self.graph.nodes[idx].out_scales()[*outlet])
                    .collect_vec()
            );

            let mut values: Vec<ValTensor<Fp>> = if !node.is_input() {
                node.inputs()
                    .iter()
                    .map(|(idx, outlet)| {
                        Ok(results.get(idx).ok_or(GraphError::MissingResults)?[*outlet].clone())
                    })
                    .collect::<Result<Vec<_>, GraphError>>()?
            } else {
                // we re-assign inputs, always from the 0 outlet
                vec![results.get(idx).ok_or(GraphError::MissingResults)?[0].clone()]
            };
            debug!("output dims: {:?}", node.out_dims());
            debug!(
                "input dims {:?}",
                values.iter().map(|v| v.dims()).collect_vec()
            );

            match &node {
                NodeType::Node(n) => {
                    let res = if node.is_constant() && node.num_uses() == 1 {
                        log::debug!("node {} is a constant with 1 use", n.idx);
                        let mut node = n.clone();
                        let c = node
                            .opkind
                            .get_mutable_constant()
                            .ok_or(GraphError::MissingConstants)?;
                        Some(c.quantized_values.clone().try_into()?)
                    } else {
                        config
                            .base
                            .layout(region, &values, n.opkind.clone_dyn())
                            .map_err(|e| {
                                error!("{}", e);
                                halo2_proofs::plonk::Error::Synthesis
                            })?
                    };

                    if let Some(mut vt) = res {
                        vt.reshape(&node.out_dims()[0])?;
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
                    let original_values = values.clone();
                    let input_mappings = input_mappings.clone();

                    let input_dims = values.iter().map(|inp| inp.dims());
                    let num_iter = number_of_iterations(&input_mappings, input_dims.collect());

                    debug!(
                        "{} iteration(s) in a subgraph with inputs {:?}, sources {:?}, and outputs {:?}",
                        num_iter, inputs, model.graph.inputs, model.graph.outputs
                    );

                    let mut full_results: Vec<ValTensor<Fp>> = vec![];

                    for i in 0..num_iter {
                        debug!(" -------------- subgraph iteration: {}", i);
                        // replace the Stacked input with the current chunk iter
                        for ((mapping, inp), og_inp) in
                            input_mappings.iter().zip(&mut values).zip(&original_values)
                        {
                            if let InputMapping::Stacked { axis, chunk } = mapping {
                                let start = i * chunk;
                                let end = (i + 1) * chunk;
                                let mut sliced_input = og_inp.clone();
                                sliced_input.slice(axis, &start, &end)?;
                                *inp = sliced_input;
                            }
                        }

                        let mut subgraph_results = BTreeMap::from_iter(
                            model
                                .graph
                                .inputs
                                .clone()
                                .into_iter()
                                .zip(values.clone().into_iter().map(|v| vec![v])),
                        );

                        let res = model.layout_nodes(config, region, &mut subgraph_results)?;

                        let mut outlets = BTreeMap::new();
                        let mut stacked_outlets = BTreeMap::new();

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
                                            stacked_outlets.insert(outlet, stacked_res);
                                        }
                                        outlets.insert(outlet, outlet_res.clone());
                                    }
                                }
                            }
                        }

                        // now extend with stacked elements
                        let mut pre_stacked_outlets = outlets.clone();
                        pre_stacked_outlets.extend(stacked_outlets);

                        let outlets = outlets.into_values().collect_vec();

                        full_results = pre_stacked_outlets.into_values().collect_vec();

                        let output_states = output_state_idx(output_mappings);
                        let input_states = input_state_idx(&input_mappings);

                        assert_eq!(
                            input_states.len(),
                            output_states.len(),
                            "input and output states must be the same length, got {:?} and {:?}",
                            input_mappings,
                            output_mappings
                        );

                        for (input_idx, output_idx) in input_states.iter().zip(output_states) {
                            assert_eq!(
                                values[*input_idx].dims(),
                                outlets[output_idx].dims(),
                                "input and output dims must be the same, got {:?} and {:?}",
                                values[*input_idx].dims(),
                                outlets[output_idx].dims()
                            );
                            values[*input_idx] = outlets[output_idx].clone();
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
            .map(|(idx, outlet)| {
                Ok(results.get(idx).ok_or(GraphError::MissingResults)?[*outlet].clone())
            })
            .collect::<Result<Vec<_>, GraphError>>()?;

        Ok(outputs)
    }

    /// Assigns dummy values to the regions created when calling `configure`.
    /// # Arguments
    /// * `input_shapes` - The shapes of the inputs to the model.
    pub fn dummy_layout(
        &self,
        run_args: &RunArgs,
        inputs: &[ValTensor<Fp>],
        witness_gen: bool,
        check_lookup: bool,
    ) -> Result<DummyPassRes, GraphError> {
        debug!("calculating num of constraints using dummy model layout...");

        let start_time = instant::Instant::now();

        let mut results = BTreeMap::<usize, Vec<ValTensor<Fp>>>::new();

        for (i, input_idx) in self.graph.inputs.iter().enumerate() {
            results.insert(*input_idx, vec![inputs[i].clone()]);
        }

        let mut dummy_config =
            PolyConfig::dummy(run_args.logrows as usize, run_args.num_inner_cols);
        let mut model_config = ModelConfig {
            base: dummy_config.clone(),
            vars: ModelVars::new_dummy(),
        };

        let mut region =
            RegionCtx::new_dummy(0, run_args.num_inner_cols, witness_gen, check_lookup);

        let outputs = self.layout_nodes(&mut model_config, &mut region, &mut results)?;

        if self.visibility.output.is_public() || self.visibility.output.is_fixed() {
            let output_scales = self.graph.get_output_scales()?;
            let res = outputs
                .iter()
                .enumerate()
                .map(|(i, output)| {
                    let mut comparator: ValTensor<Fp> = (0..output.len())
                        .map(|_| {
                            if !self.visibility.output.is_fixed() {
                                ValType::Value(Value::<Fp>::unknown())
                            } else {
                                ValType::Constant(Fp::random(&mut rand::thread_rng()))
                            }
                        })
                        .collect::<Vec<_>>()
                        .into();
                    comparator.reshape(output.dims())?;

                    let mut tolerance = run_args.tolerance;
                    tolerance.scale = scale_to_multiplier(output_scales[i]).into();

                    dummy_config.layout(
                        &mut region,
                        &[output.clone(), comparator],
                        Box::new(HybridOp::RangeCheck(tolerance)),
                    )
                })
                .collect::<Result<Vec<_>, _>>();
            res?;
        } else if !self.visibility.output.is_private() {
            for output in &outputs {
                region.update_constants(output.create_constants_map());
            }
        }

        let duration = start_time.elapsed();
        trace!("dummy model layout took: {:?}", duration);

        // Then number of columns in the circuits
        #[cfg(not(target_arch = "wasm32"))]
        region.debug_report();

        let outputs = outputs
            .iter()
            .map(|x| {
                x.get_felt_evals()
                    .unwrap_or(Tensor::new(Some(&[Fp::ZERO]), &[1]).unwrap())
            })
            .collect();

        let res = DummyPassRes {
            num_rows: region.row(),
            linear_coord: region.linear_coord(),
            total_const_size: region.total_constants(),
            lookup_ops: region.used_lookups(),
            range_checks: region.used_range_checks(),
            max_lookup_inputs: region.max_lookup_inputs(),
            min_lookup_inputs: region.min_lookup_inputs(),
            max_range_size: region.max_range_size(),
            num_dynamic_lookups: region.dynamic_lookup_index(),
            dynamic_lookup_col_coord: region.dynamic_lookup_col_coord(),
            num_shuffles: region.shuffle_index(),
            shuffle_col_coord: region.shuffle_col_coord(),
            outputs,
        };

        Ok(res)
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
                NodeType::Node(n) => {
                    if let SupportedOp::Constant(c) = &n.opkind {
                        let mut op = crate::circuit::Constant::new(
                            c.quantized_values.clone(),
                            c.raw_values.clone(),
                        );
                        op.pre_assign(consts[const_idx].clone());
                        n.opkind = SupportedOp::Constant(op);

                        const_idx += 1;
                    }
                }
                NodeType::SubGraph { model, .. } => {
                    let total_consts = model.replace_consts(&consts[const_idx..]);
                    const_idx += total_consts;
                }
            }
        }
        const_idx
    }

    /// Shapes of the computational graph's public inputs (if any)
    pub fn instance_shapes(&self) -> Result<Vec<Vec<usize>>, GraphError> {
        let mut instance_shapes = vec![];
        if self.visibility.input.is_public() {
            instance_shapes.extend(self.graph.input_shapes()?);
        }
        if self.visibility.output.is_public() {
            instance_shapes.extend(self.graph.output_shapes()?);
        }
        Ok(instance_shapes)
    }
}
