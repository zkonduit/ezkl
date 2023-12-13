/// Representations of a computational graph's inputs.
pub mod input;
/// Crate for defining a computational graph and building a ZK-circuit from it.
pub mod model;
/// Representations of a computational graph's modules.
pub mod modules;
/// Inner elements of a computational graph that represent a single operation / constraints.
pub mod node;
/// Helper functions
pub mod utilities;
/// Representations of a computational graph's variables.
pub mod vars;
#[cfg(not(target_arch = "wasm32"))]
use colored_json::ToColoredJson;
use halo2_proofs::plonk::VerifyingKey;
use halo2_proofs::poly::kzg::commitment::ParamsKZG;
pub use input::DataSource;
use itertools::Itertools;

#[cfg(not(target_arch = "wasm32"))]
use self::input::OnChainSource;
use self::input::{FileSource, GraphData};
use self::modules::{
    GraphModules, ModuleConfigs, ModuleForwardResult, ModuleSettings, ModuleSizes,
};
use crate::circuit::lookup::LookupOp;
use crate::circuit::modules::ModulePlanner;
use crate::circuit::table::{Table, RESERVED_BLINDING_ROWS_PAD};
use crate::circuit::{CheckMode, InputType};
use crate::pfsys::RescaledInstances;
use crate::tensor::{Tensor, ValTensor};
use crate::RunArgs;
use halo2_proofs::{
    circuit::Layouter,
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use halo2curves::bn256::{self, Bn256, Fr as Fp, G1Affine};
use halo2curves::ff::PrimeField;
use log::{debug, error, info, trace, warn};
pub use model::*;
pub use node::*;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::ops::Deref;
use thiserror::Error;
pub use utilities::*;
pub use vars::*;

#[cfg(feature = "python-bindings")]
use crate::pfsys::field_to_vecu64_montgomery;

/// The safety factor for the range of the lookup table.
pub const RANGE_MULTIPLIER: i128 = 2;

/// Max representation of a lookup table input
pub const MAX_LOOKUP_ABS: i128 = 8 * 2_i128.pow(MAX_PUBLIC_SRS);

/// circuit related errors.
#[derive(Debug, Error)]
pub enum GraphError {
    /// The wrong inputs were passed to a lookup node
    #[error("invalid inputs for a lookup node")]
    InvalidLookupInputs,
    /// Shape mismatch in circuit construction
    #[error("invalid dimensions used for node {0} ({1})")]
    InvalidDims(usize, String),
    /// Wrong method was called to configure an op
    #[error("wrong method was called to configure node {0} ({1})")]
    WrongMethod(usize, String),
    /// A requested node is missing in the graph
    #[error("a requested node is missing in the graph: {0}")]
    MissingNode(usize),
    /// The wrong method was called on an operation
    #[error("an unsupported method was called on node {0} ({1})")]
    OpMismatch(usize, String),
    /// This operation is unsupported
    #[error("unsupported operation in graph")]
    UnsupportedOp,
    /// This operation is unsupported
    #[error("unsupported datatype in graph")]
    UnsupportedDataType,
    /// A node has missing parameters
    #[error("a node is missing required params: {0}")]
    MissingParams(String),
    /// A node has missing parameters
    #[error("a node is has misformed params: {0}")]
    MisformedParams(String),
    /// Error in the configuration of the visibility of variables
    #[error("there should be at least one set of public variables")]
    Visibility,
    /// Ezkl only supports divisions by constants
    #[error("ezkl currently only supports division by constants")]
    NonConstantDiv,
    /// Ezkl only supports constant powers
    #[error("ezkl currently only supports constant exponents")]
    NonConstantPower,
    /// Error when attempting to rescale an operation
    #[error("failed to rescale inputs for {0}")]
    RescalingError(String),
    /// Error when attempting to load a model
    #[error("failed to load model")]
    ModelLoad,
    /// Packing exponent is too large
    #[error("largest packing exponent exceeds max. try reducing the scale")]
    PackingExponent,
    /// Invalid Input Types
    #[error("invalid input types")]
    InvalidInputTypes,
    /// Missing results
    #[error("missing results")]
    MissingResults,
}

const ASSUMED_BLINDING_FACTORS: usize = 5;
/// The minimum number of rows in the grid
pub const MIN_LOGROWS: u32 = 6;

/// 26
pub const MAX_PUBLIC_SRS: u32 = bn256::Fr::S - 2;

/// Lookup deg
pub const LOOKUP_DEG: usize = 5;

use std::cell::RefCell;

thread_local!(
    /// This is a global variable that holds the settings for the graph
    /// This is used to pass settings to the layouter and other parts of the circuit without needing to heavily modify the Halo2 API in a new fork
    pub static GLOBAL_SETTINGS: RefCell<Option<GraphSettings>> = RefCell::new(None)
);

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphWitness {
    /// The inputs of the forward pass
    pub inputs: Vec<Vec<Fp>>,
    /// The rescaled inputs of the forward pass
    pub rescaled_inputs: Option<Vec<Vec<f64>>>,
    /// The output of the forward pass
    pub outputs: Vec<Vec<Fp>>,
    /// The rescaled outputs of the forward pass
    pub rescaled_outputs: Option<Vec<Vec<f64>>>,
    /// Any hashes of inputs generated during the forward pass
    pub processed_inputs: Option<ModuleForwardResult>,
    /// Any hashes of params generated during the forward pass
    pub processed_params: Option<ModuleForwardResult>,
    /// Any hashes of outputs generated during the forward pass
    pub processed_outputs: Option<ModuleForwardResult>,
    /// max lookup input
    pub max_lookup_inputs: i128,
    /// max lookup input
    pub min_lookup_inputs: i128,
}

impl GraphWitness {
    ///
    pub fn new(inputs: Vec<Vec<Fp>>, outputs: Vec<Vec<Fp>>) -> Self {
        GraphWitness {
            inputs,
            outputs,
            rescaled_inputs: None,
            rescaled_outputs: None,
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
        }
    }

    ///
    pub fn get_kzg_commitments(&self) -> Vec<G1Affine> {
        let mut commitments = vec![];
        if let Some(processed_inputs) = &self.processed_inputs {
            if let Some(commits) = &processed_inputs.kzg_commit {
                commitments.extend(commits.iter().flatten());
            }
        }
        if let Some(processed_params) = &self.processed_params {
            if let Some(commits) = &processed_params.kzg_commit {
                commitments.extend(commits.iter().flatten());
            }
        }
        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(commits) = &processed_outputs.kzg_commit {
                commitments.extend(commits.iter().flatten());
            }
        }
        commitments
    }

    /// Export the ezkl witness as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }

    /// Load the model input from a file
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path.clone())
            .map_err(|_| format!("failed to load model at {}", path.display()))?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;
        serde_json::from_str(&data).map_err(|e| e.into())
    }

    /// Save the model input to a file
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        serde_json::to_writer(std::fs::File::create(path)?, &self).map_err(|e| e.into())
    }

    ///
    pub fn get_input_tensor(&self) -> Vec<Tensor<Fp>> {
        self.inputs
            .clone()
            .into_iter()
            .map(|i| Tensor::from(i.into_iter()))
            .collect::<Vec<Tensor<Fp>>>()
    }

    ///
    pub fn get_output_tensor(&self) -> Vec<Tensor<Fp>> {
        self.outputs
            .clone()
            .into_iter()
            .map(|i| Tensor::from(i.into_iter()))
            .collect::<Vec<Tensor<Fp>>>()
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for GraphWitness {
    fn to_object(&self, py: Python) -> PyObject {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let dict_inputs = PyDict::new(py);
        let dict_params = PyDict::new(py);
        let dict_outputs = PyDict::new(py);

        let inputs: Vec<Vec<[u64; 4]>> = self
            .inputs
            .iter()
            .map(|x| x.iter().map(field_to_vecu64_montgomery).collect())
            .collect();

        let outputs: Vec<Vec<[u64; 4]>> = self
            .outputs
            .iter()
            .map(|x| x.iter().map(field_to_vecu64_montgomery).collect())
            .collect();

        dict.set_item("inputs", inputs).unwrap();
        dict.set_item("outputs", outputs).unwrap();
        dict.set_item("max_lookup_inputs", self.max_lookup_inputs)
            .unwrap();

        if let Some(processed_inputs) = &self.processed_inputs {
            //poseidon_hash
            if let Some(processed_inputs_poseidon_hash) = &processed_inputs.poseidon_hash {
                insert_poseidon_hash_pydict(dict_inputs, processed_inputs_poseidon_hash).unwrap();
            }
            if let Some(processed_inputs_elgamal) = &processed_inputs.elgamal {
                insert_elgamal_results_pydict(py, dict_inputs, processed_inputs_elgamal).unwrap();
            }
            if let Some(processed_inputs_kzg_commit) = &processed_inputs.kzg_commit {
                insert_kzg_commit_pydict(dict_inputs, processed_inputs_kzg_commit).unwrap();
            }

            dict.set_item("processed_inputs", dict_inputs).unwrap();
        }

        if let Some(processed_params) = &self.processed_params {
            if let Some(processed_params_poseidon_hash) = &processed_params.poseidon_hash {
                insert_poseidon_hash_pydict(dict_params, processed_params_poseidon_hash).unwrap();
            }
            if let Some(processed_params_elgamal) = &processed_params.elgamal {
                insert_elgamal_results_pydict(py, dict_params, processed_params_elgamal).unwrap();
            }
            if let Some(processed_params_kzg_commit) = &processed_params.kzg_commit {
                insert_kzg_commit_pydict(dict_inputs, processed_params_kzg_commit).unwrap();
            }

            dict.set_item("processed_params", dict_params).unwrap();
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(processed_outputs_poseidon_hash) = &processed_outputs.poseidon_hash {
                insert_poseidon_hash_pydict(dict_outputs, processed_outputs_poseidon_hash).unwrap();
            }
            if let Some(processed_outputs_elgamal) = &processed_outputs.elgamal {
                insert_elgamal_results_pydict(py, dict_outputs, processed_outputs_elgamal).unwrap();
            }
            if let Some(processed_outputs_kzg_commit) = &processed_outputs.kzg_commit {
                insert_kzg_commit_pydict(dict_inputs, processed_outputs_kzg_commit).unwrap();
            }

            dict.set_item("processed_outputs", dict_outputs).unwrap();
        }

        dict.to_object(py)
    }
}

#[cfg(feature = "python-bindings")]
fn insert_poseidon_hash_pydict(pydict: &PyDict, poseidon_hash: &Vec<Fp>) -> Result<(), PyErr> {
    let poseidon_hash: Vec<[u64; 4]> = poseidon_hash
        .iter()
        .map(field_to_vecu64_montgomery)
        .collect();
    pydict.set_item("poseidon_hash", poseidon_hash)?;

    Ok(())
}

#[cfg(feature = "python-bindings")]
fn insert_kzg_commit_pydict(pydict: &PyDict, commits: &Vec<Vec<G1Affine>>) -> Result<(), PyErr> {
    use crate::python::PyG1Affine;
    let poseidon_hash: Vec<Vec<PyG1Affine>> = commits
        .iter()
        .map(|c| c.iter().map(|x| PyG1Affine::from(*x)).collect())
        .collect();
    pydict.set_item("kzg_commit", poseidon_hash)?;

    Ok(())
}

#[cfg(feature = "python-bindings")]
use modules::ElGamalResult;
#[cfg(feature = "python-bindings")]
fn insert_elgamal_results_pydict(
    py: Python,
    pydict: &PyDict,
    elgamal_results: &ElGamalResult,
) -> Result<(), PyErr> {
    let results_dict = PyDict::new(py);
    let cipher_text: Vec<Vec<[u64; 4]>> = elgamal_results
        .ciphertexts
        .iter()
        .map(|v| {
            v.iter()
                .map(field_to_vecu64_montgomery)
                .collect::<Vec<[u64; 4]>>()
        })
        .collect::<Vec<Vec<[u64; 4]>>>();
    results_dict.set_item("ciphertexts", cipher_text)?;

    let encrypted_messages: Vec<Vec<[u64; 4]>> = elgamal_results
        .encrypted_messages
        .iter()
        .map(|v| {
            v.iter()
                .map(field_to_vecu64_montgomery)
                .collect::<Vec<[u64; 4]>>()
        })
        .collect::<Vec<Vec<[u64; 4]>>>();
    results_dict.set_item("encrypted_messages", encrypted_messages)?;

    let variables: crate::python::PyElGamalVariables = elgamal_results.variables.clone().into();

    results_dict.set_item("variables", variables)?;

    pydict.set_item("elgamal", results_dict)?;

    Ok(())

    //elgamal
}

/// model parameters
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphSettings {
    /// run args
    pub run_args: RunArgs,
    /// the potential number of rows used by the circuit
    pub num_rows: usize,
    /// total linear coordinate of assignments
    pub total_assignments: usize,
    /// total const size
    pub total_const_size: usize,
    /// the shape of public inputs to the model (in order of appearance)
    pub model_instance_shapes: Vec<Vec<usize>>,
    /// model output scales
    pub model_output_scales: Vec<crate::Scale>,
    /// model input scales
    pub model_input_scales: Vec<crate::Scale>,
    /// the of instance cells used by modules
    pub module_sizes: ModuleSizes,
    /// required_lookups
    pub required_lookups: Vec<LookupOp>,
    /// check mode
    pub check_mode: CheckMode,
    /// ezkl version used
    pub version: String,
    /// num blinding factors
    pub num_blinding_factors: Option<usize>,
    /// unix time timestamp
    pub timestamp: Option<u128>,
}

impl GraphSettings {
    /// calculate the total number of instances
    pub fn total_instances(&self) -> Vec<usize> {
        let mut instances: Vec<usize> = self
            .model_instance_shapes
            .iter()
            .map(|x| x.iter().product())
            .collect();
        instances.extend(self.module_sizes.num_instances());

        instances
    }

    /// save params to file
    pub fn save(&self, path: &std::path::PathBuf) -> Result<(), std::io::Error> {
        let encoded = serde_json::to_string(&self)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(encoded.as_bytes())
    }
    /// load params from file
    pub fn load(path: &std::path::PathBuf) -> Result<Self, std::io::Error> {
        let mut file = std::fs::File::open(path).map_err(|e| {
            error!("failed to open settings file at {}", e);
            e
        })?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;
        let res = serde_json::from_str(&data)?;
        Ok(res)
    }

    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
    /// Parse an ezkl configuration from a json
    pub fn from_json(arg_json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(arg_json)
    }

    fn set_num_blinding_factors(&mut self, num_blinding_factors: usize) {
        self.num_blinding_factors = Some(num_blinding_factors);
    }

    ///
    pub fn available_col_size(&self) -> usize {
        let base = 2u32;
        if let Some(num_blinding_factors) = self.num_blinding_factors {
            base.pow(self.run_args.logrows) as usize - num_blinding_factors - 1
        } else {
            log::error!("num_blinding_factors not set");
            log::warn!("using default available_col_size");
            base.pow(self.run_args.logrows) as usize - ASSUMED_BLINDING_FACTORS - 1
        }
    }

    ///
    pub fn uses_modules(&self) -> bool {
        !self.module_sizes.max_constraints() > 0
    }

    /// if any visibility is encrypted or hashed
    pub fn module_requires_fixed(&self) -> bool {
        self.run_args.input_visibility.is_encrypted()
            || self.run_args.input_visibility.is_hashed()
            || self.run_args.output_visibility.is_encrypted()
            || self.run_args.output_visibility.is_hashed()
            || self.run_args.param_visibility.is_encrypted()
            || self.run_args.param_visibility.is_hashed()
    }

    /// any kzg visibility
    pub fn module_requires_kzg(&self) -> bool {
        self.run_args.input_visibility.is_kzgcommit()
            || self.run_args.output_visibility.is_kzgcommit()
            || self.run_args.param_visibility.is_kzgcommit()
    }
}

/// Configuration for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    model_config: ModelConfig,
    module_configs: ModuleConfigs,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoreCircuit {
    /// The model / graph of computations.
    pub model: Model,
    /// The settings of the model.
    pub settings: GraphSettings,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphCircuit {
    /// Core circuit
    pub core: CoreCircuit,
    /// The witness data for the model.
    pub graph_witness: GraphWitness,
    /// The settings of the model's modules.
    pub module_settings: ModuleSettings,
}

impl GraphCircuit {
    /// Settings for the graph
    pub fn settings(&self) -> &GraphSettings {
        &self.core.settings
    }
    /// Settings for the graph (mutable)
    pub fn settings_mut(&mut self) -> &mut GraphSettings {
        &mut self.core.settings
    }
    /// The model
    pub fn model(&self) -> &Model {
        &self.core.model
    }
    ///
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let f = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(f);
        bincode::serialize_into(writer, &self)?;
        Ok(())
    }

    ///
    pub fn load(path: std::path::PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        // read bytes from file
        let mut f = std::fs::File::open(&path)?;
        let metadata = std::fs::metadata(&path)?;
        let mut buffer = vec![0; metadata.len() as usize];
        f.read_exact(&mut buffer)?;
        let result = bincode::deserialize(&buffer)?;
        Ok(result)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, PartialOrd)]
/// The data source for a test
pub enum TestDataSource {
    /// The data is loaded from a file
    File,
    /// The data is loaded from the chain
    #[default]
    OnChain,
}

impl From<String> for TestDataSource {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "file" => TestDataSource::File,
            "on-chain" => TestDataSource::OnChain,
            _ => {
                error!("invalid data source: {}", value);
                warn!("using default data source: on-chain");
                TestDataSource::default()
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
///
pub struct TestSources {
    ///
    pub input: TestDataSource,
    ///
    pub output: TestDataSource,
}

///
#[derive(Clone, Debug, Default)]
pub struct TestOnChainData {
    /// The path to the test witness
    pub data: std::path::PathBuf,
    /// rpc endpoint
    pub rpc: Option<String>,
    ///
    pub data_sources: TestSources,
}

impl GraphCircuit {
    ///
    pub fn new(
        model: Model,
        run_args: &RunArgs,
    ) -> Result<GraphCircuit, Box<dyn std::error::Error>> {
        // // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Vec<Fp>> = vec![];
        for shape in model.graph.input_shapes()? {
            let t: Vec<Fp> = vec![Fp::zero(); shape.iter().product::<usize>()];
            inputs.push(t);
        }

        // dummy module settings, must load from GraphData after
        let module_settings = ModuleSettings::default();
        let mut settings = model.gen_params(run_args, CheckMode::UNSAFE)?;

        let mut num_params = 0;
        if !model.const_shapes().is_empty() {
            for shape in model.const_shapes() {
                num_params += shape.iter().product::<usize>();
            }
        }

        let sizes = GraphModules::num_constraints_and_instances(
            model.graph.input_shapes()?,
            vec![vec![num_params]],
            model.graph.output_shapes()?,
            VarVisibility::from_args(run_args)?,
        );

        // number of instances used by modules
        settings.module_sizes = sizes.clone();

        // as they occupy independent rows
        settings.num_rows = std::cmp::max(settings.num_rows, sizes.max_constraints());

        let core = CoreCircuit {
            model,
            settings: settings.clone(),
        };

        Ok(GraphCircuit {
            core,
            graph_witness: GraphWitness::new(inputs, vec![]),
            module_settings,
        })
    }

    ///
    pub fn new_from_settings(
        model: Model,
        mut settings: GraphSettings,
        check_mode: CheckMode,
    ) -> Result<GraphCircuit, Box<dyn std::error::Error>> {
        // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Vec<Fp>> = vec![];
        for shape in model.graph.input_shapes()? {
            let t: Vec<Fp> = vec![Fp::zero(); shape.iter().product::<usize>()];
            inputs.push(t);
        }

        // dummy module settings, must load from GraphData after
        let module_settings = ModuleSettings::default();

        settings.check_mode = check_mode;

        let core = CoreCircuit {
            model,
            settings: settings.clone(),
        };

        Ok(GraphCircuit {
            core,
            graph_witness: GraphWitness::new(inputs, vec![]),
            module_settings,
        })
    }

    /// load inputs and outputs for the model
    pub fn load_graph_witness(
        &mut self,
        data: &GraphWitness,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_witness = data.clone();
        // load the module settings
        self.module_settings = ModuleSettings::from(data);

        Ok(())
    }

    /// Prepare the public inputs for the circuit.
    pub fn prepare_public_inputs(
        &self,
        data: &GraphWitness,
    ) -> Result<Vec<Fp>, Box<dyn std::error::Error>> {
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>
        let mut public_inputs: Vec<Fp> = vec![];
        if self.settings().run_args.input_visibility.is_public() {
            public_inputs.extend(self.graph_witness.inputs.clone().into_iter().flatten())
        } else if let Some(processed_inputs) = &data.processed_inputs {
            public_inputs.extend(processed_inputs.get_instances().into_iter().flatten());
        }

        if let Some(processed_params) = &data.processed_params {
            public_inputs.extend(processed_params.get_instances().into_iter().flatten());
        }

        if self.settings().run_args.output_visibility.is_public() {
            public_inputs.extend(self.graph_witness.outputs.clone().into_iter().flatten());
        } else if let Some(processed_outputs) = &data.processed_outputs {
            public_inputs.extend(processed_outputs.get_instances().into_iter().flatten());
        }

        if public_inputs.len() < 11 {
            debug!("public inputs: {:?}", public_inputs);
        } else {
            debug!("public inputs: {:?} ...", &public_inputs[0..10]);
        }

        Ok(public_inputs)
    }

    /// get rescaled public inputs as floating points for the circuit.
    pub fn rescaled_public_inputs(
        &self,
        data: &GraphWitness,
    ) -> Result<RescaledInstances, Box<dyn std::error::Error>> {
        // dequantize the supplied data using the provided scale.
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>

        let mut public_inputs = RescaledInstances::default();
        if self.settings().run_args.input_visibility.is_public() {
            let inputs = self.graph_witness.inputs.clone();
            let scales = self.model().graph.get_input_scales();
            public_inputs.rescaled_inputs = inputs
                .iter()
                .zip(scales)
                .map(|(i, s)| i.iter().map(|x| dequantize(*x, s, 0.)).collect_vec())
                .collect_vec();
        } else if let Some(processed_inputs) = &data.processed_inputs {
            public_inputs.processed_inputs = processed_inputs.get_instances();
        }

        if let Some(processed_params) = &data.processed_params {
            public_inputs.processed_params = processed_params.get_instances();
        }

        if self.settings().run_args.output_visibility.is_public() {
            let outputs = self.graph_witness.outputs.clone();
            let scales = self.model().graph.get_output_scales()?;
            public_inputs.rescaled_outputs = outputs
                .iter()
                .zip(scales)
                .map(|(i, s)| i.iter().map(|x| dequantize(*x, s, 0.)).collect_vec())
                .collect_vec();
        } else if let Some(processed_outputs) = &data.processed_outputs {
            public_inputs.processed_outputs = processed_outputs.get_instances();
        } else if let Some(processed_outputs) = &data.processed_outputs {
            public_inputs.processed_outputs = processed_outputs.get_instances();
        }

        debug!(
            "rescaled and processed public inputs: {}",
            serde_json::to_string(&public_inputs)?.to_colored_json_auto()?
        );

        Ok(public_inputs)
    }

    ///
    #[cfg(target_arch = "wasm32")]
    pub fn load_graph_input(
        &mut self,
        data: &GraphData,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        let shapes = self.model().graph.input_shapes()?;
        let scales = self.model().graph.get_input_scales();
        let input_types = self.model().graph.get_input_types()?;
        self.process_data_source(&data.input_data, shapes, scales, input_types)
    }

    ///
    pub fn load_graph_from_file_exclusively(
        &mut self,
        data: &GraphData,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        let shapes = self.model().graph.input_shapes()?;
        let scales = self.model().graph.get_input_scales();
        let input_types = self.model().graph.get_input_types()?;
        info!("input scales: {:?}", scales);

        match &data.input_data {
            DataSource::File(file_data) => {
                self.load_file_data(file_data, &shapes, scales, input_types)
            }
            _ => Err("Cannot use non-file data source as input for this method.".into()),
        }
    }

    ///
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_graph_input(
        &mut self,
        data: &GraphData,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        let shapes = self.model().graph.input_shapes()?;
        let scales = self.model().graph.get_input_scales();
        let input_types = self.model().graph.get_input_types()?;
        info!("input scales: {:?}", scales);

        self.process_data_source(&data.input_data, shapes, scales, input_types)
            .await
    }

    #[cfg(target_arch = "wasm32")]
    /// Process the data source for the model
    fn process_data_source(
        &mut self,
        data: &DataSource,
        shapes: Vec<Vec<usize>>,
        scales: Vec<crate::Scale>,
        input_types: Vec<InputType>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        match &data {
            DataSource::File(file_data) => {
                self.load_file_data(file_data, &shapes, scales, input_types)
            }
            DataSource::OnChain(_) => {
                Err("Cannot use on-chain data source as input for this method.".into())
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Process the data source for the model
    async fn process_data_source(
        &mut self,
        data: &DataSource,
        shapes: Vec<Vec<usize>>,
        scales: Vec<crate::Scale>,
        input_types: Vec<InputType>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        match &data {
            DataSource::OnChain(source) => {
                let mut per_item_scale = vec![];
                for (i, shape) in shapes.iter().enumerate() {
                    per_item_scale.extend(vec![scales[i]; shape.iter().product::<usize>()]);
                }
                self.load_on_chain_data(source.clone(), &shapes, per_item_scale)
                    .await
            }
            DataSource::File(file_data) => {
                self.load_file_data(file_data, &shapes, scales, input_types)
            }
            DataSource::DB(pg) => {
                let data = pg.fetch_and_format_as_file()?;
                self.load_file_data(&data, &shapes, scales, input_types)
            }
        }
    }

    /// Prepare on chain test data
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_on_chain_data(
        &mut self,
        source: OnChainSource,
        shapes: &Vec<Vec<usize>>,
        scales: Vec<crate::Scale>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        use crate::eth::{evm_quantize, read_on_chain_inputs, setup_eth_backend};
        let (_, client) = setup_eth_backend(Some(&source.rpc), None).await?;
        let inputs = read_on_chain_inputs(client.clone(), client.address(), &source.calls).await?;
        // quantize the supplied data using the provided scale + QuantizeData.sol
        let quantized_evm_inputs = evm_quantize(client, scales, &inputs).await?;
        // on-chain data has already been quantized at this point. Just need to reshape it and push into tensor vector
        let mut inputs: Vec<Tensor<Fp>> = vec![];
        for (input, shape) in [quantized_evm_inputs].iter().zip(shapes) {
            let mut t: Tensor<Fp> = input.iter().cloned().collect();
            t.reshape(shape)?;
            inputs.push(t);
        }

        Ok(inputs)
    }

    ///
    pub fn load_file_data(
        &mut self,
        file_data: &FileSource,
        shapes: &Vec<Vec<usize>>,
        scales: Vec<crate::Scale>,
        input_types: Vec<InputType>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        let mut data: Vec<Tensor<Fp>> = vec![];
        for (((d, shape), scale), input_type) in file_data
            .iter()
            .zip(shapes)
            .zip(scales)
            .zip(input_types.iter())
        {
            let t: Vec<Fp> = d
                .par_iter()
                .map(|x| {
                    let mut x = x.clone();
                    x.as_type(input_type);
                    x.to_field(scale)
                })
                .collect();

            let mut t: Tensor<Fp> = t.into_iter().into();
            t.reshape(shape)?;

            data.push(t);
        }
        Ok(data)
    }

    ///
    pub fn load_witness_file_data(
        &mut self,
        file_data: &[Vec<Fp>],
        shapes: &[Vec<usize>],
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        let mut data: Vec<Tensor<Fp>> = vec![];
        for (d, shape) in file_data.iter().zip(shapes) {
            let mut t: Tensor<Fp> = d.clone().into_iter().into();
            t.reshape(shape)?;
            data.push(t);
        }
        Ok(data)
    }

    fn reserved_blinding_rows() -> f64 {
        (ASSUMED_BLINDING_FACTORS + RESERVED_BLINDING_ROWS_PAD) as f64
    }

    fn calc_safe_range(res: &GraphWitness) -> (i128, i128) {
        (
            RANGE_MULTIPLIER * res.min_lookup_inputs,
            RANGE_MULTIPLIER * res.max_lookup_inputs,
        )
    }

    fn calc_num_cols(safe_range: (i128, i128), max_logrows: u32) -> usize {
        let max_col_size = Table::<Fp>::cal_col_size(
            max_logrows as usize,
            Self::reserved_blinding_rows() as usize,
        );
        Table::<Fp>::num_cols_required(safe_range, max_col_size)
    }

    fn calc_min_logrows(
        &mut self,
        res: &GraphWitness,
        max_logrows: Option<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // load the max logrows
        let max_logrows = max_logrows.unwrap_or(MAX_PUBLIC_SRS);
        let max_logrows = std::cmp::min(max_logrows, MAX_PUBLIC_SRS);
        let mut max_logrows = std::cmp::max(max_logrows, MIN_LOGROWS);

        let reserved_blinding_rows = Self::reserved_blinding_rows();
        // check if has overflowed max lookup input
        if res.max_lookup_inputs > MAX_LOOKUP_ABS / RANGE_MULTIPLIER
            || res.min_lookup_inputs < -MAX_LOOKUP_ABS / RANGE_MULTIPLIER
        {
            let err_string = format!("max lookup input ({}) is too large", res.max_lookup_inputs);
            return Err(err_string.into());
        }

        let safe_range = Self::calc_safe_range(res);
        let mut min_logrows = MIN_LOGROWS;
        // degrade the max logrows until the extended k is small enough
        while min_logrows < max_logrows
            && !self.extended_k_is_small_enough(
                min_logrows,
                Self::calc_num_cols(safe_range, min_logrows),
            )
        {
            min_logrows += 1;
        }

        if !self
            .extended_k_is_small_enough(min_logrows, Self::calc_num_cols(safe_range, min_logrows))
        {
            let err_string = format!(
                "extended k is too large to accomodate the quotient polynomial with logrows {}",
                min_logrows
            );
            return Err(err_string.into());
        }

        // degrade the max logrows until the extended k is small enough
        while max_logrows > min_logrows
            && !self.extended_k_is_small_enough(
                max_logrows,
                Self::calc_num_cols(safe_range, max_logrows),
            )
        {
            max_logrows -= 1;
        }

        let min_bits = ((safe_range.1 - safe_range.0) as f64 + reserved_blinding_rows + 1.)
            .log2()
            .ceil() as usize;

        let min_rows_from_constraints = (self.settings().num_rows as f64 + reserved_blinding_rows)
            .log2()
            .ceil() as usize;

        let mut logrows = std::cmp::max(min_bits, min_rows_from_constraints);

        // if public input then public inputs col will have public inputs len
        if self.settings().run_args.input_visibility.is_public()
            || self.settings().run_args.output_visibility.is_public()
        {
            let mut max_instance_len = self
                .model()
                .instance_shapes()?
                .iter()
                .fold(0, |acc, x| std::cmp::max(acc, x.iter().product::<usize>()))
                as f64
                + reserved_blinding_rows;
            // if there are modules then we need to add the max module size
            if self.settings().uses_modules() {
                max_instance_len += self
                    .settings()
                    .module_sizes
                    .num_instances()
                    .iter()
                    .sum::<usize>() as f64;
            }
            let instance_len_logrows = (max_instance_len).log2().ceil() as usize;
            logrows = std::cmp::max(logrows, instance_len_logrows);
            // this is for fixed const columns
        }

        // ensure logrows is at least 4
        logrows = std::cmp::max(logrows, min_logrows as usize);
        logrows = std::cmp::min(logrows, max_logrows as usize);

        let model = self.model().clone();
        let settings_mut = self.settings_mut();
        settings_mut.run_args.lookup_range = safe_range;
        settings_mut.run_args.logrows = logrows as u32;

        *settings_mut = GraphCircuit::new(model, &settings_mut.run_args)?
            .settings()
            .clone();

        // recalculate the total const size give nthe new logrows
        let total_const_len = settings_mut.total_const_size;
        let const_len_logrows = (total_const_len as f64).log2().ceil() as u32;
        settings_mut.run_args.logrows =
            std::cmp::max(settings_mut.run_args.logrows, const_len_logrows);
        // recalculate the total number of constraints given the new logrows
        let min_rows_from_constraints = (settings_mut.num_rows as f64 + reserved_blinding_rows)
            .log2()
            .ceil() as u32;
        settings_mut.run_args.logrows =
            std::cmp::max(settings_mut.run_args.logrows, min_rows_from_constraints);

        settings_mut.run_args.logrows = std::cmp::min(max_logrows, settings_mut.run_args.logrows);

        info!(
            "setting lookup_range to: {:?}, setting logrows to: {}",
            self.settings().run_args.lookup_range,
            self.settings().run_args.logrows
        );

        Ok(())
    }

    fn extended_k_is_small_enough(&self, k: u32, num_lookup_cols: usize) -> bool {
        let max_degree = self.settings().run_args.num_inner_cols + 2;
        let max_lookup_degree = LOOKUP_DEG + num_lookup_cols - 1; // num_lookup_cols - 1 is the degree of the lookup synthetic selector

        let max_degree = std::cmp::max(max_degree, max_lookup_degree);
        // quotient_poly_degree * params.n - 1 is the degree of the quotient polynomial
        let quotient_poly_degree = (max_degree - 1) as u64;
        // n = 2^k
        let n = 1u64 << k;
        let mut extended_k = k;
        while (1 << extended_k) < (n * quotient_poly_degree) {
            extended_k += 1;
        }
        extended_k <= bn256::Fr::S
    }

    /// Calibrate the circuit to the supplied data.
    pub fn calibrate(
        &mut self,
        input: &[Tensor<Fp>],
        max_logrows: Option<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.forward(&mut input.to_vec(), None, None)?;
        self.calc_min_logrows(&res, max_logrows)
    }

    /// Runs the forward pass of the model / graph of computations and any associated hashing.
    pub fn forward(
        &self,
        inputs: &mut [Tensor<Fp>],
        vk: Option<&VerifyingKey<G1Affine>>,
        srs: Option<&ParamsKZG<Bn256>>,
    ) -> Result<GraphWitness, Box<dyn std::error::Error>> {
        let original_inputs = inputs.to_vec();

        let visibility = VarVisibility::from_args(&self.settings().run_args)?;
        let mut processed_inputs = None;
        let mut processed_params = None;
        let mut processed_outputs = None;

        if visibility.input.requires_processing() {
            let module_outlets = visibility.input.overwrites_inputs();
            if !module_outlets.is_empty() {
                let mut module_inputs = vec![];
                for outlet in &module_outlets {
                    module_inputs.push(inputs[*outlet].clone());
                }
                let res = GraphModules::forward(&module_inputs, visibility.input.clone(), vk, srs)?;
                processed_inputs = Some(res.clone());
                let module_results = res.get_result(visibility.input.clone());

                for (i, outlet) in module_outlets.iter().enumerate() {
                    inputs[*outlet] = Tensor::from(module_results[i].clone().into_iter());
                }
            } else {
                processed_inputs = Some(GraphModules::forward(inputs, visibility.input, vk, srs)?);
            }
        }

        if visibility.params.requires_processing() {
            let params = self.model().get_all_params();
            if !params.is_empty() {
                let flattened_params = Tensor::new(Some(&params), &[params.len()])?.combine()?;
                processed_params = Some(GraphModules::forward(
                    &[flattened_params],
                    visibility.params,
                    vk,
                    srs,
                )?);
            }
        }

        let mut model_results = self.model().forward(inputs)?;

        if visibility.output.requires_processing() {
            let module_outlets = visibility.output.overwrites_inputs();
            if !module_outlets.is_empty() {
                let mut module_inputs = vec![];
                for outlet in &module_outlets {
                    module_inputs.push(model_results.outputs[*outlet].clone());
                }
                let res =
                    GraphModules::forward(&module_inputs, visibility.output.clone(), vk, srs)?;
                processed_outputs = Some(res.clone());
                let module_results = res.get_result(visibility.output.clone());

                for (i, outlet) in module_outlets.iter().enumerate() {
                    model_results.outputs[*outlet] =
                        Tensor::from(module_results[i].clone().into_iter());
                }
            } else {
                processed_outputs = Some(GraphModules::forward(
                    &model_results.outputs,
                    visibility.output,
                    vk,
                    srs,
                )?);
            }
        }

        let rescaled_inputs = original_inputs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let scale = self.model().graph.get_input_scales()[i];
                t.map(|x| dequantize(x, scale, 0.)).to_vec()
            })
            .collect();

        let rescaled_outputs: Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> = model_results
            .outputs
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let scale = self.model().graph.get_output_scales()?[i];
                Ok(t.map(|x| dequantize(x, scale, 0.)).to_vec())
            })
            .collect();

        let witness = GraphWitness {
            inputs: original_inputs
                .iter()
                .map(|t| t.deref().to_vec())
                .collect_vec(),
            rescaled_inputs: Some(rescaled_inputs),
            rescaled_outputs: Some(rescaled_outputs?),
            outputs: model_results
                .outputs
                .iter()
                .map(|t| t.deref().to_vec())
                .collect_vec(),
            processed_inputs,
            processed_params,
            processed_outputs,
            max_lookup_inputs: model_results.max_lookup_inputs,
            min_lookup_inputs: model_results.min_lookup_inputs,
        };

        #[cfg(not(target_arch = "wasm32"))]
        log::trace!(
            "witness: \n {}",
            &witness.as_json()?.to_colored_json_auto()?
        );

        Ok(witness)
    }

    /// Create a new circuit from a set of input data and [RunArgs].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_run_args(
        run_args: &RunArgs,
        model_path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Model::from_run_args(run_args, model_path)?;
        Self::new(model, run_args)
    }

    /// Create a new circuit from a set of input data and [GraphSettings].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_settings(
        params: &GraphSettings,
        model_path: &std::path::Path,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        params.run_args.validate()?;
        let model = Model::from_run_args(&params.run_args, model_path)?;
        Self::new_from_settings(model, params.clone(), check_mode)
    }

    ///
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn populate_on_chain_test_data(
        &mut self,
        data: &mut GraphData,
        test_on_chain_data: TestOnChainData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Set up local anvil instance for reading on-chain data

        let input_scales = self.model().graph.get_input_scales();
        let output_scales = self.model().graph.get_output_scales()?;
        let input_shapes = self.model().graph.input_shapes()?;
        let output_shapes = self.model().graph.output_shapes()?;

        if matches!(
            test_on_chain_data.data_sources.input,
            TestDataSource::OnChain
        ) {
            // if not public then fail
            if self.settings().run_args.input_visibility.is_private() {
                return Err("Cannot use on-chain data source as private data".into());
            }

            let input_data = match &data.input_data {
                DataSource::File(input_data) => input_data,
                _ => {
                    return Err("Cannot use non file source as input for on-chain test.
                    Manually populate on-chain data from file source instead"
                        .into())
                }
            };
            // Get the flatten length of input_data
            // if the input source is a field then set scale to 0

            let datam: (Vec<Tensor<Fp>>, OnChainSource) = OnChainSource::test_from_file_data(
                input_data,
                input_scales,
                input_shapes,
                test_on_chain_data.rpc.as_deref(),
            )
            .await?;
            data.input_data = datam.1.into();
        }
        if matches!(
            test_on_chain_data.data_sources.output,
            TestDataSource::OnChain
        ) {
            // if not public then fail
            if self.settings().run_args.output_visibility.is_private() {
                return Err("Cannot use on-chain data source as private data".into());
            }

            let output_data = match &data.output_data {
                Some(DataSource::File(output_data)) => output_data,
                Some(DataSource::OnChain(_)) => {
                    return Err(
                        "Cannot use on-chain data source as output for on-chain test. 
                    Will manually populate on-chain data from file source instead"
                            .into(),
                    )
                }
                _ => return Err("No output data found".into()),
            };
            let datum: (Vec<Tensor<Fp>>, OnChainSource) = OnChainSource::test_from_file_data(
                output_data,
                output_scales,
                output_shapes,
                test_on_chain_data.rpc.as_deref(),
            )
            .await?;
            data.output_data = Some(datum.1.into());
        }
        // Save the updated GraphData struct to the data_path
        data.save(test_on_chain_data.data)?;
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CircuitSize {
    num_instances: usize,
    num_advice_columns: usize,
    num_fixed: usize,
    num_challenges: usize,
    num_selectors: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl CircuitSize {
    pub fn from_cs(cs: &ConstraintSystem<Fp>) -> Self {
        CircuitSize {
            num_instances: cs.num_instance_columns(),
            num_advice_columns: cs.num_advice_columns(),
            num_fixed: cs.num_fixed_columns(),
            num_challenges: cs.num_challenges(),
            num_selectors: cs.num_selectors(),
        }
    }

    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(Box::new(e));
            }
        };
        Ok(serialized)
    }
}

impl Circuit<Fp> for GraphCircuit {
    type Config = GraphConfig;
    type FloorPlanner = ModulePlanner;
    type Params = GraphSettings;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn params(&self) -> Self::Params {
        // safe to clone because the model is Arc'd
        self.settings().clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<Fp>, params: Self::Params) -> Self::Config {
        let mut params = params.clone();
        params.set_num_blinding_factors(cs.blinding_factors());
        GLOBAL_SETTINGS.with(|settings| {
            *settings.borrow_mut() = Some(params.clone());
        });
        let visibility = match VarVisibility::from_args(&params.run_args) {
            Ok(v) => v,
            Err(e) => {
                log::error!("failed to create visibility: {:?}", e);
                log::warn!("using default visibility");
                VarVisibility::default()
            }
        };

        let mut module_configs = ModuleConfigs::from_visibility(
            cs,
            params.module_sizes.clone(),
            params.run_args.logrows as usize,
        );

        let mut vars = ModelVars::new(
            cs,
            params.run_args.logrows as usize,
            params.total_assignments,
            params.run_args.num_inner_cols,
            params.total_const_size,
            params.module_requires_fixed(),
        );

        module_configs.configure_complex_modules(cs, visibility, params.module_sizes.clone());

        vars.instantiate_instance(
            cs,
            params.model_instance_shapes,
            params.run_args.input_scale,
            module_configs.instance,
        );

        let base = Model::configure(
            cs,
            &vars,
            params.run_args.lookup_range,
            params.run_args.logrows as usize,
            params.required_lookups,
            params.check_mode,
        )
        .unwrap();

        let model_config = ModelConfig { base, vars };

        debug!(
            "degree: {}, log2_ceil of degrees: {:?}",
            cs.degree(),
            (cs.degree() as f32).log2().ceil()
        );

        #[cfg(not(target_arch = "wasm32"))]
        info!(
            "circuit size: \n {}",
            CircuitSize::from_cs(cs)
                .as_json()
                .unwrap()
                .to_colored_json_auto()
                .unwrap()
        );

        GraphConfig {
            model_config,
            module_configs,
        }
    }

    fn configure(_: &mut ConstraintSystem<Fp>) -> Self::Config {
        unimplemented!("you should call configure_with_params instead")
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), PlonkError> {
        trace!("Setting input in synthesize");
        let input_vis = &self.settings().run_args.input_visibility;
        let output_vis = &self.settings().run_args.output_visibility;
        let mut graph_modules = GraphModules::new();

        let mut config = config.clone();

        let mut inputs = self
            .graph_witness
            .get_input_tensor()
            .iter_mut()
            .map(|i| {
                i.set_visibility(input_vis);
                ValTensor::try_from(i.clone()).map_err(|e| {
                    log::error!("failed to convert input to valtensor: {:?}", e);
                    PlonkError::Synthesis
                })
            })
            .collect::<Result<Vec<ValTensor<Fp>>, PlonkError>>()?;

        let outputs = self
            .graph_witness
            .get_output_tensor()
            .iter_mut()
            .map(|i| {
                i.set_visibility(output_vis);
                ValTensor::try_from(i.clone()).map_err(|e| {
                    log::error!("failed to convert output to valtensor: {:?}", e);
                    PlonkError::Synthesis
                })
            })
            .collect::<Result<Vec<ValTensor<Fp>>, PlonkError>>()?;

        let mut instance_offset = 0;
        trace!("running input module layout");

        let input_visibility = &self.settings().run_args.input_visibility;
        let outlets = input_visibility.overwrites_inputs();

        if !outlets.is_empty() {
            let mut input_outlets = vec![];
            for outlet in &outlets {
                input_outlets.push(inputs[*outlet].clone());
            }
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut input_outlets,
                input_visibility,
                &mut instance_offset,
                &self.module_settings.input,
            )?;
            // replace inputs with the outlets
            for (i, outlet) in outlets.iter().enumerate() {
                inputs[*outlet] = input_outlets[i].clone();
            }
        } else {
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut inputs,
                input_visibility,
                &mut instance_offset,
                &self.module_settings.input,
            )?;
        }

        // now we need to assign the flattened params to the model
        let mut model = self.model().clone();
        let param_visibility = &self.settings().run_args.param_visibility;
        trace!("running params module layout");
        if !self.model().get_all_params().is_empty() && param_visibility.requires_processing() {
            // now we need to flatten the params
            let consts = self.model().get_all_params();

            let mut flattened_params = {
                let mut t = Tensor::new(Some(&consts), &[consts.len()])
                    .map_err(|_| {
                        log::error!("failed to flatten params");
                        PlonkError::Synthesis
                    })?
                    .combine()
                    .map_err(|_| {
                        log::error!("failed to combine params");
                        PlonkError::Synthesis
                    })?;
                t.set_visibility(param_visibility);
                vec![t.try_into().map_err(|_| {
                    log::error!("failed to convert params to valtensor");
                    PlonkError::Synthesis
                })?]
            };

            // now do stuff to the model params
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut flattened_params,
                param_visibility,
                &mut instance_offset,
                &self.module_settings.params,
            )?;

            let shapes = self.model().const_shapes();
            trace!("replacing processed consts");
            let split_params = split_valtensor(&flattened_params[0], shapes).map_err(|_| {
                log::error!("failed to split params");
                PlonkError::Synthesis
            })?;

            // now the flattened_params have been assigned to and we-assign them to the model consts such that they are constrained to be equal
            model.replace_consts(&split_params);
        }

        // create a new module for the model (space 2)
        layouter.assign_region(|| "_enter_module_2", |_| Ok(()))?;
        trace!("laying out model");

        let mut vars = config.model_config.vars.clone();
        vars.set_initial_instance_offset(instance_offset);

        let mut outputs = model
            .layout(
                config.model_config.clone(),
                &mut layouter,
                &self.settings().run_args,
                &inputs,
                &mut vars,
                &outputs,
            )
            .map_err(|e| {
                log::error!("{}", e);
                PlonkError::Synthesis
            })?;
        trace!("running output module layout");

        let output_visibility = &self.settings().run_args.output_visibility;
        let outlets = output_visibility.overwrites_inputs();

        instance_offset += vars.get_instance_len();

        if !outlets.is_empty() {
            let mut output_outlets = vec![];
            for outlet in &outlets {
                output_outlets.push(outputs[*outlet].clone());
            }
            // this will re-enter module 0
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut output_outlets,
                &self.settings().run_args.output_visibility,
                &mut instance_offset,
                &self.module_settings.output,
            )?;

            // replace outputs with the outlets
            for (i, outlet) in outlets.iter().enumerate() {
                outputs[*outlet] = output_outlets[i].clone();
            }
        } else {
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut outputs,
                &self.settings().run_args.output_visibility,
                &mut instance_offset,
                &self.module_settings.output,
            )?;
        }

        Ok(())
    }
}
