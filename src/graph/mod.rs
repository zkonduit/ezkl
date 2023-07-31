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

use halo2_proofs::circuit::Value;
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
use crate::circuit::CheckMode;
use crate::commands::RunArgs;
use crate::graph::modules::ModuleInstanceOffset;
use crate::pfsys::field_to_vecu64;
use crate::tensor::{Tensor, ValTensor};
use halo2_proofs::{
    circuit::Layouter,
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use halo2curves::bn256::{self, Fr as Fp};
use halo2curves::ff::PrimeField;
use log::{error, info, trace};
pub use model::*;
pub use node::*;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::{Read, Write};
use std::ops::Deref;
use thiserror::Error;
pub use utilities::*;
pub use vars::*;

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
}

/// Inner elements of witness coming from a witness
#[derive(Clone, Debug, Default)]
pub struct WitnessFileSource(Vec<Vec<Fp>>);

impl From<Vec<Vec<Fp>>> for WitnessFileSource {
    fn from(value: Vec<Vec<Fp>>) -> Self {
        WitnessFileSource(value)
    }
}

// !!! ALWAYS USE JSON SERIALIZATION FOR GRAPH INPUT
// UNTAGGED ENUMS WONT WORK :( as highlighted here:
impl<'de> Deserialize<'de> for WitnessFileSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        let t: Vec<Vec<[u64; 4]>> = serde_json::from_str(this_json.get())
            .map_err(|_| serde::de::Error::custom("failed to deserialize WitnessSource"))?;

        let t: Vec<Vec<Fp>> = t
            .iter()
            .map(|x| x.iter().map(|fp| Fp::from_raw(*fp)).collect())
            .collect();
        Ok(WitnessFileSource(t))
    }
}

impl Serialize for WitnessFileSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let field_elems: Vec<Vec<[u64; 4]>> = self
            .0
            .iter()
            .map(|x| x.iter().map(field_to_vecu64).collect())
            .collect::<Vec<_>>();
        field_elems.serialize(serializer)
    }
}

const ASSUMED_BLINDING_FACTORS: usize = 6;

/// 26
const MAX_PUBLIC_SRS: u32 = bn256::Fr::S - 2;

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphWitness {
    /// The inputs of the forward pass
    pub inputs: WitnessFileSource,
    /// The output of the forward pass
    pub outputs: WitnessFileSource,
    /// Any hashes of inputs generated during the forward pass
    pub processed_inputs: Option<ModuleForwardResult>,
    /// Any hashes of params generated during the forward pass
    pub processed_params: Option<ModuleForwardResult>,
    /// Any hashes of outputs generated during the forward pass
    pub processed_outputs: Option<ModuleForwardResult>,
    /// max lookup input
    pub max_lookup_inputs: i128,
}

impl GraphWitness {
    ///
    pub fn new(inputs: Vec<Vec<Fp>>, outputs: Vec<Vec<Fp>>) -> Self {
        GraphWitness {
            inputs: inputs.into(),
            outputs: outputs.into(),
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
            max_lookup_inputs: 0,
        }
    }
    /// Load the model input from a file
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(path)?;
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
            .0
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
            .0
            .iter()
            .map(|x| x.iter().map(field_to_vecu64).collect())
            .collect();

        let outputs: Vec<Vec<[u64; 4]>> = self
            .outputs
            .0
            .iter()
            .map(|x| x.iter().map(field_to_vecu64).collect())
            .collect();

        dict.set_item("inputs", &inputs).unwrap();
        dict.set_item("outputs", &outputs).unwrap();
        dict.set_item("max_lookup_inputs", &self.max_lookup_inputs)
            .unwrap();

        if let Some(processed_inputs) = &self.processed_inputs {
            //poseidon_hash
            if let Some(processed_inputs_poseidon_hash) = &processed_inputs.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_inputs, &processed_inputs_poseidon_hash.0[0]);
            }
            if let Some(processed_inputs_elgamal) = &processed_inputs.elgamal {
                insert_elgamal_results_pydict(py, dict_inputs, processed_inputs_elgamal);
            }

            dict.set_item("processed_inputs", dict_inputs).unwrap();
        }

        if let Some(processed_params) = &self.processed_params {
            if let Some(processed_params_poseidon_hash) = &processed_params.poseidon_hash {
                insert_poseidon_hash_pydict(dict_params, &processed_params_poseidon_hash.0[0]);
            }
            if let Some(processed_params_elgamal) = &processed_params.elgamal {
                insert_elgamal_results_pydict(py, dict_params, processed_params_elgamal);
            }

            dict.set_item("processed_params", dict_params).unwrap();
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(processed_outputs_poseidon_hash) = &processed_outputs.poseidon_hash {
                insert_poseidon_hash_pydict(dict_outputs, &processed_outputs_poseidon_hash.0[0]);
            }
            if let Some(processed_outputs_elgamal) = &processed_outputs.elgamal {
                insert_elgamal_results_pydict(py, dict_outputs, processed_outputs_elgamal);
            }

            dict.set_item("processed_outputs", dict_outputs).unwrap();
        }

        dict.to_object(py)
    }
}

#[cfg(feature = "python-bindings")]
fn insert_poseidon_hash_pydict(pydict: &PyDict, poseidon_hash: &Vec<Fp>) {
    let poseidon_hash: Vec<[u64; 4]> = poseidon_hash.iter().map(field_to_vecu64).collect();
    pydict.set_item("poseidon_hash", poseidon_hash).unwrap();
}

#[cfg(feature = "python-bindings")]
use halo2curves::bn256::G1Affine;
#[cfg(feature = "python-bindings")]
fn g1affine_to_pydict(g1affine_dict: &PyDict, g1affine: &G1Affine) {
    let g1affine_x = field_to_vecu64(&g1affine.x);
    let g1affine_y = field_to_vecu64(&g1affine.y);
    g1affine_dict.set_item("x", g1affine_x).unwrap();
    g1affine_dict.set_item("y", g1affine_y).unwrap();
}

#[cfg(feature = "python-bindings")]
use modules::ElGamalResult;
#[cfg(feature = "python-bindings")]
fn insert_elgamal_results_pydict(py: Python, pydict: &PyDict, elgamal_results: &ElGamalResult) {
    let results_dict = PyDict::new(py);
    let cipher_text: Vec<Vec<[u64; 4]>> = elgamal_results
        .ciphertexts
        .0
        .iter()
        .map(|v| v.iter().map(field_to_vecu64).collect::<Vec<[u64; 4]>>())
        .collect::<Vec<Vec<[u64; 4]>>>();
    results_dict.set_item("ciphertexts", cipher_text).unwrap();

    let encrypted_messages: Vec<Vec<[u64; 4]>> = elgamal_results
        .encrypted_messages
        .0
        .iter()
        .map(|v| v.iter().map(field_to_vecu64).collect::<Vec<[u64; 4]>>())
        .collect::<Vec<Vec<[u64; 4]>>>();
    results_dict
        .set_item("encrypted_messages", encrypted_messages)
        .unwrap();

    let variables_dict = PyDict::new(py);
    let variables = &elgamal_results.variables;

    let r = field_to_vecu64(&variables.r);
    variables_dict.set_item("r", r).unwrap();
    // elgamal secret key
    let sk = field_to_vecu64(&variables.sk);
    variables_dict.set_item("sk", sk).unwrap();

    let pk_dict = PyDict::new(py);
    // elgamal public key
    g1affine_to_pydict(pk_dict, &variables.pk);
    variables_dict.set_item("pk", pk_dict).unwrap();

    let aux_generator_dict = PyDict::new(py);
    // elgamal aux generator used in ecc chip
    g1affine_to_pydict(aux_generator_dict, &variables.aux_generator);
    variables_dict
        .set_item("aux_generator", aux_generator_dict)
        .unwrap();

    // elgamal window size used in ecc chip
    variables_dict
        .set_item("window_size", variables.window_size)
        .unwrap();

    results_dict.set_item("variables", variables_dict).unwrap();

    pydict.set_item("elgamal", results_dict).unwrap();

    //elgamal
}

/// model parameters
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphSettings {
    /// run args
    pub run_args: RunArgs,
    /// the potential number of constraints in the circuit
    pub num_constraints: usize,
    /// the shape of public inputs to the model (in order of appearance)
    pub model_instance_shapes: Vec<Vec<usize>>,
    /// model output scales
    pub model_output_scales: Vec<u32>,
    /// the of instance cells used by modules
    pub module_sizes: ModuleSizes,
    /// required_lookups
    pub required_lookups: Vec<LookupOp>,
    /// check mode
    pub check_mode: CheckMode,
    /// ezkl version used
    pub version: String,
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
        let mut file = std::fs::File::open(path)?;
        let mut data = String::new();
        file.read_to_string(&mut data)?;
        let res = serde_json::from_str(&data)?;
        Ok(res)
    }
}

/// Configuration for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    model_config: ModelConfig,
    module_configs: ModuleConfigs,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GraphCircuit {
    /// The model / graph of computations.
    pub model: Model,
    /// Vector of input tensors to the model / graph of computations.
    pub graph_witness: GraphWitness,
    /// The settings of the model / graph of computations.
    pub settings: GraphSettings,
    /// The settings of the model's modules.
    pub module_settings: ModuleSettings,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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
            _ => panic!("not a valid test data source"),
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
        run_args: RunArgs,
        check_mode: CheckMode,
    ) -> Result<GraphCircuit, Box<dyn std::error::Error>> {
        // // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Vec<Fp>> = vec![];
        for shape in model.graph.input_shapes() {
            let t: Vec<Fp> = vec![Fp::zero(); shape.iter().product::<usize>()];
            inputs.push(t);
        }

        // dummy module settings, must load from GraphData after
        let module_settings = ModuleSettings::default();

        let mut settings = model.gen_params(run_args, check_mode)?;

        let mut num_params = 0;
        if !model.const_shapes().is_empty() {
            for shape in model.const_shapes() {
                num_params += shape.iter().product::<usize>();
            }
        }

        let sizes = GraphModules::num_constraints_and_instances(
            model.graph.input_shapes(),
            vec![vec![num_params]],
            model.graph.output_shapes(),
            VarVisibility::from_args(run_args).unwrap(),
        );

        // number of instances used by modules
        settings.module_sizes = sizes.clone();

        // as they occupy independent rows
        settings.num_constraints = std::cmp::max(settings.num_constraints, sizes.max_constraints());

        Ok(GraphCircuit {
            model,
            graph_witness: GraphWitness::new(inputs, vec![]),
            settings,
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
        for shape in model.graph.input_shapes() {
            let t: Vec<Fp> = vec![Fp::zero(); shape.iter().product::<usize>()];
            inputs.push(t);
        }

        // dummy module settings, must load from GraphData after
        let module_settings = ModuleSettings::default();

        settings.check_mode = check_mode;

        Ok(GraphCircuit {
            model,
            graph_witness: GraphWitness::new(inputs, vec![]),
            settings,
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
        &mut self,
        data: &GraphWitness,
    ) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>
        let mut public_inputs = vec![];
        if self.settings.run_args.input_visibility.is_public() {
            public_inputs = self.graph_witness.inputs.0.clone();
        }
        if self.settings.run_args.output_visibility.is_public() {
            public_inputs.extend(self.graph_witness.outputs.0.clone());
        }
        info!(
            "public inputs lengths: {:?}",
            public_inputs
                .iter()
                .map(|i| i.len())
                .collect::<Vec<usize>>()
        );
        trace!("{:?}", public_inputs);

        let mut pi_inner: Vec<Vec<Fp>> = public_inputs
            .iter()
            .map(|i| i.clone().into_iter().collect::<Vec<Fp>>())
            .collect::<Vec<Vec<Fp>>>();

        let module_instances =
            GraphModules::public_inputs(data, VarVisibility::from_args(self.settings.run_args)?);

        if !module_instances.is_empty() {
            pi_inner.extend(module_instances);
        }

        Ok(pi_inner)
    }

    ///
    #[cfg(target_arch = "wasm32")]
    pub fn load_graph_input(
        &mut self,
        data: &GraphData,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        let shapes = self.model.graph.input_shapes();
        let scales = vec![self.settings.run_args.scale; shapes.len()];
        self.process_data_source(&data.input_data, shapes, scales)
    }

    ///
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_graph_input(
        &mut self,
        data: &GraphData,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        let shapes = self.model.graph.input_shapes();
        let scales = vec![self.settings.run_args.scale; shapes.len()];
        self.process_data_source(&data.input_data, shapes, scales)
            .await
    }

    #[cfg(target_arch = "wasm32")]
    /// Process the data source for the model
    fn process_data_source(
        &mut self,
        data: &DataSource,
        shapes: Vec<Vec<usize>>,
        scales: Vec<u32>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        match &data {
            DataSource::OnChain(_) => {
                panic!("Cannot use on-chain data source as input for wasm rn.")
            }
            DataSource::File(file_data) => self.load_file_data(file_data, &shapes, scales),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Process the data source for the model
    async fn process_data_source(
        &mut self,
        data: &DataSource,
        shapes: Vec<Vec<usize>>,
        scales: Vec<u32>,
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
            DataSource::File(file_data) => self.load_file_data(file_data, &shapes, scales),
        }
    }

    /// Prepare on chain test data
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn load_on_chain_data(
        &mut self,
        source: OnChainSource,
        shapes: &Vec<Vec<usize>>,
        scales: Vec<u32>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        use crate::eth::{evm_quantize, read_on_chain_inputs, setup_eth_backend};
        let (_, client) = setup_eth_backend(Some(&source.rpc)).await?;
        let inputs = read_on_chain_inputs(client.clone(), client.address(), &source.calls).await?;
        // quantize the supplied data using the provided scale + QuantizeData.sol
        let quantized_evm_inputs = evm_quantize(
            client,
            scales.into_iter().map(scale_to_multiplier).collect(),
            &inputs,
        )
        .await?;
        // on-chain data has already been quantized at this point. Just need to reshape it and push into tensor vector
        let mut inputs: Vec<Tensor<Fp>> = vec![];
        for (input, shape) in vec![quantized_evm_inputs].iter().zip(shapes) {
            let mut t: Tensor<Fp> = input.iter().cloned().collect();
            t.reshape(shape);
            inputs.push(t);
        }

        Ok(inputs)
    }

    ///
    pub fn load_file_data(
        &mut self,
        file_data: &FileSource,
        shapes: &Vec<Vec<usize>>,
        scales: Vec<u32>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        let mut data: Vec<Tensor<Fp>> = vec![];
        for ((d, shape), scale) in file_data.iter().zip(shapes).zip(scales) {
            let t: Vec<Fp> = d.par_iter().map(|x| x.to_field(scale)).collect();

            let mut t: Tensor<Fp> = t.into_iter().into();
            t.reshape(shape);

            data.push(t);
        }
        Ok(data)
    }

    ///
    pub fn load_witness_file_data(
        &mut self,
        file_data: &WitnessFileSource,
        shapes: &Vec<Vec<usize>>,
    ) -> Result<Vec<Tensor<Fp>>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        let mut data: Vec<Tensor<Fp>> = vec![];
        for (d, shape) in file_data.0.iter().zip(shapes) {
            let mut t: Tensor<Fp> = d.clone().into_iter().into();
            t.reshape(shape);
            data.push(t);
        }
        Ok(data)
    }

    /// Calibrate the circuit to the supplied data.
    pub fn calibrate(&mut self, input: &[Tensor<Fp>]) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.forward(input)?;

        let max_range = 2i128.pow(self.settings.run_args.bits as u32 - 1);
        if res.max_lookup_inputs > max_range {
            let recommended_bits = (res.max_lookup_inputs as f64).log2().ceil() as usize + 1;

            if recommended_bits <= (MAX_PUBLIC_SRS - 1) as usize {
                self.settings.run_args.bits = recommended_bits;
                self.settings.run_args.logrows = (recommended_bits + 1) as u32;
                return self.calibrate(input);
            } else {
                let err_string = format!("No possible value of bits (estimate {}) at scale {} can accomodate this value.", recommended_bits, self.settings.run_args.scale);
                return Err(err_string.into());
            }
        } else {
            let min_bits = (res.max_lookup_inputs as f64).log2().ceil() as usize + 1;

            let min_rows_from_constraints = (self.settings.num_constraints as f64
                + ASSUMED_BLINDING_FACTORS as f64)
                .log2()
                .ceil() as usize
                + 1;
            let mut logrows = std::cmp::max(min_bits + 1, min_rows_from_constraints);
            // if public input then public inputs col will have public inputs len
            if self.settings.run_args.input_visibility.is_public()
                || self.settings.run_args.output_visibility.is_public()
            {
                let max_instance_len = self
                    .model
                    .instance_shapes()
                    .iter()
                    .fold(0, |acc, x| std::cmp::max(acc, x.iter().product::<usize>()));
                let instance_len_logrows = (max_instance_len as f64).log2().ceil() as usize + 1;
                logrows = std::cmp::max(logrows, instance_len_logrows);
            // this is for fixed const columns
            } else if self.settings.run_args.param_visibility.is_public() {
                // if private input then public inputs col will have 0
                let total_const_len = self
                    .model
                    .const_shapes()
                    .iter()
                    .fold(0, |acc, x| std::cmp::max(acc, x.iter().product::<usize>()));
                let const_len_logrows = (total_const_len as f64).log2().ceil() as usize + 1;
                logrows = std::cmp::max(logrows, const_len_logrows);
            }

            // ensure logrows is at least 4
            logrows = std::cmp::max(
                logrows,
                (ASSUMED_BLINDING_FACTORS as f64).ceil() as usize + 1,
            );

            logrows = std::cmp::min(logrows, MAX_PUBLIC_SRS as usize);

            info!(
                "setting bits to: {}, setting logrows to: {}",
                min_bits, logrows
            );
            self.settings.run_args.bits = min_bits;
            self.settings.run_args.logrows = logrows as u32;
        }

        self.settings = GraphCircuit::new(
            self.model.clone(),
            self.settings.run_args,
            self.settings.check_mode,
        )?
        .settings;

        Ok(())
    }

    /// Runs the forward pass of the model / graph of computations and any associated hashing.
    pub fn forward(
        &self,
        inputs: &[Tensor<Fp>],
    ) -> Result<GraphWitness, Box<dyn std::error::Error>> {
        let visibility = VarVisibility::from_args(self.settings.run_args)?;
        let mut processed_inputs = None;
        let mut processed_params = None;
        let mut processed_outputs = None;

        if visibility.input.requires_processing() {
            processed_inputs = Some(GraphModules::forward(inputs, visibility.input)?);
        }

        if visibility.params.requires_processing() {
            let params = self.model.get_all_consts();
            if !params.is_empty() {
                let flattened_params = Tensor::new(Some(&params), &[params.len()])?.combine()?;
                processed_params = Some(GraphModules::forward(
                    &[flattened_params],
                    visibility.params,
                )?);
            }
        }

        let model_results = self.model.forward(inputs)?;

        if visibility.output.requires_processing() {
            processed_outputs = Some(GraphModules::forward(
                &model_results.outputs,
                visibility.output,
            )?);
        }

        Ok(GraphWitness {
            inputs: inputs
                .to_vec()
                .iter()
                .map(|t| t.deref().to_vec())
                .collect_vec()
                .into(),
            outputs: model_results
                .outputs
                .iter()
                .map(|t| t.deref().to_vec())
                .collect_vec()
                .into(),
            processed_inputs,
            processed_params,
            processed_outputs,
            max_lookup_inputs: model_results.max_lookup_inputs,
        })
    }

    /// Create a new circuit from a set of input data and [RunArgs].
    pub fn from_run_args(
        run_args: &RunArgs,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Model::from_run_args(run_args, model_path)?;
        Self::new(model, *run_args, check_mode)
    }

    ///
    pub fn preprocessed_from_run_args(
        run_args: &RunArgs,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Model::load(model_path.clone()).map_err(|e| {
            error!("failed to deserialize compiled model. have you called compile-model ?");
            e
        })?;
        Self::new(model, *run_args, check_mode)
    }

    /// Create a new circuit from a set of input data and [GraphSettings].
    pub fn from_settings(
        params: &GraphSettings,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Model::from_run_args(&params.run_args, model_path)?;
        Self::new_from_settings(model, params.clone(), check_mode)
    }

    /// Create a new circuit from a set of input data and [GraphSettings].
    pub fn preprocessed_from_settings(
        params: &GraphSettings,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Model::load(model_path.clone()).map_err(|e| {
            error!("failed to deserialize compiled model. have you called compile-model ?");
            e
        })?;
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

        if matches!(
            test_on_chain_data.data_sources.input,
            TestDataSource::OnChain
        ) {
            // if not public then fail
            if !self.settings.run_args.input_visibility.is_public() {
                return Err("Cannot use on-chain data source as private data".into());
            }

            let input_data = match &data.input_data {
                DataSource::File(input_data) => input_data,
                DataSource::OnChain(_) => panic!(
                    "Cannot use on-chain data source as input for on-chain test. 
                    Will manually populate on-chain data from file source instead"
                ),
            };
            // Get the flatten length of input_data
            let length = input_data.iter().map(|x| x.len()).sum();
            let scales = vec![self.settings.run_args.scale; length];
            let datam: (Vec<Tensor<Fp>>, OnChainSource) = OnChainSource::test_from_file_data(
                input_data,
                scales,
                self.model.graph.input_shapes(),
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
            if !self.settings.run_args.output_visibility.is_public() {
                return Err("Cannot use on-chain data source as private data".into());
            }

            let output_data = match &data.output_data {
                Some(DataSource::File(output_data)) => output_data,
                Some(DataSource::OnChain(_)) => panic!(
                    "Cannot use on-chain data source as output for on-chain test. 
                    Will manually populate on-chain data from file source instead"
                ),
                _ => panic!("No output data to populate"),
            };
            let datum: (Vec<Tensor<Fp>>, OnChainSource) = OnChainSource::test_from_file_data(
                output_data,
                self.model.graph.get_output_scales(),
                self.model.graph.output_shapes(),
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

impl Circuit<Fp> for GraphCircuit {
    type Config = GraphConfig;
    type FloorPlanner = ModulePlanner;
    type Params = GraphSettings;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn params(&self) -> Self::Params {
        // safe to clone because the model is Arc'd
        self.settings.clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<Fp>, params: Self::Params) -> Self::Config {
        let visibility = VarVisibility::from_args(params.run_args).unwrap();

        let mut vars = ModelVars::new(
            cs,
            params.run_args.logrows as usize,
            params.num_constraints,
            params.model_instance_shapes.clone(),
            visibility.clone(),
            params.run_args.scale,
        );

        let base = Model::configure(
            cs,
            &mut vars,
            params.run_args.bits,
            params.required_lookups,
            params.check_mode,
        )
        .unwrap();

        let model_config = ModelConfig { base, vars };

        let module_configs = ModuleConfigs::from_visibility(cs, visibility, params.module_sizes);

        trace!(
            "log2_ceil of degrees {:?}",
            (cs.degree() as f32).log2().ceil()
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
        let mut inputs = self
            .graph_witness
            .get_input_tensor()
            .iter()
            .map(|i| ValTensor::from(i.map(Value::known)))
            .collect::<Vec<ValTensor<Fp>>>();

        let mut instance_offset = ModuleInstanceOffset::new();
        trace!("running input module layout");
        // we reserve module 0 for poseidon
        // we reserve module 1 for elgamal
        GraphModules::layout(
            &mut layouter,
            &config.module_configs,
            &mut inputs,
            self.settings.run_args.input_visibility,
            &mut instance_offset,
            &self.module_settings.input,
        )?;

        // now we need to assign the flattened params to the model
        let mut model = self.model.clone();
        let param_visibility = self.settings.run_args.param_visibility;
        trace!("running params module layout");
        if !self.model.get_all_consts().is_empty() && param_visibility.requires_processing() {
            // now we need to flatten the params
            let consts = self.model.get_all_consts();

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
                vec![t.into()]
            };

            // now do stuff to the model params
            GraphModules::layout(
                &mut layouter,
                &config.module_configs,
                &mut flattened_params,
                param_visibility,
                &mut instance_offset,
                &self.module_settings.params,
            )?;

            let shapes = self.model.const_shapes();
            trace!("replacing processed consts");
            let split_params = split_valtensor(&flattened_params[0], shapes).map_err(|_| {
                log::error!("failed to split params");
                PlonkError::Synthesis
            })?;

            // now the flattened_params have been assigned to and we-assign them to the model consts such that they are constrained to be equal
            model.replace_consts(split_params);
        }

        // create a new module for the model (space 2)
        layouter.assign_region(|| "_new_module", |_| Ok(()))?;
        trace!("laying out model");
        let mut outputs = model
            .layout(
                config.model_config.clone(),
                &mut layouter,
                &self.settings.run_args,
                &inputs,
                &config.model_config.vars,
            )
            .map_err(|e| {
                log::error!("{}", e);
                PlonkError::Synthesis
            })?;
        trace!("running output module layout");

        // this will re-enter module 0
        GraphModules::layout(
            &mut layouter,
            &config.module_configs,
            &mut outputs,
            self.settings.run_args.output_visibility,
            &mut instance_offset,
            &self.module_settings.output,
        )?;

        Ok(())
    }
}
