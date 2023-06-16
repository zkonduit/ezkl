/// Helper functions
pub mod utilities;
use halo2curves::ff::PrimeField;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
pub use utilities::*;
/// Crate for defining a computational graph and building a ZK-circuit from it.
pub mod model;
/// Inner elements of a computational graph that represent a single operation / constraints.
pub mod node;
/// Representations of a computational graph's variables.
pub mod vars;

use crate::circuit::lookup::LookupOp;
use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{witness_hash, PoseidonChip, PoseidonConfig};
use crate::circuit::modules::ModulePlanner;
use crate::circuit::CheckMode;
use crate::commands::RunArgs;
use crate::fieldutils::i128_to_felt;
use crate::tensor::ops::pack;
use crate::tensor::{Tensor, ValTensor};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use halo2curves::bn256::{self, Fr as Fp};
use log::{error, info, trace};
pub use model::*;
pub use node::*;
use std::io::{Read, Write};
use std::sync::Arc;
use thiserror::Error;
pub use vars::*;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;

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

const ASSUMED_BLINDING_FACTORS: usize = 6;

type Decimals = u8;
type Call = String;
type RPCUrl = String;
/// Defines the view only calls to accounts to fetch the on-chain input data.
/// This data will be included as part of the first elements in the publicInputs
/// for the sol evm verifier and will be  verifyWithDataAttestation.sol
#[derive(Clone, Debug, Deserialize, Serialize, Default)]

pub struct CallsToAccount {
    /// A vector of tuples, where index 0 of tuples
    /// are the byte strings representing the ABI encoded function calls to 
    /// read the data from the address. This call must return a single
    /// elementary type (https://docs.soliditylang.org/en/v0.8.20/abi-spec.html#types).
    /// The second index of the tuple is the number of decimals for f32 conversion.
    /// We don't support dynamic types currently.
    pub call_data: Vec<(Call, Decimals)>,
    /// Address of the contract to read the data from.
    pub address: String,
}
/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct GraphInput {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are coming from on-chain).
    /// TODO: Add retrieve from on-chain functionality
    pub input_data: Vec<Vec<f32>>, 
    /// The expected output of the model (can be empty vectors if outputs are not being constrained).
    pub output_data: Vec<Vec<f32>>,
    /// Optional hashes of the inputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub input_hashes: Option<Vec<Fp>>,
    /// Optional hashes of the inputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub output_hashes: Option<Vec<Fp>>,
    /// Optional on-chain inputs. (can be None if there are no on-chain inputs). Wrapped as Option for backwards compatibility
    pub on_chain_input_data: Option<(Vec<CallsToAccount>, RPCUrl)>,
}

impl GraphInput {
    ///
    pub fn new(input_data: Vec<Vec<f32>>, output_data: Vec<Vec<f32>>) -> Self {
        GraphInput {
            input_data,
            output_data,
            input_hashes: None,
            output_hashes: None,
            on_chain_input_data: None,
        }
    }
    ///
    pub fn split_into_batches(
        &self,
        batch_size: usize,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        // split input data into batches
        let mut batched_inputs = vec![];

        for (i, input) in self.input_data.iter().enumerate() {
            // ensure the input is devenly divisible by batch_size
            if input.len() % batch_size != 0 {
                return Err(Box::new(GraphError::InvalidDims(
                    0,
                    "input data length must be evenly divisible by batch size".to_string(),
                )));
            }
            let input_size = input_shapes[i].clone().iter().product::<usize>();
            let mut batches = vec![];
            for batch in input.chunks(batch_size * input_size) {
                batches.push(batch.to_vec());
            }
            batched_inputs.push(batches);
        }
        // now merge all the batches for each input into a vector of batches
        // first assert each input has the same number of batches
        let num_batches = batched_inputs[0].len();
        for input in batched_inputs.iter() {
            assert_eq!(input.len(), num_batches);
        }
        // now merge the batches
        let mut input_batches = vec![];
        for i in 0..num_batches {
            let mut batch = vec![];
            for input in batched_inputs.iter() {
                batch.push(input[i].clone());
            }
            input_batches.push(batch);
        }

        // split output data into batches
        let mut batched_outputs = vec![];

        for (i, output) in self.output_data.iter().enumerate() {
            // ensure the input is devenly divisible by batch_size
            if output.len() % batch_size != 0 {
                return Err(Box::new(GraphError::InvalidDims(
                    0,
                    "input data length must be evenly divisible by batch size".to_string(),
                )));
            }

            let output_size = output_shapes[i].clone().iter().product::<usize>();
            let mut batches = vec![];
            for batch in output.chunks(batch_size * output_size) {
                batches.push(batch.to_vec());
            }
            batched_outputs.push(batches);
        }

        // now merge all the batches for each output into a vector of batches
        // first assert each output has the same number of batches
        let num_batches = batched_outputs[0].len();
        for output in batched_outputs.iter() {
            assert_eq!(output.len(), num_batches);
        }
        // now merge the batches
        let mut output_batches = vec![];
        for i in 0..num_batches {
            let mut batch = vec![];
            for output in batched_outputs.iter() {
                batch.push(output[i].clone());
            }
            output_batches.push(batch);
        }

        // create a new GraphInput for each batch
        let batches = input_batches
            .into_iter()
            .zip(output_batches.into_iter())
            .map(|(input, output)| GraphInput::new(input, output))
            .collect::<Vec<GraphInput>>();

        Ok(batches)
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for GraphInput {
    fn to_object(&self, py: Python) -> PyObject {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let input_data_mut = &self.input_data;
        let output_data_mut = &self.output_data;
        dict.set_item("input_data", truncate_nested_vector(&input_data_mut))
            .unwrap();
        dict.set_item("output_data", truncate_nested_vector(&output_data_mut))
            .unwrap();

        dict.to_object(py)
    }
}

/// Truncates nested vector due to omit junk floating point values in python
#[cfg(feature = "python-bindings")]
fn truncate_nested_vector(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut input_mut = input.clone();
    for inner_vec in input_mut.iter_mut() {
        for value in inner_vec.iter_mut() {
            // truncate 6 decimal places
            *value = (*value * 10000000.0).trunc() / 10000000.0;
        }
    }
    input_mut
}

const POSEIDON_LEN_GRAPH: usize = 4;
// TODO: make this a function of the number of constraints this is a bit of a hack
const POSEIDON_CONSTRAINTS_ESTIMATE: usize = 44;

impl GraphInput {
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
}

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ForwardResult {
    /// The inputs of the forward pass
    pub inputs: Vec<Tensor<i128>>,
    /// The output of the forward pass
    pub outputs: Vec<Tensor<i128>>,
    /// Any hashes of inputs generated during the forward pass
    pub input_hashes: Vec<Fp>,
    /// Any hashes of outputs generated during the forward pass
    pub output_hashes: Vec<Fp>,
    /// max lookup input
    pub max_lookup_input: i128,
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
    /// the number of hashes generated
    pub num_hashes: usize,
    /// required_lookups
    pub required_lookups: Vec<LookupOp>,
    /// check mode
    pub check_mode: CheckMode,
}

impl GraphSettings {
    /// calculate the total number of instances
    pub fn total_instances(&self) -> Vec<usize> {
        let mut instances: Vec<usize> = self
            .model_instance_shapes
            .iter()
            .map(|x| x.iter().product())
            .collect();
        if self.num_hashes > 0 {
            instances.push(self.num_hashes)
        }
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
    poseidon_config: Option<PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>>,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default)]
pub struct GraphCircuit {
    /// The model / graph of computations.
    pub model: Arc<Model>,
    /// Vector of input tensors to the model / graph of computations.
    pub inputs: Vec<Tensor<i128>>,
    /// The settings of the model / graph of computations.
    pub settings: GraphSettings,
}

impl GraphCircuit {
    ///
    pub fn new(
        model: Arc<Model>,
        run_args: RunArgs,
        check_mode: CheckMode,
    ) -> Result<GraphCircuit, Box<dyn std::error::Error>> {
        // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for shape in model.graph.input_shapes() {
            let t: Tensor<i128> = Tensor::new(None, &shape).unwrap();
            inputs.push(t);
        }

        let mut hashing_constraints = 0;
        let mut settings = model.gen_params(run_args, check_mode)?;
        if settings.run_args.input_visibility.is_hashed() {
            settings.num_hashes += model.graph.num_inputs();
            for input in model.graph.input_shapes() {
                hashing_constraints +=
                    POSEIDON_CONSTRAINTS_ESTIMATE * input.iter().product::<usize>();
            }
        }
        if settings.run_args.output_visibility.is_hashed() {
            settings.num_hashes += model.graph.num_outputs();
            for output in model.graph.output_shapes() {
                hashing_constraints +=
                    POSEIDON_CONSTRAINTS_ESTIMATE * output.iter().product::<usize>();
            }
        }

        // as they occupy independent rows
        settings.num_constraints = std::cmp::max(settings.num_constraints, hashing_constraints);

        Ok(GraphCircuit {
            model,
            inputs,
            settings,
        })
    }

    ///
    pub fn new_from_settings(
        model: Arc<Model>,
        mut settings: GraphSettings,
        check_mode: CheckMode,
    ) -> Result<GraphCircuit, Box<dyn std::error::Error>> {
        // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for shape in model.graph.input_shapes() {
            let t: Tensor<i128> = Tensor::new(None, &shape).unwrap();
            inputs.push(t);
        }

        settings.check_mode = check_mode;

        Ok(GraphCircuit {
            model,
            inputs,
            settings,
        })
    }
    ///
    pub fn load_inputs(&mut self, data: &GraphInput) {
        // quantize the supplied data using the provided scale.
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for (input, shape) in data.input_data.iter().zip(self.model.graph.input_shapes()) {
            let t: Vec<i128> = input
            .par_iter()
            .map(|x| quantize_float(x, 0.0, self.settings.run_args.scale).unwrap())
            .collect();
        
            let mut t: Tensor<i128> = t.into_iter().into();
            t.reshape(&shape);
            
            inputs.push(t);
        }
        self.inputs = inputs;
    }
    ///
    pub fn load_on_chain_inputs(&mut self, data: Vec<Vec<i128>>) {
        // on-chain data has already been quantized at this point. Just need to reshape it and push into tensor vector
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for (input, shape) in data.iter().zip(self.model.graph.input_shapes()) {
            let mut t: Tensor<i128> = input.iter().cloned().collect();
            t.reshape(&shape);
            inputs.push(t);
        }
        self.inputs = inputs;
    }

    /// Calibrate the circuit to the supplied data.
    pub fn calibrate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.forward()?;
        let max_range = 2i128.pow(self.settings.run_args.bits as u32 - 1);
        if res.max_lookup_input > max_range {
            let recommended_bits = (res.max_lookup_input as f64).log2().ceil() as usize + 1;
            assert!(res.max_lookup_input <= 2i128.pow(recommended_bits as u32 - 1));

            if recommended_bits <= (bn256::Fr::S - 1) as usize {
                self.settings.run_args.bits = recommended_bits;
                self.settings.run_args.logrows = (recommended_bits + 1) as u32;
                info!(
                    "increasing bits to: {}, increasing logrows to: {}",
                    recommended_bits,
                    recommended_bits + 1
                );
                return self.calibrate();
            } else {
                let err_string = format!("No possible value of bits (estimate {}) at scale {} can accomodate this value.", recommended_bits, self.settings.run_args.scale);
                error!("{}", err_string);

                return Err(err_string.into());
            }
        } else {
            let min_bits = (res.max_lookup_input as f64).log2().ceil() as usize + 1;

            let min_rows_from_constraints = (self.settings.num_constraints as f64
                + ASSUMED_BLINDING_FACTORS as f64)
                .log2()
                .ceil() as usize
                + 1;
            let mut logrows = std::cmp::max(min_bits + 1, min_rows_from_constraints);

            // ensure logrows is at least 4
            logrows = std::cmp::max(
                logrows,
                (ASSUMED_BLINDING_FACTORS as f64).ceil() as usize + 1,
            );

            logrows = std::cmp::min(logrows, bn256::Fr::S as usize);

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
    pub fn forward(&self) -> Result<ForwardResult, Box<dyn std::error::Error>> {
        let mut input_hashes = vec![];
        for input in self.inputs.iter() {
            input_hashes.push(witness_hash::<POSEIDON_LEN_GRAPH>(
                input.iter().map(|x| i128_to_felt(*x)).collect(),
            )?);
        }

        let mut output_hashes = vec![];
        let outputs = self.model.forward(&self.inputs)?;

        for input in outputs.outputs.iter() {
            output_hashes.push(witness_hash::<POSEIDON_LEN_GRAPH>(
                input.iter().map(|x| i128_to_felt(*x)).collect(),
            )?);
        }

        Ok(ForwardResult {
            inputs: self.inputs.clone(),
            outputs: outputs.outputs,
            input_hashes,
            output_hashes,
            max_lookup_input: outputs.max_lookup_inputs,
        })
    }

    /// Create a new circuit from a set of input data and [RunArgs].
    pub fn from_run_args(
        run_args: &RunArgs,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Arc::new(Model::from_run_args(run_args, model_path)?);
        Self::new(model, *run_args, check_mode)
    }

    /// Create a new circuit from a set of input data and [GraphSettings].
    pub fn from_settings(
        params: &GraphSettings,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Arc::new(Model::from_run_args(&params.run_args, model_path)?);
        Self::new_from_settings(model, params.clone(), check_mode)
    }

    /// Prepare the public inputs for the circuit.
    pub fn prepare_public_inputs(
        &mut self,
        data: &GraphInput,
        on_chain_inputs: Option<Vec<Vec<i128>>>
    ) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        let out_scales = self.model.graph.get_output_scales();
        
            // quantize the supplied data using the provided scale.
            if let Some(on_chain_inputs) = on_chain_inputs {
                self.load_on_chain_inputs(on_chain_inputs)
            } else {
                self.load_inputs(&data);
            }

            // quantize the supplied data using the provided scale.
            // the ordering here is important, we want the inputs to come before the outputs
            // as they are configured in that order as Column<Instances>
            let mut public_inputs = vec![];
            if self.settings.run_args.input_visibility.is_public() {
                public_inputs = self.inputs.clone();
            }
            if self.settings.run_args.output_visibility.is_public() {
                for (idx, v) in data.output_data.iter().enumerate() {
                    let t: Vec<i128> = v
                        .par_iter()
                        .map(|x| quantize_float(x, 0.0, out_scales[idx]).unwrap())
                        .collect();

                    let mut t: Tensor<i128> = t.into_iter().into();

                    let len = t.len();
                    if self.settings.run_args.pack_base > 1 {
                        let max_exponent =
                            (((len - 1) as u32) * (self.settings.run_args.scale + 1)) as f64;
                        if max_exponent
                        > (i128::MAX as f64).log(self.settings.run_args.pack_base as f64)
                    {
                            return Err(Box::new(GraphError::PackingExponent));
                        }
                        t = pack(
                            &t,
                            self.settings.run_args.pack_base as i128,
                            self.settings.run_args.scale,
                        )?;
                    }
                    public_inputs.push(t);
                }
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
            .map(|i| {
                i.iter()
                    .map(|e| i128_to_felt::<Fp>(*e))
                    .collect::<Vec<Fp>>()
            })
            .collect::<Vec<Vec<Fp>>>();

        if self.settings.run_args.input_visibility.is_hashed()
            || self.settings.run_args.output_visibility.is_hashed()
        {
            let mut hash_pi = vec![];
            if self.settings.run_args.input_visibility.is_hashed() {
                // should unwrap safely
                hash_pi.extend(data.input_hashes.as_deref().unwrap().to_vec());
            }
            if self.settings.run_args.output_visibility.is_hashed() {
                // should unwrap safely
                hash_pi.extend(data.output_hashes.as_deref().unwrap().to_vec());
            };
            if !hash_pi.is_empty() {
                pi_inner.push(hash_pi);
            }
        }
        Ok(pi_inner)
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
            params.run_args.tolerance,
            params.required_lookups,
            params.check_mode,
        )
        .unwrap();

        let model_config = ModelConfig { base, vars };

        let poseidon_config = if visibility.input.is_hashed()
            || visibility.output.is_hashed()
            || visibility.params.is_hashed()
        {
            Some(PoseidonChip::<
                PoseidonSpec,
                POSEIDON_WIDTH,
                POSEIDON_RATE,
                POSEIDON_LEN_GRAPH,
            >::configure(cs))
        } else {
            None
        };

        GraphConfig {
            model_config,
            poseidon_config,
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
            .inputs
            .iter()
            .map(|i| ValTensor::from(<Tensor<i128> as Into<Tensor<Value<Fp>>>>::into(i.clone())))
            .collect::<Vec<ValTensor<Fp>>>();

        if self.settings.run_args.input_visibility.is_hashed() {
            // instantiate new poseidon module in chip
            let chip = PoseidonChip::<
                PoseidonSpec,
                POSEIDON_WIDTH,
                POSEIDON_RATE,
                POSEIDON_LEN_GRAPH,
            >::construct(config.poseidon_config.as_ref().unwrap().clone());
            for (i, input) in inputs.clone().iter().enumerate() {
                // hash the input and replace the constrained cells in the inputs
                inputs[i] = chip.hash(&mut layouter, input, i)?;
                inputs[i]
                    .reshape(input.dims())
                    .map_err(|_| PlonkError::Synthesis)?;
            }
            // instantiate new module in chip
            layouter.assign_region(|| "_new_module", |_| Ok(()))?;
        }

        trace!("Laying out model");
        let mut outputs = self
            .model
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

        if self.settings.run_args.output_visibility.is_hashed() {
            let chip = PoseidonChip::<
                PoseidonSpec,
                POSEIDON_WIDTH,
                POSEIDON_RATE,
                POSEIDON_LEN_GRAPH,
            >::construct(config.poseidon_config.unwrap());
            let mut hash_offset = 0;
            if self.settings.run_args.input_visibility.is_hashed() {
                hash_offset += inputs.len();
                // re-enter the first module
                layouter.assign_region(|| "_enter_module_0", |_| Ok(()))?;
            } else {
                layouter.assign_region(|| "_new_module", |_| Ok(()))?;
            }
            for (i, output) in outputs.clone().iter().enumerate() {
                // hash the output and replace the constrained cells in the outputs
                outputs[i] = chip.hash(&mut layouter, output, hash_offset + i)?;
                outputs[i]
                    .reshape(output.dims())
                    .map_err(|_| PlonkError::Synthesis)?;
            }
        }

        Ok(())
    }
}

////////////////////////
