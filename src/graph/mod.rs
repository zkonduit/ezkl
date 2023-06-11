/// Helper functions
pub mod utilities;
use itertools::Itertools;
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
use crate::commands::{Cli, RunArgs};
use crate::fieldutils::i128_to_felt;
use crate::tensor::ops::pack;
use crate::tensor::{Tensor, ValTensor};
use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use halo2curves::bn256::Fr as Fp;
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

/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct GraphInput {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are not being constrained).
    pub input_data: Vec<Vec<f32>>,
    /// The expected output of the model (can be empty vectors if outputs are not being constrained).
    pub output_data: Vec<Vec<f32>>,
    /// Optional hashes of the inputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub input_hashes: Option<Vec<Fp>>,
    /// Optional hashes of the outputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub output_hashes: Option<Vec<Fp>>,
}

impl GraphInput {
    ///
    pub fn new(input_data: Vec<Vec<f32>>, output_data: Vec<Vec<f32>>) -> Self {
        GraphInput {
            input_data,
            output_data,
            input_hashes: None,
            output_hashes: None,
        }
    }
    ///
    pub fn split_into_batches(
        &self,
        batch_size: usize,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        // ensure the input is devenly divisible by batch_size
        if self.input_data.len() % batch_size != 0 || self.output_data.len() % batch_size != 0 {
            return Err(Box::new(GraphError::InvalidDims(
                0,
                "input data length must be evenly divisible by batch size".to_string(),
            )));
        }

        // split input data into batches
        let mut input_batches = vec![];

        for input in &self.input_data.iter().chunks(batch_size) {
            let mut batch = vec![];
            for i in input.into_iter() {
                batch.push(i.clone());
            }
            input_batches.push(batch);
        }

        // split output data into batches
        let mut output_batches = vec![];

        for output in &self.output_data.iter().chunks(batch_size) {
            let mut batch = vec![];
            for i in output.into_iter() {
                batch.push(i.clone());
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

const POSEIDON_LEN_GRAPH: usize = 10;

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
        serde_json::to_writer(std::fs::File::create(&path)?, &self).map_err(|e| e.into())
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphParams {
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

impl GraphParams {
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
        file.write_all(&encoded.as_bytes())
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
    /// The parameters of the model / graph of computations.
    pub params: GraphParams,
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

        let mut params = model.gen_params(run_args, check_mode)?;
        if params.run_args.input_visibility.is_hashed() {
            params.num_hashes += model.graph.num_inputs();
        }
        if params.run_args.output_visibility.is_hashed() {
            params.num_hashes += model.graph.num_outputs();
        }

        Ok(GraphCircuit {
            model: model.clone(),
            inputs,
            params,
        })
    }
    ///
    pub fn load_inputs(&mut self, data: &GraphInput) {
        // quantize the supplied data using the provided scale.
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for (input, shape) in data.input_data.iter().zip(self.model.graph.input_shapes()) {
            let t: Vec<i128> = input
                .par_iter()
                .map(|x| quantize_float(x, 0.0, self.params.run_args.scale).unwrap())
                .collect();

            let mut t: Tensor<i128> = t.into_iter().into();
            t.reshape(&shape);

            inputs.push(t);
        }
        self.inputs = inputs;
    }

    /// Calibrate the circuit to the supplied data.
    pub fn calibrate(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let res = self.forward()?;
        let max_range = 2i128.pow(self.params.run_args.bits as u32 - 1);
        if res.max_lookup_input > max_range {
            let recommended_bits = (res.max_lookup_input as f64).log2().ceil() as usize + 1;
            assert!(res.max_lookup_input <= 2i128.pow(recommended_bits as u32 - 1));

            if recommended_bits <= 27 {
                self.params.run_args.bits = recommended_bits;
                self.params.run_args.logrows = (recommended_bits + 1) as u32;
                info!(
                    "increasing bits to: {}, increasing logrows to: {}",
                    recommended_bits,
                    recommended_bits + 1
                );
                return self.calibrate();
            } else {
                let err_string = format!("No possible value of bits (estimate {}) at scale {} can accomodate this value. Try changing the scale and re-running calibration.", recommended_bits, self.params.run_args.scale);
                error!("{}", err_string);
                return Err(err_string.into());
            }
        } else {
            let min_bits = (res.max_lookup_input as f64).log2().ceil() as usize + 1;
            let min_rows_from_constraints =
                (self.params.num_constraints as f64).log2().ceil() as usize + 1;
            let logrows = std::cmp::max(min_bits + 1, min_rows_from_constraints);
            info!(
                "setting bits to: {}, setting logrows to: {}",
                min_bits, logrows
            );
            self.params.run_args.bits = min_bits;
            self.params.run_args.logrows = logrows as u32;
        }
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

    /// Create a new circuit from a set of input data and cli arguments.
    pub fn from_arg(check_mode: CheckMode) -> Result<Self, Box<dyn std::error::Error>> {
        let cli = Cli::create()?;
        let model = Arc::new(Model::from_cli(cli.clone())?);
        let run_args = RunArgs::from_cli(cli)?;
        Self::new(model, run_args, check_mode)
    }

    /// Create a new circuit from a set of input data and [GraphParams].
    pub fn from_params(
        params: &GraphParams,
        model_path: &std::path::PathBuf,
        check_mode: CheckMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Arc::new(Model::from_model_params(params, model_path)?);
        Self::new(model, params.run_args, check_mode)
    }

    /// Prepare the public inputs for the circuit.
    pub fn prepare_public_inputs(
        &mut self,
        data: &GraphInput,
    ) -> Result<Vec<Vec<Fp>>, Box<dyn std::error::Error>> {
        let out_scales = self.model.graph.get_output_scales();

        self.load_inputs(data);

        // quantize the supplied data using the provided scale.
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>
        let mut public_inputs = vec![];
        if self.params.run_args.input_visibility.is_public() {
            public_inputs = self.inputs.clone();
        }
        if self.params.run_args.output_visibility.is_public() {
            for (idx, v) in data.output_data.iter().enumerate() {
                let t: Vec<i128> = v
                    .par_iter()
                    .map(|x| quantize_float(x, 0.0, out_scales[idx]).unwrap())
                    .collect();

                let mut t: Tensor<i128> = t.into_iter().into();

                let len = t.len();
                if self.params.run_args.pack_base > 1 {
                    let max_exponent =
                        (((len - 1) as u32) * (self.params.run_args.scale + 1)) as f64;
                    if max_exponent > (i128::MAX as f64).log(self.params.run_args.pack_base as f64)
                    {
                        return Err(Box::new(GraphError::PackingExponent));
                    }
                    t = pack(
                        &t,
                        self.params.run_args.pack_base as i128,
                        self.params.run_args.scale,
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

        if self.params.run_args.input_visibility.is_hashed()
            || self.params.run_args.output_visibility.is_hashed()
        {
            let mut hash_pi = vec![];
            if self.params.run_args.input_visibility.is_hashed() {
                // should unwrap safely
                hash_pi.extend(data.input_hashes.as_deref().unwrap().to_vec());
            }
            if self.params.run_args.output_visibility.is_hashed() {
                // should unwrap safely
                hash_pi.extend(data.output_hashes.as_deref().unwrap().to_vec());
            };
            if hash_pi.len() > 0 {
                pi_inner.push(hash_pi);
            }
        }
        Ok(pi_inner)
    }
}

impl Circuit<Fp> for GraphCircuit {
    type Config = GraphConfig;
    type FloorPlanner = ModulePlanner;
    type Params = GraphParams;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn params(&self) -> Self::Params {
        // safe to clone because the model is Arc'd
        self.params.clone()
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

        if self.params.run_args.input_visibility.is_hashed() {
            // instantiate new poseidon module in chip
            let chip = PoseidonChip::<
                PoseidonSpec,
                POSEIDON_WIDTH,
                POSEIDON_RATE,
                POSEIDON_LEN_GRAPH,
            >::construct(config.poseidon_config.as_ref().unwrap().clone());
            for (i, input) in inputs.clone().iter().enumerate() {
                // hash the input and replace the constrained cells in the inputs
                inputs[i] = chip.hash(&mut layouter, &input, i)?;
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
                &self.params.run_args,
                &inputs,
                &config.model_config.vars,
            )
            .map_err(|e| {
                log::error!("{}", e);
                PlonkError::Synthesis
            })?;

        if self.params.run_args.output_visibility.is_hashed() {
            let chip = PoseidonChip::<
                PoseidonSpec,
                POSEIDON_WIDTH,
                POSEIDON_RATE,
                POSEIDON_LEN_GRAPH,
            >::construct(config.poseidon_config.unwrap());
            let mut hash_offset = 0;
            if self.params.run_args.input_visibility.is_hashed() {
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
