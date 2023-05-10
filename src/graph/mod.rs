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
use crate::commands::{Cli, RunArgs};
use crate::fieldutils::i128_to_felt;
use crate::pfsys::ModelInput;
use crate::tensor::ops::pack;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor};
use anyhow::Result;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use log::{info, trace};
pub use model::*;
pub use node::*;
use std::io::Write;
use std::sync::Arc;
use thiserror::Error;
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

/// model parameters
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelParams {
    /// run args
    pub run_args: RunArgs,
    /// the visibility of the variables in the circuit
    pub visibility: VarVisibility,
    /// the potential number of constraints in the circuit
    pub num_constraints: usize,
    /// the shape of public inputs to the circuit (in order of appearance)
    pub instance_shapes: Vec<Vec<usize>>,
    /// required_lookups
    pub required_lookups: Vec<LookupOp>,
}

impl ModelParams {
    /// save params to file
    pub fn save(&self, path: &std::path::PathBuf) {
        let mut file = std::fs::File::create(path).unwrap();
        let encoded: Vec<u8> = bincode::serialize(&self).unwrap();
        file.write_all(&encoded).unwrap();
    }
    /// load params from file
    pub fn load(path: &std::path::PathBuf) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let decoded: Self = bincode::deserialize_from(file).unwrap();
        decoded
    }
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default)]
pub struct ModelCircuit<F: PrimeField + TensorType + PartialOrd> {
    /// The model / graph of computations.
    pub model: Arc<Model<F>>,
    /// Vector of input tensors to the model / graph of computations.
    pub inputs: Vec<Tensor<i128>>,
    /// The parameters of the model / graph of computations.
    pub params: ModelParams,
}

impl<F: PrimeField + TensorType + PartialOrd> ModelCircuit<F> {
    ///
    pub fn new(
        data: &ModelInput,
        model: Arc<Model<F>>,
    ) -> Result<ModelCircuit<F>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for (input, shape) in data.input_data.iter().zip(data.input_shapes.clone()) {
            let t: Vec<i128> = input
                .par_iter()
                .map(|x| quantize_float(x, 0.0, model.run_args.scale).unwrap())
                .collect();

            let mut t: Tensor<i128> = t.into_iter().into();
            t.reshape(&shape);

            inputs.push(t);
        }

        Ok(ModelCircuit::<F> {
            model: model.clone(),
            inputs,
            params: model.gen_params()?,
        })
    }

    /// Create a new circuit from a set of input data and cli arguments.
    pub fn from_arg(data: &ModelInput) -> Result<Self, Box<dyn std::error::Error>> {
        let cli = Cli::create()?;
        let model = Arc::new(Model::from_ezkl_conf(cli)?);
        Self::new(data, model)
    }

    /// Create a new circuit from a set of input data and [ModelParams].
    pub fn from_model_params(
        data: &ModelInput,
        params: &ModelParams,
        model_path: &std::path::PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model = Arc::new(Model::from_model_params(params, model_path)?);
        Self::new(data, model)
    }

    /// Prepare the public inputs for the circuit.
    pub fn prepare_public_inputs(
        &self,
        data: &ModelInput,
    ) -> Result<Vec<Vec<F>>, Box<dyn std::error::Error>> {
        let out_scales = self.model.get_output_scales();

        // quantize the supplied data using the provided scale.
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>
        let mut public_inputs = vec![];
        if self.model.visibility.input.is_public() {
            for v in data.input_data.iter() {
                let t: Vec<i128> = v
                    .par_iter()
                    .map(|x| quantize_float(x, 0.0, self.model.run_args.scale).unwrap())
                    .collect();

                let t: Tensor<i128> = t.into_iter().into();

                public_inputs.push(t);
            }
        }
        if self.model.visibility.output.is_public() {
            for (idx, v) in data.output_data.iter().enumerate() {
                let t: Vec<i128> = v
                    .par_iter()
                    .map(|x| quantize_float(x, 0.0, out_scales[idx]).unwrap())
                    .collect();

                let mut t: Tensor<i128> = t.into_iter().into();

                let len = t.len();
                if self.model.run_args.pack_base > 1 {
                    let max_exponent =
                        (((len - 1) as u32) * (self.model.run_args.scale + 1)) as f64;
                    if max_exponent > (i128::MAX as f64).log(self.model.run_args.pack_base as f64) {
                        return Err(Box::new(GraphError::PackingExponent));
                    }
                    t = pack(
                        &t,
                        self.model.run_args.pack_base as i128,
                        self.model.run_args.scale,
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

        let pi_inner: Vec<Vec<F>> = public_inputs
            .iter()
            .map(|i| i.iter().map(|e| i128_to_felt::<F>(*e)).collect::<Vec<F>>())
            .collect::<Vec<Vec<F>>>();

        Ok(pi_inner)
    }
}

impl<F: PrimeField + TensorType + PartialOrd> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ModelParams;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn params(&self) -> Self::Params {
        // safe to clone because the model is Arc'd
        self.params.clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<F>, params: Self::Params) -> Self::Config {
        let mut vars = ModelVars::new(
            cs,
            params.run_args.logrows as usize,
            params.num_constraints,
            params.instance_shapes.clone(),
            params.visibility.clone(),
            params.run_args.scale,
        );

        let base =
            Model::<F>::configure(cs, &mut vars, params.run_args, params.required_lookups).unwrap();

        ModelConfig { base, vars }
    }

    fn configure(_: &mut ConstraintSystem<F>) -> Self::Config {
        unimplemented!("you should call configure_with_params instead")
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), PlonkError> {
        trace!("Setting input in synthesize");
        let inputs = self
            .inputs
            .iter()
            .map(|i| ValTensor::from(<Tensor<i128> as Into<Tensor<Value<F>>>>::into(i.clone())))
            .collect::<Vec<ValTensor<F>>>();
        trace!("Laying out model");
        self.model
            .layout(config.clone(), &mut layouter, &inputs, &config.vars)
            .unwrap();

        Ok(())
    }
}

////////////////////////
