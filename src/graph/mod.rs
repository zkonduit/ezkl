/// Helper functions
pub mod utilities;
use serde::{Deserialize, Serialize};
pub use utilities::*;
/// Crate for defining a computational graph and building a ZK-circuit from it.
pub mod model;
/// Inner elements of a computational graph that represent a single operation / constraints.
pub mod node;
/// Representations of a computational graph's variables.
pub mod vars;

use crate::circuit::OpKind;
use crate::commands::Cli;
use crate::fieldutils::i128_to_felt;
use crate::pfsys::ModelInput;
use crate::tensor::ops::pack;
use crate::tensor::TensorType;
use crate::tensor::{Tensor, ValTensor};
use anyhow::Result;
use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use log::{info, trace};
pub use model::*;
pub use node::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::PathBuf;
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
    InvalidDims(usize, OpKind),
    /// Wrong method was called to configure an op
    #[error("wrong method was called to configure node {0} ({1})")]
    WrongMethod(usize, OpKind),
    /// A requested node is missing in the graph
    #[error("a requested node is missing in the graph: {0}")]
    MissingNode(usize),
    /// The wrong method was called on an operation
    #[error("an unsupported method was called on node {0} ({1})")]
    OpMismatch(usize, OpKind),
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
    RescalingError(OpKind),
    /// Error when attempting to load a model
    #[error("failed to load model")]
    ModelLoad,
    /// Packing exponent is too large
    #[error("largest packing exponent exceeds max. try reducing the scale")]
    PackingExponent,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCircuit<F: FieldExt> {
    /// Vector of input tensors to the model / graph of computations.
    pub inputs: Vec<Tensor<i128>>,
    ///
    pub model: Model,
    /// Represents the Field we are using.
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> ModelCircuit<F> {
    ///
    pub fn new(
        data: &ModelInput,
        model: Model,
    ) -> Result<ModelCircuit<F>, Box<dyn std::error::Error>> {
        // quantize the supplied data using the provided scale.
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for (input, shape) in data.input_data.iter().zip(data.input_shapes.clone()) {
            let t = vector_to_quantized(input, &shape, 0.0, model.run_args.scale)?;
            inputs.push(t);
        }

        Ok(ModelCircuit::<F> {
            inputs,
            model,
            _marker: PhantomData,
        })
    }

    ///
    pub fn write<W: Write>(
        &self,
        mut writer: BufWriter<W>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let circuit_bytes = bincode::serialize(&self)?;
        writer.write(&circuit_bytes)?;
        writer.flush()?;
        Ok(())
    }

    ///
    pub fn write_to_file(&self, path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        let fs = File::create(path)?;
        let buffer = BufWriter::new(fs);
        self.write(buffer)
    }

    ///
    pub fn read<R: Read>(mut reader: BufReader<R>) -> Result<Self, Box<dyn std::error::Error>> {
        let buffer: &mut Vec<u8> = &mut vec![];
        reader.read_to_end(buffer)?;

        let circuit = bincode::deserialize(&buffer)?;
        Ok(circuit)
    }
    ///
    pub fn read_from_file(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        Self::read(reader)
    }

    ///
    pub fn from_arg(data: &ModelInput) -> Result<Self, Box<dyn std::error::Error>> {
        let cli = Cli::create()?;
        let model = Model::from_ezkl_conf(cli)?;
        Self::new(data, model)
    }

    ///
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
                let t =
                    vector_to_quantized(v, &Vec::from([v.len()]), 0.0, self.model.run_args.scale)?;
                public_inputs.push(t);
            }
        }
        if self.model.visibility.output.is_public() {
            for (idx, v) in data.output_data.iter().enumerate() {
                let mut t = vector_to_quantized(v, &Vec::from([v.len()]), 0.0, out_scales[idx])?;
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

impl<F: FieldExt + TensorType> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let model = Model::from_arg().expect("model should load");

        // for now the number of instances corresponds to the number of graph / model outputs
        let instance_shapes = model.instance_shapes();
        let var_len = model.total_var_len();

        info!("total var len: {:?}", var_len);
        info!("instance_shapes: {:?}", instance_shapes);

        let mut vars = ModelVars::new(
            cs,
            model.run_args.logrows as usize,
            var_len,
            instance_shapes.clone(),
            model.visibility.clone(),
        );
        info!(
            "number of advices used: {:?}",
            vars.advices.iter().map(|a| a.num_cols()).sum::<usize>()
        );
        info!(
            "number of fixed used: {:?}",
            vars.fixed.iter().map(|a| a.num_cols()).sum::<usize>()
        );
        info!("number of instances used: {:?}", instance_shapes.len());
        model.configure(cs, &mut vars).unwrap()
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
        trace!("Setting output in synthesize");
        config
            .model
            .layout(config.clone(), &mut layouter, &inputs, &config.vars)
            .unwrap();

        Ok(())
    }
}

////////////////////////
