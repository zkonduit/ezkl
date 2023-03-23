/// Helper functions
pub mod utilities;
pub use utilities::*;
/// Crate for defining a computational graph and building a ZK-circuit from it.
pub mod model;
/// Inner elements of a computational graph that represent a single operation / constraints.
pub mod node;
/// Representations of a computational graph's variables.
pub mod vars;

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
use std::marker::PhantomData;
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
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug)]
pub struct ModelCircuit<F: FieldExt> {
    /// Vector of input tensors to the model / graph of computations.
    pub inputs: Vec<Tensor<i128>>,
    /// Represents the Field we are using.
    pub _marker: PhantomData<F>,
}

impl<F: FieldExt + TensorType> Circuit<F> for ModelCircuit<F> {
    type Config = ModelConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
        let model = Model::from_arg().expect("model should load from args");

        // for now the number of instances corresponds to the number of graph / model outputs
        let instance_shapes = model.instance_shapes();

        let mut vars = ModelVars::new(cs, model.run_args.logrows as usize, instance_shapes.clone());
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
