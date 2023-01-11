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
use std::cmp::max;
use std::marker::PhantomData;
use thiserror::Error;
pub use vars::*;

/// circuit related errors.
#[derive(Debug, Error)]
pub enum GraphError {
    /// Shape mismatch in circuit construction
    DimMismatch(String),
    /// Wrong method was called to configure an op
    WrongMethod(OpKind),
    /// A requested node is missing in the graph
    MissingNode(usize),
    /// A requested node is missing in the graph
    OpMismatch(OpKind),
    /// A requested node is missing in the graph
    UnsupportedOp,
    /// A requested node is missing in the graph
    MissingParams(String),
    /// Error in the configuration of the visibility of variables
    Visibility,
    ///
    NonConstantDiv,
    ///
    NonConstantPower,
    ///
    RescalingError(OpKind),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GraphError::DimMismatch(op) => {
                write!(f, "dimension mismatch in circuit construction: {}", op)
            }
            GraphError::WrongMethod(op) => {
                write!(f, "wrong method was called to configure: {}", op)
            }
            GraphError::MissingNode(id) => {
                write!(f, "a requested node is missing in the graph: {}", id)
            }
            GraphError::OpMismatch(id) => {
                write!(f, "a requested node is missing in the graph: {}", id)
            }
            GraphError::UnsupportedOp => {
                write!(f, "unsupported operation in graph")
            }
            GraphError::MissingParams(id) => {
                write!(f, "a requested node is missing in the graph: {}", id)
            }
            GraphError::Visibility => {
                write!(f, "there should be at least 1 set of public variables")
            }
            GraphError::NonConstantDiv => {
                write!(f, "ezkl currently only supports division by a constant")
            }
            GraphError::NonConstantPower => {
                write!(f, "ezkl currently only supports constant exponents")
            }
            GraphError::RescalingError(op) => {
                write!(f, "failed to rescale inputs to {}", op)
            }
        }
    }
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug)]
pub struct ModelCircuit<F: FieldExt> {
    /// Vector of input tensors to the model / graph of computations.
    pub inputs: Vec<Tensor<i32>>,
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
        let mut num_fixed = 0;
        let row_cap = model.max_node_size();

        // TODO: extract max number of params in a given fused layer
        let num_advice: usize = if model.visibility.params.is_public() {
            num_fixed += model.max_node_params();
            // this is the maximum of variables in non-fused layer, and the maximum of variables (non-params) in fused layers
            max(model.max_node_vars_non_fused(), model.max_node_vars_fused())
        } else {
            // this is the maximum of variables in non-fused layer, and the maximum of variables (non-params) in fused layers
            //  + the max number of params in a fused layer
            max(
                model.max_node_vars_non_fused(),
                model.max_node_params() + model.max_node_vars_fused(),
            )
        };
        // for now the number of instances corresponds to the number of graph / model outputs
        let mut num_instances = 0;
        let mut instance_shapes = vec![];
        if model.visibility.input.is_public() {
            num_instances += model.num_inputs();
            instance_shapes.extend(model.input_shapes());
        }
        if model.visibility.output.is_public() {
            num_instances += model.num_outputs();
            instance_shapes.extend(model.output_shapes());
        }
        let mut vars = ModelVars::new(
            cs,
            model.logrows as usize,
            model.max_rotations,
            (num_advice, row_cap),
            (num_fixed, row_cap),
            (num_instances, instance_shapes),
        );
        info!("row cap: {:?}", row_cap);
        info!(
            "number of advices used: {:?}",
            vars.advices.iter().map(|a| a.num_cols()).sum::<usize>()
        );
        info!(
            "number of fixed used: {:?}",
            vars.fixed.iter().map(|a| a.num_cols()).sum::<usize>()
        );
        info!("number of instances used: {:?}", num_instances);
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
            .map(|i| ValTensor::from(<Tensor<i32> as Into<Tensor<Value<F>>>>::into(i.clone())))
            .collect::<Vec<ValTensor<F>>>();
        trace!("Setting output in synthesize");
        config
            .model
            .layout(config.clone(), &mut layouter, &inputs, &config.vars)
            .unwrap();

        Ok(())
    }
}
