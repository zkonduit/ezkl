use std::error::Error;

use crate::commands::RunArgs;
use crate::tensor::TensorType;
use crate::tensor::{ValTensor, VarTensor};
use halo2_proofs::plonk::ConstraintSystem;
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::*;

/// Label enum to track whether model input, model parameters, and model output are public or private
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Visibility {
    /// Mark an item as private to the prover (not in the proof submitted for verification)
    #[default]
    Private,
    /// Mark an item as public (sent in the proof submitted for verification)
    Public,
}
impl Visibility {
    #[allow(missing_docs)]
    pub fn is_public(&self) -> bool {
        matches!(&self, Visibility::Public)
    }
}
impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Visibility::Private => write!(f, "private"),
            Visibility::Public => write!(f, "public"),
        }
    }
}

/// Represents whether the model input, model parameters, and model output are Public or Private to the prover.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct VarVisibility {
    /// Input to the model or computational graph
    pub input: Visibility,
    /// Parameters, such as weights and biases, in the model
    pub params: Visibility,
    /// Output of the model or computational graph
    pub output: Visibility,
}
impl std::fmt::Display for VarVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "(inputs: {}, params: {}, outputs: {})",
            self.input, self.params, self.output
        )
    }
}

impl VarVisibility {
    /// Read from cli args whether the model input, model parameters, and model output are Public or Private to the prover.
    /// Place in [VarVisibility] struct.
    pub fn from_args(args: RunArgs) -> Result<Self, Box<dyn Error>> {
        let input_vis = if args.public_inputs {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let params_vis = if args.public_params {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let output_vis = if args.public_outputs {
            Visibility::Public
        } else {
            Visibility::Private
        };
        if !output_vis.is_public() & !params_vis.is_public() & !input_vis.is_public() {
            return Err(Box::new(GraphError::Visibility));
        }
        Ok(Self {
            input: input_vis,
            params: params_vis,
            output: output_vis,
        })
    }
}

/// A wrapper for holding all columns that will be assigned to by a model.
#[derive(Clone, Debug)]
pub struct ModelVars<F: PrimeField + TensorType + PartialOrd> {
    #[allow(missing_docs)]
    pub advices: Vec<VarTensor>,
    #[allow(missing_docs)]
    pub fixed: Vec<VarTensor>,
    #[allow(missing_docs)]
    pub instances: Vec<ValTensor<F>>,
}

impl<F: PrimeField + TensorType + PartialOrd> ModelVars<F> {
    /// Allocate all columns that will be assigned to by a model.
    pub fn new(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        var_len: usize,
        instance_dims: Vec<Vec<usize>>,
        visibility: VarVisibility,
        scale: u32,
    ) -> Self {
        let advices = (0..3)
            .map(|_| VarTensor::new_advice(cs, logrows, var_len))
            .collect_vec();
        let mut fixed = vec![];
        if visibility.params == Visibility::Public {
            fixed = (0..1)
                .map(|_| VarTensor::new_fixed(cs, logrows, var_len))
                .collect_vec();
        }
        // will be empty if instances dims has len 0
        let instances = (0..instance_dims.len())
            .map(|i| ValTensor::new_instance(cs, instance_dims[i].clone(), scale))
            .collect_vec();
        ModelVars {
            advices,
            fixed,
            instances,
        }
    }
}
