use std::fmt::Display;

use crate::tensor::TensorType;
use crate::tensor::{ValTensor, VarTensor};
use crate::RunArgs;
use halo2_proofs::plonk::{Column, ConstraintSystem, Instance};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::debug;
#[cfg(feature = "python-bindings")]
use pyo3::{
    exceptions::PyValueError, FromPyObject, IntoPy, PyObject, PyResult, Python, ToPyObject,
};
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToFlags;

use self::errors::GraphError;
use super::*;

/// Defines the visibility level of values within the zero-knowledge circuit
/// Controls how values are handled during proof generation and verification
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Visibility {
    /// Value is private to the prover and not included in proof
    #[default]
    Private,
    /// Value is public and included in proof for verification
    Public,
    /// Value is hashed and the hash is included in proof
    Hashed {
        /// Controls how the hash is handled in proof
        /// true - hash is included directly in proof (public)
        /// false - hash is used as advice and passed to computational graph
        hash_is_public: bool,
        /// Specifies which outputs this hash affects
        outlets: Vec<usize>,
    },
    /// Value is committed using KZG commitment scheme
    KZGCommit,
    /// Value is assigned as a constant in the circuit
    Fixed,
}

impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Visibility::KZGCommit => write!(f, "polycommit"),
            Visibility::Private => write!(f, "private"),
            Visibility::Public => write!(f, "public"),
            Visibility::Fixed => write!(f, "fixed"),
            Visibility::Hashed {
                hash_is_public,
                outlets,
            } => {
                if *hash_is_public {
                    write!(f, "hashed/public")
                } else {
                    write!(f, "hashed/private/{}", outlets.iter().join(","))
                }
            }
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for Visibility {
    /// Converts visibility to command line flags
    fn to_flags(&self) -> Vec<String> {
        vec![format!("{}", self)]
    }
}

impl<'a> From<&'a str> for Visibility {
    /// Converts string representation to Visibility
    fn from(s: &'a str) -> Self {
        if s.contains("hashed/private") {
            // Split on last occurrence of '/'
            let (_, outlets) = s.split_at(s.rfind('/').unwrap());
            let outlets = outlets
                .trim_start_matches('/')
                .split(',')
                .map(|s| s.parse::<usize>().unwrap())
                .collect_vec();

            return Visibility::Hashed {
                hash_is_public: false,
                outlets,
            };
        }
        match s {
            "private" => Visibility::Private,
            "public" => Visibility::Public,
            "polycommit" => Visibility::KZGCommit,
            "fixed" => Visibility::Fixed,
            "hashed" | "hashed/public" => Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            },
            _ => {
                log::error!("Invalid value for Visibility: {}", s);
                log::warn!("Defaulting to private");
                Visibility::Private
            }
        }
    }
}

#[cfg(feature = "python-bindings")]
impl IntoPy<PyObject> for Visibility {
    /// Converts Visibility to Python object
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Visibility::Private => "private".to_object(py),
            Visibility::Public => "public".to_object(py),
            Visibility::Fixed => "fixed".to_object(py),
            Visibility::KZGCommit => "polycommit".to_object(py),
            Visibility::Hashed {
                hash_is_public,
                outlets,
            } => {
                if hash_is_public {
                    "hashed/public".to_object(py)
                } else {
                    let outlets = outlets
                        .iter()
                        .map(|o| o.to_string())
                        .collect_vec()
                        .join(",");
                    format!("hashed/private/{}", outlets).to_object(py)
                }
            }
        }
    }
}

#[cfg(feature = "python-bindings")]
impl<'source> FromPyObject<'source> for Visibility {
    /// Extracts Visibility from Python object
    fn extract_bound(ob: &pyo3::Bound<'source, pyo3::PyAny>) -> PyResult<Self> {
        let strval = String::extract_bound(ob)?;
        let strval = strval.as_str();

        if strval.contains("hashed/private") {
            let (_, outlets) = strval.split_at(strval.rfind('/').unwrap());
            let outlets = outlets
                .trim_start_matches('/')
                .split(',')
                .map(|s| s.parse::<usize>().unwrap())
                .collect_vec();

            return Ok(Visibility::Hashed {
                hash_is_public: false,
                outlets,
            });
        }

        match strval.to_lowercase().as_str() {
            "private" => Ok(Visibility::Private),
            "public" => Ok(Visibility::Public),
            "polycommit" => Ok(Visibility::KZGCommit),
            "hashed" => Ok(Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            }),
            "hashed/public" => Ok(Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            }),
            "fixed" => Ok(Visibility::Fixed),
            _ => Err(PyValueError::new_err("Invalid value for Visibility")),
        }
    }
}

impl Visibility {
    /// Returns true if visibility is Fixed
    pub fn is_fixed(&self) -> bool {
        matches!(&self, Visibility::Fixed)
    }

    /// Returns true if visibility is Private or hashed private
    pub fn is_private(&self) -> bool {
        matches!(&self, Visibility::Private) || self.is_hashed_private()
    }

    /// Returns true if visibility is Public
    pub fn is_public(&self) -> bool {
        matches!(&self, Visibility::Public)
    }

    /// Returns true if visibility involves hashing
    pub fn is_hashed(&self) -> bool {
        matches!(&self, Visibility::Hashed { .. })
    }

    /// Returns true if visibility uses KZG commitment
    pub fn is_polycommit(&self) -> bool {
        matches!(&self, Visibility::KZGCommit)
    }

    /// Returns true if visibility is hashed with public hash
    pub fn is_hashed_public(&self) -> bool {
        if let Visibility::Hashed {
            hash_is_public: true,
            ..
        } = self
        {
            return true;
        }
        false
    }

    /// Returns true if visibility is hashed with private hash
    pub fn is_hashed_private(&self) -> bool {
        if let Visibility::Hashed {
            hash_is_public: false,
            ..
        } = self
        {
            return true;
        }
        false
    }

    /// Returns true if visibility requires additional processing
    pub fn requires_processing(&self) -> bool {
        matches!(&self, Visibility::Hashed { .. }) | matches!(&self, Visibility::KZGCommit)
    }

    /// Returns vector of output indices that this visibility setting affects
    pub fn overwrites_inputs(&self) -> Vec<usize> {
        if let Visibility::Hashed { outlets, .. } = self {
            return outlets.clone();
        }
        vec![]
    }
}

/// Manages scaling factors for different parts of the model
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct VarScales {
    /// Scale factor for input values
    pub input: crate::Scale,
    /// Scale factor for parameter values
    pub params: crate::Scale,
    /// Multiplier for scale rebasing
    pub rebase_multiplier: u32,
}

impl std::fmt::Display for VarScales {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(inputs: {}, params: {})", self.input, self.params)
    }
}

impl VarScales {
    /// Returns maximum scale value
    pub fn get_max(&self) -> crate::Scale {
        std::cmp::max(self.input, self.params)
    }

    /// Returns minimum scale value
    pub fn get_min(&self) -> crate::Scale {
        std::cmp::min(self.input, self.params)
    }

    /// Creates VarScales from runtime arguments
    pub fn from_args(args: &RunArgs) -> Self {
        Self {
            input: args.input_scale,
            params: args.param_scale,
            rebase_multiplier: args.scale_rebase_multiplier,
        }
    }
}

/// Controls visibility settings for different parts of the model
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct VarVisibility {
    /// Visibility of model inputs
    pub input: Visibility,
    /// Visibility of model parameters (weights, biases)
    pub params: Visibility,
    /// Visibility of model outputs
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

impl Default for VarVisibility {
    fn default() -> Self {
        Self {
            input: Visibility::Private,
            params: Visibility::Private,
            output: Visibility::Public,
        }
    }
}

impl VarVisibility {
    /// Creates visibility settings from runtime arguments
    pub fn from_args(args: &RunArgs) -> Result<Self, GraphError> {
        let input_vis = &args.input_visibility;
        let params_vis = &args.param_visibility;
        let output_vis = &args.output_visibility;

        if params_vis.is_public() {
            return Err(GraphError::ParamsPublicVisibility);
        }

        if !output_vis.is_public()
            && !params_vis.is_public()
            && !input_vis.is_public()
            && !output_vis.is_fixed()
            && !params_vis.is_fixed()
            && !input_vis.is_fixed()
            && !output_vis.is_hashed()
            && !params_vis.is_hashed()
            && !input_vis.is_hashed()
            && !output_vis.is_polycommit()
            && !params_vis.is_polycommit()
            && !input_vis.is_polycommit()
        {
            return Err(GraphError::Visibility);
        }
        Ok(Self {
            input: input_vis.clone(),
            params: params_vis.clone(),
            output: output_vis.clone(),
        })
    }
}

/// Container for circuit columns used by a model
#[derive(Clone, Debug)]
pub struct ModelVars<F: PrimeField + TensorType + PartialOrd> {
    /// Advice columns for circuit assignments
    pub advices: Vec<VarTensor>,
    /// Optional instance column for public inputs
    pub instance: Option<ValTensor<F>>,
}

impl<F: PrimeField + TensorType + PartialOrd + std::hash::Hash> ModelVars<F> {
    /// Gets reference to instance column if it exists
    pub fn get_instance_col(&self) -> Option<&Column<Instance>> {
        if let Some(instance) = &self.instance {
            match instance {
                ValTensor::Instance { inner, .. } => Some(inner),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Sets initial offset for instance values
    pub fn set_initial_instance_offset(&mut self, offset: usize) {
        if let Some(instance) = &mut self.instance {
            instance.set_initial_instance_offset(offset);
        }
    }

    /// Gets total length of instance data
    pub fn get_instance_len(&self) -> usize {
        if let Some(instance) = &self.instance {
            instance.get_total_instance_len()
        } else {
            0
        }
    }

    /// Increments instance index
    pub fn increment_instance_idx(&mut self) {
        if let Some(instance) = &mut self.instance {
            instance.increment_idx();
        }
    }

    /// Sets instance index to specific value
    pub fn set_instance_idx(&mut self, val: usize) {
        if let Some(instance) = &mut self.instance {
            instance.set_idx(val);
        }
    }

    /// Gets current instance index
    pub fn get_instance_idx(&self) -> usize {
        if let Some(instance) = &self.instance {
            instance.get_idx()
        } else {
            0
        }
    }

    /// Initializes instance column with specified dimensions and scale
    pub fn instantiate_instance(
        &mut self,
        cs: &mut ConstraintSystem<F>,
        instance_dims: Vec<Vec<usize>>,
        instance_scale: crate::Scale,
        existing_instance: Option<Column<Instance>>,
    ) {
        debug!("model uses {:?} instance dims", instance_dims);
        self.instance = if let Some(existing_instance) = existing_instance {
            debug!("using existing instance");
            Some(ValTensor::new_instance_from_col(
                instance_dims,
                instance_scale,
                existing_instance,
            ))
        } else {
            Some(ValTensor::new_instance(cs, instance_dims, instance_scale))
        };
    }

    /// Creates new ModelVars with allocated columns based on settings
    pub fn new(cs: &mut ConstraintSystem<F>, params: &GraphSettings) -> Self {
        debug!("number of blinding factors: {}", cs.blinding_factors());

        let logrows = params.run_args.logrows as usize;
        let var_len = params.total_assignments;
        let num_inner_cols = params.run_args.num_inner_cols;
        let num_constants = params.total_const_size;
        let module_requires_fixed = params.module_requires_fixed();
        let requires_dynamic_lookup = params.requires_dynamic_lookup();
        let requires_shuffle = params.requires_shuffle();
        let dynamic_lookup_and_shuffle_size = params.dynamic_lookup_and_shuffle_col_size();

        let mut advices = (0..3)
            .map(|_| VarTensor::new_advice(cs, logrows, num_inner_cols, var_len))
            .collect_vec();

        if requires_dynamic_lookup || requires_shuffle {
            let num_cols = 3;
            for _ in 0..num_cols {
                let dynamic_lookup =
                    VarTensor::new_advice(cs, logrows, 1, dynamic_lookup_and_shuffle_size);
                if dynamic_lookup.num_blocks() > 1 {
                    warn!("dynamic lookup has {} blocks", dynamic_lookup.num_blocks());
                };
                advices.push(dynamic_lookup);
            }
        }

        debug!(
            "model uses {} advice blocks (size={})",
            advices.iter().map(|v| v.num_blocks()).sum::<usize>(),
            num_inner_cols
        );

        let num_const_cols =
            VarTensor::constant_cols(cs, logrows, num_constants, module_requires_fixed);
        debug!("model uses {} fixed columns", num_const_cols);

        ModelVars {
            advices,
            instance: None,
        }
    }

    /// Allocate all columns that will be assigned to by a model.
    pub fn new_dummy() -> Self {
        ModelVars {
            advices: vec![],
            instance: None,
        }
    }
}
