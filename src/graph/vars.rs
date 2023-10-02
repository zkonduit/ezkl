use std::error::Error;

use crate::tensor::TensorType;
use crate::tensor::{ValTensor, VarTensor};
use crate::RunArgs;
use halo2_proofs::plonk::{Column, ConstraintSystem, Instance};
use halo2curves::ff::PrimeField;
use itertools::Itertools;
use log::debug;
#[cfg(feature = "python-bindings")]
use pyo3::{
    exceptions::PyValueError, types::PyString, FromPyObject, IntoPy, PyAny, PyObject, PyResult,
    PyTryFrom, Python, ToPyObject,
};

use serde::{Deserialize, Serialize};

use super::*;

/// Label enum to track whether model input, model parameters, and model output are public, private, or hashed
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Visibility {
    /// Mark an item as private to the prover (not in the proof submitted for verification)
    #[default]
    Private,
    /// Mark an item as public (sent in the proof submitted for verification)
    Public,
    /// Mark an item as publicly committed to (hash sent in the proof submitted for verification)
    Hashed {
        /// Whether the hash is used as an instance (sent in the proof submitted for verification)
        /// if false the hash is used as an advice (not in the proof submitted for verification) and is then sent to the computational graph
        /// if true the hash is used as an instance (sent in the proof submitted for verification) the *inputs* to the hashing function are then sent to the computational graph
        hash_is_public: bool,
        ///
        outlets: Vec<usize>,
    },
    /// Mark an item as encrypted (public key and encrypted message sent in the proof submitted for verificatio)
    Encrypted,
    /// assigned as a constant in the circuit
    Fixed,
}

impl<'a> From<&'a str> for Visibility {
    fn from(s: &'a str) -> Self {
        if s.contains("hashed/private") {
            // split on last occurence of '/'
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
            "fixed" => Visibility::Fixed,
            "hashed" | "hashed/public" => Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            },
            "encrypted" => Visibility::Encrypted,
            _ => panic!("Invalid visibility string"),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Converts Visibility into a PyObject (Required for Visibility to be compatible with Python)
impl IntoPy<PyObject> for Visibility {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Visibility::Private => "private".to_object(py),
            Visibility::Public => "public".to_object(py),
            Visibility::Fixed => "fixed".to_object(py),
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
            Visibility::Encrypted => "encrypted".to_object(py),
        }
    }
}

#[cfg(feature = "python-bindings")]
/// Obtains Visibility from PyObject (Required for Visibility to be compatible with Python)
impl<'source> FromPyObject<'source> for Visibility {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let trystr = <PyString as PyTryFrom>::try_from(ob)?;
        let strval = trystr.to_string();

        let strval = strval.as_str();

        if strval.contains("hashed/private") {
            // split on last occurence of '/'
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
            "hashed" => Ok(Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            }),
            "hashed/public" => Ok(Visibility::Hashed {
                hash_is_public: true,
                outlets: vec![],
            }),
            "fixed" => Ok(Visibility::Fixed),
            "encrypted" => Ok(Visibility::Encrypted),
            _ => Err(PyValueError::new_err("Invalid value for Visibility")),
        }
    }
}

impl Visibility {
    #[allow(missing_docs)]
    pub fn is_fixed(&self) -> bool {
        matches!(&self, Visibility::Fixed)
    }
    #[allow(missing_docs)]
    pub fn is_public(&self) -> bool {
        matches!(&self, Visibility::Public)
    }
    #[allow(missing_docs)]
    pub fn is_hashed(&self) -> bool {
        matches!(&self, Visibility::Hashed { .. })
    }
    #[allow(missing_docs)]
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
    #[allow(missing_docs)]
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
    #[allow(missing_docs)]
    pub fn is_encrypted(&self) -> bool {
        matches!(&self, Visibility::Encrypted)
    }
    #[allow(missing_docs)]
    pub fn requires_processing(&self) -> bool {
        matches!(&self, Visibility::Encrypted) | matches!(&self, Visibility::Hashed { .. })
    }
    #[allow(missing_docs)]
    pub fn overwrites_inputs(&self) -> Vec<usize> {
        if let Visibility::Hashed { outlets, .. } = self {
            return outlets.clone();
        }
        vec![]
    }
}
impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Visibility::Private => write!(f, "private"),
            Visibility::Public => write!(f, "public"),
            Visibility::Fixed => write!(f, "fixed"),
            Visibility::Hashed { .. } => write!(f, "hashed"),
            Visibility::Encrypted => write!(f, "encrypted"),
        }
    }
}

/// Represents the scale of the model input, model parameters.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, PartialOrd)]
pub struct VarScales {
    ///
    pub input: u32,
    ///
    pub params: u32,
    ///
    pub rebase_multiplier: u32,
}

impl std::fmt::Display for VarScales {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "(inputs: {}, params: {})", self.input, self.params)
    }
}

impl VarScales {
    /// Place in [VarScales] struct.
    pub fn from_args(args: &RunArgs) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            input: args.input_scale,
            params: args.param_scale,
            rebase_multiplier: args.scale_rebase_multiplier,
        })
    }
}

/// Represents whether the model input, model parameters, and model output are Public or Private to the prover.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, PartialOrd)]
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
    pub fn from_args(args: &RunArgs) -> Result<Self, Box<dyn Error>> {
        let input_vis = &args.input_visibility;
        let params_vis = &args.param_visibility;
        let output_vis = &args.output_visibility;

        if !output_vis.is_public()
            & !params_vis.is_public()
            & !input_vis.is_public()
            & !output_vis.is_fixed()
            & !params_vis.is_fixed()
            & !input_vis.is_fixed()
            & !output_vis.is_hashed()
            & !params_vis.is_hashed()
            & !input_vis.is_hashed()
            & !output_vis.is_encrypted()
            & !params_vis.is_encrypted()
            & !input_vis.is_encrypted()
        {
            return Err(Box::new(GraphError::Visibility));
        }
        Ok(Self {
            input: input_vis.clone(),
            params: params_vis.clone(),
            output: output_vis.clone(),
        })
    }
}

/// A wrapper for holding all columns that will be assigned to by a model.
#[derive(Clone, Debug)]
pub struct ModelVars<F: PrimeField + TensorType + PartialOrd> {
    #[allow(missing_docs)]
    pub advices: Vec<VarTensor>,
    #[allow(missing_docs)]
    pub instance: ValTensor<F>,
}

impl<F: PrimeField + TensorType + PartialOrd> ModelVars<F> {
    /// Set the initial instance offset
    pub fn set_initial_instance_offset(&mut self, offset: usize) {
        self.instance.set_initial_instance_offset(offset);
    }

    /// Increment the instance offset
    pub fn increment_instance_idx(&mut self) {
        self.instance.increment_idx();
    }

    /// Reset the instance offset
    pub fn set_instance_idx(&mut self, val: usize) {
        self.instance.set_idx(val);
    }

    /// Get the instance offset
    pub fn get_instance_idx(&self) -> usize {
        self.instance.get_idx()
    }

    /// Allocate all columns that will be assigned to by a model.
    pub fn new(
        cs: &mut ConstraintSystem<F>,
        logrows: usize,
        var_len: usize,
        num_constants: usize,
        instance_dims: Vec<Vec<usize>>,
        instance_scale: u32,
        uses_modules: bool,
        existing_instance: Option<Column<Instance>>,
    ) -> Self {
        info!("number of blinding factors: {}", cs.blinding_factors());

        let advices = (0..3)
            .map(|_| VarTensor::new_advice(cs, logrows, var_len))
            .collect_vec();

        debug!(
            "model uses {} advice columns",
            advices.iter().map(|v| v.num_cols()).sum::<usize>()
        );

        let instance = if let Some(existing_instance) = existing_instance {
            debug!("using existing instance");
            ValTensor::new_instance_from_col(instance_dims, instance_scale, existing_instance)
        } else {
            ValTensor::new_instance(cs, instance_dims, instance_scale)
        };

        let num_const_cols = VarTensor::constant_cols(cs, logrows, num_constants, uses_modules);
        debug!("model uses {} fixed columns", num_const_cols);

        ModelVars { advices, instance }
    }

    /// Allocate all columns that will be assigned to by a model.
    pub fn new_dummy() -> Self {
        ModelVars {
            advices: vec![],
            instance: ValTensor::new_dummy(),
        }
    }
}
