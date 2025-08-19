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

/// errors for the graph
pub mod errors;

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use colored_json::ToColoredJson;
#[cfg(all(not(not(feature = "ezkl")), unix))]
use gag::Gag;
use halo2_proofs::plonk::VerifyingKey;
use halo2_proofs::poly::commitment::CommitmentScheme;
pub use input::DataSource;
use itertools::Itertools;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tosubcommand::ToFlags;

#[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
use self::input::GraphData;

use self::errors::GraphError;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use self::input::GraphData;
use self::modules::{GraphModules, ModuleConfigs, ModuleForwardResult, ModuleSizes};
use crate::circuit::lookup::LookupOp;
use crate::circuit::modules::ModulePlanner;
use crate::circuit::region::{ConstantsMap, RegionSettings};
use crate::circuit::table::{num_cols_required, Range, Table, RESERVED_BLINDING_ROWS_PAD};
use crate::circuit::{CheckMode, InputType};
use crate::fieldutils::{felt_to_f64, IntegerRep};
use crate::pfsys::PrettyElements;
use crate::tensor::{Tensor, ValTensor};
use crate::{RunArgs, EZKL_BUF_CAPACITY};

use halo2_proofs::{
    circuit::Layouter,
    plonk::{Circuit, ConstraintSystem, Error as PlonkError},
};
use halo2curves::bn256::{self, Fr as Fp, G1Affine};
use halo2curves::ff::{Field, PrimeField};
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use lazy_static::lazy_static;
use log::{debug, error, trace, warn};
use maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
pub use model::*;
pub use node::*;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDictMethods;
#[cfg(feature = "python-bindings")]
use pyo3::IntoPyObject;

use serde::{Deserialize, Serialize};
use std::ops::Deref;
pub use utilities::*;
pub use vars::*;

#[cfg(feature = "python-bindings")]
use crate::pfsys::field_to_string;

/// The safety factor for the range of the lookup table.
pub const RANGE_MULTIPLIER: IntegerRep = 2;

/// The maximum number of columns in a lookup table.
pub const MAX_NUM_LOOKUP_COLS: usize = 12;

/// Max representation of a lookup table input
pub const MAX_LOOKUP_ABS: IntegerRep =
    (MAX_NUM_LOOKUP_COLS as IntegerRep) * 2_i128.pow(MAX_PUBLIC_SRS);

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
lazy_static! {
    /// Max circuit area
    pub static ref EZKL_MAX_CIRCUIT_AREA: Option<usize> =
        if let Ok(max_circuit_area) = std::env::var("EZKL_MAX_CIRCUIT_AREA") {
            Some(max_circuit_area.parse().unwrap_or(0))
        } else {
            None
        };
}

#[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
const EZKL_MAX_CIRCUIT_AREA: Option<usize> = None;

///
pub const ASSUMED_BLINDING_FACTORS: usize = 5;
/// The minimum number of rows in the grid
pub const MIN_LOGROWS: u32 = 6;

/// 26
pub const MAX_PUBLIC_SRS: u32 = bn256::Fr::S - 2;

///
pub const RESERVED_BLINDING_ROWS: usize = ASSUMED_BLINDING_FACTORS + RESERVED_BLINDING_ROWS_PAD;

use std::cell::RefCell;

thread_local!(
    /// This is a global variable that holds the settings for the graph
    /// This is used to pass settings to the layouter and other parts of the circuit without needing to heavily modify the Halo2 API in a new fork
    pub static GLOBAL_SETTINGS: RefCell<Option<GraphSettings>> = const { RefCell::new(None) }
);

/// Result from a forward pass
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GraphWitness {
    /// The inputs of the forward pass
    pub inputs: Vec<Vec<Fp>>,
    /// The prettified outputs of the forward pass, we use a String to maximize compatibility with Python and JS clients
    pub pretty_elements: Option<PrettyElements>,
    /// The output of the forward pass
    pub outputs: Vec<Vec<Fp>>,
    /// Any hashes of inputs generated during the forward pass
    pub processed_inputs: Option<ModuleForwardResult>,
    /// Any hashes of params generated during the forward pass
    pub processed_params: Option<ModuleForwardResult>,
    /// Any hashes of outputs generated during the forward pass
    pub processed_outputs: Option<ModuleForwardResult>,
    /// max lookup input
    pub max_lookup_inputs: IntegerRep,
    /// max lookup input
    pub min_lookup_inputs: IntegerRep,
    /// max range check size
    pub max_range_size: IntegerRep,
    /// (optional) version of ezkl used
    pub version: Option<String>,
}

impl GraphWitness {
    ///
    pub fn get_float_outputs(&self, scales: &[crate::Scale]) -> Vec<Tensor<f32>> {
        self.outputs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                x.iter()
                    .map(|y| (felt_to_f64(*y) / scale_to_multiplier(scales[i])) as f32)
                    .collect::<Tensor<f32>>()
            })
            .collect()
    }

    ///
    pub fn new(inputs: Vec<Vec<Fp>>, outputs: Vec<Vec<Fp>>) -> Self {
        GraphWitness {
            inputs,
            outputs,
            pretty_elements: None,
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
            max_lookup_inputs: 0,
            min_lookup_inputs: 0,
            max_range_size: 0,
            version: None,
        }
    }

    /// Generate the rescaled elements for the witness
    pub fn generate_rescaled_elements(
        &mut self,
        input_scales: Vec<crate::Scale>,
        output_scales: Vec<crate::Scale>,
        visibility: VarVisibility,
    ) {
        let mut pretty_elements = PrettyElements {
            rescaled_inputs: self
                .inputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let scale = input_scales[i];
                    t.iter()
                        .map(|x| dequantize(*x, scale, 0.).to_string())
                        .collect()
                })
                .collect(),
            inputs: self
                .inputs
                .iter()
                .map(|t| t.iter().map(|x| format!("{:?}", x)).collect())
                .collect(),
            rescaled_outputs: self
                .outputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let scale = output_scales[i];
                    t.iter()
                        .map(|x| dequantize(*x, scale, 0.).to_string())
                        .collect()
                })
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|t| t.iter().map(|x| format!("{:?}", x)).collect())
                .collect(),
            ..Default::default()
        };

        if let Some(processed_inputs) = self.processed_inputs.clone() {
            pretty_elements.processed_inputs = processed_inputs
                .get_result(visibility.input)
                .iter()
                // gets printed as hex string
                .map(|x| x.iter().map(|y| format!("{:?}", y)).collect())
                .collect();
        }

        if let Some(processed_params) = self.processed_params.clone() {
            pretty_elements.processed_params = processed_params
                .get_result(visibility.params)
                .iter()
                // gets printed as hex string
                .map(|x| x.iter().map(|y| format!("{:?}", y)).collect())
                .collect();
        }

        if let Some(processed_outputs) = self.processed_outputs.clone() {
            pretty_elements.processed_outputs = processed_outputs
                .get_result(visibility.output)
                .iter()
                // gets printed as hex string
                .map(|x| x.iter().map(|y| format!("{:?}", y)).collect())
                .collect();
        }

        self.pretty_elements = Some(pretty_elements);
    }

    ///
    pub fn get_polycommitments(&self) -> Vec<G1Affine> {
        let mut commitments = vec![];
        if let Some(processed_inputs) = &self.processed_inputs {
            if let Some(commits) = &processed_inputs.polycommit {
                commitments.extend(commits.iter().flatten());
            }
        }
        if let Some(processed_params) = &self.processed_params {
            if let Some(commits) = &processed_params.polycommit {
                commitments.extend(commits.iter().flatten());
            }
        }
        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(commits) = &processed_outputs.polycommit {
                commitments.extend(commits.iter().flatten());
            }
        }
        commitments
    }

    /// Export the ezkl witness as json
    pub fn as_json(&self) -> Result<String, GraphError> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => return Err(e.into()),
        };
        Ok(serialized)
    }

    /// Load the model input from a file
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, GraphError> {
        let file = std::fs::File::open(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;

        let reader = std::io::BufReader::with_capacity(*EZKL_BUF_CAPACITY, file);
        let witness: GraphWitness =
            serde_json::from_reader(reader).map_err(Into::<GraphError>::into)?;

        // check versions match
        crate::check_version_string_matches(witness.version.as_deref().unwrap_or(""));

        Ok(witness)
    }

    /// Save the model input to a file
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), GraphError> {
        let file = std::fs::File::create(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        // use buf writer
        let writer = std::io::BufWriter::with_capacity(*EZKL_BUF_CAPACITY, file);

        serde_json::to_writer(writer, &self).map_err(|e| e.into())
    }

    ///
    pub fn get_input_tensor(&self) -> Vec<Tensor<Fp>> {
        self.inputs
            .clone()
            .into_iter()
            .map(|i| Tensor::from(i.into_iter()))
            .collect::<Vec<Tensor<Fp>>>()
    }

    ///
    pub fn get_output_tensor(&self) -> Vec<Tensor<Fp>> {
        self.outputs
            .clone()
            .into_iter()
            .map(|i| Tensor::from(i.into_iter()))
            .collect::<Vec<Tensor<Fp>>>()
    }
}

#[cfg(feature = "python-bindings")]
impl<'py> IntoPyObject<'py> for GraphWitness {
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let dict_inputs = PyDict::new(py);
        let dict_params = PyDict::new(py);
        let dict_outputs = PyDict::new(py);

        let inputs: Vec<Vec<String>> = self
            .inputs
            .iter()
            .map(|x| x.iter().map(field_to_string).collect())
            .collect();

        let outputs: Vec<Vec<String>> = self
            .outputs
            .iter()
            .map(|x| x.iter().map(field_to_string).collect())
            .collect();

        dict.set_item("inputs", inputs).unwrap();
        dict.set_item("outputs", outputs).unwrap();
        dict.set_item("max_lookup_inputs", self.max_lookup_inputs)
            .unwrap();
        dict.set_item("min_lookup_inputs", self.min_lookup_inputs)
            .unwrap();
        dict.set_item("max_range_size", self.max_range_size)
            .unwrap();

        if let Some(processed_inputs) = &self.processed_inputs {
            //poseidon_hash
            if let Some(processed_inputs_poseidon_hash) = &processed_inputs.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_inputs, processed_inputs_poseidon_hash).unwrap();
            }
            if let Some(processed_inputs_polycommit) = &processed_inputs.polycommit {
                insert_polycommit_pydict(&dict_inputs, processed_inputs_polycommit).unwrap();
            }

            dict.set_item("processed_inputs", dict_inputs).unwrap();
        }

        if let Some(processed_params) = &self.processed_params {
            if let Some(processed_params_poseidon_hash) = &processed_params.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_params, processed_params_poseidon_hash).unwrap();
            }
            if let Some(processed_params_polycommit) = &processed_params.polycommit {
                insert_polycommit_pydict(&dict_params, processed_params_polycommit).unwrap();
            }

            dict.set_item("processed_params", dict_params).unwrap();
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(processed_outputs_poseidon_hash) = &processed_outputs.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_outputs, processed_outputs_poseidon_hash)
                    .unwrap();
            }
            if let Some(processed_outputs_polycommit) = &processed_outputs.polycommit {
                insert_polycommit_pydict(&dict_outputs, processed_outputs_polycommit).unwrap();
            }

            dict.set_item("processed_outputs", dict_outputs).unwrap();
        }

        Ok(dict.into_any())
    }
}

#[cfg(feature = "python-bindings")]
fn insert_poseidon_hash_pydict(
    pydict: &Bound<'_, PyDict>,
    poseidon_hash: &Vec<Fp>,
) -> Result<(), PyErr> {
    let poseidon_hash: Vec<String> = poseidon_hash.iter().map(field_to_string).collect();
    pydict.set_item("poseidon_hash", poseidon_hash)?;

    Ok(())
}

#[cfg(feature = "python-bindings")]
fn insert_polycommit_pydict(
    pydict: &Bound<'_, PyDict>,
    commits: &Vec<Vec<G1Affine>>,
) -> Result<(), PyErr> {
    use crate::bindings::python::PyG1Affine;
    let poseidon_hash: Vec<Vec<PyG1Affine>> = commits
        .iter()
        .map(|c| c.iter().map(|x| PyG1Affine::from(*x)).collect())
        .collect();
    pydict.set_item("polycommit", poseidon_hash)?;

    Ok(())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// Parameters for dynamic lookups
/// serde should flatten this struct
pub struct DynamicLookupParams {
    /// total dynamic column size
    pub total_dynamic_col_size: usize,
    /// max dynamic column input length
    pub max_dynamic_input_len: usize,
    /// number of dynamic lookups
    pub num_dynamic_lookups: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
/// Parameters for shuffle operations
pub struct ShuffleParams {
    /// number of shuffles
    pub num_shuffles: usize,
    /// total shuffle column size
    pub total_shuffle_col_size: usize,
}

/// model parameters
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphSettings {
    /// run args
    pub run_args: RunArgs,
    /// the potential number of rows used by the circuit
    pub num_rows: usize,
    /// total linear coordinate of assignments
    pub total_assignments: usize,
    /// total const size
    pub total_const_size: usize,
    /// dynamic lookup parameters, flattened for backwards compatibility, serialize and deserialize flattened for backwards compatibility
    pub dynamic_lookup_params: DynamicLookupParams,
    /// shuffle parameters, flattened for backwards compatibility
    pub shuffle_params: ShuffleParams,
    /// the shape of public inputs to the model (in order of appearance)
    pub model_instance_shapes: Vec<Vec<usize>>,
    /// model output scales
    pub model_output_scales: Vec<crate::Scale>,
    /// model input scales
    pub model_input_scales: Vec<crate::Scale>,
    /// the of instance cells used by modules
    pub module_sizes: ModuleSizes,
    /// required_lookups
    pub required_lookups: Vec<LookupOp>,
    /// required range_checks
    pub required_range_checks: Vec<Range>,
    /// check mode
    pub check_mode: CheckMode,
    /// ezkl version used
    pub version: String,
    /// num blinding factors
    pub num_blinding_factors: Option<usize>,
    /// unix time timestamp
    pub timestamp: Option<u128>,
    /// Model inputs types (if any)
    pub input_types: Option<Vec<InputType>>,
    /// Model outputs types (if any)
    pub output_types: Option<Vec<InputType>>,
}

impl Serialize for GraphSettings {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            // JSON format - use flattened fields for backwards compatibility
            use serde::ser::SerializeStruct;
            let mut state = serializer.serialize_struct("GraphSettings", 21)?;
            state.serialize_field("run_args", &self.run_args)?;
            state.serialize_field("num_rows", &self.num_rows)?;
            state.serialize_field("total_assignments", &self.total_assignments)?;
            state.serialize_field("total_const_size", &self.total_const_size)?;

            // Flatten DynamicLookupParams fields
            state.serialize_field(
                "total_dynamic_col_size",
                &self.dynamic_lookup_params.total_dynamic_col_size,
            )?;
            state.serialize_field(
                "max_dynamic_input_len",
                &self.dynamic_lookup_params.max_dynamic_input_len,
            )?;
            state.serialize_field(
                "num_dynamic_lookups",
                &self.dynamic_lookup_params.num_dynamic_lookups,
            )?;

            // Flatten ShuffleParams fields
            state.serialize_field("num_shuffles", &self.shuffle_params.num_shuffles)?;
            state.serialize_field(
                "total_shuffle_col_size",
                &self.shuffle_params.total_shuffle_col_size,
            )?;

            state.serialize_field("model_instance_shapes", &self.model_instance_shapes)?;
            state.serialize_field("model_output_scales", &self.model_output_scales)?;
            state.serialize_field("model_input_scales", &self.model_input_scales)?;
            state.serialize_field("module_sizes", &self.module_sizes)?;
            state.serialize_field("required_lookups", &self.required_lookups)?;
            state.serialize_field("required_range_checks", &self.required_range_checks)?;
            state.serialize_field("check_mode", &self.check_mode)?;
            state.serialize_field("version", &self.version)?;
            state.serialize_field("num_blinding_factors", &self.num_blinding_factors)?;
            state.serialize_field("timestamp", &self.timestamp)?;
            state.serialize_field("input_types", &self.input_types)?;
            state.serialize_field("output_types", &self.output_types)?;
            state.end()
        } else {
            // Binary format (bincode) - use nested struct format
            use serde::ser::SerializeTuple;
            let mut state = serializer.serialize_tuple(18)?;
            state.serialize_element(&self.run_args)?;
            state.serialize_element(&self.num_rows)?;
            state.serialize_element(&self.total_assignments)?;
            state.serialize_element(&self.total_const_size)?;
            state.serialize_element(&self.dynamic_lookup_params)?;
            state.serialize_element(&self.shuffle_params)?;
            state.serialize_element(&self.model_instance_shapes)?;
            state.serialize_element(&self.model_output_scales)?;
            state.serialize_element(&self.model_input_scales)?;
            state.serialize_element(&self.module_sizes)?;
            state.serialize_element(&self.required_lookups)?;
            state.serialize_element(&self.required_range_checks)?;
            state.serialize_element(&self.check_mode)?;
            state.serialize_element(&self.version)?;
            state.serialize_element(&self.num_blinding_factors)?;
            state.serialize_element(&self.timestamp)?;
            state.serialize_element(&self.input_types)?;
            state.serialize_element(&self.output_types)?;
            state.end()
        }
    }
}

impl<'de> Deserialize<'de> for GraphSettings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            RunArgs,
            NumRows,
            TotalAssignments,
            TotalConstSize,
            // Flattened DynamicLookupParams fields
            TotalDynamicColSize,
            MaxDynamicInputLen,
            NumDynamicLookups,
            // Flattened ShuffleParams fields
            NumShuffles,
            TotalShuffleColSize,
            ModelInstanceShapes,
            ModelOutputScales,
            ModelInputScales,
            ModuleSizes,
            RequiredLookups,
            RequiredRangeChecks,
            CheckMode,
            Version,
            NumBlindingFactors,
            Timestamp,
            InputTypes,
            OutputTypes,
            // Legacy nested struct fields for backwards compatibility
            DynamicLookupParams,
            ShuffleParams,
        }

        struct GraphSettingsVisitor;

        impl<'de> Visitor<'de> for GraphSettingsVisitor {
            type Value = GraphSettings;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct GraphSettings")
            }

            fn visit_map<V>(self, mut map: V) -> Result<GraphSettings, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut run_args = None;
                let mut num_rows = None;
                let mut total_assignments = None;
                let mut total_const_size = None;
                let mut total_dynamic_col_size = None;
                let mut max_dynamic_input_len = None;
                let mut num_dynamic_lookups = None;
                let mut num_shuffles = None;
                let mut total_shuffle_col_size = None;
                let mut model_instance_shapes = None;
                let mut model_output_scales = None;
                let mut model_input_scales = None;
                let mut module_sizes = None;
                let mut required_lookups = None;
                let mut required_range_checks = None;
                let mut check_mode = None;
                let mut version = None;
                let mut num_blinding_factors = None;
                let mut timestamp = None;
                let mut input_types = None;
                let mut output_types = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::RunArgs => {
                            if run_args.is_some() {
                                return Err(de::Error::duplicate_field("run_args"));
                            }
                            run_args = Some(map.next_value()?);
                        }
                        Field::NumRows => {
                            if num_rows.is_some() {
                                return Err(de::Error::duplicate_field("num_rows"));
                            }
                            num_rows = Some(map.next_value()?);
                        }
                        Field::TotalAssignments => {
                            if total_assignments.is_some() {
                                return Err(de::Error::duplicate_field("total_assignments"));
                            }
                            total_assignments = Some(map.next_value()?);
                        }
                        Field::TotalConstSize => {
                            if total_const_size.is_some() {
                                return Err(de::Error::duplicate_field("total_const_size"));
                            }
                            total_const_size = Some(map.next_value()?);
                        }
                        Field::TotalDynamicColSize => {
                            if total_dynamic_col_size.is_some() {
                                return Err(de::Error::duplicate_field("total_dynamic_col_size"));
                            }
                            total_dynamic_col_size = Some(map.next_value()?);
                        }
                        Field::MaxDynamicInputLen => {
                            if max_dynamic_input_len.is_some() {
                                return Err(de::Error::duplicate_field("max_dynamic_input_len"));
                            }
                            max_dynamic_input_len = Some(map.next_value()?);
                        }
                        Field::NumDynamicLookups => {
                            if num_dynamic_lookups.is_some() {
                                return Err(de::Error::duplicate_field("num_dynamic_lookups"));
                            }
                            num_dynamic_lookups = Some(map.next_value()?);
                        }
                        Field::NumShuffles => {
                            if num_shuffles.is_some() {
                                return Err(de::Error::duplicate_field("num_shuffles"));
                            }
                            num_shuffles = Some(map.next_value()?);
                        }
                        Field::TotalShuffleColSize => {
                            if total_shuffle_col_size.is_some() {
                                return Err(de::Error::duplicate_field("total_shuffle_col_size"));
                            }
                            total_shuffle_col_size = Some(map.next_value()?);
                        }
                        Field::ModelInstanceShapes => {
                            if model_instance_shapes.is_some() {
                                return Err(de::Error::duplicate_field("model_instance_shapes"));
                            }
                            model_instance_shapes = Some(map.next_value()?);
                        }
                        Field::ModelOutputScales => {
                            if model_output_scales.is_some() {
                                return Err(de::Error::duplicate_field("model_output_scales"));
                            }
                            model_output_scales = Some(map.next_value()?);
                        }
                        Field::ModelInputScales => {
                            if model_input_scales.is_some() {
                                return Err(de::Error::duplicate_field("model_input_scales"));
                            }
                            model_input_scales = Some(map.next_value()?);
                        }
                        Field::ModuleSizes => {
                            if module_sizes.is_some() {
                                return Err(de::Error::duplicate_field("module_sizes"));
                            }
                            module_sizes = Some(map.next_value()?);
                        }
                        Field::RequiredLookups => {
                            if required_lookups.is_some() {
                                return Err(de::Error::duplicate_field("required_lookups"));
                            }
                            required_lookups = Some(map.next_value()?);
                        }
                        Field::RequiredRangeChecks => {
                            if required_range_checks.is_some() {
                                return Err(de::Error::duplicate_field("required_range_checks"));
                            }
                            required_range_checks = Some(map.next_value()?);
                        }
                        Field::CheckMode => {
                            if check_mode.is_some() {
                                return Err(de::Error::duplicate_field("check_mode"));
                            }
                            check_mode = Some(map.next_value()?);
                        }
                        Field::Version => {
                            if version.is_some() {
                                return Err(de::Error::duplicate_field("version"));
                            }
                            version = Some(map.next_value()?);
                        }
                        Field::NumBlindingFactors => {
                            if num_blinding_factors.is_some() {
                                return Err(de::Error::duplicate_field("num_blinding_factors"));
                            }
                            num_blinding_factors = map.next_value()?;
                        }
                        Field::Timestamp => {
                            if timestamp.is_some() {
                                return Err(de::Error::duplicate_field("timestamp"));
                            }
                            timestamp = Some(map.next_value()?);
                        }
                        Field::InputTypes => {
                            if input_types.is_some() {
                                return Err(de::Error::duplicate_field("input_types"));
                            }
                            input_types = map.next_value()?;
                        }
                        Field::OutputTypes => {
                            if output_types.is_some() {
                                return Err(de::Error::duplicate_field("output_types"));
                            }
                            output_types = map.next_value()?;
                        }
                        // Handle legacy nested struct fields for backwards compatibility
                        Field::DynamicLookupParams => {
                            let legacy_params: DynamicLookupParams = map.next_value()?;
                            if total_dynamic_col_size.is_none() {
                                total_dynamic_col_size = Some(legacy_params.total_dynamic_col_size);
                            }
                            if max_dynamic_input_len.is_none() {
                                max_dynamic_input_len = Some(legacy_params.max_dynamic_input_len);
                            }
                            if num_dynamic_lookups.is_none() {
                                num_dynamic_lookups = Some(legacy_params.num_dynamic_lookups);
                            }
                        }
                        Field::ShuffleParams => {
                            let legacy_params: ShuffleParams = map.next_value()?;
                            if num_shuffles.is_none() {
                                num_shuffles = Some(legacy_params.num_shuffles);
                            }
                            if total_shuffle_col_size.is_none() {
                                total_shuffle_col_size = Some(legacy_params.total_shuffle_col_size);
                            }
                        }
                    }
                }

                let run_args = run_args.ok_or_else(|| de::Error::missing_field("run_args"))?;
                let num_rows = num_rows.ok_or_else(|| de::Error::missing_field("num_rows"))?;
                let total_assignments = total_assignments
                    .ok_or_else(|| de::Error::missing_field("total_assignments"))?;
                let total_const_size =
                    total_const_size.ok_or_else(|| de::Error::missing_field("total_const_size"))?;
                let model_instance_shapes = model_instance_shapes
                    .ok_or_else(|| de::Error::missing_field("model_instance_shapes"))?;
                let model_output_scales = model_output_scales
                    .ok_or_else(|| de::Error::missing_field("model_output_scales"))?;
                let model_input_scales = model_input_scales
                    .ok_or_else(|| de::Error::missing_field("model_input_scales"))?;
                let module_sizes =
                    module_sizes.ok_or_else(|| de::Error::missing_field("module_sizes"))?;
                let required_lookups =
                    required_lookups.ok_or_else(|| de::Error::missing_field("required_lookups"))?;
                let required_range_checks = required_range_checks
                    .ok_or_else(|| de::Error::missing_field("required_range_checks"))?;
                let check_mode =
                    check_mode.ok_or_else(|| de::Error::missing_field("check_mode"))?;
                let version = version.ok_or_else(|| de::Error::missing_field("version"))?;

                // Build the nested structs from flattened fields, with defaults if missing
                let dynamic_lookup_params = DynamicLookupParams {
                    total_dynamic_col_size: total_dynamic_col_size.unwrap_or_default(),
                    max_dynamic_input_len: max_dynamic_input_len.unwrap_or_default(),
                    num_dynamic_lookups: num_dynamic_lookups.unwrap_or_default(),
                };

                let shuffle_params = ShuffleParams {
                    num_shuffles: num_shuffles.unwrap_or_default(),
                    total_shuffle_col_size: total_shuffle_col_size.unwrap_or_default(),
                };

                Ok(GraphSettings {
                    run_args,
                    num_rows,
                    total_assignments,
                    total_const_size,
                    dynamic_lookup_params,
                    shuffle_params,
                    model_instance_shapes,
                    model_output_scales,
                    model_input_scales,
                    module_sizes,
                    required_lookups,
                    required_range_checks,
                    check_mode,
                    version,
                    num_blinding_factors,
                    timestamp,
                    input_types,
                    output_types,
                })
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<GraphSettings, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                use serde::de::Error;

                // For bincode compatibility, deserialize in the same order as tuple serialization
                let run_args = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(0, &self))?;
                let num_rows = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(1, &self))?;
                let total_assignments = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(2, &self))?;
                let total_const_size = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(3, &self))?;
                let dynamic_lookup_params = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(4, &self))?;
                let shuffle_params = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(5, &self))?;
                let model_instance_shapes = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(6, &self))?;
                let model_output_scales = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(7, &self))?;
                let model_input_scales = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(8, &self))?;
                let module_sizes = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(9, &self))?;
                let required_lookups = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(10, &self))?;
                let required_range_checks = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(11, &self))?;
                let check_mode = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(12, &self))?;
                let version = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(13, &self))?;
                let num_blinding_factors = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(14, &self))?;
                let timestamp = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(15, &self))?;
                let input_types = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(16, &self))?;
                let output_types = seq
                    .next_element()?
                    .ok_or_else(|| Error::invalid_length(17, &self))?;

                Ok(GraphSettings {
                    run_args,
                    num_rows,
                    total_assignments,
                    total_const_size,
                    dynamic_lookup_params,
                    shuffle_params,
                    model_instance_shapes,
                    model_output_scales,
                    model_input_scales,
                    module_sizes,
                    required_lookups,
                    required_range_checks,
                    check_mode,
                    version,
                    num_blinding_factors,
                    timestamp,
                    input_types,
                    output_types,
                })
            }
        }

        // Universal deserializer that works with both JSON (map) and bincode (tuple)
        if deserializer.is_human_readable() {
            // JSON format - use struct/map deserialization with flattened fields
            const FIELDS: &'static [&'static str] = &[
                "run_args",
                "num_rows",
                "total_assignments",
                "total_const_size",
                "total_dynamic_col_size",
                "max_dynamic_input_len",
                "num_dynamic_lookups",
                "num_shuffles",
                "total_shuffle_col_size",
                "model_instance_shapes",
                "model_output_scales",
                "model_input_scales",
                "module_sizes",
                "required_lookups",
                "required_range_checks",
                "check_mode",
                "version",
                "num_blinding_factors",
                "timestamp",
                "input_types",
                "output_types",
                "dynamic_lookup_params",
                "shuffle_params",
            ];
            deserializer.deserialize_struct("GraphSettings", FIELDS, GraphSettingsVisitor)
        } else {
            // Binary format (bincode) - use tuple deserialization
            deserializer.deserialize_tuple(18, GraphSettingsVisitor)
        }
    }
}

impl GraphSettings {
    /// Calc the number of rows required for lookup tables
    pub fn lookup_log_rows(&self) -> u32 {
        ((self.run_args.lookup_range.1 - self.run_args.lookup_range.0) as f32)
            .log2()
            .ceil() as u32
    }

    /// Calc the number of rows required for lookup tables
    pub fn lookup_log_rows_with_blinding(&self) -> u32 {
        ((self.run_args.lookup_range.1 - self.run_args.lookup_range.0) as f32
            + RESERVED_BLINDING_ROWS as f32)
            .log2()
            .ceil() as u32
    }

    /// Calc the number of rows required for the range checks
    pub fn range_check_log_rows_with_blinding(&self) -> u32 {
        let max_range = self
            .required_range_checks
            .iter()
            .map(|x| x.1 - x.0)
            .max()
            .unwrap_or(0);

        (max_range as f32).log2().ceil() as u32
    }

    fn model_constraint_logrows_with_blinding(&self) -> u32 {
        (self.num_rows as f64 + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    fn dynamic_lookup_and_shuffle_logrows(&self) -> u32 {
        (self.dynamic_lookup_params.total_dynamic_col_size as f64
            + self.shuffle_params.total_shuffle_col_size as f64)
            .log2()
            .ceil() as u32
    }

    /// calculate the number of rows required for the dynamic lookup and shuffle
    pub fn dynamic_lookup_and_shuffle_logrows_with_blinding(&self) -> u32 {
        (self.dynamic_lookup_params.total_dynamic_col_size as f64
            + self.shuffle_params.total_shuffle_col_size as f64
            + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    /// calculate the number of rows required for the dynamic lookup and shuffle
    pub fn min_dynamic_lookup_and_shuffle_logrows_with_blinding(&self) -> u32 {
        (self.dynamic_lookup_params.max_dynamic_input_len as f64 + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    fn dynamic_lookup_and_shuffle_col_size(&self) -> usize {
        self.dynamic_lookup_params.total_dynamic_col_size
            + self.shuffle_params.total_shuffle_col_size
    }

    /// calculate the number of rows required for the module constraints
    pub fn module_constraint_logrows(&self) -> u32 {
        (self.module_sizes.max_constraints() as f64).log2().ceil() as u32
    }

    /// calculate the number of rows required for the module constraints
    pub fn module_constraint_logrows_with_blinding(&self) -> u32 {
        (self.module_sizes.max_constraints() as f64 + RESERVED_BLINDING_ROWS as f64)
            .log2()
            .ceil() as u32
    }

    fn constants_logrows(&self) -> u32 {
        (self.total_const_size as f64 / self.run_args.num_inner_cols as f64)
            .log2()
            .ceil() as u32
    }

    /// calculate the total number of instances
    pub fn total_instances(&self) -> Vec<usize> {
        let mut instances: Vec<usize> = self.module_sizes.num_instances();
        instances.extend(
            self.model_instance_shapes
                .iter()
                .map(|x| x.iter().product::<usize>()),
        );

        instances
    }

    /// get the scale data for instances
    pub fn get_model_instance_scales(&self) -> Vec<crate::Scale> {
        let mut scales = vec![];
        if self.run_args.input_visibility.is_public() {
            scales.extend(
                self.model_input_scales
                    .iter()
                    .map(|x| x.clone())
                    .collect::<Vec<crate::Scale>>(),
            );
        };
        if self.run_args.output_visibility.is_public() {
            scales.extend(
                self.model_output_scales
                    .iter()
                    .map(|x| x.clone())
                    .collect::<Vec<crate::Scale>>(),
            );
        };
        scales
    }

    /// calculate the log2 of the total number of instances
    pub fn log2_total_instances(&self) -> u32 {
        let sum = self.total_instances().iter().sum::<usize>();

        // max between 1 and the log2 of the sums
        std::cmp::max((sum as f64).log2().ceil() as u32, 1)
    }

    /// calculate the log2 of the total number of instances
    pub fn log2_total_instances_with_blinding(&self) -> u32 {
        let sum = self.total_instances().iter().sum::<usize>() + RESERVED_BLINDING_ROWS;

        // max between 1 and the log2 of the sums
        std::cmp::max((sum as f64).log2().ceil() as u32, 1)
    }

    /// save params to file
    pub fn save(&self, path: &std::path::PathBuf) -> Result<(), std::io::Error> {
        // buf writer
        let writer =
            std::io::BufWriter::with_capacity(*EZKL_BUF_CAPACITY, std::fs::File::create(path)?);
        serde_json::to_writer(writer, &self).map_err(|e| {
            error!("failed to save settings file at {}", e);
            std::io::Error::other(e)
        })
    }
    /// load params from file
    pub fn load(path: &std::path::PathBuf) -> Result<Self, std::io::Error> {
        // buf reader
        let reader =
            std::io::BufReader::with_capacity(*EZKL_BUF_CAPACITY, std::fs::File::open(path)?);
        let settings: GraphSettings = serde_json::from_reader(reader).map_err(|e| {
            error!("failed to load settings file at {}", e);
            std::io::Error::other(e)
        })?;

        crate::check_version_string_matches(&settings.version);

        Ok(settings)
    }

    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, GraphError> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => {
                return Err(e.into());
            }
        };
        Ok(serialized)
    }
    /// Parse an ezkl configuration from a json
    pub fn from_json(arg_json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(arg_json)
    }

    fn set_num_blinding_factors(&mut self, num_blinding_factors: usize) {
        self.num_blinding_factors = Some(num_blinding_factors);
    }

    ///
    pub fn available_col_size(&self) -> usize {
        let base = 2u32;
        if let Some(num_blinding_factors) = self.num_blinding_factors {
            base.pow(self.run_args.logrows) as usize - num_blinding_factors - 1
        } else {
            log::error!("num_blinding_factors not set");
            log::warn!("using default available_col_size");
            base.pow(self.run_args.logrows) as usize - ASSUMED_BLINDING_FACTORS - 1
        }
    }

    /// if any visibility is encrypted or hashed
    pub fn module_requires_fixed(&self) -> bool {
        self.run_args.input_visibility.is_hashed()
            || self.run_args.output_visibility.is_hashed()
            || self.run_args.param_visibility.is_hashed()
    }

    /// requires dynamic lookup
    pub fn requires_dynamic_lookup(&self) -> bool {
        self.dynamic_lookup_params.num_dynamic_lookups > 0
    }

    /// requires dynamic shuffle
    pub fn requires_shuffle(&self) -> bool {
        self.shuffle_params.num_shuffles > 0
    }

    /// any kzg visibility
    pub fn module_requires_polycommit(&self) -> bool {
        self.run_args.input_visibility.is_polycommit()
            || self.run_args.output_visibility.is_polycommit()
            || self.run_args.param_visibility.is_polycommit()
    }
}

/// Configuration for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    model_config: ModelConfig,
    module_configs: ModuleConfigs,
    circuit_size: CircuitSize,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoreCircuit {
    /// The model / graph of computations.
    pub model: Model,
    /// The settings of the model.
    pub settings: GraphSettings,
}

/// Defines the circuit for a computational graph / model loaded from a `.onnx` file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphCircuit {
    /// Core circuit
    pub core: CoreCircuit,
    /// The witness data for the model.
    pub graph_witness: GraphWitness,
}

impl GraphCircuit {
    /// Settings for the graph
    pub fn settings(&self) -> &GraphSettings {
        &self.core.settings
    }
    /// Settings for the graph (mutable)
    pub fn settings_mut(&mut self) -> &mut GraphSettings {
        &mut self.core.settings
    }
    /// The model
    pub fn model(&self) -> &Model {
        &self.core.model
    }
    ///
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), GraphError> {
        let f = std::fs::File::create(&path).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let writer = std::io::BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
        bincode::serialize_into(writer, &self)?;
        Ok(())
    }

    ///
    pub fn load(path: std::path::PathBuf) -> Result<Self, GraphError> {
        // read bytes from file
        let f = std::fs::File::open(&path).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let reader = std::io::BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
        let result: GraphCircuit = bincode::deserialize_from(reader)?;

        // check that the versions match
        crate::check_version_string_matches(&result.core.settings.version);

        Ok(result)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, PartialOrd)]
/// The data source for a test
pub enum TestDataSource {
    /// The data is loaded from a file
    File,
    /// The data is loaded from the chain
    #[default]
    OnChain,
}

impl std::fmt::Display for TestDataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestDataSource::File => write!(f, "file"),
            TestDataSource::OnChain => write!(f, "on-chain"),
        }
    }
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl ToFlags for TestDataSource {}

impl From<String> for TestDataSource {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "file" => TestDataSource::File,
            "on-chain" => TestDataSource::OnChain,
            _ => {
                error!("invalid data source: {}", value);
                warn!("using default data source: on-chain");
                TestDataSource::default()
            }
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
    pub rpc: String,
    /// data sources for the on chain data
    pub data_sources: TestSources,
}

impl GraphCircuit {
    ///
    pub fn new(model: Model, run_args: &RunArgs) -> Result<GraphCircuit, GraphError> {
        // // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Vec<Fp>> = vec![];
        for shape in model.graph.input_shapes()? {
            let t: Vec<Fp> = vec![Fp::zero(); shape.iter().product::<usize>()];
            inputs.push(t);
        }

        // dummy module settings, must load from GraphData after
        let mut settings = model.gen_params(run_args, run_args.check_mode)?;

        let mut num_params = 0;
        if !model.const_shapes().is_empty() {
            for shape in model.const_shapes() {
                num_params += shape.iter().product::<usize>();
            }
        }

        let sizes = GraphModules::num_constraints_and_instances(
            model.graph.input_shapes()?,
            vec![vec![num_params]],
            model.graph.output_shapes()?,
            VarVisibility::from_args(run_args)?,
        );

        // number of instances used by modules
        settings.module_sizes = sizes.clone();

        // as they occupy independent rows
        settings.num_rows = std::cmp::max(settings.num_rows, sizes.max_constraints());

        let core = CoreCircuit {
            model,
            settings: settings.clone(),
        };

        Ok(GraphCircuit {
            core,
            graph_witness: GraphWitness::new(inputs, vec![]),
        })
    }

    ///
    pub fn new_from_settings(
        model: Model,
        mut settings: GraphSettings,
        check_mode: CheckMode,
    ) -> Result<GraphCircuit, GraphError> {
        // placeholder dummy inputs - must call prepare_public_inputs to load data afterwards
        let mut inputs: Vec<Vec<Fp>> = vec![];
        for shape in model.graph.input_shapes()? {
            let t: Vec<Fp> = vec![Fp::zero(); shape.iter().product::<usize>()];
            inputs.push(t);
        }

        // dummy module settings, must load from GraphData after

        settings.check_mode = check_mode;

        let core = CoreCircuit {
            model,
            settings: settings.clone(),
        };

        Ok(GraphCircuit {
            core,
            graph_witness: GraphWitness::new(inputs, vec![]),
        })
    }

    /// load inputs and outputs for the model
    pub fn load_graph_witness(&mut self, data: &GraphWitness) -> Result<(), GraphError> {
        self.graph_witness = data.clone();
        // load the module settings
        Ok(())
    }

    /// Prepare the public inputs for the circuit.
    pub fn prepare_public_inputs(&self, data: &GraphWitness) -> Result<Vec<Fp>, GraphError> {
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>
        let mut public_inputs: Vec<Fp> = vec![];

        // we first process the inputs
        if let Some(processed_inputs) = &data.processed_inputs {
            public_inputs.extend(processed_inputs.get_instances().into_iter().flatten());
        }

        // we then process the params
        if let Some(processed_params) = &data.processed_params {
            public_inputs.extend(processed_params.get_instances().into_iter().flatten());
        }

        // if the inputs are public, we add them to the public inputs AFTER the processed params as they are configured in that order as Column<Instances>
        if self.settings().run_args.input_visibility.is_public() {
            public_inputs.extend(self.graph_witness.inputs.clone().into_iter().flatten())
        }

        // if the outputs are public, we add them to the public inputs
        if self.settings().run_args.output_visibility.is_public() {
            public_inputs.extend(self.graph_witness.outputs.clone().into_iter().flatten());
        // if the outputs are processed, we add the processed outputs to the public inputs
        } else if let Some(processed_outputs) = &data.processed_outputs {
            public_inputs.extend(processed_outputs.get_instances().into_iter().flatten());
        }

        if public_inputs.len() < 11 {
            debug!("public inputs: {:?}", public_inputs);
        } else {
            debug!("public inputs: {:?} ...", &public_inputs[0..10]);
        }

        Ok(public_inputs)
    }

    /// get rescaled public inputs as floating points for the circuit.
    pub fn pretty_public_inputs(
        &self,
        data: &GraphWitness,
    ) -> Result<Option<PrettyElements>, GraphError> {
        // dequantize the supplied data using the provided scale.
        // the ordering here is important, we want the inputs to come before the outputs
        // as they are configured in that order as Column<Instances>

        if data.pretty_elements.is_none() {
            warn!("no rescaled elements found in witness data");
            return Ok(None);
        }

        let mut public_inputs = PrettyElements::default();
        let elements = data.pretty_elements.as_ref().unwrap();

        if self.settings().run_args.input_visibility.is_public() {
            public_inputs.rescaled_inputs = elements.rescaled_inputs.clone();
            public_inputs.inputs = elements.inputs.clone();
        } else if data.processed_inputs.is_some() {
            public_inputs.processed_inputs = elements.processed_inputs.clone();
        }

        if data.processed_params.is_some() {
            public_inputs.processed_params = elements.processed_params.clone();
        }

        if self.settings().run_args.output_visibility.is_public() {
            public_inputs.rescaled_outputs = elements.rescaled_outputs.clone();
            public_inputs.outputs = elements.outputs.clone();
        } else if data.processed_outputs.is_some() {
            public_inputs.processed_outputs = elements.processed_outputs.clone();
        }

        #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
        debug!(
            "rescaled and processed public inputs: {}",
            serde_json::to_string(&public_inputs)?.to_colored_json_auto()?
        );

        Ok(Some(public_inputs))
    }

    ///
    pub fn load_graph_input(&mut self, data: &GraphData) -> Result<Vec<Tensor<Fp>>, GraphError> {
        let shapes = self.model().graph.input_shapes()?;
        let scales = self.model().graph.get_input_scales();
        let input_types = self.model().graph.get_input_types()?;
        self.load_file_data(&data.input_data, &shapes, scales, input_types)
    }

    ///
    pub fn load_file_data(
        &mut self,
        file_data: &DataSource,
        shapes: &Vec<Vec<usize>>,
        scales: Vec<crate::Scale>,
        input_types: Vec<InputType>,
    ) -> Result<Vec<Tensor<Fp>>, GraphError> {
        // quantize the supplied data using the provided scale.
        let mut data: Vec<Tensor<Fp>> = vec![];
        for (((d, shape), scale), input_type) in file_data
            .iter()
            .zip(shapes)
            .zip(scales)
            .zip(input_types.iter())
        {
            let t: Vec<Fp> = d
                .par_iter()
                .map(|x| {
                    let mut x = x.clone();
                    x.as_type(input_type);
                    x.to_field(scale)
                })
                .collect();

            let mut t: Tensor<Fp> = t.into_iter().into();
            t.reshape(shape)?;

            data.push(t);
        }
        Ok(data)
    }

    ///
    pub fn load_witness_file_data(
        &mut self,
        file_data: &[Vec<Fp>],
        shapes: &[Vec<usize>],
    ) -> Result<Vec<Tensor<Fp>>, GraphError> {
        // quantize the supplied data using the provided scale.
        let mut data: Vec<Tensor<Fp>> = vec![];
        for (d, shape) in file_data.iter().zip(shapes) {
            let mut t: Tensor<Fp> = d.clone().into_iter().into();
            t.reshape(shape)?;
            data.push(t);
        }
        Ok(data)
    }

    fn calc_safe_lookup_range(min_max_lookup: Range, lookup_safety_margin: f64) -> Range {
        (
            (lookup_safety_margin * min_max_lookup.0 as f64).floor() as IntegerRep,
            (lookup_safety_margin * min_max_lookup.1 as f64).ceil() as IntegerRep,
        )
    }

    fn calc_num_cols(range_len: IntegerRep, max_logrows: u32) -> usize {
        let max_col_size = Table::<Fp>::cal_col_size(max_logrows as usize, RESERVED_BLINDING_ROWS);
        num_cols_required(range_len, max_col_size)
    }

    fn table_size_logrows(
        &self,
        safe_lookup_range: Range,
        max_range_size: IntegerRep,
    ) -> Result<u32, GraphError> {
        // pick the range with the largest absolute size safe_lookup_range or max_range_size
        let safe_range = std::cmp::max(
            (safe_lookup_range.1 - safe_lookup_range.0).abs(),
            max_range_size,
        );

        let min_bits = (safe_range as f64 + RESERVED_BLINDING_ROWS as f64 + 1.)
            .log2()
            .ceil() as u32;

        Ok(min_bits)
    }

    /// calculate the minimum logrows required for the circuit
    pub fn calc_min_logrows(
        &mut self,
        min_max_lookup: Range,
        max_range_size: IntegerRep,
        max_logrows: Option<u32>,
        lookup_safety_margin: f64,
    ) -> Result<(), GraphError> {
        // load the max logrows
        let max_logrows = max_logrows.unwrap_or(MAX_PUBLIC_SRS);
        let max_logrows = std::cmp::min(max_logrows, MAX_PUBLIC_SRS);
        let mut max_logrows = std::cmp::max(max_logrows, MIN_LOGROWS);
        let mut min_logrows = MIN_LOGROWS;

        let safe_lookup_range = Self::calc_safe_lookup_range(min_max_lookup, lookup_safety_margin);

        // check if subtraction overflows

        let lookup_size =
            (safe_lookup_range.1.saturating_sub(safe_lookup_range.0)).saturating_abs();
        // check if has overflowed max lookup input

        if lookup_size > (MAX_LOOKUP_ABS as f64 / lookup_safety_margin).floor() as IntegerRep {
            return Err(GraphError::LookupRangeTooLarge(
                lookup_size.unsigned_abs() as usize
            ));
        }

        if max_range_size.abs() > MAX_LOOKUP_ABS {
            return Err(GraphError::RangeCheckTooLarge(
                max_range_size.unsigned_abs() as usize,
            ));
        }

        // These are hard lower limits, we can't overflow instances or modules constraints
        let instance_logrows = self.settings().log2_total_instances();
        let module_constraint_logrows = self.settings().module_constraint_logrows();
        let dynamic_lookup_logrows = self.settings().dynamic_lookup_and_shuffle_logrows();
        min_logrows = std::cmp::max(
            min_logrows,
            // max of the instance logrows and the module constraint logrows and the dynamic lookup logrows is the lower limit
            *[
                instance_logrows,
                module_constraint_logrows,
                dynamic_lookup_logrows,
            ]
            .iter()
            .max()
            .unwrap(),
        );

        // These are upper limits, going above these is wasteful, but they are not hard limits
        let model_constraint_logrows = self.settings().model_constraint_logrows_with_blinding();
        let min_bits = self.table_size_logrows(safe_lookup_range, max_range_size)?;
        let constants_logrows = self.settings().constants_logrows();
        max_logrows = std::cmp::min(
            max_logrows,
            // max of the model constraint logrows, min_bits, and the constants logrows is the upper limit
            *[model_constraint_logrows, min_bits, constants_logrows]
                .iter()
                .max()
                .unwrap(),
        );

        // we now have a min and max logrows
        max_logrows = std::cmp::max(min_logrows, max_logrows);

        // degrade the max logrows until the extended k is small enough
        while min_logrows < max_logrows
            && !self.extended_k_is_small_enough(max_logrows, safe_lookup_range, max_range_size)
        {
            max_logrows -= 1;
        }

        if !self.extended_k_is_small_enough(max_logrows, safe_lookup_range, max_range_size) {
            return Err(GraphError::ExtendedKTooLarge(max_logrows));
        }

        let logrows = max_logrows;

        let model = self.model().clone();
        let settings_mut = self.settings_mut();
        settings_mut.run_args.lookup_range = safe_lookup_range;
        settings_mut.run_args.logrows = logrows;

        *settings_mut = GraphCircuit::new(model, &settings_mut.run_args)?
            .settings()
            .clone();

        debug!(
            "setting lookup_range to: {:?}, setting logrows to: {}",
            self.settings().run_args.lookup_range,
            self.settings().run_args.logrows
        );

        Ok(())
    }

    fn extended_k_is_small_enough(
        &self,
        k: u32,
        safe_lookup_range: Range,
        max_range_size: IntegerRep,
    ) -> bool {
        // if num cols is too large then the extended k is too large
        if Self::calc_num_cols(safe_lookup_range.1 - safe_lookup_range.0, k) > MAX_NUM_LOOKUP_COLS
            || Self::calc_num_cols(max_range_size, k) > MAX_NUM_LOOKUP_COLS
        {
            return false;
        }

        let mut settings = self.settings().clone();
        settings.run_args.lookup_range = safe_lookup_range;
        settings.run_args.logrows = k;
        settings.required_range_checks = vec![(0, max_range_size)];
        let mut cs = ConstraintSystem::default();
        // if unix get a gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _r = Gag::stdout().ok();
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _g = Gag::stderr().ok();

        Self::configure_with_params(&mut cs, settings);

        // drop the gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_r);
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_g);

        #[cfg(feature = "mv-lookup")]
        let cs = cs.chunk_lookups();
        // quotient_poly_degree * params.n - 1 is the degree of the quotient polynomial
        let max_degree = cs.degree();
        let quotient_poly_degree = (max_degree - 1) as u64;
        // n = 2^k
        let n = 1u64 << k;
        let mut extended_k = k;

        while (1 << extended_k) < (n * quotient_poly_degree) {
            extended_k += 1;
            if extended_k > bn256::Fr::S {
                return false;
            }
        }
        true
    }

    /// Runs the forward pass of the model / graph of computations and any associated hashing.
    pub fn forward<Scheme: CommitmentScheme<Scalar = Fp, Curve = G1Affine>>(
        &self,
        inputs: &mut [Tensor<Fp>],
        vk: Option<&VerifyingKey<G1Affine>>,
        srs: Option<&Scheme::ParamsProver>,
        region_settings: RegionSettings,
    ) -> Result<GraphWitness, GraphError> {
        let original_inputs = inputs.to_vec();

        let visibility = VarVisibility::from_args(&self.settings().run_args)?;
        let mut processed_inputs = None;
        let mut processed_params = None;
        let mut processed_outputs = None;

        if visibility.input.requires_processing() {
            let module_outlets = visibility.input.overwrites_inputs();
            if !module_outlets.is_empty() {
                let mut module_inputs = vec![];
                for outlet in &module_outlets {
                    module_inputs.push(inputs[*outlet].clone());
                }
                let res =
                    GraphModules::forward::<Scheme>(&module_inputs, &visibility.input, vk, srs)?;
                processed_inputs = Some(res.clone());
                let module_results = res.get_result(visibility.input.clone());

                for (i, outlet) in module_outlets.iter().enumerate() {
                    inputs[*outlet] = Tensor::from(module_results[i].clone().into_iter());
                }
            } else {
                processed_inputs = Some(GraphModules::forward::<Scheme>(
                    inputs,
                    &visibility.input,
                    vk,
                    srs,
                )?);
            }
        }

        if visibility.params.requires_processing() {
            let params = self.model().get_all_params();
            if !params.is_empty() {
                let flattened_params = Tensor::new(Some(&params), &[params.len()])?.combine()?;
                processed_params = Some(GraphModules::forward::<Scheme>(
                    &[flattened_params],
                    &visibility.params,
                    vk,
                    srs,
                )?);
            }
        }

        let mut model_results =
            self.model()
                .forward(inputs, &self.settings().run_args, region_settings)?;

        if visibility.output.requires_processing() {
            let module_outlets = visibility.output.overwrites_inputs();
            if !module_outlets.is_empty() {
                let mut module_inputs = vec![];
                for outlet in &module_outlets {
                    module_inputs.push(model_results.outputs[*outlet].clone());
                }
                let res =
                    GraphModules::forward::<Scheme>(&module_inputs, &visibility.output, vk, srs)?;
                processed_outputs = Some(res.clone());
                let module_results = res.get_result(visibility.output.clone());

                for (i, outlet) in module_outlets.iter().enumerate() {
                    model_results.outputs[*outlet] =
                        Tensor::from(module_results[i].clone().into_iter());
                }
            } else {
                processed_outputs = Some(GraphModules::forward::<Scheme>(
                    &model_results.outputs,
                    &visibility.output,
                    vk,
                    srs,
                )?);
            }
        }

        let mut witness = GraphWitness {
            inputs: original_inputs
                .iter()
                .map(|t| t.deref().to_vec())
                .collect_vec(),
            pretty_elements: None,
            outputs: model_results
                .outputs
                .iter()
                .map(|t| t.deref().to_vec())
                .collect_vec(),
            processed_inputs,
            processed_params,
            processed_outputs,
            max_lookup_inputs: model_results.max_lookup_inputs,
            min_lookup_inputs: model_results.min_lookup_inputs,
            max_range_size: model_results.max_range_size,
            version: Some(crate::version().to_string()),
        };

        witness.generate_rescaled_elements(
            self.model().graph.get_input_scales(),
            self.model().graph.get_output_scales()?,
            visibility,
        );

        #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
        log::trace!(
            "witness: \n {}",
            &witness.as_json()?.to_colored_json_auto()?
        );

        Ok(witness)
    }

    /// Create a new circuit from a set of input data and [RunArgs].
    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    pub fn from_run_args(
        run_args: &RunArgs,
        model_path: &std::path::Path,
    ) -> Result<Self, GraphError> {
        let model = Model::from_run_args(run_args, model_path)?;
        Self::new(model, run_args)
    }

    /// Create a new circuit from a set of input data and [GraphSettings].
    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    pub fn from_settings(
        params: &GraphSettings,
        model_path: &std::path::Path,
        check_mode: CheckMode,
    ) -> Result<Self, GraphError> {
        params
            .run_args
            .validate()
            .map_err(GraphError::InvalidRunArgs)?;
        let model = Model::from_run_args(&params.run_args, model_path)?;
        Self::new_from_settings(model, params.clone(), check_mode)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
/// The configuration for the graph circuit
pub struct CircuitSize {
    num_instances: usize,
    num_advice_columns: usize,
    num_fixed: usize,
    num_challenges: usize,
    num_selectors: usize,
    logrows: u32,
}

impl CircuitSize {
    ///
    pub fn from_cs<F: Field>(cs: &ConstraintSystem<F>, logrows: u32) -> Self {
        CircuitSize {
            num_instances: cs.num_instance_columns(),
            num_advice_columns: cs.num_advice_columns(),
            num_fixed: cs.num_fixed_columns(),
            num_challenges: cs.num_challenges(),
            num_selectors: cs.num_selectors(),
            logrows,
        }
    }

    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    /// Export the ezkl configuration as json
    pub fn as_json(&self) -> Result<String, GraphError> {
        let serialized = match serde_json::to_string(&self) {
            Ok(s) => s,
            Err(e) => return Err(e.into()),
        };
        Ok(serialized)
    }

    /// number of columns
    pub fn num_columns(&self) -> usize {
        self.num_instances + self.num_advice_columns + self.num_fixed
    }

    /// area of the circuit
    pub fn area(&self) -> usize {
        self.num_columns() * (1 << self.logrows)
    }

    /// area less than max
    pub fn area_less_than_max(&self) -> bool {
        if EZKL_MAX_CIRCUIT_AREA.is_some() {
            self.area() < EZKL_MAX_CIRCUIT_AREA.unwrap()
        } else {
            true
        }
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
        self.settings().clone()
    }

    fn configure_with_params(cs: &mut ConstraintSystem<Fp>, params: Self::Params) -> Self::Config {
        let mut params = params.clone();
        params.set_num_blinding_factors(cs.blinding_factors());
        GLOBAL_SETTINGS.with(|settings| {
            *settings.borrow_mut() = Some(params.clone());
        });
        let visibility = match VarVisibility::from_args(&params.run_args) {
            Ok(v) => v,
            Err(e) => {
                log::error!("failed to create visibility: {:?}", e);
                log::warn!("using default visibility");
                VarVisibility::default()
            }
        };

        let mut module_configs = ModuleConfigs::from_visibility(
            cs,
            &params.module_sizes,
            params.run_args.logrows as usize,
        );

        let mut vars = ModelVars::new(cs, &params);

        module_configs.configure_complex_modules(cs, &visibility, &params.module_sizes);

        vars.instantiate_instance(
            cs,
            params.model_instance_shapes.clone(),
            params.run_args.input_scale,
            module_configs.instance,
        );

        let base = Model::configure(cs, &vars, &params).unwrap();

        let model_config = ModelConfig { base, vars };

        debug!(
            "degree: {}, log2_ceil of degrees: {:?}",
            cs.degree(),
            (cs.degree() as f32).log2().ceil()
        );

        let circuit_size = CircuitSize::from_cs(cs, params.run_args.logrows);

        #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
        debug!(
            "circuit size: \n {}",
            circuit_size
                .as_json()
                .unwrap()
                .to_colored_json_auto()
                .unwrap()
        );

        GraphConfig {
            model_config,
            module_configs,
            circuit_size,
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
        // check if the circuit area is less than the max
        if !config.circuit_size.area_less_than_max() {
            error!(
                "circuit area {} is larger than the max allowed area {}",
                config.circuit_size.area(),
                EZKL_MAX_CIRCUIT_AREA.unwrap()
            );
            return Err(PlonkError::Synthesis);
        }

        trace!("Setting input in synthesize");
        let input_vis = &self.settings().run_args.input_visibility;
        let output_vis = &self.settings().run_args.output_visibility;
        let mut graph_modules = GraphModules::new();

        let mut constants = ConstantsMap::new();

        let mut config = config.clone();

        let mut inputs = self
            .graph_witness
            .get_input_tensor()
            .iter_mut()
            .map(|i| {
                i.set_visibility(input_vis);
                ValTensor::try_from(i.clone()).map_err(|e| {
                    log::error!("failed to convert input to valtensor: {:?}", e);
                    PlonkError::Synthesis
                })
            })
            .collect::<Result<Vec<ValTensor<Fp>>, PlonkError>>()?;

        let outputs = self
            .graph_witness
            .get_output_tensor()
            .iter_mut()
            .map(|i| {
                i.set_visibility(output_vis);
                ValTensor::try_from(i.clone()).map_err(|e| {
                    log::error!("failed to convert output to valtensor: {:?}", e);
                    PlonkError::Synthesis
                })
            })
            .collect::<Result<Vec<ValTensor<Fp>>, PlonkError>>()?;

        let mut instance_offset = 0;
        trace!("running input module layout");

        let input_visibility = &self.settings().run_args.input_visibility;
        let outlets = input_visibility.overwrites_inputs();

        if !outlets.is_empty() {
            let mut input_outlets = vec![];
            for outlet in &outlets {
                input_outlets.push(inputs[*outlet].clone());
            }
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut input_outlets,
                input_visibility,
                &mut instance_offset,
                &mut constants,
            )?;
            // replace inputs with the outlets
            for (i, outlet) in outlets.iter().enumerate() {
                inputs[*outlet] = input_outlets[i].clone();
            }
        } else {
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut inputs,
                input_visibility,
                &mut instance_offset,
                &mut constants,
            )?;
        }

        // now we need to assign the flattened params to the model
        let mut model = self.model().clone();
        let param_visibility = &self.settings().run_args.param_visibility;
        trace!("running params module layout");
        if !self.model().get_all_params().is_empty() && param_visibility.requires_processing() {
            // now we need to flatten the params
            let consts = self.model().get_all_params();

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
                vec![t.try_into().map_err(|_| {
                    log::error!("failed to convert params to valtensor");
                    PlonkError::Synthesis
                })?]
            };

            // now do stuff to the model params
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut flattened_params,
                param_visibility,
                &mut instance_offset,
                &mut constants,
            )?;

            let shapes = self.model().const_shapes();
            trace!("replacing processed consts");
            let split_params = split_valtensor(&flattened_params[0], shapes).map_err(|_| {
                log::error!("failed to split params");
                PlonkError::Synthesis
            })?;

            // now the flattened_params have been assigned to and we-assign them to the model consts such that they are constrained to be equal
            model.replace_consts(&split_params);
        }

        // create a new module for the model (space 2)
        layouter.assign_region(|| "_enter_module_2", |_| Ok(()))?;
        trace!("laying out model");

        let mut vars = config.model_config.vars.clone();
        vars.set_initial_instance_offset(instance_offset);

        let mut outputs = model
            .layout(
                config.model_config.clone(),
                &mut layouter,
                &self.settings().run_args,
                &inputs,
                &mut vars,
                &outputs,
                &mut constants,
            )
            .map_err(|e| {
                log::error!("{}", e);
                PlonkError::Synthesis
            })?;
        trace!("running output module layout");

        let output_visibility = &self.settings().run_args.output_visibility;
        let outlets = output_visibility.overwrites_inputs();

        instance_offset += vars.get_instance_len();

        if !outlets.is_empty() {
            let mut output_outlets = vec![];
            for outlet in &outlets {
                output_outlets.push(outputs[*outlet].clone());
            }
            // this will re-enter module 0
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut output_outlets,
                &self.settings().run_args.output_visibility,
                &mut instance_offset,
                &mut constants,
            )?;

            // replace outputs with the outlets
            for (i, outlet) in outlets.iter().enumerate() {
                outputs[*outlet] = output_outlets[i].clone();
            }
        } else {
            graph_modules.layout(
                &mut layouter,
                &mut config.module_configs,
                &mut outputs,
                &self.settings().run_args.output_visibility,
                &mut instance_offset,
                &mut constants,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
/// Tests for the graph module
pub mod tests {
    use super::*;

    #[test]
    fn test_graph_settings_serialization_roundtrip() {
        use crate::{CheckMode, RunArgs};

        // Create a test GraphSettings with nested structs
        let original = GraphSettings {
            run_args: RunArgs::default(),
            num_rows: 1000,
            total_assignments: 500,
            total_const_size: 100,
            dynamic_lookup_params: DynamicLookupParams {
                total_dynamic_col_size: 42,
                max_dynamic_input_len: 128,
                num_dynamic_lookups: 5,
            },
            shuffle_params: ShuffleParams {
                num_shuffles: 3,
                total_shuffle_col_size: 256,
            },
            model_instance_shapes: vec![vec![1, 2, 3]],
            model_output_scales: vec![],
            model_input_scales: vec![],
            module_sizes: ModuleSizes::default(),
            required_lookups: vec![],
            required_range_checks: vec![],
            check_mode: CheckMode::SAFE,
            version: "1.0.0".to_string(),
            num_blinding_factors: Some(5),
            timestamp: Some(123456789),
            input_types: None,
            output_types: None,
        };

        // Test 1: JSON serialization roundtrip with flattened format
        let json_str = serde_json::to_string_pretty(&original).unwrap();
        println!("JSON serialized (flattened):\n{}", json_str);

        // Verify the JSON contains flattened fields
        assert!(json_str.contains("\"total_dynamic_col_size\": 42"));
        assert!(json_str.contains("\"max_dynamic_input_len\": 128"));
        assert!(json_str.contains("\"num_dynamic_lookups\": 5"));
        assert!(json_str.contains("\"num_shuffles\": 3"));
        assert!(json_str.contains("\"total_shuffle_col_size\": 256"));

        // Verify the JSON does NOT contain nested structs
        assert!(!json_str.contains("\"dynamic_lookup_params\""));
        assert!(!json_str.contains("\"shuffle_params\""));

        // Deserialize from JSON
        let deserialized: GraphSettings = serde_json::from_str(&json_str).unwrap();
        assert_eq!(original, deserialized);

        // now do JSON bytes
        let json_bytes = serde_json::to_vec(&original).unwrap();
        let deserialized_from_bytes: GraphSettings = serde_json::from_slice(&json_bytes).unwrap();
        assert_eq!(original, deserialized_from_bytes);

        // Test 2: Bincode serialization roundtrip
        let bincode_data = bincode::serialize(&original).unwrap();
        let bincode_deserialized: GraphSettings = bincode::deserialize(&bincode_data).unwrap();
        assert_eq!(original, bincode_deserialized);

        // Test 3: Backwards compatibility - deserialize old nested format
        let old_format_json = r#"{
    "run_args": {
        "tolerance": {
            "val": 0.0,
            "scale": 1.0
        },
        "input_scale": 0,
        "param_scale": 0,
        "scale_rebase_multiplier": 10,
        "lookup_range": [
            0,
            0
        ],
        "logrows": 6,
        "num_inner_cols": 2,
        "variables": [
            [
                "batch_size",
                1
            ]
        ],
        "input_visibility": "Private",
        "output_visibility": "Public",
        "param_visibility": "Private",
        "rebase_frac_zero_constants": false,
        "check_mode": "UNSAFE",
        "commitment": "KZG",
        "decomp_base": 128,
        "decomp_legs": 2,
        "bounded_log_lookup": false,
        "ignore_range_check_inputs_outputs": false
    },
    "num_rows": 236,
    "total_assignments": 472,
    "total_const_size": 4,
    "total_dynamic_col_size": 0,
    "max_dynamic_input_len": 0,
    "num_dynamic_lookups": 0,
    "num_shuffles": 0,
    "total_shuffle_col_size": 0,
    "model_instance_shapes": [
        [
            1,
            4
        ]
    ],
    "model_output_scales": [
        0
    ],
    "model_input_scales": [
        0
    ],
    "module_sizes": {
        "polycommit": [],
        "poseidon": [
            0,
            [
                0
            ]
        ]
    },
    "required_lookups": [],
    "required_range_checks": [
        [
            -1,
            1
        ],
        [
            0,
            127
        ]
    ],
    "check_mode": "UNSAFE",
    "version": "0.0.0",
    "num_blinding_factors": null,
    "timestamp": 1741214578354
}"#;

        let _backwards_compatible: GraphSettings = serde_json::from_str(old_format_json).unwrap();
    }
}
