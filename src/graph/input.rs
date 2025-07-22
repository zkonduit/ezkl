use super::errors::GraphError;
use super::quantize_float;
use crate::circuit::InputType;
use crate::fieldutils::integer_rep_to_felt;
use crate::EZKL_BUF_CAPACITY;
use halo2curves::bn256::Fr as Fp;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::panic::UnwindSafe;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tract_onnx::tract_core::{
    tract_data::{prelude::Tensor as TractTensor, TVec},
    value::TValue,
};

/// Represents different types of values that can be stored in a file source
/// Used for handling various input types in zero-knowledge proofs
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub enum FileSourceInner {
    /// Floating point value (64-bit)
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// Field element value for direct use in circuits
    Field(Fp),
}

impl From<Fp> for FileSourceInner {
    fn from(value: Fp) -> Self {
        FileSourceInner::Field(value)
    }
}
impl From<bool> for FileSourceInner {
    fn from(value: bool) -> Self {
        FileSourceInner::Bool(value)
    }
}
impl From<f64> for FileSourceInner {
    fn from(value: f64) -> Self {
        FileSourceInner::Float(value)
    }
}

impl FileSourceInner {
    /// Returns true if the value is a floating point number
    pub fn is_float(&self) -> bool {
        matches!(self, FileSourceInner::Float(_))
    }

    /// Returns true if the value is a boolean
    pub fn is_bool(&self) -> bool {
        matches!(self, FileSourceInner::Bool(_))
    }

    /// Returns true if the value is a field element
    pub fn is_field(&self) -> bool {
        matches!(self, FileSourceInner::Field(_))
    }

    /// Creates a new floating point value
    pub fn new_float(f: f64) -> Self {
        FileSourceInner::Float(f)
    }

    /// Creates a new field element value
    pub fn new_field(f: Fp) -> Self {
        FileSourceInner::Field(f)
    }

    /// Creates a new boolean value
    pub fn new_bool(f: bool) -> Self {
        FileSourceInner::Bool(f)
    }

    /// Adjusts the value according to the specified input type
    ///
    /// # Arguments
    /// * `input_type` - Type specification to convert the value to
    pub fn as_type(&mut self, input_type: &InputType) {
        match self {
            FileSourceInner::Float(f) => input_type.roundtrip(f),
            FileSourceInner::Bool(_) => assert!(matches!(input_type, InputType::Bool)),
            FileSourceInner::Field(_) => {}
        }
    }

    /// Converts the value to a field element using appropriate scaling
    ///
    /// # Arguments
    /// * `scale` - Scaling factor for floating point conversion
    pub fn to_field(&self, scale: crate::Scale) -> Fp {
        match self {
            FileSourceInner::Float(f) => {
                integer_rep_to_felt(quantize_float(f, 0.0, scale).unwrap())
            }
            FileSourceInner::Bool(f) => {
                if *f {
                    Fp::one()
                } else {
                    Fp::zero()
                }
            }
            FileSourceInner::Field(f) => *f,
        }
    }

    /// Converts the value to a floating point number
    pub fn to_float(&self) -> f64 {
        match self {
            FileSourceInner::Float(f) => *f,
            FileSourceInner::Bool(f) => {
                if *f {
                    1.0
                } else {
                    0.0
                }
            }
            FileSourceInner::Field(f) => crate::fieldutils::felt_to_integer_rep(*f) as f64,
        }
    }
}

impl Serialize for FileSourceInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            FileSourceInner::Field(data) => data.serialize(serializer),
            FileSourceInner::Bool(data) => data.serialize(serializer),
            FileSourceInner::Float(data) => data.serialize(serializer),
        }
    }
}

// Deserialization implementation for FileSourceInner
// Uses JSON deserialization to handle the different variants
impl<'de> Deserialize<'de> for FileSourceInner {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        let bool_try: Result<bool, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = bool_try {
            return Ok(FileSourceInner::Bool(t));
        }
        let float_try: Result<f64, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = float_try {
            return Ok(FileSourceInner::Float(t));
        }
        let field_try: Result<Fp, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = field_try {
            return Ok(FileSourceInner::Field(t));
        }

        Err(serde::de::Error::custom(
            "failed to deserialize FileSourceInner",
        ))
    }
}

/// A collection of input values from a file source
/// Organized as a vector of vectors where each inner vector represents a row/entry
pub type DataSource = Vec<Vec<FileSourceInner>>;

/// Represents which parts of the model (input/output) are attested to on-chain
pub type InputOutput = (bool, bool);

/// Container for input and output data for graph computations
///
/// Important: Always use JSON serialization for GraphData to handle enum variants correctly
#[derive(Clone, Debug, Deserialize, Default, PartialEq, Serialize)]
pub struct GraphData {
    /// Input data for the model/graph
    /// Can be empty if inputs come from on-chain sources
    pub input_data: DataSource,

    /// Optional output data for the model/graph
    /// Can be empty if outputs come from on-chain sources
    pub output_data: Option<DataSource>,
}

impl UnwindSafe for GraphData {}

impl GraphData {
    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    /// Converts the input data to tract's tensor format
    ///
    /// # Arguments
    /// * `shapes` - Expected shapes for each input tensor
    /// * `datum_types` - Expected data types for each input
    pub fn to_tract_data(
        &self,
        shapes: &[Vec<usize>],
        datum_types: &[tract_onnx::prelude::DatumType],
    ) -> Result<TVec<TValue>, GraphError> {
        let mut inputs = TVec::new();
        for (i, input) in self.input_data.iter().enumerate() {
            if !input.is_empty() {
                let dt = datum_types[i];
                let input = input.iter().map(|e| e.to_float()).collect::<Vec<f64>>();
                let tt = TractTensor::from_shape(&shapes[i], &input)?;
                let tt = tt.cast_to_dt(dt)?;
                inputs.push(tt.into_owned().into());
            }
        }

        Ok(inputs)
    }

    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    /// Converts tract tensor data into GraphData format
    ///
    /// # Arguments
    /// * `tensors` - Array of tract tensors to convert
    ///
    /// # Returns
    /// A new GraphData instance containing the converted tensor data
    pub fn from_tract_data(tensors: &[TractTensor]) -> Result<Self, GraphError> {
        use tract_onnx::prelude::DatumType;

        let mut input_data = vec![];
        for tensor in tensors {
            match tensor.datum_type() {
                tract_onnx::prelude::DatumType::Bool => {
                    let tensor = tensor.to_array_view::<bool>()?;
                    let tensor = tensor.iter().map(|e| FileSourceInner::Bool(*e)).collect();
                    input_data.push(tensor);
                }
                _ => {
                    let cast_tensor = tensor.cast_to_dt(DatumType::F64)?;
                    let tensor = cast_tensor.to_array_view::<f64>()?;
                    let tensor = tensor.iter().map(|e| FileSourceInner::Float(*e)).collect();
                    input_data.push(tensor);
                }
            }
        }
        Ok(GraphData {
            input_data,
            output_data: None,
        })
    }

    /// Creates a new GraphData instance with given input data
    ///
    /// # Arguments
    /// * `input_data` - The input data source
    pub fn new(input_data: DataSource) -> Self {
        GraphData {
            input_data,
            output_data: None,
        }
    }

    /// Loads graph input data from a string, first seeing if it is a file path or JSON data
    /// If it is a file path, it will load the data from the file
    /// Otherwise, it will attempt to parse the string as JSON data
    ///
    /// # Arguments
    /// * `data` - String containing the input data
    /// # Returns
    /// A new GraphData instance containing the loaded data
    pub fn from_str(data: &str) -> Result<Self, GraphError> {
        let graph_input = serde_json::from_str(data);
        match graph_input {
            Ok(graph_input) => Ok(graph_input),
            Err(_) => {
                let path = std::path::PathBuf::from(data);
                GraphData::from_path(path)
            }
        }
    }

    /// Loads graph input data from a file
    ///
    /// # Arguments
    /// * `path` - Path to the input file
    ///
    /// # Returns
    /// A new GraphData instance containing the loaded data
    pub fn from_path(path: std::path::PathBuf) -> Result<Self, GraphError> {
        let reader = std::fs::File::open(&path).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, reader);
        let mut buf = String::new();
        reader.read_to_string(&mut buf).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let graph_input = serde_json::from_str(&buf)?;
        Ok(graph_input)
    }

    /// Saves the graph data to a file
    ///
    /// # Arguments
    /// * `path` - Path where to save the data
    pub fn save(&self, path: std::path::PathBuf) -> Result<(), GraphError> {
        let file = std::fs::File::create(path.clone()).map_err(|e| {
            GraphError::ReadWriteFileError(path.display().to_string(), e.to_string())
        })?;
        let writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Splits the input data into multiple batches based on input shapes
    ///
    /// # Arguments
    /// * `input_shapes` - Vector of shapes for each input tensor
    ///
    /// # Returns
    /// Vector of GraphData instances, one for each batch
    ///
    /// # Errors
    /// Returns error if:
    /// - Data is from on-chain source
    /// - Input size is not evenly divisible by batch size
    pub fn split_into_batches(
        &self,
        input_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Self>, GraphError> {
        let mut batched_inputs = vec![];

        let iterable = self.input_data.clone();

        // Process each input tensor according to its shape
        for (i, shape) in input_shapes.iter().enumerate() {
            let input_size = shape.clone().iter().product::<usize>();
            let input = &iterable[i];

            // Validate input size is divisible by batch size
            if input.len() % input_size != 0 {
                return Err(GraphError::InvalidDims(
                    0,
                    format!(
                        "calibration data length (={}) must be evenly divisible by the original input_size(={})",
                        input.len(),
                        input_size
                    ),
                ));
            }

            // Split input into batches
            let mut batches = vec![];
            for batch in input.chunks(input_size) {
                batches.push(batch.to_vec());
            }
            batched_inputs.push(batches);
        }

        // Merge batches across inputs
        let num_batches = if batched_inputs.is_empty() {
            0
        } else {
            let num_batches = batched_inputs[0].len();
            // Verify all inputs have same number of batches
            for input in batched_inputs.iter() {
                assert_eq!(input.len(), num_batches);
            }
            num_batches
        };

        let mut input_batches = vec![];
        for i in 0..num_batches {
            let mut batch = vec![];
            for input in batched_inputs.iter() {
                batch.push(input[i].clone());
            }
            input_batches.push(batch);
        }

        // Ensure at least one batch exists
        if input_batches.is_empty() {
            input_batches.push(vec![vec![]]);
        }

        // Create GraphData instance for each batch
        let batches = input_batches
            .into_iter()
            .map(GraphData::new)
            .collect::<Vec<GraphData>>();

        Ok(batches)
    }
}

// Additional Python bindings for various types...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_input_serialization_round_trip() {
        // Test serialization/deserialization of graph input
        let file = GraphData::new(vec![vec![
            0.05326242372393608.into(),
            0.07497056573629379.into(),
            0.05235547572374344.into(),
        ]]);

        let serialized = serde_json::to_string(&file).unwrap();
        const JSON: &str = r#"{"input_data":[[0.05326242372393608,0.07497056573629379,0.05235547572374344]],"output_data":null}"#;
        assert_eq!(serialized, JSON);

        let graph_input3 = serde_json::from_str::<GraphData>(JSON)
            .map_err(|e| e.to_string())
            .unwrap();
        assert_eq!(graph_input3, file);
    }

    #[test]
    fn test_python_compat() {
        // Test compatibility with mclbn256 library serialization
        let source = Fp::from_raw([18445520602771460712, 838677322461845011, 3079992810, 0]);
        let original_addr = "0x000000000000000000000000b794f5ea0ba39494ce839613fffba74279579268";
        assert_eq!(format!("{:?}", source), original_addr);
    }
}



#[cfg(feature = "python-bindings")]
use crate::pfsys::field_to_string;

#[cfg(feature = "python-bindings")]
impl ToPyObject for FileSourceInner {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            FileSourceInner::Field(data) => field_to_string(data).to_object(py),
            FileSourceInner::Bool(data) => data.to_object(py),
            FileSourceInner::Float(data) => data.to_object(py),
        }
    }
}
