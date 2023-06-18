#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
use serde::{Deserialize, Serialize, Serializer};
use serde::ser::SerializeStruct;
use std::io::Read;
use halo2curves::bn256::Fr as Fp;
use std::collections::HashMap;

use super::{modules::ModuleForwardResult, GraphError};

/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Default)]
pub struct GraphInput {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are not being constrained).
    pub input_data: Vec<Vec<f32>>,
    /// The expected output of the model (can be empty vectors if outputs are not being constrained).
    pub output_data: Vec<Vec<f32>>,
    /// Optional hashes of the inputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_inputs: Option<ModuleForwardResult>,
    /// Optional hashes of the params (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_params: Option<ModuleForwardResult>,
    /// Optional hashes of the outputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_outputs: Option<ModuleForwardResult>,
}

impl GraphInput {
    ///
    pub fn new(input_data: Vec<Vec<f32>>, output_data: Vec<Vec<f32>>) -> Self {
        GraphInput {
            input_data,
            output_data,
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
        }
    }
    ///
    pub fn split_into_batches(
        &self,
        batch_size: usize,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        // split input data into batches
        let mut batched_inputs = vec![];

        for (i, input) in self.input_data.iter().enumerate() {
            // ensure the input is devenly divisible by batch_size
            if input.len() % batch_size != 0 {
                return Err(Box::new(GraphError::InvalidDims(
                    0,
                    "input data length must be evenly divisible by batch size".to_string(),
                )));
            }
            let input_size = input_shapes[i].clone().iter().product::<usize>();
            let mut batches = vec![];
            for batch in input.chunks(batch_size * input_size) {
                batches.push(batch.to_vec());
            }
            batched_inputs.push(batches);
        }
        // now merge all the batches for each input into a vector of batches
        // first assert each input has the same number of batches
        let num_batches = batched_inputs[0].len();
        for input in batched_inputs.iter() {
            assert_eq!(input.len(), num_batches);
        }
        // now merge the batches
        let mut input_batches = vec![];
        for i in 0..num_batches {
            let mut batch = vec![];
            for input in batched_inputs.iter() {
                batch.push(input[i].clone());
            }
            input_batches.push(batch);
        }

        // split output data into batches
        let mut batched_outputs = vec![];

        for (i, output) in self.output_data.iter().enumerate() {
            // ensure the input is devenly divisible by batch_size
            if output.len() % batch_size != 0 {
                return Err(Box::new(GraphError::InvalidDims(
                    0,
                    "input data length must be evenly divisible by batch size".to_string(),
                )));
            }

            let output_size = output_shapes[i].clone().iter().product::<usize>();
            let mut batches = vec![];
            for batch in output.chunks(batch_size * output_size) {
                batches.push(batch.to_vec());
            }
            batched_outputs.push(batches);
        }

        // now merge all the batches for each output into a vector of batches
        // first assert each output has the same number of batches
        let num_batches = batched_outputs[0].len();
        for output in batched_outputs.iter() {
            assert_eq!(output.len(), num_batches);
        }
        // now merge the batches
        let mut output_batches = vec![];
        for i in 0..num_batches {
            let mut batch = vec![];
            for output in batched_outputs.iter() {
                batch.push(output[i].clone());
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

/// converts fp into Vec<u64>
pub fn fp_to_vecu64(fp: &Fp) -> Vec<u64> {
    let bytes = fp.to_bytes();
    let bytes_first_u64 = u64::from_le_bytes(bytes[0..8][..].try_into().unwrap());
    let bytes_second_u64 = u64::from_le_bytes(bytes[8..16][..].try_into().unwrap());
    let bytes_third_u64 = u64::from_le_bytes(bytes[16..24][..].try_into().unwrap());
    let bytes_fourth_u64 = u64::from_le_bytes(bytes[24..32][..].try_into().unwrap());

    [bytes_first_u64, bytes_second_u64, bytes_third_u64, bytes_fourth_u64].to_vec()
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for GraphInput {
    fn to_object(&self, py: Python) -> PyObject {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let dict_inputs = PyDict::new(py);
        let dict_params = PyDict::new(py);
        let dict_outputs = PyDict::new(py);

        let input_data_mut = &self.input_data;
        let output_data_mut = &self.output_data;

        dict.set_item("input_data", &input_data_mut)
            .unwrap();
        dict.set_item("output_data", &output_data_mut)
            .unwrap();

        if let Some(processed_inputs) = &self.processed_inputs {
            let processed_inputs_poseidon_hash: Vec<Vec<u64>> = processed_inputs.poseidon_hash.iter().map(fp_to_vecu64).collect();
            dict_inputs.set_item("poseidon_hash", processed_inputs_poseidon_hash).unwrap();
            dict.set_item("processed_inputs", dict_inputs).unwrap();
        }

        if let Some(processed_params) = &self.processed_params {
            let processed_params_poseidon_hash: Vec<Vec<u64>> = processed_params.poseidon_hash.iter().map(fp_to_vecu64).collect();
            dict_params.set_item("poseidon_hash", processed_params_poseidon_hash).unwrap();
            dict.set_item("processed_params", dict_params).unwrap();
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            let processed_outputs_poseidon_hash: Vec<Vec<u64>> = processed_outputs.poseidon_hash.iter().map(fp_to_vecu64).collect();
            dict_outputs.set_item("poseidon_hash", processed_outputs_poseidon_hash).unwrap();
            dict.set_item("processed_outputs", dict_outputs).unwrap();
        }
        dict.to_object(py)
    }
}

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
        serde_json::to_writer(std::fs::File::create(path)?, &self).map_err(|e| e.into())
    }
}

impl Serialize for GraphInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphInput", 4)?;
        let input_data_f64: Vec<Vec<f64>> = self.input_data.iter().map(|v| v.iter().map(|&f| f as f64).collect()).collect();
        let output_data_f64: Vec<Vec<f64>> = self.output_data.iter().map(|v| v.iter().map(|&f| f as f64).collect()).collect();
        state.serialize_field("input_data", &input_data_f64)?;
        state.serialize_field("output_data", &output_data_f64)?;

        if let Some(processed_inputs) = &self.processed_inputs {
            let processed_inputs_poseidon: Vec<Vec<u64>> = processed_inputs.poseidon_hash.iter().map(fp_to_vecu64).collect();
            let mut hash_map = HashMap::new();
            hash_map.insert("poseidon_hash", processed_inputs_poseidon);
            state.serialize_field("processed_inputs", &hash_map)?;
        }

        if let Some(processed_params) = &self.processed_params {
            let processed_params_poseidon: Vec<Vec<u64>> = processed_params.poseidon_hash.iter().map(fp_to_vecu64).collect();
            let mut hash_map = HashMap::new();
            hash_map.insert("poseidon_hash", processed_params_poseidon);
            state.serialize_field("processed_params", &hash_map)?;
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            let processed_outputs_poseidon: Vec<Vec<u64>> = processed_outputs.poseidon_hash.iter().map(fp_to_vecu64).collect();
            let mut hash_map = HashMap::new();
            hash_map.insert("poseidon_hash", processed_outputs_poseidon);
            state.serialize_field("processed_outputs", &hash_map)?;
        }
        state.end()
    }
}
