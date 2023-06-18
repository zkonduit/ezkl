#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
use serde::{Deserialize, Serialize};
use std::io::Read;

use super::{modules::ModuleForwardResult, GraphError};


type Decimals = u8;
type Call = String;
type RPCUrl = String;
/// Defines the view only calls to accounts to fetch the on-chain input data.
/// This data will be included as part of the first elements in the publicInputs
/// for the sol evm verifier and will be  verifyWithDataAttestation.sol
#[derive(Clone, Debug, Deserialize, Serialize, Default)]

pub struct CallsToAccount {
    /// A vector of tuples, where index 0 of tuples
    /// are the byte strings representing the ABI encoded function calls to 
    /// read the data from the address. This call must return a single
    /// elementary type (https://docs.soliditylang.org/en/v0.8.20/abi-spec.html#types).
    /// The second index of the tuple is the number of decimals for f32 conversion.
    /// We don't support dynamic types currently.
    pub call_data: Vec<(Call, Decimals)>,
    /// Address of the contract to read the data from.
    pub address: String,
}
/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct GraphInput {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are coming from on-chain).
    /// TODO: Add retrieve from on-chain functionality
    pub input_data: Vec<Vec<f32>>, 
    /// The expected output of the model (can be empty vectors if outputs are not being constrained).
    pub output_data: Vec<Vec<f32>>,
    /// Optional hashes of the inputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_inputs: Option<ModuleForwardResult>,
    /// Optional hashes of the params (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_params: Option<ModuleForwardResult>,
    /// Optional hashes of the outputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_outputs: Option<ModuleForwardResult>,
    /// Optional on-chain inputs. (can be None if there are no on-chain inputs). Wrapped as Option for backwards compatibility
    pub on_chain_input_data: Option<(Vec<CallsToAccount>, RPCUrl)>,
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
            on_chain_input_data: None
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

#[cfg(feature = "python-bindings")]
impl ToPyObject for GraphInput {
    fn to_object(&self, py: Python) -> PyObject {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let input_data_mut = &self.input_data;
        let output_data_mut = &self.output_data;
        dict.set_item("input_data", truncate_nested_vector(&input_data_mut))
            .unwrap();
        dict.set_item("output_data", truncate_nested_vector(&output_data_mut))
            .unwrap();

        dict.to_object(py)
    }
}

/// Truncates nested vector due to omit junk floating point values in python
#[cfg(feature = "python-bindings")]
fn truncate_nested_vector(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut input_mut = input.clone();
    for inner_vec in input_mut.iter_mut() {
        for value in inner_vec.iter_mut() {
            // truncate 6 decimal places
            *value = (*value * 10000000.0).trunc() / 10000000.0;
        }
    }
    input_mut
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