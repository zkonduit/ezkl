#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use std::io::Read;
// use std::collections::HashMap;

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
#[derive(Clone, Debug, Deserialize, Default)]
pub struct GraphWitness {
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

impl GraphWitness {
    ///
    pub fn new(input_data: Vec<Vec<f32>>, output_data: Vec<Vec<f32>>) -> Self {
        GraphWitness {
            input_data,
            output_data,
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
            on_chain_input_data: None,
        }
    }
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

/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Default, Serialize)]
pub struct GraphInput {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are coming from on-chain).
    /// TODO: Add retrieve from on-chain functionality
    pub input_data: Vec<Vec<f32>>,
}

impl GraphInput {
    ///
    pub fn new(input_data: Vec<Vec<f32>>) -> Self {
        GraphInput { input_data }
    }

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

    ///
    pub fn split_into_batches(
        &self,
        batch_size: usize,
        input_shapes: Vec<Vec<usize>>,
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

        // create a new GraphWitness for each batch
        let batches = input_batches
            .into_iter()
            .map(GraphInput::new)
            .collect::<Vec<GraphInput>>();

        Ok(batches)
    }
}

#[cfg(feature = "python-bindings")]
use halo2curves::{
    bn256::{Fr as Fp, G1Affine},
    ff::PrimeField,
    serde::SerdeObject,
};

#[cfg(feature = "python-bindings")]
/// converts fp into Vec<u64>
fn field_to_vecu64<F: PrimeField + SerdeObject + Serialize>(fp: &F) -> Vec<u64> {
    let repr = serde_json::to_string(&fp).unwrap();
    let b: Vec<u64> = serde_json::from_str(&repr).unwrap();
    b
}

#[cfg(feature = "python-bindings")]
fn insert_poseidon_hash_pydict(pydict: &PyDict, poseidon_hash: &Vec<Fp>) {
    let poseidon_hash: Vec<Vec<u64>> = poseidon_hash.iter().map(field_to_vecu64).collect();
    pydict.set_item("poseidon_hash", poseidon_hash).unwrap();
}

#[cfg(feature = "python-bindings")]
fn g1affine_to_pydict(g1affine_dict: &PyDict, g1affine: &G1Affine) {
    let g1affine_x = field_to_vecu64(&g1affine.x);
    let g1affine_y = field_to_vecu64(&g1affine.y);
    g1affine_dict.set_item("x", g1affine_x).unwrap();
    g1affine_dict.set_item("y", g1affine_y).unwrap();
}

#[cfg(feature = "python-bindings")]
use super::modules::ElGamalResult;
#[cfg(feature = "python-bindings")]
fn insert_elgamal_results_pydict(py: Python, pydict: &PyDict, elgamal_results: &ElGamalResult) {
    let results_dict = PyDict::new(py);
    let cipher_text: Vec<Vec<Vec<u64>>> = elgamal_results
        .ciphertexts
        .iter()
        .map(|v| v.iter().map(field_to_vecu64).collect::<Vec<Vec<u64>>>())
        .collect::<Vec<Vec<Vec<u64>>>>();
    results_dict.set_item("ciphertexts", cipher_text).unwrap();

    let variables_dict = PyDict::new(py);
    let variables = &elgamal_results.variables;

    let r = field_to_vecu64(&variables.r);
    variables_dict.set_item("r", r).unwrap();
    // elgamal secret key
    let sk = field_to_vecu64(&variables.sk);
    variables_dict.set_item("sk", sk).unwrap();

    let pk_dict = PyDict::new(py);
    // elgamal public key
    g1affine_to_pydict(pk_dict, &variables.pk);
    variables_dict.set_item("pk", pk_dict).unwrap();

    let aux_generator_dict = PyDict::new(py);
    // elgamal aux generator used in ecc chip
    g1affine_to_pydict(aux_generator_dict, &variables.aux_generator);
    variables_dict
        .set_item("aux_generator", aux_generator_dict)
        .unwrap();

    // elgamal window size used in ecc chip
    variables_dict
        .set_item("window_size", variables.window_size)
        .unwrap();

    results_dict.set_item("variables", variables_dict).unwrap();

    pydict.set_item("elgamal", results_dict).unwrap();

    //elgamal
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for GraphWitness {
    fn to_object(&self, py: Python) -> PyObject {
        // Create a Python dictionary
        let dict = PyDict::new(py);
        let dict_inputs = PyDict::new(py);
        let dict_params = PyDict::new(py);
        let dict_outputs = PyDict::new(py);

        let input_data_mut = &self.input_data;
        let output_data_mut = &self.output_data;

        dict.set_item("input_data", &input_data_mut).unwrap();
        dict.set_item("output_data", &output_data_mut).unwrap();

        if let Some(processed_inputs) = &self.processed_inputs {
            //poseidon_hash
            if let Some(processed_inputs_poseidon_hash) = &processed_inputs.poseidon_hash {
                insert_poseidon_hash_pydict(&dict_inputs, processed_inputs_poseidon_hash);
            }
            if let Some(processed_inputs_elgamal) = &processed_inputs.elgamal {
                insert_elgamal_results_pydict(py, dict_inputs, processed_inputs_elgamal);
            }

            dict.set_item("processed_inputs", dict_inputs).unwrap();
        }

        if let Some(processed_params) = &self.processed_params {
            if let Some(processed_params_poseidon_hash) = &processed_params.poseidon_hash {
                insert_poseidon_hash_pydict(dict_params, processed_params_poseidon_hash);
            }
            if let Some(processed_params_elgamal) = &processed_params.elgamal {
                insert_elgamal_results_pydict(py, dict_params, processed_params_elgamal);
            }

            dict.set_item("processed_params", dict_params).unwrap();
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            if let Some(processed_outputs_poseidon_hash) = &processed_outputs.poseidon_hash {
                insert_poseidon_hash_pydict(dict_outputs, processed_outputs_poseidon_hash);
            }
            if let Some(processed_outputs_elgamal) = &processed_outputs.elgamal {
                insert_elgamal_results_pydict(py, dict_outputs, processed_outputs_elgamal);
            }

            dict.set_item("processed_outputs", dict_outputs).unwrap();
        }

        dict.to_object(py)
    }
}

impl Serialize for GraphWitness {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphWitness", 4)?;
        let input_data_f64: Vec<Vec<f64>> = self
            .input_data
            .iter()
            .map(|v| v.iter().map(|&f| f as f64).collect())
            .collect();
        let output_data_f64: Vec<Vec<f64>> = self
            .output_data
            .iter()
            .map(|v| v.iter().map(|&f| f as f64).collect())
            .collect();
        state.serialize_field("input_data", &input_data_f64)?;
        state.serialize_field("output_data", &output_data_f64)?;

        if let Some(processed_inputs) = &self.processed_inputs {
            state.serialize_field("processed_inputs", &processed_inputs)?;
        }

        if let Some(processed_params) = &self.processed_params {
            state.serialize_field("processed_params", &processed_params)?;
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            state.serialize_field("processed_outputs", &processed_outputs)?;
        }

        if let Some(on_chain_input_data) = &self.on_chain_input_data {
            state.serialize_field("on_chain_input_data", &on_chain_input_data)?;
        }
        state.end()
    }
}
