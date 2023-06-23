#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
// use serde::de::{Visitor, MapAccess};
// use serde::de::{Visitor, MapAccess};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(not(target_arch = "wasm32"))]
use crate::tensor::Tensor;
// use std::collections::HashMap;
use std::io::Read;
// use std::collections::HashMap;

use super::{modules::ModuleForwardResult, GraphError};

type Decimals = u8;
type Call = String;
type RPCUrl = String;

/// Inner elements of inputs/outputs coming from a file
pub type FileSourceInner = Vec<Vec<f32>>;
/// Inner elements of inputs/outputs coming from on-chain
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct OnChainSourceInner {
    /// Vector of calls to accounts
    pub calls: Vec<CallsToAccount>,
    /// RPC url
    pub rpc: RPCUrl,
}

impl OnChainSourceInner {
    /// Create a new OnChainSourceInner
    pub fn new(calls: Vec<CallsToAccount>, rpc: RPCUrl) -> Self {
        OnChainSourceInner { calls, rpc }
    }
}

impl OnChainSourceInner {
    #[cfg(not(target_arch = "wasm32"))]
    /// Create dummy local on-chain data to test the OnChain data source
    pub async fn test_from_file_data(
        data: &FileSourceInner,
        scales: Vec<u32>,
        shapes: Vec<Vec<usize>>,
    ) -> Result<(Vec<Tensor<i128>>, Self), Box<dyn std::error::Error>> {
        use crate::eth::{evm_quantize, read_on_chain_inputs, test_on_chain_data};
        use crate::graph::scale_to_multiplier;
        use log::debug;

        // Set up local anvil instance for reading on-chain data
        let (anvil, client) = crate::eth::setup_eth_backend(None).await?;

        let address = client.address();

        let calls_to_accounts = test_on_chain_data(client.clone(), &data).await?;
        debug!("Calls to accounts: {:?}", calls_to_accounts);
        let inputs = read_on_chain_inputs(client.clone(), address, &calls_to_accounts).await?;
        debug!("Inputs: {:?}", inputs);

        let mut quantized_evm_inputs = vec![];
        let scales: Vec<f64> = scales.into_iter().map(scale_to_multiplier).collect();

        let mut prev = 0;
        for (idx, i) in data.iter().enumerate() {
            quantized_evm_inputs.extend(
                evm_quantize(
                    client.clone(),
                    vec![scales[idx]; i.len()],
                    &(inputs.0[prev..i.len()].to_vec(), inputs.1[prev..i.len()].to_vec()),
                ).await?
            );
            prev += i.len();
        }

        // on-chain data has already been quantized at this point. Just need to reshape it and push into tensor vector
        let mut inputs: Vec<Tensor<i128>> = vec![];
        for (input, shape) in vec![quantized_evm_inputs].iter().zip(shapes) {
            let mut t: Tensor<i128> = input.iter().cloned().collect();
            t.reshape(&shape);
            inputs.push(t);
        }
        // Fill the input_data field of the GraphInput struct
        Ok((
            inputs,
            OnChainSourceInner::new(
                calls_to_accounts.clone(),
                anvil.endpoint(),
            )
        ))
    }
}

/// Defines the view only calls to accounts to fetch the on-chain input data.
/// This data will be included as part of the first elements in the publicInputs
/// for the sol evm verifier and will be  verifyWithDataAttestation.sol
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
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
/// Enum that defines source of the inputs/outputs to the EZKL model
#[derive(Clone, Debug, Serialize, PartialOrd, PartialEq)]
#[serde(untagged)]
pub enum DataSource {
    /// .json File data source.
    File(FileSourceInner),
    /// On-chain data source. The first element is the calls to the account, and the second is the RPC url.
    OnChain(OnChainSourceInner),
}
impl Default for DataSource {
    fn default() -> Self {
        DataSource::File(vec![vec![]])
    }
}

impl From<FileSourceInner> for DataSource {
    fn from(data: FileSourceInner) -> Self {
        DataSource::File(data)
    }
}

impl From<OnChainSourceInner> for DataSource {
    fn from(data: OnChainSourceInner) -> Self {
        DataSource::OnChain(data)
    }
}

/// Enum that defines source of the inputs/outputs to the EZKL model
/// used for f32 to f64 conversion
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum DataSourceF64 {
    OnChain(OnChainSourceInner),
    File(Vec<Vec<f64>>),
}

impl From<DataSource> for DataSourceF64 {
    fn from(source: DataSource) -> Self {
        match source {
            DataSource::File(data) => {
                let data = data
                    .iter()
                    .map(|v| v.iter().map(|&f| f as f64).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                DataSourceF64::File(data)
            }
            DataSource::OnChain(source) => DataSourceF64::OnChain(source),
        }
    }
}

/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Default)]
pub struct GraphWitness {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are coming from on-chain).
    /// TODO: Add retrieve from on-chain functionality
    pub input_data: DataSource,
    /// The expected output of the model (can be empty vectors if outputs are not being constrained).
    pub output_data: DataSource,
    /// Optional hashes of the inputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_inputs: Option<ModuleForwardResult>,
    /// Optional hashes of the params (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_params: Option<ModuleForwardResult>,
    /// Optional hashes of the outputs (can be None if there are no commitments). Wrapped as Option for backwards compatibility
    pub processed_outputs: Option<ModuleForwardResult>,
}

impl GraphWitness {
    ///
    pub fn new(input_data: DataSource, output_data: DataSource) -> Self {
        GraphWitness {
            input_data,
            output_data,
            processed_inputs: None,
            processed_params: None,
            processed_outputs: None,
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
/// Input to graph as a datasource
/// Always use JSON serialization for GraphInput. Seriously.
#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
pub struct GraphInput {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are coming from on-chain).
    pub input_data: DataSource,
}

impl From<GraphWitness> for GraphInput {
    fn from(witness: GraphWitness) -> Self {
        GraphInput {
            input_data: witness.input_data,
        }
    }
}

impl GraphInput {
    ///
    pub fn new(input_data: DataSource) -> Self {
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

        let iterable = match self {
            GraphInput {
                input_data: DataSource::File(data),
            } => data,
            GraphInput {
                input_data: DataSource::OnChain(_),
            } => {
                todo!("on-chain data batching not implemented yet")
            }
        };

        for (i, input) in iterable.iter().enumerate() {
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
            input_batches.push(DataSource::File(batch));
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
impl ToPyObject for CallsToAccount {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("account", &self.address).unwrap();
        dict.set_item("call_data", &self.call_data).unwrap();
        dict.to_object(py)
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for DataSource {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            DataSource::File(data) => data.to_object(py),
            DataSource::OnChain(source) => {
                let dict = PyDict::new(py);
                dict.set_item("rpc_url", &source.rpc).unwrap();
                dict.set_item("calls_to_accounts", &source.calls).unwrap();
                dict.to_object(py)
            }
        }
    }
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

impl Serialize for GraphInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphInput", 4)?;
        let input_data: DataSourceF64 = self.input_data.clone().into();
        state.serialize_field("input_data", &input_data)?;
        state.end()
    }
}

// !!! ALWAYS USE JSON SERIALIZATION FOR GRAPH INPUT
// UNTAGGED ENUMS WONT WORK :( as highlighted here:
impl<'de> Deserialize<'de> for DataSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        let first_try: Result<FileSourceInner, _> = serde_json::from_str(this_json.get());

        if first_try.is_ok() {
            return Ok(DataSource::File(first_try.unwrap()));
        }
        let second_try: Result<OnChainSourceInner, _> = serde_json::from_str(this_json.get());
        if second_try.is_ok() {
            return Ok(DataSource::OnChain(second_try.unwrap()));
        }

        Err(serde::de::Error::custom("failed to deserialize DataSource"))
    }
}

impl Serialize for GraphWitness {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphWitness", 4)?;
        let input_data: DataSourceF64 = self.input_data.clone().into();
        let output_data: DataSourceF64 = self.output_data.clone().into();
        state.serialize_field("input_data", &input_data)?;
        state.serialize_field("output_data", &output_data)?;

        if let Some(processed_inputs) = &self.processed_inputs {
            state.serialize_field("processed_inputs", &processed_inputs)?;
        }

        if let Some(processed_params) = &self.processed_params {
            state.serialize_field("processed_params", &processed_params)?;
        }

        if let Some(processed_outputs) = &self.processed_outputs {
            state.serialize_field("processed_outputs", &processed_outputs)?;
        }
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // this is for backwards compatibility with the old format
    fn test_data_source_serialization_round_trip() {
        let source = DataSource::File(vec![vec![
            0.05326242372393608,
            0.07497056573629379,
            0.05235547572374344,
        ]]);

        let serialized = serde_json::to_string(&source).unwrap();

        const JSON: &str = r#"[[0.053262424,0.074970566,0.052355476]]"#;

        assert_eq!(serialized, JSON);

        println!("serialized {:?}", serialized);

        let expect = serde_json::from_str::<DataSource>(JSON)
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(expect, source);
    }

    #[test]
    // this is for backwards compatibility with the old format
    fn test_graph_input_serialization_round_trip() {
        let file = GraphInput::new(DataSource::File(vec![vec![
            0.05326242372393608,
            0.07497056573629379,
            0.05235547572374344,
        ]]));

        let serialized = serde_json::to_string(&file).unwrap();

        const JSON: &str =
            r#"{"input_data":[[0.05326242372393608,0.07497056573629379,0.05235547572374344]]}"#;

        assert_eq!(serialized, JSON);

        println!("serialized {:?}", serialized);

        let graph_input3 = serde_json::from_str::<GraphInput>(&JSON)
            .map_err(|e| e.to_string())
            .unwrap();
        println!("{:?}", graph_input3.input_data);
        assert_eq!(graph_input3, file);
    }
}
