use crate::fieldutils::i128_to_felt;
use crate::pfsys::field_to_vecu64;
#[cfg(not(target_arch = "wasm32"))]
use crate::tensor::Tensor;
use halo2curves::bn256::Fr as Fp;
#[cfg(not(target_arch = "wasm32"))]
use postgres::{Client, NoTls};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
#[cfg(feature = "python-bindings")]
use pyo3::ToPyObject;
#[cfg(not(target_arch = "wasm32"))]
use rust_decimal::prelude::ToPrimitive;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

use super::quantize_float;
use super::GraphError;

type Decimals = u8;
type Call = String;
type RPCUrl = String;

///
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub enum FileSourceInner {
    /// Inner elements of inputs coming from a file
    Float(f64),
    /// Inner elements of inputs coming from a witness
    Field(Fp),
}

impl Serialize for FileSourceInner {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            FileSourceInner::Field(data) => field_to_vecu64(data).serialize(serializer),
            FileSourceInner::Float(data) => data.serialize(serializer),
        }
    }
}

// !!! ALWAYS USE JSON SERIALIZATION FOR GRAPH INPUT
// UNTAGGED ENUMS WONT WORK :( as highlighted here:
impl<'de> Deserialize<'de> for FileSourceInner {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        let first_try: Result<f64, _> = serde_json::from_str(this_json.get());

        if let Ok(t) = first_try {
            return Ok(FileSourceInner::Float(t));
        }
        let second_try: Result<[u64; 4], _> = serde_json::from_str(this_json.get());
        if let Ok(t) = second_try {
            return Ok(FileSourceInner::Field(Fp::from_raw(t)));
        }

        Err(serde::de::Error::custom(
            "failed to deserialize FileSourceInner",
        ))
    }
}

/// Elements of inputs coming from a file
pub type FileSource = Vec<Vec<FileSourceInner>>;

impl FileSourceInner {
    /// Create a new FileSourceInner
    pub fn new_float(f: f64) -> Self {
        FileSourceInner::Float(f)
    }
    /// Create a new FileSourceInner
    pub fn new_field(f: Fp) -> Self {
        FileSourceInner::Field(f)
    }

    /// Convert to a field element
    pub fn to_field(&self, scale: u32) -> Fp {
        match self {
            FileSourceInner::Float(f) => i128_to_felt(quantize_float(f, 0.0, scale).unwrap()),
            FileSourceInner::Field(f) => *f,
        }
    }
    /// Convert to a float
    pub fn to_float(&self) -> f64 {
        match self {
            FileSourceInner::Float(f) => *f,
            FileSourceInner::Field(f) => crate::fieldutils::felt_to_i128(*f) as f64,
        }
    }
}

/// Inner elements of inputs/outputs coming from on-chain
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct OnChainSource {
    /// Vector of calls to accounts
    pub calls: Vec<CallsToAccount>,
    /// RPC url
    pub rpc: RPCUrl,
}

impl OnChainSource {
    /// Create a new OnChainSource
    pub fn new(calls: Vec<CallsToAccount>, rpc: RPCUrl) -> Self {
        OnChainSource { calls, rpc }
    }
}

#[cfg(not(target_arch = "wasm32"))]
/// Inner elements of inputs/outputs coming from postgres DB
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct PostgresSource {
    /// postgres host
    pub host: RPCUrl,
    /// user to connect to postgres
    pub user: String,
    /// query to execute
    pub query: String,
    /// dbname
    pub dbname: String,
    /// port
    pub port: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl PostgresSource {
    /// Create a new PostgresSource
    pub fn new(host: RPCUrl, port: String, user: String, query: String, dbname: String) -> Self {
        PostgresSource {
            host,
            user,
            query,
            dbname,
            port,
        }
    }

    /// Fetch data from postgres
    pub fn fetch(&self) -> Result<Vec<Vec<rust_decimal::Decimal>>, Box<dyn std::error::Error>> {
        // clone to move into thread
        let user = self.user.clone();
        let host = self.host.clone();
        let query = self.query.clone();
        let dbname = self.dbname.clone();
        let port = self.port.clone();

        println!("fetching data from postgres: {}", query);
        println!("host: {}", host);
        print!("user: {}", user);
        println!("dbname: {}", dbname);

        let res: Vec<rust_decimal::Decimal> = thread::spawn(move || {
            let mut client = Client::connect(
                &format!(
                    "host={} user={} dbname={} port={}",
                    host, user, dbname, port
                ),
                NoTls,
            )
            .unwrap();
            let mut res: Vec<rust_decimal::Decimal> = Vec::new();
            // extract rows from query
            for row in client.query(&query, &[]).unwrap() {
                // extract features from row
                for i in 0..row.len() {
                    res.push(row.get(i));
                }
            }
            res
        })
        .join()
        .map_err(|_| "failed to fetch data from postgres")?;

        Ok(vec![res])
    }

    /// Fetch data from postgres and format it as a FileSource
    pub fn fetch_and_format_as_file(
        &self,
    ) -> Result<Vec<Vec<FileSourceInner>>, Box<dyn std::error::Error>> {
        Ok(self
            .fetch()?
            .iter()
            .map(|d| {
                d.iter()
                    .map(|d| {
                        FileSourceInner::Float(
                            d.to_f64()
                                .ok_or("could not convert decimal to f64")
                                .unwrap(),
                        )
                    })
                    .collect()
            })
            .collect())
    }
}

impl OnChainSource {
    #[cfg(not(target_arch = "wasm32"))]
    /// Create dummy local on-chain data to test the OnChain data source
    pub async fn test_from_file_data(
        data: &FileSource,
        scales: Vec<u32>,
        shapes: Vec<Vec<usize>>,
        rpc: Option<&str>,
    ) -> Result<(Vec<Tensor<Fp>>, Self), Box<dyn std::error::Error>> {
        use crate::eth::{evm_quantize, read_on_chain_inputs, test_on_chain_data};
        use crate::graph::scale_to_multiplier;
        use itertools::Itertools;
        use log::debug;

        // Set up local anvil instance for reading on-chain data
        let (anvil, client) = crate::eth::setup_eth_backend(rpc).await?;

        let address = client.address();

        let scales: Vec<f64> = scales.into_iter().map(scale_to_multiplier).collect();

        // unquantize data
        let float_data = data
            .iter()
            .map(|t| t.iter().map(|e| (e.to_float() as f32)).collect_vec())
            .collect::<Vec<Vec<f32>>>();

        let calls_to_accounts = test_on_chain_data(client.clone(), &float_data).await?;
        debug!("Calls to accounts: {:?}", calls_to_accounts);
        let inputs = read_on_chain_inputs(client.clone(), address, &calls_to_accounts).await?;
        debug!("Inputs: {:?}", inputs);

        let mut quantized_evm_inputs = vec![];

        let mut prev = 0;
        for (idx, i) in data.iter().enumerate() {
            quantized_evm_inputs.extend(
                evm_quantize(
                    client.clone(),
                    vec![scales[idx]; i.len()],
                    &(
                        inputs.0[prev..i.len()].to_vec(),
                        inputs.1[prev..i.len()].to_vec(),
                    ),
                )
                .await?,
            );
            prev += i.len();
        }

        // on-chain data has already been quantized at this point. Just need to reshape it and push into tensor vector
        let mut inputs: Vec<Tensor<Fp>> = vec![];
        for (input, shape) in vec![quantized_evm_inputs].iter().zip(shapes) {
            let mut t: Tensor<Fp> = input.iter().cloned().collect();
            t.reshape(&shape);
            inputs.push(t);
        }

        let used_rpc = rpc.unwrap_or(&anvil.endpoint()).to_string();

        // Fill the input_data field of the GraphData struct
        Ok((
            inputs,
            OnChainSource::new(calls_to_accounts.clone(), used_rpc),
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
    /// elementary type (<https://docs.soliditylang.org/en/v0.8.20/abi-spec.html#types>).
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
    File(FileSource),
    /// On-chain data source. The first element is the calls to the account, and the second is the RPC url.
    OnChain(OnChainSource),
    /// Postgres DB
    #[cfg(not(target_arch = "wasm32"))]
    DB(PostgresSource),
}
impl Default for DataSource {
    fn default() -> Self {
        DataSource::File(vec![vec![]])
    }
}

impl From<FileSource> for DataSource {
    fn from(data: FileSource) -> Self {
        DataSource::File(data)
    }
}

impl From<Vec<Vec<Fp>>> for DataSource {
    fn from(data: Vec<Vec<Fp>>) -> Self {
        DataSource::File(
            data.iter()
                .map(|e| e.iter().map(|e| FileSourceInner::Field(*e)).collect())
                .collect(),
        )
    }
}

impl From<Vec<Vec<f64>>> for DataSource {
    fn from(data: Vec<Vec<f64>>) -> Self {
        DataSource::File(
            data.iter()
                .map(|e| e.iter().map(|e| FileSourceInner::Float(*e)).collect())
                .collect(),
        )
    }
}

impl From<OnChainSource> for DataSource {
    fn from(data: OnChainSource) -> Self {
        DataSource::OnChain(data)
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

        let first_try: Result<FileSource, _> = serde_json::from_str(this_json.get());

        if let Ok(t) = first_try {
            return Ok(DataSource::File(t));
        }
        let second_try: Result<OnChainSource, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = second_try {
            return Ok(DataSource::OnChain(t));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let third_try: Result<PostgresSource, _> = serde_json::from_str(this_json.get());
            if let Ok(t) = third_try {
                return Ok(DataSource::DB(t));
            }
        }

        Err(serde::de::Error::custom("failed to deserialize DataSource"))
    }
}

/// Input to graph as a datasource
/// Always use JSON serialization for GraphData. Seriously.
#[derive(Clone, Debug, Deserialize, Default, PartialEq)]
pub struct GraphData {
    /// Inputs to the model / computational graph (can be empty vectors if inputs are coming from on-chain).
    pub input_data: DataSource,
    /// Outputs of the model / computational graph (can be empty vectors if outputs are coming from on-chain).
    pub output_data: Option<DataSource>,
}

impl GraphData {
    ///
    pub fn new(input_data: DataSource) -> Self {
        GraphData {
            input_data,
            output_data: None,
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

    ///
    pub fn split_into_batches(
        &self,
        batch_size: usize,
        input_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        // split input data into batches
        let mut batched_inputs = vec![];

        let iterable = match self {
            GraphData {
                input_data: DataSource::File(data),
                output_data: _,
            } => data.clone(),
            GraphData {
                input_data: DataSource::OnChain(_),
                output_data: _,
            } => todo!("on-chain data batching not implemented yet"),
            #[cfg(not(target_arch = "wasm32"))]
            GraphData {
                input_data: DataSource::DB(data),
                output_data: _,
            } => data.fetch_and_format_as_file()?,
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
            .map(GraphData::new)
            .collect::<Vec<GraphData>>();

        Ok(batches)
    }
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
            DataSource::DB(source) => {
                let dict = PyDict::new(py);
                dict.set_item("host", &source.host).unwrap();
                dict.set_item("user", &source.user).unwrap();
                dict.set_item("query", &source.query).unwrap();
                dict.to_object(py)
            }
        }
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for FileSourceInner {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            FileSourceInner::Field(data) => field_to_vecu64(data).to_object(py),
            FileSourceInner::Float(data) => data.to_object(py),
        }
    }
}

impl Serialize for GraphData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphData", 4)?;
        state.serialize_field("input_data", &self.input_data)?;
        state.serialize_field("output_data", &self.output_data)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // this is for backwards compatibility with the old format
    fn test_data_source_serialization_round_trip() {
        let source = DataSource::from(vec![vec![0.053_262_424, 0.074_970_566, 0.052_355_476]]);

        let serialized = serde_json::to_string(&source).unwrap();

        const JSON: &str = r#"[[0.053262424,0.074970566,0.052355476]]"#;

        assert_eq!(serialized, JSON);

        let expect = serde_json::from_str::<DataSource>(JSON)
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(expect, source);
    }

    #[test]
    // this is for backwards compatibility with the old format
    fn test_graph_input_serialization_round_trip() {
        let file = GraphData::new(DataSource::from(vec![vec![
            0.05326242372393608,
            0.07497056573629379,
            0.05235547572374344,
        ]]));

        let serialized = serde_json::to_string(&file).unwrap();

        const JSON: &str = r#"{"input_data":[[0.05326242372393608,0.07497056573629379,0.05235547572374344]],"output_data":null}"#;

        assert_eq!(serialized, JSON);

        let graph_input3 = serde_json::from_str::<GraphData>(JSON)
            .map_err(|e| e.to_string())
            .unwrap();
        assert_eq!(graph_input3, file);
    }

    //  test for the compatibility with the serialized elements from the mclbn256 library
    #[test]
    fn test_python_compat() {
        let source = Fp::from_raw([18445520602771460712, 838677322461845011, 3079992810, 0]);

        let original_addr = "0x000000000000000000000000b794f5ea0ba39494ce839613fffba74279579268";

        assert_eq!(format!("{:?}", source), original_addr);
    }
}
