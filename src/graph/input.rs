use super::errors::GraphError;
use super::quantize_float;
use crate::circuit::InputType;
use crate::fieldutils::integer_rep_to_felt;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use crate::graph::postgres::Client;
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use crate::tensor::Tensor;
use crate::EZKL_BUF_CAPACITY;
use halo2curves::bn256::Fr as Fp;
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::types::PyDict;
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

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
use tract_onnx::tract_hir::tract_num_traits::ToPrimitive;

type Decimals = u8;
type Call = String;
type RPCUrl = String;

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
pub type FileSource = Vec<Vec<FileSourceInner>>;

/// Represents different types of calls for fetching on-chain data
#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub enum Calls {
    /// Multiple calls to different accounts, each returning individual values
    Multiple(Vec<CallsToAccount>),
    /// Single call returning an array of values
    Single(CallToAccount),
}

impl Default for Calls {
    fn default() -> Self {
        Calls::Multiple(Vec::new())
    }
}

impl Serialize for Calls {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Calls::Single(data) => data.serialize(serializer),
            Calls::Multiple(data) => data.serialize(serializer),
        }
    }
}

// !!! ALWAYS USE JSON SERIALIZATION FOR GRAPH INPUT
// UNTAGGED ENUMS WONT WORK :( as highlighted here:
impl<'de> Deserialize<'de> for Calls {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;
        let multiple_try: Result<Vec<CallsToAccount>, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = multiple_try {
            return Ok(Calls::Multiple(t));
        }
        let single_try: Result<CallToAccount, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = single_try {
            return Ok(Calls::Single(t));
        }

        Err(serde::de::Error::custom("failed to deserialize Calls"))
    }
}
/// Configuration for accessing on-chain data sources
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct OnChainSource {
    /// Call specifications for fetching data
    pub calls: Calls,
    /// RPC endpoint URL for accessing the chain
    pub rpc: RPCUrl,
}

impl OnChainSource {
    /// Creates a new OnChainSource with multiple calls
    ///
    /// # Arguments
    /// * `calls` - Vector of call specifications
    /// * `rpc` - RPC endpoint URL
    pub fn new_multiple(calls: Vec<CallsToAccount>, rpc: RPCUrl) -> Self {
        OnChainSource {
            calls: Calls::Multiple(calls),
            rpc,
        }
    }

    /// Creates a new OnChainSource with a single call
    ///
    /// # Arguments
    /// * `call` - Call specification
    /// * `rpc` - RPC endpoint URL
    pub fn new_single(call: CallToAccount, rpc: RPCUrl) -> Self {
        OnChainSource {
            calls: Calls::Single(call),
            rpc,
        }
    }

    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
    /// Creates test data for the OnChain data source
    /// Used for testing and development purposes
    ///
    /// # Arguments
    /// * `data` - Sample file data to use
    /// * `scales` - Scaling factors for each input
    /// * `shapes` - Shapes of the input tensors
    /// * `rpc` - Optional RPC endpoint override
    pub async fn test_from_file_data(
        data: &FileSource,
        scales: Vec<crate::Scale>,
        mut shapes: Vec<Vec<usize>>,
        rpc: Option<&str>,
    ) -> Result<(Vec<Tensor<Fp>>, Self), GraphError> {
        use crate::eth::{
            evm_quantize_multi, read_on_chain_inputs_multi, test_on_chain_data,
            DEFAULT_ANVIL_ENDPOINT,
        };
        use log::debug;

        // Set up local anvil instance for reading on-chain data
        let (client, client_address) = crate::eth::setup_eth_backend(rpc, None).await?;

        let mut scales = scales;
        // set scales to 1 where data is a field element
        for (idx, i) in data.iter().enumerate() {
            if i.iter().all(|e| e.is_field()) {
                scales[idx] = 0;
                shapes[idx] = vec![i.len()];
            }
        }

        let calls_to_accounts = test_on_chain_data(client.clone(), data).await?;
        debug!("Calls to accounts: {:?}", calls_to_accounts);
        let inputs =
            read_on_chain_inputs_multi(client.clone(), client_address, &calls_to_accounts).await?;
        debug!("Inputs: {:?}", inputs);

        let mut quantized_evm_inputs = vec![];

        let mut prev = 0;
        for (idx, i) in data.iter().enumerate() {
            quantized_evm_inputs.extend(
                evm_quantize_multi(
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
        for (input, shape) in [quantized_evm_inputs].iter().zip(shapes) {
            let mut t: Tensor<Fp> = input.iter().cloned().collect();
            t.reshape(&shape)?;
            inputs.push(t);
        }

        let used_rpc = rpc.unwrap_or(DEFAULT_ANVIL_ENDPOINT).to_string();

        // Fill the input_data field of the GraphData struct
        Ok((
            inputs,
            OnChainSource::new_multiple(calls_to_accounts.clone(), used_rpc),
        ))
    }
}

/// Specification for view-only calls to fetch on-chain data
/// Used for data attestation in smart contract verification
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct CallsToAccount {
    /// Vector of (call data, decimals) pairs
    /// call_data: ABI-encoded function call
    /// decimals: Number of decimal places for float conversion
    pub call_data: Vec<(Call, Decimals)>,
    /// Contract address to call
    pub address: String,
}

/// Specification for a single view-only call returning an array
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct CallToAccount {
    /// ABI-encoded function call data
    pub call_data: Call,
    /// Number of decimal places for float conversion
    pub decimals: Decimals,
    /// Contract address to call
    pub address: String,
    /// Expected length of returned array
    pub len: usize,
}

/// Represents different sources of input/output data for the EZKL model
#[derive(Clone, Debug, Serialize, PartialOrd, PartialEq)]
#[serde(untagged)]
pub enum DataSource {
    /// Data from a JSON file containing arrays of values
    File(FileSource),
    /// Data fetched from blockchain contracts
    OnChain(OnChainSource),
    /// Data from a PostgreSQL database
    #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
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

// Note: Always use JSON serialization for untagged enums
impl<'de> Deserialize<'de> for DataSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let this_json: Box<serde_json::value::RawValue> = Deserialize::deserialize(deserializer)?;

        // Try deserializing as FileSource first
        let first_try: Result<FileSource, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = first_try {
            return Ok(DataSource::File(t));
        }

        // Try deserializing as OnChainSource
        let second_try: Result<OnChainSource, _> = serde_json::from_str(this_json.get());
        if let Ok(t) = second_try {
            return Ok(DataSource::OnChain(t));
        }

        // Try deserializing as PostgresSource if feature enabled
        #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
        {
            let third_try: Result<PostgresSource, _> = serde_json::from_str(this_json.get());
            if let Ok(t) = third_try {
                return Ok(DataSource::DB(t));
            }
        }

        Err(serde::de::Error::custom("failed to deserialize DataSource"))
    }
}

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
        match &self.input_data {
            DataSource::File(data) => {
                for (i, input) in data.iter().enumerate() {
                    if !input.is_empty() {
                        let dt = datum_types[i];
                        let input = input.iter().map(|e| e.to_float()).collect::<Vec<f64>>();
                        let tt = TractTensor::from_shape(&shapes[i], &input)?;
                        let tt = tt.cast_to_dt(dt)?;
                        inputs.push(tt.into_owned().into());
                    }
                }
            }
            _ => {
                return Err(GraphError::InvalidDims(
                    0,
                    "non file data cannot be split into batches".to_string(),
                ))
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
            input_data: DataSource::File(input_data),
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
    pub async fn split_into_batches(
        &self,
        input_shapes: Vec<Vec<usize>>,
    ) -> Result<Vec<Self>, GraphError> {
        let mut batched_inputs = vec![];

        let iterable = match self {
            GraphData {
                input_data: DataSource::File(data),
                output_data: _,
            } => data.clone(),
            GraphData {
                input_data: DataSource::OnChain(_),
                output_data: _,
            } => {
                return Err(GraphError::InvalidDims(
                    0,
                    "on-chain data cannot be split into batches".to_string(),
                ))
            }
            #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
            GraphData {
                input_data: DataSource::DB(data),
                output_data: _,
            } => data.fetch_and_format_as_file().await?,
        };

        // Process each input tensor according to its shape
        for (i, shape) in input_shapes.iter().enumerate() {
            let input_size = shape.clone().iter().product::<usize>();
            let input = &iterable[i];

            // Validate input size is divisible by batch size
            if input.len() % input_size != 0 {
                return Err(GraphError::InvalidDims(
                    0,
                    "calibration data length must be evenly divisible by the original input_size"
                        .to_string(),
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
            input_batches.push(DataSource::File(batch));
        }

        // Ensure at least one batch exists
        if input_batches.is_empty() {
            input_batches.push(DataSource::File(vec![vec![]]));
        }

        // Create GraphData instance for each batch
        let batches = input_batches
            .into_iter()
            .map(GraphData::new)
            .collect::<Vec<GraphData>>();

        Ok(batches)
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for CallsToAccount {
    /// Converts CallsToAccount to Python object
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("account", &self.address).unwrap();
        dict.set_item("call_data", &self.call_data).unwrap();
        dict.to_object(py)
    }
}

// Additional Python bindings for various types...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postgres_source_new() {
        #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
        {
            let source = PostgresSource::new(
                "localhost".to_string(),
                "5432".to_string(),
                "user".to_string(),
                "SELECT * FROM table".to_string(),
                "database".to_string(),
                "password".to_string(),
            );

            assert_eq!(source.host, "localhost");
            assert_eq!(source.port, "5432");
            assert_eq!(source.user, "user");
            assert_eq!(source.query, "SELECT * FROM table");
            assert_eq!(source.dbname, "database");
            assert_eq!(source.password, "password");
        }
    }

    #[test]
    fn test_data_source_serialization_round_trip() {
        // Test backwards compatibility with old format
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
    fn test_graph_input_serialization_round_trip() {
        // Test serialization/deserialization of graph input
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

    #[test]
    fn test_python_compat() {
        // Test compatibility with mclbn256 library serialization
        let source = Fp::from_raw([18445520602771460712, 838677322461845011, 3079992810, 0]);
        let original_addr = "0x000000000000000000000000b794f5ea0ba39494ce839613fffba74279579268";
        assert_eq!(format!("{:?}", source), original_addr);
    }
}

/// Source data from a PostgreSQL database
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
#[derive(Clone, Debug, Deserialize, Serialize, Default, PartialOrd, PartialEq)]
pub struct PostgresSource {
    /// Database host address
    pub host: RPCUrl,
    /// Database user name
    pub user: String,
    /// Database password
    pub password: String,
    /// SQL query to execute
    pub query: String,
    /// Database name
    pub dbname: String,
    /// Database port
    pub port: String,
}

#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
impl PostgresSource {
    /// Creates a new PostgreSQL data source
    pub fn new(
        host: RPCUrl,
        port: String,
        user: String,
        query: String,
        dbname: String,
        password: String,
    ) -> Self {
        PostgresSource {
            host,
            user,
            password,
            query,
            dbname,
            port,
        }
    }

    /// Fetches data from the PostgreSQL database
    pub async fn fetch(&self) -> Result<Vec<Vec<pg_bigdecimal::PgNumeric>>, GraphError> {
        // Configuration string
        let config = if self.password.is_empty() {
            format!(
                "host={} user={} dbname={} port={}",
                self.host, self.user, self.dbname, self.port
            )
        } else {
            format!(
                "host={} user={} dbname={} port={} password={}",
                self.host, self.user, self.dbname, self.port, self.password
            )
        };

        let mut client = Client::connect(&config).await?;
        let mut res: Vec<pg_bigdecimal::PgNumeric> = Vec::new();

        // Extract rows from query
        for row in client.query(&self.query, &[]).await? {
            for i in 0..row.len() {
                res.push(row.get(i));
            }
        }
        Ok(vec![res])
    }

    /// Fetches and formats data as FileSource
    pub async fn fetch_and_format_as_file(&self) -> Result<Vec<Vec<FileSourceInner>>, GraphError> {
        Ok(self
            .fetch()
            .await?
            .iter()
            .map(|d| {
                d.iter()
                    .map(|d| {
                        FileSourceInner::Float(
                            d.n.as_ref()
                                .unwrap()
                                .to_f64()
                                .ok_or("could not convert decimal to f64")
                                .unwrap(),
                        )
                    })
                    .collect()
            })
            .collect())
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for CallToAccount {
    fn to_object(&self, py: Python) -> PyObject {
        let dict = PyDict::new(py);
        dict.set_item("account", &self.address).unwrap();
        dict.set_item("call_data", &self.call_data).unwrap();
        dict.set_item("decimals", &self.decimals).unwrap();
        dict.set_item("len", &self.len).unwrap();
        dict.to_object(py)
    }
}

#[cfg(feature = "python-bindings")]
impl ToPyObject for Calls {
    fn to_object(&self, py: Python) -> PyObject {
        match self {
            Calls::Multiple(calls) => calls.to_object(py),
            Calls::Single(call) => call.to_object(py),
        }
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
                dict.set_item("calls_to_accounts", &source.calls.to_object(py))
                    .unwrap();
                dict.to_object(py)
            }
            #[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
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
