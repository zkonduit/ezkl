use std::convert::Infallible;

use thiserror::Error;

/// circuit related errors.
#[derive(Debug, Error)]
pub enum GraphError {
    /// The wrong inputs were passed to a lookup node
    #[error("invalid inputs for a lookup node")]
    InvalidLookupInputs,
    /// Shape mismatch in circuit construction
    #[error("invalid dimensions used for node {0} ({1})")]
    InvalidDims(usize, String),
    /// Wrong method was called to configure an op
    #[error("wrong method was called to configure node {0} ({1})")]
    WrongMethod(usize, String),
    /// A requested node is missing in the graph
    #[error("a requested node is missing in the graph: {0}")]
    MissingNode(usize),
    /// The wrong method was called on an operation
    #[error("an unsupported method was called on node {0} ({1})")]
    OpMismatch(usize, String),
    /// This operation is unsupported
    #[error("unsupported datatype in graph node {0} ({1})")]
    UnsupportedDataType(usize, String),
    /// A node has missing parameters
    #[error("a node is missing required params: {0}")]
    MissingParams(String),
    /// A node has missing parameters
    #[error("a node is has misformed params: {0}")]
    MisformedParams(String),
    /// Error in the configuration of the visibility of variables
    #[error("there should be at least one set of public variables")]
    Visibility,
    /// Ezkl only supports divisions by constants
    #[error("ezkl currently only supports division by constants")]
    NonConstantDiv,
    /// Ezkl only supports constant powers
    #[error("ezkl currently only supports constant exponents")]
    NonConstantPower,
    /// Error when attempting to rescale an operation
    #[error("failed to rescale inputs for {0}")]
    RescalingError(String),
    /// Reading a file failed
    #[error("[io] ({0}) {1}")]
    ReadWriteFileError(String, String),
    /// Model serialization error
    #[error("failed to ser/deser model: {0}")]
    ModelSerialize(#[from] bincode::Error),
    /// Tract error
    #[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
    #[error("[tract] {0}")]
    TractError(#[from] tract_onnx::prelude::TractError),
    /// Packing exponent is too large
    #[error("largest packing exponent exceeds max. try reducing the scale")]
    PackingExponent,
    /// Invalid Input Types
    #[error("invalid input types")]
    InvalidInputTypes,
    /// Missing results
    #[error("missing results")]
    MissingResults,
    /// Tensor error
    #[error("[tensor] {0}")]
    TensorError(#[from] crate::tensor::TensorError),
    /// Public visibility for params is deprecated
    #[error("public visibility for params is deprecated, please use `fixed` instead")]
    ParamsPublicVisibility,
    /// Slice length mismatch
    #[error("slice length mismatch: {0}")]
    SliceLengthMismatch(#[from] std::array::TryFromSliceError),
    /// Bad conversion
    #[error("invalid conversion: {0}")]
    InvalidConversion(#[from] Infallible),
    /// Circuit error
    #[error("[circuit] {0}")]
    CircuitError(#[from] crate::circuit::CircuitError),
    /// Halo2 error
    #[error("[halo2] {0}")]
    Halo2Error(#[from] halo2_proofs::plonk::Error),
    /// System time error
    #[error("[system time] {0}")]
    SystemTimeError(#[from] std::time::SystemTimeError),
    /// Missing Batch Size
    #[error("unknown dimension batch_size in model inputs, set batch_size in variables")]
    MissingBatchSize,
    /// Tokio postgres error
    #[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
    #[error("[tokio postgres] {0}")]
    TokioPostgresError(#[from] tokio_postgres::Error),
    /// Eth error
    #[cfg(not(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown"))))]
    #[error("[eth] {0}")]
    EthError(#[from] crate::eth::EthError),
    /// Json error
    #[error("[json] {0}")]
    JsonError(#[from] serde_json::Error),
    /// Missing instances
    #[error("missing instances")]
    MissingInstances,
    /// Missing constants
    #[error("missing constants")]
    MissingConstants,
    /// Missing input for a node
    #[error("missing input for node {0}")]
    MissingInput(usize),
    ///
    #[error("range only supports constant inputs in a zk circuit")]
    NonConstantRange,
    ///
    #[error("trilu only supports constant diagonals in a zk circuit")]
    NonConstantTrilu,
    ///
    #[error("insufficient witness values to generate a fixed output")]
    InsufficientWitnessValues,
    /// Missing scale
    #[error("missing scale")]
    MissingScale,
    /// Extended k is too large
    #[error("extended k is too large to accommodate the quotient polynomial with logrows {0}")]
    ExtendedKTooLarge(u32),
    /// Max lookup input is too large
    #[error("lookup range {0} is too large")]
    LookupRangeTooLarge(usize),
    /// Max range check input is too large
    #[error("range check {0} is too large")]
    RangeCheckTooLarge(usize),
    ///Cannot use on-chain data source as private data
    #[error("cannot use on-chain data source as 1) output for on-chain test 2) as private data 3) as input when using wasm.")]
    OnChainDataSource,
    /// Missing data source
    #[error("missing data source")]
    MissingDataSource,
    /// Invalid RunArg
    #[error("invalid RunArgs: {0}")]
    InvalidRunArgs(String),
}
