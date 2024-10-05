/// Universal bindings for all platforms
#[cfg(any(target_os = "ios", all(target_arch = "wasm32", target_os = "unknown")))]
pub mod universal;
/// wasm prover and verifier
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm;
/// Python bindings
#[cfg(feature = "python-bindings")]
pub mod python;
