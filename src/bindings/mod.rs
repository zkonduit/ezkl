mod logic;
/// wasm prover and verifier
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub mod wasm;
/// Universal bindings for all platforms
#[cfg(target_os = "ios")]
pub mod universal;