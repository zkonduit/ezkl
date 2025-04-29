///
pub mod modules;

///
pub mod table;

///
pub mod utils;

///
pub mod ops;

pub use ops::chip::*;
pub use ops::*;

/// Tests
#[cfg(test)]
mod tests;

/// Metal optimizations for age verification circuit
#[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
pub mod metal_optimize;
