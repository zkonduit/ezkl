//! Metal GPU acceleration for tensor operations
//! 
//! This module provides Metal GPU acceleration for tensor operations,
//! specifically optimized for the age verification circuit.

#[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
pub mod bridge;

#[cfg(any(feature = "macos-metal", feature = "ios-metal"))]
pub use bridge::*; 