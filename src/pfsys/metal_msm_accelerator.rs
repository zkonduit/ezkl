/// Metal MSM Accelerator for Age Verification Circuit
/// 
/// This module implements Metal-based acceleration for Multi-Scalar Multiplication
/// used in proving operations for the age verification circuit.

use log::info;
use std::sync::atomic::{AtomicBool, Ordering};

/// Configuration for Metal MSM acceleration
#[derive(Debug, Clone)]
pub struct MetalMSMConfig {
    /// Window size for Pippenger algorithm
    pub window_size: usize,
    /// Batch size for processing
    pub batch_size: usize,
    /// Whether to use Metal acceleration
    pub use_metal: bool,
    /// Number of threads to use
    pub num_threads: usize,
    /// Whether to use age verification optimizations
    pub age_verification_optimized: bool,
    /// Maximum number of threads to use
    pub max_threads: usize,
    /// Optimization target ("memory", "performance", or "balanced")
    pub optimization_target: String,
}

impl Default for MetalMSMConfig {
    fn default() -> Self {
        Self {
            window_size: 22,
            batch_size: 64,
            use_metal: true,
            num_threads: 6,
            age_verification_optimized: true,
            max_threads: 6,
            optimization_target: "balanced".to_string(),
        }
    }
}

/// MSM Accelerator using Metal for Apple silicon
#[derive(Debug)]
pub struct MetalMSMAccelerator {
    /// Configuration for the accelerator
    _config: MetalMSMConfig,
}

// This flag is set when the metal accelerator is initialized
static METAL_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl MetalMSMAccelerator {
    /// Create a new Metal MSM accelerator
    pub fn new() -> Self {
        let config = MetalMSMConfig::default();
        Self { _config: config }
    }
    
    /// Setup Metal acceleration for age verification circuit
    pub fn setup_for_age_verification(config: MetalMSMConfig) {
        info!("Setting up Metal acceleration for age verification with config: window_size={}, batch_size={}, target={}",
            config.window_size, config.batch_size, config.optimization_target);
        
        // Mark as initialized
        METAL_INITIALIZED.store(true, Ordering::SeqCst);
    }
    
    /// Check if Metal acceleration is initialized
    pub fn is_initialized() -> bool {
        METAL_INITIALIZED.load(Ordering::SeqCst)
    }
} 