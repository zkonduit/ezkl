/// Metal optimizations for age verification circuit.

use log::info;
use crate::RunArgs;

/// Optimize the age verification circuit with balanced performance and memory usage
pub fn optimize_age_verification_circuit(settings: &mut RunArgs) {
    info!("Applying balanced optimizations for age verification circuit");
    
    // Set optimal parameters for balanced performance/memory usage
    settings.input_scale = 10;
    settings.param_scale = 10;
    settings.logrows = 16;
    settings.decomp_legs = 2;
    settings.lookup_range = (-16384, 16384);
}

/// Optimize the age verification circuit for reduced memory usage
pub fn optimize_age_verification_circuit_for_memory(settings: &mut RunArgs) {
    info!("Applying memory-optimized settings for age verification circuit");
    
    // Set memory-efficient parameters
    settings.input_scale = 9;
    settings.param_scale = 9;
    settings.logrows = 15;
    settings.decomp_legs = 1;
    settings.lookup_range = (-8192, 8192);
}

/// Optimize the age verification circuit for maximum performance
pub fn optimize_age_verification_circuit_for_performance(settings: &mut RunArgs) {
    info!("Applying performance-optimized settings for age verification circuit");
    
    // Set high-performance parameters
    settings.input_scale = 12;
    settings.param_scale = 12;
    settings.logrows = 17;
    settings.decomp_legs = 3;
    settings.lookup_range = (-32768, 32768);
} 