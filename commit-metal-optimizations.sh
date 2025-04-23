#!/bin/bash

# This script creates multiple focused commits for the Metal acceleration optimizations

echo "Starting commit process with multiple focused commits..."

# 1. Initial setup for Metal module structure
git add src/tensor/mod.rs
git commit -m "feat(metal): Add Metal module structure" -m "Initialize the module structure for Metal acceleration support.
- Add Metal module to tensor directory
- Configure feature gates for macos-metal and ios-metal"

# 2. Bridge implementation for Metal
git add src/tensor/metal/bridge.rs
git commit -m "feat(metal): Implement Metal bridge for MSM acceleration" -m "Add Rust-to-Metal bridge for Multi-Scalar Multiplication operations.
- Create FFI interface to Metal shaders
- Implement Metal device management
- Add thread-safe device initialization"

# 3. MSM shader implementation
git add src/tensor/metal/msm_shader.metal
git commit -m "feat(metal): Add Metal shaders for MSM operations" -m "Implement Metal compute shaders for accelerating MSM operations.
- Add standard MSM kernel
- Implement field arithmetic in Metal
- Create basic point operations for elliptic curves"

# 4. Memory-optimized MSM implementation
git add -p src/pfsys/metal_msm_accelerator.rs
git commit -m "feat(metal): Add memory-optimized MSM implementation" -m "Implement memory-efficient MSM operations to reduce memory usage.
- Create smaller batch sizes for reduced memory footprint
- Implement sequential batch processing
- Optimize memory allocations for age verification circuit"

# 5. Performance-optimized MSM implementation
git add -p src/pfsys/metal_msm_accelerator.rs
git commit -m "feat(metal): Add performance-optimized MSM implementation" -m "Implement performance-focused MSM operations for faster proving.
- Optimize window sizes for maximum performance
- Implement parallel batch processing
- Create custom data structures for faster MSM"

# 6. Age verification specific optimizations
git add -p src/pfsys/metal_msm_accelerator.rs
git commit -m "feat(metal): Add age verification specific optimizations" -m "Implement specialized optimizations for the age verification circuit.
- Create age verification specific MSM implementation
- Optimize for 64x64x3 input tensors
- Tune algorithm parameters specifically for age circuit"

# 7. Metal accelerator configuration
git add -p src/pfsys/metal_msm_accelerator.rs
git commit -m "feat(metal): Add Metal MSM accelerator configuration" -m "Implement configuration system for Metal acceleration.
- Create MetalMSMConfig for customization
- Add window size and batch size configuration
- Implement thread count control"

# 8. Circuit optimization for Metal
git add src/circuit/metal_optimize.rs
git commit -m "feat(metal): Add circuit optimization for Metal acceleration" -m "Implement circuit-level optimizations for Metal acceleration.
- Create AgeVerificationOptimizer
- Add memory and performance optimization modes
- Implement tensor sparsity optimizations"

# 9. Circuit module integration
git add src/circuit/mod.rs
git commit -m "feat(metal): Add Metal optimization module to circuit" -m "Integrate Metal optimization module with the circuit system.
- Add metal_optimize module to circuit
- Configure feature gates
- Add proper documentation"

# 10. Thread safety improvements
git add -p src/tensor/metal/bridge.rs
git commit -m "feat(metal): Improve thread safety for Metal device" -m "Enhance thread safety for Metal device access.
- Replace static mutable references with thread-local storage
- Add proper synchronization for device access
- Fix potential race conditions"

# 11. Add benchmarking and testing
git add tests/metal_msm_tests.rs
git commit -m "test(metal): Add benchmarking for Metal acceleration" -m "Add testing and benchmarking utilities for Metal acceleration.
- Create correctness tests for MSM implementations
- Add benchmarking for standard, memory-optimized, and performance-optimized MSM
- Implement memory usage measurement tools"

# 12. Final cleanup and documentation
git add .
git commit -m "docs(metal): Add documentation and final cleanup" -m "Add comprehensive documentation and final cleanup.
- Add detailed documentation for all Metal-related modules
- Fix linting issues and warnings
- Ensure proper attributes for unsafe code
- Complete method documentation"

echo "All changes have been committed in multiple focused commits! 🎉" 