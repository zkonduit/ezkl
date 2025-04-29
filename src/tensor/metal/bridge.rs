//! Rust bridge to Metal shader code for MSM operations
//! 
//! This module provides the bridge between Rust and Metal, allowing the Metal shaders
//! to be used for accelerating MSM operations in the age verification circuit.

use std::sync::Arc;
// use std::ptr::NonNull;
use std::ffi::c_void;

use halo2curves::bn256::{Fr, G1Affine, G1};
use halo2curves::group::Group;

// Since we can't easily make halo2curves types FFI-safe, we'll have to allow this lint
#[allow(improper_ctypes)]
// FFI declarations for Metal interface
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn initialize_metal_device() -> *mut c_void;
    fn create_msm_command_queue(device: *mut c_void) -> *mut c_void;
    fn execute_standard_msm(
        device: *mut c_void,
        command_queue: *mut c_void,
        bases_ptr: *const G1Affine,
        scalars_ptr: *const Fr,
        result_ptr: *mut G1,
        length: u32,
        window_size: u32,
    ) -> bool;
    fn execute_memory_optimized_msm(
        device: *mut c_void,
        command_queue: *mut c_void,
        bases_ptr: *const G1Affine,
        scalars_ptr: *const Fr,
        result_ptr: *mut G1,
        length: u32,
        batch_size: u32,
    ) -> bool;
    fn execute_performance_optimized_msm(
        device: *mut c_void,
        command_queue: *mut c_void,
        bases_ptr: *const G1Affine,
        scalars_ptr: *const Fr,
        result_ptr: *mut G1,
        length: u32,
        window_size: u32,
        num_windows: u32,
    ) -> bool;
    fn release_metal_device(device: *mut c_void);
    fn release_command_queue(command_queue: *mut c_void);
    fn execute_age_verification_msm(
        device: *mut c_void,
        command_queue: *mut c_void,
        bases_ptr: *const G1Affine,
        scalars_ptr: *const Fr,
        result_ptr: *mut G1,
        length: u32,
        batch_size: u32,
    ) -> bool;
}

/// Wrapper for Metal device
#[derive(Debug)]
pub struct MetalDevice {
    device_ptr: *mut c_void,
    command_queue_ptr: *mut c_void,
}

#[allow(unsafe_code)]
unsafe impl Send for MetalDevice {}
#[allow(unsafe_code)]
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    /// Create a new Metal device
    #[allow(unsafe_code)]
    pub fn new() -> Option<Self> {
        unsafe {
            let device_ptr = initialize_metal_device();
            if device_ptr.is_null() {
                return None;
            }
            
            let command_queue_ptr = create_msm_command_queue(device_ptr);
            if command_queue_ptr.is_null() {
                release_metal_device(device_ptr);
                return None;
            }
            
            Some(Self {
                device_ptr,
                command_queue_ptr,
            })
        }
    }
    
    /// Execute standard MSM using Metal
    #[allow(unsafe_code)]
    pub fn execute_standard_msm(
        &self,
        bases: &[G1Affine],
        scalars: &[Fr],
        window_size: u32,
    ) -> Option<G1> {
        if bases.len() != scalars.len() {
            return None;
        }
        
        let length = bases.len() as u32;
        let mut result = G1::identity();
        
        unsafe {
            let success = execute_standard_msm(
                self.device_ptr,
                self.command_queue_ptr,
                bases.as_ptr(),
                scalars.as_ptr(),
                &mut result,
                length,
                window_size,
            );
            
            if success {
                Some(result)
            } else {
                None
            }
        }
    }
    
    /// Execute memory-optimized MSM using Metal
    #[allow(unsafe_code)]
    pub fn execute_memory_optimized_msm(
        &self,
        bases: &[G1Affine],
        scalars: &[Fr],
        batch_size: u32,
    ) -> Option<G1> {
        if bases.len() != scalars.len() {
            return None;
        }
        
        let length = bases.len() as u32;
        let mut result = G1::identity();
        
        unsafe {
            let success = execute_memory_optimized_msm(
                self.device_ptr,
                self.command_queue_ptr,
                bases.as_ptr(),
                scalars.as_ptr(),
                &mut result,
                length,
                batch_size,
            );
            
            if success {
                Some(result)
            } else {
                None
            }
        }
    }
    
    /// Execute performance-optimized MSM using Metal
    #[allow(unsafe_code)]
    pub fn execute_performance_optimized_msm(
        &self,
        bases: &[G1Affine],
        scalars: &[Fr],
        window_size: u32,
    ) -> Option<G1> {
        if bases.len() != scalars.len() {
            return None;
        }
        
        let length = bases.len() as u32;
        let num_windows = (256 + window_size - 1) / window_size; // Ceiling division
        let mut result = G1::identity();
        
        unsafe {
            let success = execute_performance_optimized_msm(
                self.device_ptr,
                self.command_queue_ptr,
                bases.as_ptr(),
                scalars.as_ptr(),
                &mut result,
                length,
                window_size,
                num_windows,
            );
            
            if success {
                Some(result)
            } else {
                None
            }
        }
    }
    
    /// Execute memory-optimized MSM using Metal specifically tuned for age verification circuit
    #[allow(unsafe_code)]
    pub fn execute_age_verification_msm(
        &self,
        bases: &[G1Affine],
        scalars: &[Fr],
        batch_size: u32,
    ) -> Option<G1> {
        if bases.len() != scalars.len() {
            return None;
        }
        
        let length = bases.len() as u32;
        let mut result = G1::identity();
        
        unsafe {
            let success = execute_age_verification_msm(
                self.device_ptr,
                self.command_queue_ptr,
                bases.as_ptr(),
                scalars.as_ptr(),
                &mut result,
                length,
                batch_size,
            );
            
            if success {
                Some(result)
            } else {
                None
            }
        }
    }
}

impl Drop for MetalDevice {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        unsafe {
            release_command_queue(self.command_queue_ptr);
            release_metal_device(self.device_ptr);
        }
    }
}

// Replace the global static with a thread-local for safety
use std::cell::RefCell;
thread_local! {
    static METAL_DEVICE: RefCell<Option<Arc<MetalDevice>>> = RefCell::new(None);
}

/// Initialize global Metal device
#[allow(unsafe_code)]
pub fn initialize_global_metal_device() -> bool {
    METAL_DEVICE.with(|device| {
        let mut device_ref = device.borrow_mut();
        if device_ref.is_none() {
            if let Some(new_device) = MetalDevice::new() {
                *device_ref = Some(Arc::new(new_device));
                true
            } else {
                false
            }
        } else {
            true
        }
    })
}

/// Get global Metal device
pub fn get_global_metal_device() -> Option<Arc<MetalDevice>> {
    METAL_DEVICE.with(|device| {
        device.borrow().clone()
    })
}

/// Execute MSM with Metal acceleration
pub fn metal_msm(bases: &[G1Affine], scalars: &[Fr], window_size: u32) -> Option<G1> {
    if let Some(device) = get_global_metal_device() {
        device.execute_standard_msm(bases, scalars, window_size)
    } else {
        None
    }
}

/// Execute MSM with memory-optimized Metal acceleration
pub fn metal_memory_optimized_msm(bases: &[G1Affine], scalars: &[Fr], batch_size: u32) -> Option<G1> {
    if let Some(device) = get_global_metal_device() {
        device.execute_memory_optimized_msm(bases, scalars, batch_size)
    } else {
        None
    }
}

/// Execute MSM with performance-optimized Metal acceleration
pub fn metal_performance_optimized_msm(bases: &[G1Affine], scalars: &[Fr], window_size: u32) -> Option<G1> {
    if let Some(device) = get_global_metal_device() {
        device.execute_performance_optimized_msm(bases, scalars, window_size)
    } else {
        None
    }
}

/// Execute MSM with age-verification-optimized Metal acceleration
pub fn metal_age_verification_msm(bases: &[G1Affine], scalars: &[Fr], batch_size: u32) -> Option<G1> {
    if let Some(device) = get_global_metal_device() {
        device.execute_age_verification_msm(bases, scalars, batch_size)
    } else {
        None
    }
} 