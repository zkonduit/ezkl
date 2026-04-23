//! Integration tests for the pre-flight detection of ONNX post-training
//! quantization operators added in response to issue #942.
//!
//! ONNX Runtime's PTQ inserts ops such as `QuantizeLinear`, `DequantizeLinear`,
//! `DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`, etc. into the
//! exported graph. Tract cannot analyse those models, and EZKL already does its
//! own quantization internally via the `scale` run argument — feeding it a
//! pre-quantized model is both redundant and broken.
//!
//! These tests assert that loading a pre-quantized model now surfaces a clear,
//! actionable [`GraphError::UnsupportedQuantizationOps`] instead of an opaque
//! tract panic.

#![cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]

use ezkl::graph::errors::GraphError;
use ezkl::graph::Model;
use ezkl::RunArgs;

fn assert_quantization_error(fixture: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("assets")
        .join(fixture);
    let mut file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {}", path.display(), e));

    let run_args = RunArgs::default();
    let result = Model::new(&mut file, &run_args);

    match result {
        Err(GraphError::UnsupportedQuantizationOps(report)) => {
            assert!(
                !report.is_empty(),
                "error report should name at least one offending node, got empty string",
            );
        }
        Err(other) => panic!(
            "expected UnsupportedQuantizationOps for fixture {}, got: {other:?}",
            fixture
        ),
        Ok(_) => panic!(
            "expected UnsupportedQuantizationOps for fixture {}, but model loaded successfully",
            fixture
        ),
    }
}

#[test]
fn detects_static_quantize_dequantize_pair() {
    assert_quantization_error("quantized_qdq.onnx");
}

#[test]
fn detects_dynamic_quantize_with_matmul_integer() {
    assert_quantization_error("quantized_dynamic.onnx");
}
