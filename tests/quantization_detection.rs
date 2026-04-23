//! Behavioural tests for EZKL's handling of pre-quantized ONNX models.
//! These cover the two visible behaviours of the loader:
//!
//! 1. **Default path:** the automatic
//!    [`apply`](ezkl::graph::dequantize::apply) dequantize pass
//!    canonicalises the supported PTQ patterns transparently, so a model
//!    that ORT pre-quantized loads successfully without any user
//!    intervention.
//!
//! 2. **Safety net:** when the user opts out via `--disable-quantization-fixup`
//!    *or* the model contains a quantization pattern we cannot rewrite,
//!    [`GraphError::UnsupportedQuantizationOps`] surfaces with an actionable
//!    message instead of an opaque tract panic.

#![cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]

use ezkl::graph::errors::GraphError;
use ezkl::graph::Model;
use ezkl::RunArgs;

const STATIC_QDQ_FIXTURE: &str = "quantized_qdq.onnx";
const DYNAMIC_QUANT_FIXTURE: &str = "quantized_dynamic.onnx";

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("assets")
        .join(name)
}

fn open(name: &str) -> std::fs::File {
    let p = fixture_path(name);
    std::fs::File::open(&p)
        .unwrap_or_else(|e| panic!("missing fixture {}: {}", p.display(), e))
}

// ---------------------------------------------------------------------------
// Default path: auto-dequantize transparently handles the quantized fixtures.
// ---------------------------------------------------------------------------

#[test]
fn default_load_handles_static_qdq_pair() {
    let mut file = open(STATIC_QDQ_FIXTURE);
    let run_args = RunArgs::default();
    Model::new(&mut file, &run_args)
        .expect("default dequantize pass should accept static Q/DQ fixture");
}

#[test]
fn default_load_handles_dynamic_quantize_matmul_integer() {
    let mut file = open(DYNAMIC_QUANT_FIXTURE);
    let run_args = RunArgs::default();
    Model::new(&mut file, &run_args).expect(
        "default dequantize pass should accept DynamicQuantizeLinear + MatMulInteger fixture",
    );
}

// ---------------------------------------------------------------------------
// Safety net: explicitly disabling the dequantize pass re-exposes the
// original failure mode and triggers the actionable error.
// ---------------------------------------------------------------------------

fn assert_unsupported_quant_ops(result: Result<Model, GraphError>, fixture: &str) {
    match result {
        Err(GraphError::UnsupportedQuantizationOps(report)) => {
            assert!(
                !report.is_empty(),
                "error report should name at least one offending node, got empty string for {fixture}",
            );
        }
        Err(other) => panic!(
            "expected UnsupportedQuantizationOps for fixture {fixture}, got: {other:?}",
        ),
        Ok(_) => panic!(
            "expected UnsupportedQuantizationOps for fixture {fixture}, but model loaded successfully",
        ),
    }
}

fn run_args_with_fixup_disabled() -> RunArgs {
    RunArgs {
        disable_quantization_fixup: true,
        ..RunArgs::default()
    }
}

#[test]
fn disable_fixup_then_static_qdq_is_rejected_by_safety_net() {
    let mut file = open(STATIC_QDQ_FIXTURE);
    assert_unsupported_quant_ops(Model::new(&mut file, &run_args_with_fixup_disabled()), STATIC_QDQ_FIXTURE);
}

#[test]
fn disable_fixup_then_dynamic_quantize_is_rejected_by_safety_net() {
    let mut file = open(DYNAMIC_QUANT_FIXTURE);
    assert_unsupported_quant_ops(
        Model::new(&mut file, &run_args_with_fixup_disabled()),
        DYNAMIC_QUANT_FIXTURE,
    );
}
