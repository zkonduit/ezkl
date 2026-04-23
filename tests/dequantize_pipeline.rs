//! End-to-end tests for the `ezkl dequantize` subcommand. The same
//! dequantize logic runs automatically inside `Model::new`; this subcommand
//! exposes it as a one-shot tool that writes a cleaned ONNX file to disk
//! (useful for debugging, inspection, or sharing with non-EZKL tools).
//!
//! Each test:
//! 1. invokes `target/<profile>/ezkl dequantize -M <fixture> -O <tmp>`
//! 2. asserts the binary exited 0 and produced a non-empty file
//! 3. loads the cleaned ONNX through `Model::new` (with the auto-dequantize
//!    pass *disabled* so we know the cleaned bytes alone are accepted by the
//!    loader, not just re-cleaned by it).
//!
//! The tests are skipped (with a printed reason) when the `ezkl` binary
//! isn't available in any of the standard cargo target locations, so the
//! suite stays runnable in environments where only `cargo test --lib` was
//! built.

#![cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::process::Command;

use ezkl::graph::Model;
use ezkl::RunArgs;

const STATIC_QDQ_FIXTURE: &str = "quantized_qdq.onnx";
const DYNAMIC_QUANT_FIXTURE: &str = "quantized_dynamic.onnx";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Locate the `ezkl` binary built by cargo in any of the profiles we use.
/// The integration test harness also builds at `target/test-runs/ezkl`, so
/// we honour that too — running `cargo test --test dequantize_pipeline`
/// after `cargo build --profile=test-runs --bin ezkl` should succeed.
fn locate_ezkl_binary() -> Option<PathBuf> {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root().join("target"));
    for profile in ["test-runs", "release", "debug"] {
        let cand = target_dir.join(profile).join("ezkl");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests").join("assets").join(name)
}

fn tmp_output(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ezkl-dequantize-tests");
    std::fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir.join(format!("{name}.cleaned.onnx"))
}

fn run_dequantize(binary: &Path, input: &Path, output: &Path) {
    let result = Command::new(binary)
        .args([
            "dequantize",
            "-M",
            input.to_str().unwrap(),
            "-O",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("failed to spawn ezkl dequantize");
    assert!(
        result.status.success(),
        "ezkl dequantize failed on {} (exit {:?}):\nstderr: {}",
        input.display(),
        result.status.code(),
        String::from_utf8_lossy(&result.stderr),
    );
    let metadata =
        std::fs::metadata(output).expect("ezkl dequantize did not produce an output file");
    assert!(
        metadata.len() > 0,
        "ezkl dequantize produced an empty file at {}",
        output.display()
    );
}

/// Loads `path` through `Model::new` with the auto-dequantize pass *disabled*,
/// so the assertion proves the cleaned bytes are themselves sufficient for
/// the loader (rather than being cleaned again on the way in).
fn assert_cleaned_loads_without_fixup(path: &Path) {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("missing cleaned model {}: {}", path.display(), e));
    let run_args = RunArgs {
        disable_quantization_fixup: true,
        ..RunArgs::default()
    };
    Model::new(&mut file, &run_args).unwrap_or_else(|e| {
        panic!(
            "Model::new (dequantize disabled) rejected cleaned {}: {e:?}",
            path.display()
        )
    });
}

fn dequantize_and_load(fixture: &str) {
    let binary = match locate_ezkl_binary() {
        Some(p) => p,
        None => {
            eprintln!(
                "skipping dequantize subcommand test for {fixture}: no ezkl binary found in target/{{test-runs,release,debug}}",
            );
            return;
        }
    };

    let input = fixture_path(fixture);
    let output = tmp_output(fixture);
    run_dequantize(&binary, &input, &output);
    assert_cleaned_loads_without_fixup(&output);
}

#[test]
fn dequantize_subcommand_strips_static_qdq_pattern() {
    dequantize_and_load(STATIC_QDQ_FIXTURE);
}

#[test]
fn dequantize_subcommand_strips_dynamic_quantize_pattern() {
    dequantize_and_load(DYNAMIC_QUANT_FIXTURE);
}
