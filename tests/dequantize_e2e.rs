//! End-to-end integration test for the auto-dequantize path.
//!
//! Users do not run a separate dequantize step before the rest of the EZKL
//! pipeline — the dequantize pass fires transparently inside `Model::new`.
//! This test proves that the auto-rewrite produces a graph that:
//!
//!   1. Survives `gen-settings → calibrate → compile-circuit → gen-witness
//!      → mock` end-to-end.
//!   2. Yields outputs (after dequantising the witness via the calibrated
//!      output scales) that agree numerically with a tract inference of the
//!      equivalent original float model — within EZKL's quantization
//!      tolerance.
//!   3. Halts at `gen-settings` with `UnsupportedQuantizationOps` when the
//!      user opts out via `--disable-quantization-fixup`, proving the gating
//!      flag works at the CLI boundary as well as in `Model::new`.
//!
//! A heavier `#[ignore]`-gated companion test ([`auto_dequantize_pipeline_through_full_snark`])
//! drives the full `setup → prove → verify` SNARK on top, downloading SRS
//! the first time it runs. Skip it on routine `cargo test` runs; opt in
//! with `cargo test -- --ignored`.
//!
//! Float-vs-quantized tolerance: the fixture's input quantization
//! (`x_scale = 0.05`, per-tensor) plus EZKL's default `scale = 7`
//! (~1/128 quantum) plus the small accumulation across a 3×3 conv puts the
//! per-element error comfortably under `0.5` for our deterministic input.
//! That's the absolute tolerance asserted below — tight enough to catch a
//! genuine regression, loose enough to absorb calibration noise.

#![cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ezkl::graph::input::GraphData;
use ezkl::graph::{GraphSettings, GraphWitness, Model};
use ezkl::RunArgs;

const FIXTURE_DIR: &str = "examples/onnx/quantized_qdq";
const NETWORK_ONNX: &str = "network.onnx";
const INPUT_JSON: &str = "input.json";
/// Per-element tolerance between the proven (mock-witnessed) output and the
/// tract-computed float reference. See module-level docs for the breakdown.
const FLOAT_TOLERANCE: f32 = 0.5;

// ---------------------------------------------------------------------------
// Plumbing — locate the binary, stage a working directory, run subprocess
// pipeline steps. Mirrors the pattern in `tests/dequantize_pipeline.rs` so
// the suite stays runnable in environments where only `cargo test --lib`
// has been executed (no binary built, no SRS downloaded → tests skip with
// a printed reason).
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

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

fn stage_fixture(prefix: &str) -> PathBuf {
    // Each test gets its own work dir to avoid races between parallel cargo
    // test threads. We don't use tempdir here because the path is shown in
    // assertion messages on failure — leaving the dir behind helps debug.
    let dir = std::env::temp_dir().join(format!("ezkl-dequantize-e2e-{prefix}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create work dir");
    let src = repo_root().join(FIXTURE_DIR);
    for name in [NETWORK_ONNX, INPUT_JSON] {
        std::fs::copy(src.join(name), dir.join(name))
            .unwrap_or_else(|e| panic!("failed to stage {name}: {e}"));
    }
    dir
}

fn run_step(binary: &Path, args: &[&str]) -> Output {
    Command::new(binary)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn ezkl {:?}: {e}", args))
}

fn assert_step(binary: &Path, args: &[&str]) {
    let result = run_step(binary, args);
    assert!(
        result.status.success(),
        "ezkl {:?} failed (exit {:?}):\nstdout: {}\nstderr: {}",
        args,
        result.status.code(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
}

/// Run `gen-settings → calibrate-settings → compile-circuit → gen-witness`
/// against `network.onnx` in `work_dir`. Returns the path to `witness.json`
/// and `settings.json` that the caller can deserialize for downstream
/// assertions.
fn run_pipeline_to_witness(binary: &Path, work_dir: &Path) -> (PathBuf, PathBuf) {
    let net = work_dir.join(NETWORK_ONNX);
    let input = work_dir.join(INPUT_JSON);
    let settings = work_dir.join("settings.json");
    let compiled = work_dir.join("network.compiled");
    let witness = work_dir.join("witness.json");

    assert_step(
        binary,
        &[
            "gen-settings",
            "-M",
            net.to_str().unwrap(),
            &format!("--settings-path={}", settings.display()),
        ],
    );
    assert_step(
        binary,
        &[
            "calibrate-settings",
            "-D",
            input.to_str().unwrap(),
            "-M",
            net.to_str().unwrap(),
            &format!("--settings-path={}", settings.display()),
            "--target=resources",
            "--lookup-safety-margin=2",
        ],
    );
    assert_step(
        binary,
        &[
            "compile-circuit",
            "-M",
            net.to_str().unwrap(),
            "--compiled-circuit",
            compiled.to_str().unwrap(),
            &format!("--settings-path={}", settings.display()),
        ],
    );
    assert_step(
        binary,
        &[
            "gen-witness",
            "-D",
            input.to_str().unwrap(),
            "-M",
            compiled.to_str().unwrap(),
            "-O",
            witness.to_str().unwrap(),
        ],
    );

    (witness, settings)
}

// ---------------------------------------------------------------------------
// Float reference — strip Q/DQ from the quantized fixture via the
// `ezkl dequantize` subcommand, then run tract inference on the cleaned
// model. The dequantize pass is the same one `Model::new` runs internally;
// invoking it explicitly here lets us produce a side-by-side float graph
// without depending on Python or hand-coded fixtures.
// ---------------------------------------------------------------------------

fn compute_float_reference(binary: &Path, work_dir: &Path) -> Vec<f32> {
    let cleaned = work_dir.join("network_cleaned.onnx");
    assert_step(
        binary,
        &[
            "dequantize",
            "-M",
            work_dir.join(NETWORK_ONNX).to_str().unwrap(),
            "-O",
            cleaned.to_str().unwrap(),
        ],
    );

    let input_data =
        GraphData::from_path(work_dir.join(INPUT_JSON)).expect("failed to read input.json");
    let run_args = RunArgs::default();
    // The quantized_qdq fixture has a fixed input shape of [1,1,4,4].
    let predictions = Model::run_onnx_predictions(
        &run_args,
        &cleaned,
        std::slice::from_ref(&input_data),
        vec![vec![1, 1, 4, 4]],
    )
    .expect("tract inference on the dequantized float model failed");

    // Single chunk, single model output → flatten to a Vec<f32>.
    assert_eq!(
        predictions.len(),
        1,
        "expected exactly one prediction chunk from tract"
    );
    assert_eq!(
        predictions[0].len(),
        1,
        "expected exactly one output tensor from the conv fixture"
    );
    predictions[0][0].iter().copied().collect()
}

fn read_witness_outputs(witness_path: &Path, settings_path: &Path) -> Vec<f32> {
    let witness: GraphWitness = serde_json::from_str(
        &std::fs::read_to_string(witness_path).expect("failed to read witness.json"),
    )
    .expect("failed to deserialize witness.json");
    let settings: GraphSettings = serde_json::from_str(
        &std::fs::read_to_string(settings_path).expect("failed to read settings.json"),
    )
    .expect("failed to deserialize settings.json");

    let float_outputs = witness.get_float_outputs(&settings.model_output_scales);
    assert_eq!(
        float_outputs.len(),
        1,
        "expected exactly one output tensor from the conv fixture"
    );
    float_outputs[0].iter().copied().collect()
}

fn assert_outputs_match(proven: &[f32], reference: &[f32], tolerance: f32) {
    assert_eq!(
        proven.len(),
        reference.len(),
        "proven output has {} elements but float reference has {}",
        proven.len(),
        reference.len(),
    );
    let mut max_err: f32 = 0.0;
    let mut argmax: usize = 0;
    for (i, (p, r)) in proven.iter().zip(reference.iter()).enumerate() {
        let err = (p - r).abs();
        if err > max_err {
            max_err = err;
            argmax = i;
        }
    }
    assert!(
        max_err <= tolerance,
        "proven output diverges from float reference at index {argmax}: proven={} reference={} abs_err={} > tolerance={}",
        proven[argmax],
        reference[argmax],
        max_err,
        tolerance,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn auto_dequantize_pipeline_through_mock_matches_float_reference() {
    let binary = match locate_ezkl_binary() {
        Some(p) => p,
        None => {
            eprintln!(
                "skipping: no ezkl binary found in target/{{test-runs,release,debug}} \
                 — build with `cargo build --profile=test-runs --bin ezkl` first"
            );
            return;
        }
    };
    let work_dir = stage_fixture("mock");

    let (witness_path, settings_path) = run_pipeline_to_witness(&binary, &work_dir);

    // Mock prove validates that every circuit constraint is satisfied for the
    // generated witness — i.e. the rewritten graph is internally consistent
    // and matches the witness produced by gen-witness. No SRS needed.
    assert_step(
        &binary,
        &[
            "mock",
            "-W",
            witness_path.to_str().unwrap(),
            "-M",
            work_dir.join("network.compiled").to_str().unwrap(),
        ],
    );

    let proven = read_witness_outputs(&witness_path, &settings_path);
    let reference = compute_float_reference(&binary, &work_dir);
    assert_outputs_match(&proven, &reference, FLOAT_TOLERANCE);
}

#[test]
fn disable_dequantize_pipeline_halts_at_gen_settings() {
    let binary = match locate_ezkl_binary() {
        Some(p) => p,
        None => {
            eprintln!(
                "skipping: no ezkl binary found in target/{{test-runs,release,debug}} \
                 — build with `cargo build --profile=test-runs --bin ezkl` first"
            );
            return;
        }
    };
    let work_dir = stage_fixture("negative");
    let result = run_step(
        &binary,
        &[
            "gen-settings",
            "-M",
            work_dir.join(NETWORK_ONNX).to_str().unwrap(),
            &format!(
                "--settings-path={}",
                work_dir.join("settings.json").display()
            ),
            "--disable-quantization-fixup",
        ],
    );
    assert!(
        !result.status.success(),
        "expected gen-settings to fail when --disable-quantization-fixup is set, but it succeeded\n\
         stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        combined.contains("quantization operators")
            || combined.contains("UnsupportedQuantizationOps"),
        "expected the safety-net error to mention quantization operators, got:\n{combined}",
    );
}

#[test]
#[ignore = "downloads SRS and runs full SNARK setup/prove/verify (~minutes); enable with `cargo test -- --ignored`"]
fn auto_dequantize_pipeline_through_full_snark() {
    let binary = match locate_ezkl_binary() {
        Some(p) => p,
        None => {
            eprintln!(
                "skipping: no ezkl binary found in target/{{test-runs,release,debug}} \
                 — build with `cargo build --profile=test-runs --bin ezkl` first"
            );
            return;
        }
    };
    let work_dir = stage_fixture("snark");

    let (witness_path, settings_path) = run_pipeline_to_witness(&binary, &work_dir);

    // SRS lookup uses ezkl's standard cache location; this is a no-op once
    // it's been downloaded for the logrows the calibrated settings call for.
    let settings: GraphSettings = serde_json::from_str(
        &std::fs::read_to_string(&settings_path).expect("failed to read settings.json"),
    )
    .expect("failed to deserialize settings.json");
    assert_step(
        &binary,
        &[
            "get-srs",
            "--logrows",
            &settings.run_args.logrows.to_string(),
        ],
    );

    let compiled = work_dir.join("network.compiled");
    let pk = work_dir.join("key.pk");
    let vk = work_dir.join("key.vk");
    let proof = work_dir.join("proof.pf");

    assert_step(
        &binary,
        &[
            "setup",
            "-M",
            compiled.to_str().unwrap(),
            "--pk-path",
            pk.to_str().unwrap(),
            "--vk-path",
            vk.to_str().unwrap(),
            "--disable-selector-compression",
        ],
    );
    assert_step(
        &binary,
        &[
            "prove",
            "-W",
            witness_path.to_str().unwrap(),
            "-M",
            compiled.to_str().unwrap(),
            "--proof-path",
            proof.to_str().unwrap(),
            "--pk-path",
            pk.to_str().unwrap(),
        ],
    );
    assert_step(
        &binary,
        &[
            "verify",
            &format!("--settings-path={}", settings_path.display()),
            "--proof-path",
            proof.to_str().unwrap(),
            "--vk-path",
            vk.to_str().unwrap(),
        ],
    );

    // Float-vs-proven check after a full SNARK round-trip — same assertion
    // as the mock test, but now the witness has cleared the cryptographic
    // path too.
    let proven = read_witness_outputs(&witness_path, &settings_path);
    let reference = compute_float_reference(&binary, &work_dir);
    assert_outputs_match(&proven, &reference, FLOAT_TOLERANCE);
}
