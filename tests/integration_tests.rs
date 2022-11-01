use std::process::Command;

// Mock prove (fast, but does not cover some potential issues)
fn test_onnx_mock(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "--bits",
            "16",
            "-K",
            "17",
            "mock",
            "-D",
            format!("./examples/onnx_models/{}_input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx_models/{}.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_mock() {
    test_onnx_mock("ff".to_string());
}

#[test]
fn test_relusig_mock() {
    test_onnx_mock("relusig".to_string());
}

#[test]
#[ignore]
fn test_1lcnvrl_mock() {
    test_onnx_mock("1lcnvrl".to_string());
}

#[test]
fn test_2lcnvrl_mock() {
    test_onnx_mock("2lcnvrl_relusig".to_string());
}

// full prove (slower, covers more, but still reuses the pk)
fn test_onnx_fullprove(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "--bits",
            "16",
            "-K",
            "17",
            "fullprove",
            "-D",
            format!("./examples/onnx_models/{}_input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx_models/{}.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_fullprove() {
    test_onnx_fullprove("ff".to_string());
}

#[test]
#[ignore]
fn test_relusig_fullprove() {
    test_onnx_fullprove("relusig".to_string());
}

#[test]
fn test_relurelu_fullprove() {
    test_onnx_fullprove("relurelu_small".to_string());
}

#[test]
fn test_relusig_small_fullprove() {
    test_onnx_fullprove("relusig_small".to_string());
}

#[test]
fn test_relu_fullprove() {
    test_onnx_fullprove("relu".to_string());
}

#[test]
fn test_sig_fullprove() {
    test_onnx_fullprove("sig".to_string());
}

// These require too much memory for Github CI right now
#[test]
#[ignore]
fn test_1lcnvrl_fullprove() {
    test_onnx_fullprove("1lcnvrl".to_string());
}

#[test]
#[ignore]
fn test_2lcnvrl_fullprove() {
    test_onnx_fullprove("2lcnvrl_relusig".to_string());
}

// prove-serialize-verify, the usual full path
fn test_onnx_prove_and_verify(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "--bits",
            "16",
            "-K",
            "17",
            "prove",
            "-D",
            format!("./examples/onnx_models/{}_input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx_models/{}.onnx", example_name).as_str(),
            "-O",
            format!("pav_{}.pf", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "--bits",
            "16",
            "-K",
            "17",
            "verify",
            "-M",
            format!("./examples/onnx_models/{}.onnx", example_name).as_str(),
            "-P",
            format!("pav_{}.pf", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_pav() {
    test_onnx_prove_and_verify("ff".to_string());
}

#[test]
fn test_relusig_pav() {
    test_onnx_prove_and_verify("relusig_small".to_string());
}

#[test]
fn test_relurelu_pav() {
    test_onnx_prove_and_verify("relurelu_small".to_string());
}

#[test]
fn test_relu_pav() {
    test_onnx_prove_and_verify("relu".to_string());
}

#[test]
fn test_sig_pav() {
    test_onnx_prove_and_verify("sig".to_string());
}
