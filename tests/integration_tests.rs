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
            format!("./examples/onnx/examples/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_mock() {
    test_onnx_mock("1l_mlp".to_string());
}

#[test]
fn test_flatten_mock() {
    test_onnx_mock("1l_flatten".to_string());
}

#[test]
fn test_avg_mock() {
    test_onnx_mock("1l_average".to_string());
}

#[test]
fn test_relusig_mock() {
    test_onnx_mock("2l_relu_sigmoid".to_string());
}

#[test]
#[ignore]
fn test_1lcnvrl_mock() {
    test_onnx_mock("1l_conv".to_string());
}

#[test]
fn test_2lcnvrl_mock() {
    test_onnx_mock("2l_relu_sigmoid_conv".to_string());
}

#[test]
fn test_reshape_mock() {
    test_onnx_mock("1l_reshape".to_string());
}

// Mock prove (fast, but does not cover some potential issues)
fn test_onnx_mock_public_inputs(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "--public-inputs",
            "--bits",
            "16",
            "-K",
            "17",
            "mock",
            "-D",
            format!("./examples/onnx/examples/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_mock_public_inputs() {
    test_onnx_mock_public_inputs("1l_mlp".to_string());
}

#[test]
fn test_flatten_mock_public_inputs() {
    test_onnx_mock_public_inputs("1l_flatten".to_string());
}

#[test]
fn test_avg_mock_public_inputs() {
    test_onnx_mock_public_inputs("1l_average".to_string());
}

#[test]
fn test_relusig_mock_public_inputs() {
    test_onnx_mock_public_inputs("2l_relu_sigmoid".to_string());
}

#[test]
#[ignore]
fn test_1lcnvrl_mock_public_inputs() {
    test_onnx_mock_public_inputs("1l_conv".to_string());
}

#[test]
fn test_2lcnvrl_mock_public_inputs() {
    test_onnx_mock_public_inputs("2l_relu_sigmoid_conv".to_string());
}

#[test]
fn test_reshape_mock_public_inputs() {
    test_onnx_mock_public_inputs("1l_reshape".to_string());
}

// Mock prove (fast, but does not cover some potential issues)
fn test_onnx_mock_public_params(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "--public-params",
            "--bits",
            "16",
            "-K",
            "17",
            "mock",
            "-D",
            format!("./examples/onnx/examples/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_mock_public_params() {
    test_onnx_mock_public_params("1l_mlp".to_string());
}

#[test]
fn test_flatten_mock_public_params() {
    test_onnx_mock_public_params("1l_flatten".to_string());
}

#[test]
fn test_avg_mock_public_params() {
    test_onnx_mock_public_params("1l_average".to_string());
}

#[test]
fn test_relusig_mock_public_params() {
    test_onnx_mock_public_params("2l_relu_sigmoid".to_string());
}

#[test]
#[ignore]
fn test_1lcnvrl_mock_public_params() {
    test_onnx_mock_public_params("1l_conv".to_string());
}

#[test]
fn test_2lcnvrl_mock_public_params() {
    test_onnx_mock_public_params("2l_relu_sigmoid_conv".to_string());
}

#[test]
fn test_reshape_mock_public_params() {
    test_onnx_mock_public_params("1l_reshape".to_string());
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
            format!("./examples/onnx/examples/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_fullprove() {
    test_onnx_fullprove("1l_mlp".to_string());
}

#[test]
fn test_avg_fullprove() {
    test_onnx_fullprove("1l_average".to_string());
}

#[test]
#[ignore]
fn test_relusig_fullprove() {
    test_onnx_fullprove("2l_relu_sigmoid".to_string());
}

#[test]
fn test_relurelu_fullprove() {
    test_onnx_fullprove("2l_relu_small".to_string());
}

#[test]
fn test_relusig_small_fullprove() {
    test_onnx_fullprove("2l_relu_sigmoid_small".to_string());
}

#[test]
fn test_relu_fullprove() {
    test_onnx_fullprove("1l_relu".to_string());
}

#[test]
fn test_sig_fullprove() {
    test_onnx_fullprove("1l_sigmoid".to_string());
}

#[test]
fn test_reshape_fullprove() {
    test_onnx_fullprove("1l_reshape".to_string());
}

// These require too much memory for Github CI right now
#[test]
#[ignore]
fn test_1lcnvrl_fullprove() {
    test_onnx_fullprove("1l_conv".to_string());
}

#[test]
#[ignore]
fn test_2lcnvrl_fullprove() {
    test_onnx_fullprove("2l_relu_sigmoid_conv".to_string());
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
            format!("./examples/onnx/examples/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
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
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
            "-P",
            format!("pav_{}.pf", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
fn test_ff_pav() {
    test_onnx_prove_and_verify("1l_mlp".to_string());
}

#[test]
fn test_relusig_pav() {
    test_onnx_prove_and_verify("2l_relu_sigmoid_small".to_string());
}

#[test]
fn test_relurelu_pav() {
    test_onnx_prove_and_verify("2l_relu_small".to_string());
}

#[test]
fn test_relu_pav() {
    test_onnx_prove_and_verify("1l_relu".to_string());
}

#[test]
fn test_sig_pav() {
    test_onnx_prove_and_verify("1l_sigmoid".to_string());
}

// KZG / EVM tests
// full prove (slower, covers more, but still reuses the pk)
fn test_kzg_fullprove(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--features",
            "evm",
            "--bin",
            "ezkl",
            "--",
            "--bits",
            "16",
            "-K",
            "17",
            "fullprove",
            "-D",
            format!("./examples/onnx/examples/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/examples/{}/network.onnx", example_name).as_str(),
            "--pfsys",
            "kzg",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

#[test]
#[ignore]
fn test_avg_fullprove_kzg() {
    test_kzg_fullprove("1l_average".to_string());
}

#[test]
#[ignore]
fn test_ff_fullprove_kzg() {
    test_kzg_fullprove("1l_mlp".to_string());
}

#[test]
#[ignore]
fn test_relusig_fullprove_kzg() {
    test_kzg_fullprove("2l_relu_sigmoid".to_string());
}

#[test]
#[ignore]
fn test_relurelu_fullprove_kzg() {
    test_kzg_fullprove("2l_relu_small".to_string());
}

#[test]
#[ignore]
fn test_relusig_small_fullprove_kzg() {
    test_kzg_fullprove("2l_relu_sigmoid_small".to_string());
}

#[test]
#[ignore]
fn test_relu_fullprove_kzg() {
    test_kzg_fullprove("1l_relu".to_string());
}

#[test]
#[ignore]
fn test_sig_fullprove_kzg() {
    test_kzg_fullprove("1l_sigmoid".to_string());
}

// These require too much memory for Github CI right now
#[test]
#[ignore]
fn test_1lcnvrl_fullprove_kzg() {
    test_kzg_fullprove("1l_conv".to_string());
}

#[test]
#[ignore]
fn test_2lcnvrl_fullprove_kzg() {
    test_kzg_fullprove("2l_relu_sigmoid_conv".to_string());
}
