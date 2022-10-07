use std::process::Command;

fn test_onnx_example(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
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
fn test_ff_example() {
    test_onnx_example("ff".to_string());
}

#[test]
fn test_relusig_example() {
    test_onnx_example("relusig".to_string());
}

#[test]
fn test_1lcnvrl_example() {
    test_onnx_example("1lcnvrl".to_string());
}

#[test]
fn test_relusig_example() {
    test_onnx_example("2lcnvrl_sigmoid".to_string());
}
