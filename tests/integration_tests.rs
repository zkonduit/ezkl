use std::fs::remove_file;
use std::process::{Command, Output};

fn test_onnx_example(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
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
fn test_ff_example() {
    test_onnx_example("ff".to_string());
}

#[test]
fn test_relusig_example() {
    test_onnx_example("relusig".to_string());
}

#[test]
#[ignore]
fn test_1lcnvrl_example() {
    test_onnx_example("1lcnvrl".to_string());
}

#[test]
fn test_2lcnvrl_example() {
    test_onnx_example("2lcnvrl_relusig".to_string());
}

fn test_onnx_prove_and_verify(example_name: String) {
    let status = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
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
    let output = Command::new("cargo")
        .args([
            "run",
            "--release",
            "--bin",
            "ezkl",
            "--",
            "verify",
            "-M",
            format!("./examples/onnx_models/{}.onnx", example_name).as_str(),
            "-P",
            format!("pav_{}.pf", example_name).as_str(),
        ])
        .output()
        .expect("failed to execute process");
    let sout = String::from_utf8(output.stdout).unwrap();
    println!("{}", sout);
    assert_eq!("Verified: true\n", sout);
}

#[test]
fn test_ff_pav() {
    test_onnx_prove_and_verify("ff".to_string());
}
