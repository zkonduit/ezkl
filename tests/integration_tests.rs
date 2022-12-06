use std::process::Command;

const TESTS: [&str; 12] = [
    "1l_mlp",
    "1l_flatten",
    "1l_average",
    "2l_relu_sigmoid",
    "1l_conv",
    "2l_relu_sigmoid_conv",
    "1l_reshape",
    "1l_sigmoid",
    "1l_relu",
    "2l_relu_sigmoid_small",
    "2l_relu_small",
    "2l_relu_sigmoid",
];

macro_rules! test_func {
    ($name:ident, $func:ident) => {
        #[cfg(test)]
        mod tests {
            use seq_macro::seq;
            use crate::TESTS;
            use test_case::test_case;
            use crate::test_onnx_mock;
            use crate::test_onnx_mock_public_inputs;
            use crate::test_onnx_mock_public_params;
            use crate::test_onnx_fullprove;
            use crate::test_onnx_prove_and_verify;
            use crate::test_kzg_fullprove;

            seq!(N in 0..=11 {
            #(#[test_case(TESTS[N])])*
            fn test_onnx_mock_(test: &str) {
                test_onnx_mock(test.to_string());
            }
            });
            seq!(N in 0..=11 {
                #(#[test_case(TESTS[N])])*
                fn test_onnx_mock_public_inputs_(test: &str) {
                    test_onnx_mock_public_inputs(test.to_string());
                }
            });

            seq!(N in 0..=11 {
                #(#[test_case(TESTS[N])])*
                fn test_onnx_mock_public_params_(test: &str) {
                    test_onnx_mock_public_params(test.to_string());
                }
            });

            seq!(N in 0..=11 {
                #(#[test_case(TESTS[N])])*
                fn test_onnx_fullprove_(test: &str) {
                    test_onnx_fullprove(test.to_string());
                }
            });

            seq!(N in 0..=11 {
                #(#[test_case(TESTS[N])])*
                fn test_onnx_prove_and_verify_(test: &str) {
                    test_onnx_prove_and_verify(test.to_string());
                }
            });

            seq!(N in 0..=11 {
                #(#[test_case(TESTS[N])])*
                fn test_kzg_fullprove_(test: &str) {
                    test_kzg_fullprove(test.to_string());
                }
            });

    }
    };
}

test_func!(onnx_mock_tests, test_onnx_mock);

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
