use lazy_static::lazy_static;
use std::env::var;
use std::process::Command;

lazy_static! {
    static ref CARGO_TARGET_DIR: String =
        var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
}

#[cfg(test)]
#[ctor::ctor]
fn init() {
    println!("using cargo target dir: {}", *CARGO_TARGET_DIR);
    build_ezkl_wasm();
}

const TESTS: [&str; 19] = [
    "1l_mlp",
    "1l_flatten",
    "1l_average",
    "1l_div",
    "1l_pad",
    "1l_reshape",
    "1l_sigmoid",
    "1l_sqrt",
    "1l_leakyrelu",
    "1l_relu",
    "2l_relu_sigmoid_small",
    "2l_relu_fc",
    "2l_relu_small",
    "2l_relu_sigmoid",
    "1l_conv",
    "2l_sigmoid_small",
    "2l_relu_sigmoid_conv",
    "3l_relu_conv_fc",
    "4l_relu_conv_fc",
];

const PACKING_TESTS: [&str; 11] = [
    "1l_mlp",
    "1l_average",
    "1l_div",
    "1l_reshape",
    "1l_sigmoid",
    "1l_sqrt",
    "1l_leakyrelu",
    "1l_relu",
    "2l_relu_sigmoid_small",
    "2l_relu_fc",
    "2l_relu_small",
];

const NEG_TESTS: [(&str, &str); 2] = [
    ("2l_relu_sigmoid_small", "2l_relu_small"),
    ("2l_relu_small", "2l_relu_sigmoid_small"),
];

macro_rules! wasi_test_packed_func {
    () => {
        #[cfg(test)]
        mod packed_tests_wasi {
            use seq_macro::seq;
            use test_case::test_case;
            use crate::PACKING_TESTS;
            use crate::mock_packed_outputs;
            use crate::mock_everything;

            seq!(N in 0..=10 {


            #(#[test_case(PACKING_TESTS[N])])*
            fn mock_packed_outputs_(test: &str) {
                mock_packed_outputs(test.to_string());
            }

            #(#[test_case(PACKING_TESTS[N])])*
            fn mock_everything_(test: &str) {
                mock_everything(test.to_string());
            }

            });


    }
    };
}

macro_rules! wasi_test_func {
    () => {
        #[cfg(test)]
        mod tests_wasi {
            use seq_macro::seq;
            use crate::TESTS;
            use test_case::test_case;
            use crate::mock;
            use crate::mock_public_inputs;
            use crate::mock_public_params;
            use crate::forward_pass;
            use crate::mock_single_lookup;

            seq!(N in 0..=18 {


            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                mock(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_single_lookup_(test: &str) {
                mock_single_lookup(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_inputs_(test: &str) {
                mock_public_inputs(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_params_(test: &str) {
                mock_public_params(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn forward_pass_(test: &str) {
                forward_pass(test.to_string());
            }

            });


    }
    };
}

macro_rules! wasi_test_neg_examples {
    () => {
        #[cfg(test)]
        mod neg_tests_wasi {
            use seq_macro::seq;
            use crate::NEG_TESTS;
            use test_case::test_case;
            use crate::neg_mock as run;
            use crate::neg_mock_single_lookup as run_single_lookup;
            seq!(N in 0..=1 {
            #(#[test_case(NEG_TESTS[N])])*
            fn neg_examples_(test: (&str, &str)) {
                run(test.0.to_string(), test.1.to_string());
            }

            #(#[test_case(NEG_TESTS[N])])*
            fn neg_examples_single_lookup_(test: (&str, &str)) {
                run_single_lookup(test.0.to_string(), test.1.to_string());
            }
            });
    }
    };
}

wasi_test_func!();
wasi_test_neg_examples!();
wasi_test_packed_func!();

// Mock prove (fast, but does not cover some potential issues)
fn neg_mock(example_name: String, counter_example: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", counter_example).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(!status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn neg_mock_single_lookup(example_name: String, counter_example: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "--single-lookup",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", counter_example).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(!status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn forward_pass(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "forward",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "-O",
            format!("./examples/onnx/{}/input_forward.json", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());

    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input_forward.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_packed_outputs(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "--pack-base=2",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_everything(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "--single-lookup",
            "--public-inputs",
            "--pack-base=2",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_single_lookup(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--bits=16",
            "-K=17",
            "--single-lookup",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_public_inputs(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--public-inputs",
            "--public-outputs=false",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_public_params(example_name: String) {
    let status = Command::new("wasmtime")
        .args([
            &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
            "--dir",
            ".",
            "--",
            "--public-params",
            "--public-outputs=false",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

fn build_ezkl_wasm() {
    let status = Command::new("wasm-pack")
        .args([
            "build",
            "--release",
            "--bin",
            "ezkl",
            "--target",
            "wasm32-wasi",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}
