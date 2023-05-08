#[cfg(test)]
mod wasi_tests {
    use lazy_static::lazy_static;
    use std::env::var;
    use std::process::Command;
    use std::sync::Once;
    static COMPILE: Once = Once::new();

    lazy_static! {
        static ref CARGO_TARGET_DIR: String =
            var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
    }

    fn init() {
        COMPILE.call_once(|| {
            println!("using cargo target dir: {}", *CARGO_TARGET_DIR);
            build_ezkl_wasm();
        });
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
            use crate::wasi_tests::PACKING_TESTS;
            use crate::wasi_tests::mock_packed_outputs;
            use crate::wasi_tests::mock_everything;

            seq!(N in 0..=10 {
            #(#[test_case(PACKING_TESTS[N])])*
            fn mock_packed_outputs_(test: &str) {
                crate::wasi_tests::init();
                mock_packed_outputs(test.to_string());
            }

            #(#[test_case(PACKING_TESTS[N])])*
            fn mock_everything_(test: &str) {
                crate::wasi_tests::init();
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
            use crate::wasi_tests::TESTS;
            use test_case::test_case;
            use crate::wasi_tests::mock;
            use crate::wasi_tests::mock_public_inputs;
            use crate::wasi_tests::mock_public_params;
            use crate::wasi_tests::forward_pass;

            seq!(N in 0..=18 {

            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                crate::wasi_tests::init();
                mock(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_inputs_(test: &str) {
                crate::wasi_tests::init();
                mock_public_inputs(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_params_(test: &str) {
                crate::wasi_tests::init();
                mock_public_params(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn forward_pass_(test: &str) {
                crate::wasi_tests::init();
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
            use crate::wasi_tests::NEG_TESTS;
            use test_case::test_case;
            use crate::wasi_tests::neg_mock as run;
            seq!(N in 0..=1 {
            #(#[test_case(NEG_TESTS[N])])*
            fn neg_examples_(test: (&str, &str)) {
                crate::wasi_tests::init();
                run(test.0.to_string(), test.1.to_string());
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
                "-K=20",
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
                "-K=20",
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
                "-K=20",
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
                "-K=20",
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
                "-K=20",
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
                "-K=20",
                "--public-inputs=true",
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
    fn mock_public_inputs(example_name: String) {
        let status = Command::new("wasmtime")
            .args([
                &format!("{}/wasm32-wasi/release/ezkl.wasm", *CARGO_TARGET_DIR),
                "--dir",
                ".",
                "--",
                "--public-inputs=true",
                "--public-outputs=false",
                "--bits=16",
                "-K=20",
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
                "--public-params=true",
                "--public-outputs=false",
                "--bits=16",
                "-K=20",
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
        let status = Command::new("cargo")
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
}
