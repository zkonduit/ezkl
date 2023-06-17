#[cfg(test)]
mod wasi_tests {
    use lazy_static::lazy_static;
    use std::env::var;
    use std::process::Command;
    use std::sync::Once;
    use tempdir::TempDir;
    static COMPILE: Once = Once::new();

    lazy_static! {
        static ref CARGO_TARGET_DIR: String =
            var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
        static ref TEST_DIR: TempDir = TempDir::new("example").unwrap();
    }

    fn mv_test_(test: &str) {
        let test_dir = TEST_DIR.path().to_str().unwrap();
        let path: std::path::PathBuf = format!("{}/{}", test_dir, test).into();
        if !path.exists() {
            let status = Command::new("cp")
                .args([
                    "-R",
                    &format!("./examples/onnx/{}", test),
                    &format!("{}/{}", test_dir, test),
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        }
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

    macro_rules! wasi_test_func {
    () => {
        #[cfg(test)]
        mod tests_wasi {
            use seq_macro::seq;
            use crate::wasi_tests::TESTS;
            use test_case::test_case;
            use crate::wasi_tests::mock;

            seq!(N in 0..=18 {

            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                crate::wasi_tests::init();
                crate::wasi_tests::mv_test_(test);
                mock(test.to_string(), 7, 16, 17, "private", "private", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_inputs_(test: &str) {
                crate::wasi_tests::init();
                crate::wasi_tests::mv_test_(test);
                mock(test.to_string(), 7, 16, 17, "public", "private", "private", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_params_(test: &str) {
                crate::wasi_tests::init();
                crate::wasi_tests::mv_test_(test);
                mock(test.to_string(), 7, 16, 17, "private", "public", "private", 1);
            }


            });

    }
    };
}

    wasi_test_func!();

    // Mock prove (fast, but does not cover some potential issues)
    fn mock(
        example_name: String,
        scale: usize,
        bits: usize,
        logrows: usize,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        batch_size: usize,
    ) {
        let test_dir = TEST_DIR.path().to_str().unwrap();

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-settings",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                &format!("--bits={}", bits),
                &format!("--logrows={}", logrows),
                &format!("--scale={}", scale),
                &format!("--batch-size={}", batch_size),
                &format!("--input-visibility={}", input_visibility),
                &format!("--param-visibility={}", param_visibility),
                &format!("--output-visibility={}", output_visibility),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "forward",
                "-D",
                &format!("{}/{}/input.json", test_dir, example_name),
                "-M",
                &format!("{}/{}/network.onnx", test_dir, example_name),
                "-O",
                &format!("{}/{}/input_forward.json", test_dir, example_name),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("{}/{}/input_forward.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
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
