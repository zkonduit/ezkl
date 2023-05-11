#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod native_tests {

    use lazy_static::lazy_static;
    use std::env::var;
    use std::process::Command;
    use std::sync::Once;
    use tempdir::TempDir;
    static COMPILE: Once = Once::new();
    static KZG17: Once = Once::new();
    static KZG23: Once = Once::new();
    //Sure to run this once

    lazy_static! {
        static ref CARGO_TARGET_DIR: String =
            var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
        static ref TEST_DIR: TempDir = TempDir::new("example").unwrap();
    }

    fn init_binary() {
        COMPILE.call_once(|| {
            println!("using cargo target dir: {}", *CARGO_TARGET_DIR);
            build_ezkl();
        });
    }

    fn init_params_17() {
        KZG17.call_once(|| {
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "gen-srs",
                    &format!(
                        "--params-path={}/kzg17.params",
                        TEST_DIR.path().to_str().unwrap()
                    ),
                    "--logrows=17",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        });
    }

    fn init_params_23() {
        KZG23.call_once(|| {
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "gen-srs",
                    &format!(
                        "--params-path={}/kzg23.params",
                        TEST_DIR.path().to_str().unwrap()
                    ),
                    "--logrows=23",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        })
    }

    const TESTS: [&str; 30] = [
        "1l_mlp",
        "1l_flatten",
        "1l_average",
        "1l_div",
        "1l_pad",
        "1l_reshape",
        "1l_eltwise_div",
        "1l_sigmoid",
        "1l_sqrt",
        "1l_softmax",
        // "1l_instance_norm",
        "1l_batch_norm",
        "1l_prelu",
        "1l_leakyrelu",
        "1l_gelu_noappx",
        // "1l_gelu_tanh_appx",
        "1l_relu",
        "1l_tanh",
        "2l_relu_sigmoid_small",
        "2l_relu_fc",
        "2l_relu_small",
        "2l_relu_sigmoid",
        "1l_conv",
        "2l_sigmoid_small",
        "2l_relu_sigmoid_conv",
        "3l_relu_conv_fc",
        "4l_relu_conv_fc",
        "1l_erf",
        "1l_var",
        "min",
        "max",
        "1l_max_pool",
    ];

    const PACKING_TESTS: [&str; 14] = [
        "1l_mlp",
        "1l_average",
        "1l_div",
        "1l_reshape",
        "1l_sigmoid",
        "1l_sqrt",
        "1l_leakyrelu",
        // "1l_prelu",
        "1l_var",
        "1l_relu",
        "1l_tanh",
        "1l_gelu_noappx",
        "2l_relu_sigmoid_small",
        "2l_relu_fc",
        "2l_relu_small",
    ];

    const TESTS_AGGR: [&str; 20] = [
        "1l_mlp",
        "1l_flatten",
        "1l_average",
        "1l_reshape",
        "1l_div",
        "1l_pad",
        "1l_sigmoid",
        "1l_gelu_noappx",
        "1l_sqrt",
        // "1l_prelu",
        "1l_var",
        "1l_leakyrelu",
        "1l_relu",
        "1l_tanh",
        "2l_relu_fc",
        "2l_relu_sigmoid_small",
        "2l_relu_small",
        "1l_conv",
        "min",
        "max",
        "1l_max_pool",
    ];

    const NEG_TESTS: [(&str, &str); 2] = [
        ("2l_relu_sigmoid_small", "2l_relu_small"),
        ("2l_relu_small", "2l_relu_sigmoid_small"),
    ];

    const TESTS_EVM: [&str; 18] = [
        "1l_mlp",
        "1l_flatten",
        "1l_average",
        "1l_reshape",
        "1l_sigmoid",
        "1l_div",
        "1l_sqrt",
        // "1l_prelu",
        "1l_var",
        "1l_leakyrelu",
        "1l_gelu_noappx",
        "1l_relu",
        "1l_tanh",
        "2l_relu_sigmoid_small",
        "2l_relu_small",
        "2l_relu_fc",
        "min",
        "max",
        "1l_max_pool",
    ];

    const EXAMPLES: [&str; 2] = ["mlp_4d", "conv2d_mnist"];

    macro_rules! test_func_aggr {
    () => {
        #[cfg(test)]
        mod tests_aggr {
            use seq_macro::seq;
            use crate::native_tests::TESTS_AGGR;
            use test_case::test_case;
            use crate::native_tests::kzg_aggr_prove_and_verify;


            seq!(N in 0..=17 {

            #(#[test_case(TESTS_AGGR[N])])*
            fn kzg_aggr_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::init_params_23();
                kzg_aggr_prove_and_verify(test.to_string());
            }

            });
    }
    };
}

    macro_rules! test_packed_func {
    () => {
        #[cfg(test)]
        mod packed_tests {
            use seq_macro::seq;
            use test_case::test_case;
            use crate::native_tests::PACKING_TESTS;
            use crate::native_tests::mock_packed_outputs;
            use crate::native_tests::mock_everything;

            seq!(N in 0..=13 {

            #(#[test_case(PACKING_TESTS[N])])*
            fn mock_packed_outputs_(test: &str) {
                crate::native_tests::init_binary();
                mock_packed_outputs(test.to_string());
            }

            #(#[test_case(PACKING_TESTS[N])])*
            fn mock_everything_(test: &str) {
                crate::native_tests::init_binary();
                mock_everything(test.to_string());
            }

            });

    }
    };
}

    macro_rules! test_func {
    () => {
        #[cfg(test)]
        mod tests {
            use seq_macro::seq;
            use crate::native_tests::TESTS;
            use test_case::test_case;
            use crate::native_tests::mock;
            use crate::native_tests::mock_public_inputs;
            use crate::native_tests::mock_public_params;
            use crate::native_tests::forward_pass;
            use crate::native_tests::kzg_prove_and_verify;
            use crate::native_tests::render_circuit;
            use crate::native_tests::tutorial as run_tutorial;


            #[test]
            fn tutorial_() {
                run_tutorial();
            }


            seq!(N in 0..=29 {

            #(#[test_case(TESTS[N])])*
            fn render_circuit_(test: &str) {
                crate::native_tests::init_binary();
                render_circuit(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                mock(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_inputs_(test: &str) {
                crate::native_tests::init_binary();
                mock_public_inputs(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_params_(test: &str) {
                crate::native_tests::init_binary();
                mock_public_params(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn forward_pass_(test: &str) {
                crate::native_tests::init_binary();
                forward_pass(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::init_params_17();
                kzg_prove_and_verify(test.to_string());
            }

            });

    }
    };
}

    macro_rules! test_func_evm {
    () => {
        #[cfg(test)]
        mod tests_evm {
            use seq_macro::seq;
            use crate::native_tests::TESTS_EVM;
            use test_case::test_case;
            use crate::native_tests::kzg_evm_prove_and_verify;
            use crate::native_tests::kzg_evm_aggr_prove_and_verify;

            /// Not all models will pass VerifyEVM because their contract size exceeds the limit, so we only
            /// specify a few that will
            const TESTS_SOLIDITY: [&str; 9] = [
                "1l_relu",
                "1l_div",
                "1l_leakyrelu",
                "1l_sqrt",
                // "1l_prelu",
                "1l_gelu_noappx",
                "1l_sigmoid",
                "1l_reshape",
                "2l_relu_fc",
                "1l_var"
            ];


            seq!(N in 0..=17 {

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    kzg_evm_prove_and_verify(test.to_string(), TESTS_SOLIDITY.contains(&test));
                }
                // these take a particularly long time to run
                #(#[test_case(TESTS_EVM[N])])*
                #[ignore]
                fn kzg_evm_aggr_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_23();
                    kzg_evm_aggr_prove_and_verify(test.to_string());
                }

            });
    }
    };
}

    macro_rules! test_func_examples {
    () => {
        #[cfg(test)]
        mod tests_examples {
            use seq_macro::seq;
            use crate::native_tests::EXAMPLES;
            use test_case::test_case;
            use crate::native_tests::run_example as run;
            seq!(N in 0..=1 {
            #(#[test_case(EXAMPLES[N])])*
            fn example_(test: &str) {
                run(test.to_string());
            }
            });
    }
    };
}

    macro_rules! test_neg_examples {
    () => {
        #[cfg(test)]
        mod neg_tests {
            use seq_macro::seq;
            use crate::native_tests::NEG_TESTS;
            use test_case::test_case;
            use crate::native_tests::neg_mock as run;
            seq!(N in 0..=1 {
            #(#[test_case(NEG_TESTS[N])])*
            fn neg_examples_(test: (&str, &str)) {
                crate::native_tests::init_binary();
                run(test.0.to_string(), test.1.to_string());
            }

            });
    }
    };
}

    test_func!();
    test_func_aggr!();
    test_func_evm!();
    test_func_examples!();
    test_neg_examples!();
    test_packed_func!();

    // Mock prove (fast, but does not cover some potential issues)
    fn neg_mock(example_name: String, counter_example: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("./examples/onnx/{}/input.json", counter_example).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(!status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn run_example(example_name: String) {
        let status = Command::new("cargo")
            .args(["run", "--release", "--example", example_name.as_str()])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn forward_pass(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "forward",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "-O",
                format!(
                    "{}/{}_input_forward.json",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                )
                .as_str(),
                "--bits=16",
                "-K=17",
                // "-K",
                // "2",  //causes failure
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!(
                    "{}/{}_input_forward.json",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                )
                .as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn render_circuit(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "render-circuit",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "-O",
                format!(
                    "{}/{}_render.png",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                )
                .as_str(),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn tutorial() {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                "./examples/onnx/tutorial/input.json".to_string().as_str(),
                "-M",
                "./examples/onnx/tutorial/network.onnx".to_string().as_str(),
                "--tolerance=2",
                "--scale=4",
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn mock(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn mock_packed_outputs(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--bits=16",
                "-K=17",
                "--pack-base=2",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn mock_everything(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--bits=16",
                "-K=17",
                "--public-inputs=true",
                "--pack-base=2",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn mock_public_inputs(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--public-inputs=true",
                "--public-outputs=false",
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn mock_public_params(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--public-params=true",
                "--public-outputs=false",
                "-K=17",
                "--bits=16",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_aggr_prove_and_verify(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--pk-path",
                &format!("{}/{}.pk", TEST_DIR.path().to_str().unwrap(), example_name),
                "--vk-path",
                &format!("{}/{}.vk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--proof-path",
                &format!("{}/{}.pf", TEST_DIR.path().to_str().unwrap(), example_name),
                "--pk-path",
                &format!("{}/{}.pk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                "--transcript=poseidon",
                "--strategy=accum",
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "aggregate",
                "--logrows=23",
                &format!(
                    "--circuit-params-paths={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--aggregation-snarks",
                &format!("{}/{}.pf", TEST_DIR.path().to_str().unwrap(), example_name),
                "--aggregation-vk-paths",
                &format!("{}/{}.vk", TEST_DIR.path().to_str().unwrap(), example_name),
                "--proof-path",
                &format!(
                    "{}/{}_aggr.pf",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--vk-path",
                &format!(
                    "{}/{}_aggr.vk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                "--transcript=blake",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "verify-aggr",
                "--logrows=23",
                "--proof-path",
                &format!(
                    "{}/{}_aggr.pf",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--vk-path",
                &format!(
                    "{}/{}_aggr.vk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_evm_aggr_prove_and_verify(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--vk-path",
                &format!(
                    "{}/{}_evm.vk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--pk-path",
                &format!(
                    "{}/{}_evm.pk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--proof-path",
                &format!(
                    "{}/{}_evm.pf",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--pk-path",
                &format!(
                    "{}/{}_evm.pk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--transcript=poseidon",
                "--strategy=accum",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "aggregate",
                "--logrows=23",
                &format!(
                    "--circuit-params-paths={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--aggregation-snarks",
                &format!(
                    "{}/{}_evm.pf",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--aggregation-vk-paths",
                &format!(
                    "{}/{}_evm.vk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--proof-path",
                &format!(
                    "{}/{}_evm_aggr.pf",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--vk-path",
                &format!(
                    "{}/{}_evm_aggr.vk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                "--transcript=evm",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "create-evm-verifier-aggr",
                "--deployment-code-path",
                &format!(
                    "{}/{}_evm_aggr.code",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                &format!(
                    "--params-path={}/kzg23.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                "--vk-path",
                &format!(
                    "{}/{}_evm_aggr.vk",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "verify-evm",
                "--proof-path",
                &format!(
                    "{}/{}_evm_aggr.pf",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--deployment-code-path",
                &format!(
                    "{}/{}_evm_aggr.code",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_prove_and_verify(example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--pk-path",
                &format!("{}/{}.pk", TEST_DIR.path().to_str().unwrap(), example_name),
                "--vk-path",
                &format!("{}/{}.vk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg17.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--proof-path",
                &format!("{}/{}.pf", TEST_DIR.path().to_str().unwrap(), example_name),
                "--pk-path",
                &format!("{}/{}.pk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg17.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                "--transcript=blake",
                "--strategy=single",
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "verify",
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--proof-path",
                &format!("{}/{}.pf", TEST_DIR.path().to_str().unwrap(), example_name),
                "--vk-path",
                &format!("{}/{}.vk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg17.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_evm_prove_and_verify(example_name: String, with_solidity: bool) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--pk-path",
                &format!("{}/{}.pk", TEST_DIR.path().to_str().unwrap(), example_name),
                "--vk-path",
                &format!("{}/{}.vk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg17.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-D",
                format!("./examples/onnx/{}/input.json", example_name).as_str(),
                "-M",
                format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
                "--proof-path",
                &format!("{}/{}.pf", TEST_DIR.path().to_str().unwrap(), example_name),
                "--pk-path",
                &format!("{}/{}.pk", TEST_DIR.path().to_str().unwrap(), example_name),
                &format!(
                    "--params-path={}/kzg17.params",
                    TEST_DIR.path().to_str().unwrap()
                ),
                "--transcript=evm",
                "--strategy=single",
                &format!(
                    "--circuit-params-path={}/{}.params",
                    TEST_DIR.path().to_str().unwrap(),
                    example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let circuit_params = format!(
            "--circuit-params-path={}/{}.params",
            TEST_DIR.path().to_str().unwrap(),
            example_name
        );
        let code_arg = format!(
            "{}/{}.code",
            TEST_DIR.path().to_str().unwrap(),
            example_name
        );
        let vk_arg = format!("{}/{}.vk", TEST_DIR.path().to_str().unwrap(), example_name);
        let param_arg = format!(
            "--params-path={}/kzg17.params",
            TEST_DIR.path().to_str().unwrap()
        );

        let mut args = vec![
            "create-evm-verifier",
            circuit_params.as_str(),
            "--deployment-code-path",
            code_arg.as_str(),
            param_arg.as_str(),
            "--vk-path",
            vk_arg.as_str(),
        ];

        let sol_arg = format!("kzg_{}.sol", example_name);

        if with_solidity {
            args.push("--sol-code-path");
            args.push(sol_arg.as_str());
        }
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let pf_arg = format!("{}/{}.pf", TEST_DIR.path().to_str().unwrap(), example_name);

        let mut args = vec![
            "verify-evm",
            "--proof-path",
            pf_arg.as_str(),
            "--deployment-code-path",
            code_arg.as_str(),
        ];
        if with_solidity {
            args.push("--sol-code-path");
            //args.push(format!("kzg_{}.sol", example_name).as_str());
            args.push(sol_arg.as_str());
        }
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    fn build_ezkl() {
        let status = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--features",
                "render",
                "--bin",
                "ezkl",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }
}
