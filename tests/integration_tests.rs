#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod native_tests {

    use core::panic;
    use ezkl::graph::input::{FileSource, GraphData};
    use ezkl::graph::DataSource;
    use lazy_static::lazy_static;
    use std::env::var;
    use std::process::Command;
    use std::sync::Once;
    use tempdir::TempDir;
    static COMPILE: Once = Once::new();
    static KZG19: Once = Once::new();
    static KZG17: Once = Once::new();
    static KZG23: Once = Once::new();
    static KZG26: Once = Once::new();
    static START_ANVIL: Once = Once::new();

    //Sure to run this once

    lazy_static! {
        static ref CARGO_TARGET_DIR: String =
            var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
        static ref TEST_DIR: TempDir = TempDir::new("example").unwrap();
        static ref ANVIL_URL: String = "http://localhost:3030".to_string();
    }

    fn start_anvil() {
        START_ANVIL.call_once(|| {
            let _ = Command::new("anvil")
                .args(["-p", "3030"])
                // .stdout(Stdio::piped())
                .spawn()
                .expect("failed to start anvil process");

            std::thread::sleep(std::time::Duration::from_secs(3));
        });
    }

    fn init_binary() {
        COMPILE.call_once(|| {
            println!("using cargo target dir: {}", *CARGO_TARGET_DIR);
            build_ezkl();
        });
    }

    fn init_params_19() {
        KZG19.call_once(|| {
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "gen-srs",
                    &format!("--srs-path={}/kzg19.srs", TEST_DIR.path().to_str().unwrap()),
                    "--logrows=19",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        });
    }

    fn init_params_17() {
        KZG17.call_once(|| {
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "gen-srs",
                    &format!("--srs-path={}/kzg17.srs", TEST_DIR.path().to_str().unwrap()),
                    "--logrows=17",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        });
    }

    fn init_params_23() {
        KZG23.call_once(|| {
            let status = Command::new("curl")
                .args([
                    "-o",
                    &format!(
                        "{}/kzg23.srs",
                        TEST_DIR.path().to_str().unwrap()
                    ),
                    "https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/perpetual-powers-of-tau-raw-23",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        })
    }

    fn init_params_26() {
        KZG26.call_once(|| {
            let status = Command::new("curl")
                .args([
                    "-o",
                    &format!(
                        "{}/kzg24.srs",
                        TEST_DIR.path().to_str().unwrap()
                    ),
                    "https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/perpetual-powers-of-tau-raw-26",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        })
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

    fn mk_data_batches_(test: &str, output_dir: &str, num_batches: usize) {
        let test_dir = TEST_DIR.path().to_str().unwrap();
        let path: std::path::PathBuf = format!("{}/{}", test_dir, test).into();
        if !path.exists() {
            panic!("test_dir does not exist")
        } else {
            // copy the directory
            let status = Command::new("cp")
                .args([
                    "-R",
                    &format!("{}/{}", test_dir, test),
                    &format!("{}/{}", test_dir, output_dir),
                ])
                .status()
                .expect("failed to execute process");

            assert!(status.success());

            let data = GraphData::from_path(format!("{}/{}/input.json", test_dir, test).into())
                .expect("failed to load input data");

            let input_data = match data.input_data {
                DataSource::File(data) => data,
                DataSource::OnChain(_) => panic!("Only File data sources support batching"),
            };

            let duplicated_input_data: FileSource = input_data
                .iter()
                .map(|data| (0..num_batches).flat_map(|_| data.clone()).collect())
                .collect();

            let duplicated_data = GraphData::new(DataSource::File(duplicated_input_data));

            let res =
                duplicated_data.save(format!("{}/{}/input.json", test_dir, output_dir).into());

            assert!(res.is_ok());
        }
    }

    const PF_FAILURE: &str = "examples/test_failure.proof";

    const PF_FAILURE_AGGR: &str = "examples/test_failure_aggr.proof";

    const LARGE_TESTS: [&str; 3] = ["self_attention", "nanoGPT", "mobilenet"];

    const TESTS: [&str; 36] = [
        "1l_mlp",
        "1l_slice",
        "1l_concat",
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
        "1l_downsample",
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
        "1l_conv_transpose",
        "1l_upsample",
        "idolmodel",
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

    const TESTS_EVM: [&str; 19] = [
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
        "idolmodel",
    ];

    const EXAMPLES: [&str; 2] = ["mlp_4d_einsum", "conv2d_mnist"];

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
                crate::native_tests::mv_test_(test);
                kzg_aggr_prove_and_verify(test.to_string());
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
            use crate::native_tests::LARGE_TESTS;
            use test_case::test_case;
            use crate::native_tests::mock;
            use crate::native_tests::kzg_prove_and_verify;
            use crate::native_tests::kzg_fuzz;
            use crate::native_tests::render_circuit;
            use crate::native_tests::tutorial as run_tutorial;

            #[test]
            fn tutorial_() {
                crate::native_tests::mv_test_("tutorial");
                // percent tolerance test
                run_tutorial("1.0");
            }


            seq!(N in 0..=35 {

            #(#[test_case(TESTS[N])])*
            fn render_circuit_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                render_circuit(test.to_string());
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "private", "private", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_large_batch_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                let large_batch_dir = &format!("large_batches_{}", test);
                crate::native_tests::mk_data_batches_(test, &large_batch_dir, 10);
                mock(large_batch_dir.to_string(), "private", "private", "public", 10);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_inputs_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "public", "private", "private", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "private", "public", "private", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_input_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "hashed", "private", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_encrypted_input_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "encrypted", "private", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "private", "hashed", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_encrypted_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "private", "hashed", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_output_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "public", "private", "hashed", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_encrypted_output_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "public", "private", "encrypted", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_encrypted_input_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "encrypted", "encrypted", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_encrypted_all_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "encrypted", "encrypted", "encrypted", 1);
            }



            #(#[test_case(TESTS[N])])*
            fn mock_encrypted_input_hashed_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "encrypted", "hashed", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_input_output_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "hashed", "private", "hashed", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_input_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                // needs an extra row for the large model
                mock(test.to_string(),"hashed", "hashed", "public", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_all_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                // needs an extra row for the large model
                mock(test.to_string(),"hashed", "hashed", "hashed", 1);
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::init_params_17();
                crate::native_tests::mv_test_(test);
                kzg_prove_and_verify(test.to_string(), 17, "safe", "private", "private", "public");
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_hashed_output(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::init_params_17();
                crate::native_tests::mv_test_(test);
                kzg_prove_and_verify(test.to_string(), 17, "safe", "private", "private", "hashed");
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_encrypted_output(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::init_params_19();
                crate::native_tests::mv_test_(test);
                kzg_prove_and_verify(test.to_string(), 19, "safe", "private", "private", "encrypted");
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_fuzz_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                kzg_fuzz(test.to_string(), 7, 16, 17, "blake");
            }

            });


            seq!(N in 0..=2 {

            #(#[test_case(LARGE_TESTS[N])])*
            #[ignore]
            fn large_kzg_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::init_params_26();
                crate::native_tests::mv_test_(test);
                kzg_prove_and_verify(test.to_string(), 24, "unsafe", "private", "private", "public");
            }

            #(#[test_case(LARGE_TESTS[N])])*
            #[ignore]
            fn large_mock_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::mv_test_(test);
                mock(test.to_string(), "private", "private", "public", 1);
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
            use crate::native_tests::kzg_evm_on_chain_input_prove_and_verify;
            use crate::native_tests::kzg_evm_aggr_prove_and_verify;
            use crate::native_tests::kzg_fuzz;

            /// Currently only on chain inputs that return a non-negative value are supported.
            const TESTS_ON_CHAIN_INPUT: [&str; 11] = [
                "1l_mlp",
                "1l_average",
                "1l_reshape",
                // "1l_sigmoid",
                "1l_div",
                "1l_sqrt",
                // "1l_prelu",
                "1l_var",
                "1l_leakyrelu",
                "1l_gelu_noappx",
                "1l_relu",
                //"1l_tanh",
                // "2l_relu_sigmoid_small",
                // "2l_relu_small",
                // "2l_relu_fc",
                "min",
                "max",
            ];



            seq!(N in 0..= 10 {
                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_input_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_on_chain_input_prove_and_verify(test.to_string(), 200, "on-chain", "file", 17);
                }
            });

            seq!(N in 0..= 10 {
                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_on_chain_input_prove_and_verify(test.to_string(), 200, "file", "on-chain", 17);
                }
            });


            seq!(N in 0..= 10 {
                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_input_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_on_chain_input_prove_and_verify(test.to_string(), 200, "on-chain", "on-chain", 17);
                }
            });


            seq!(N in 0..= 18 {

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_prove_and_verify(test.to_string(), "private", "private", "public", 1, 17);
                }

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_hashed_input_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_prove_and_verify(test.to_string(), "hashed", "private", "private", 1, 17);
                }

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_hashed_params_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_prove_and_verify(test.to_string(), "private", "hashed", "public", 1, 17);
                }

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_hashed_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_17();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_evm_prove_and_verify(test.to_string(), "private", "private", "hashed", 1, 17);
                }


                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_fuzz_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
                    kzg_fuzz(test.to_string(), 7, 16, 17, "evm");
                }

                // these take a particularly long time to run
                #(#[test_case(TESTS_EVM[N])])*
                #[ignore]
                fn kzg_evm_aggr_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    crate::native_tests::init_params_23();
                    crate::native_tests::mv_test_(test);
                    crate::native_tests::start_anvil();
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
                crate::native_tests::mv_test_(test.0);
                crate::native_tests::mv_test_(test.1);
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

    // Mock prove (fast, but does not cover some potential issues)
    fn neg_mock(example_name: String, counter_example: String) {
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
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-W",
                format!("{}/{}/input.json", test_dir, counter_example).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
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
    fn mock(
        example_name: String,
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
                "calibrate-settings",
                "--data",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
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

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                &format!("{}/{}/input.json", test_dir, example_name),
                "-M",
                &format!("{}/{}/network.onnx", test_dir, example_name),
                "-O",
                &format!("{}/{}/witness_mock.json", test_dir, example_name),
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
                "-W",
                format!("{}/{}/witness_mock.json", test_dir, example_name).as_str(),
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

    // Mock prove (fast, but does not cover some potential issues)
    fn render_circuit(example_name: String) {
        let test_dir = TEST_DIR.path().to_str().unwrap();
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "render-circuit",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "-O",
                format!("{}/{}/render.png", test_dir, example_name).as_str(),
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn tutorial(tolerance: &str) {
        let test_dir = TEST_DIR.path().to_str().unwrap();

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-settings",
                "-M",
                format!("{}/tutorial/network.onnx", test_dir).as_str(),
                &format!("--settings-path={}/tutorial/settings.json", test_dir),
                "--bits=16",
                "--logrows=17",
                "--scale=4",
                &format!("--tolerance={}", tolerance),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                &format!("{}/tutorial/input.json", test_dir),
                "-M",
                &format!("{}/tutorial/network.onnx", test_dir),
                "-O",
                &format!("{}/tutorial/witness_tutorial.json", test_dir),
                &format!("--settings-path={}/tutorial/settings.json", test_dir),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock",
                "-W",
                format!("{}/tutorial/witness_tutorial.json", test_dir).as_str(),
                "-M",
                format!("{}/tutorial/network.onnx", test_dir).as_str(),
                &format!("--settings-path={}/tutorial/settings.json", test_dir),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_aggr_prove_and_verify(example_name: String) {
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
                "--bits=2",
                "-K=3",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "calibrate-settings",
                "--data",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
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

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                "-O",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", TEST_DIR.path().to_str().unwrap()),
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
                "prove",
                "-W",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", TEST_DIR.path().to_str().unwrap()),
                "--transcript=poseidon",
                "--strategy=accum",
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
                "aggregate",
                "--logrows=23",
                "--aggregation-snarks",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--proof-path",
                &format!("{}/{}/aggr.pf", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/aggr.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", TEST_DIR.path().to_str().unwrap()),
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
                &format!("{}/{}/aggr.pf", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/aggr.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", TEST_DIR.path().to_str().unwrap()),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_evm_aggr_prove_and_verify(example_name: String) {
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
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "calibrate-settings",
                "--data",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
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

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                "-O",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--vk-path",
                &format!("{}/{}/evm.vk", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/evm.pk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", test_dir),
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
                "prove",
                "-W",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--proof-path",
                &format!("{}/{}/evm.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/evm.pk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", test_dir),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
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
                "--aggregation-snarks",
                &format!("{}/{}/evm.pf", test_dir, example_name),
                "--proof-path",
                &format!("{}/{}/evm_aggr.pf", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/evm_aggr.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg23.srs", test_dir),
                "--transcript=evm",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let code_arg = format!("{}/{}/evm_aggr.code", test_dir, example_name);
        let param_arg = format!("--srs-path={}/kzg23.srs", test_dir);
        let vk_arg = format!("{}/{}/evm_aggr.vk", test_dir, example_name);

        fn build_args<'a>(
            base_args: Vec<&'a str>,
            sol_arg: &'a str,
            sol_bytecode_arg: &'a str,
        ) -> Vec<&'a str> {
            let mut args = base_args;

            args.push("--sol-code-path");
            args.push(sol_arg);
            args.push("--sol-bytecode-path");
            args.push(sol_bytecode_arg);

            args
        }

        let sol_arg = format!("{}/{}/kzg_aggr.sol", test_dir, example_name);
        let sol_bytecode_arg = format!("{}/{}/kzg_aggr.code", test_dir, example_name);
        let addr_path_arg = format!("--addr-path={}/{}/addr.txt", test_dir, example_name);
        let rpc_arg = format!("--rpc-url={}", *ANVIL_URL);

        let base_args = vec![
            "create-evm-verifier-aggr",
            "--deployment-code-path",
            code_arg.as_str(),
            param_arg.as_str(),
            "--vk-path",
            vk_arg.as_str(),
            "--optimizer-runs=1",
        ];

        let args = build_args(base_args, &sol_arg, &sol_bytecode_arg);

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // deploy the verifier
        let args = vec![
            "deploy-evm-verifier",
            rpc_arg.as_str(),
            addr_path_arg.as_str(),
            "--sol-code-path",
            sol_arg.as_str(),
        ];

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // read in the address
        let addr = std::fs::read_to_string(format!("{}/{}/addr.txt", test_dir, example_name))
            .expect("failed to read address file");

        let deployed_addr_arg = format!("--addr={}", addr);

        let pf_arg = format!("{}/{}/evm_aggr.pf", test_dir, example_name);

        let mut base_args = vec![
            "verify-evm",
            "--proof-path",
            pf_arg.as_str(),
            deployed_addr_arg.as_str(),
            rpc_arg.as_str(),
        ];

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&base_args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        // As sanity check, add example that should fail.
        base_args[2] = PF_FAILURE_AGGR;
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(base_args)
            .status()
            .expect("failed to execute process");
        assert!(!status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_prove_and_verify(
        example_name: String,
        logrows: usize,
        checkmode: &str,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
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
                &format!("--input-visibility={}", input_visibility),
                &format!("--param-visibility={}", param_visibility),
                &format!("--output-visibility={}", output_visibility),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "calibrate-settings",
                "--data",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
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

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                "-O",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
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
                "prove",
                "-W",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
                "--transcript=blake",
                "--strategy=single",
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                &format!("--check-mode={}", checkmode),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "verify",
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_fuzz(example_name: String, scale: usize, bits: usize, logrows: usize, transcript: &str) {
        let test_dir = TEST_DIR.path().to_str().unwrap();

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-settings",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "-O",
                format!("{}/{}/settings_fuzz.json", test_dir, example_name).as_str(),
                &format!("--scale={}", scale),
                "--batch-size=1",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--settings-path",
                format!("{}/{}/settings_fuzz.json", test_dir, example_name).as_str(),
                "-O",
                format!("{}/{}/witness_fuzz.json", test_dir, example_name).as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "fuzz",
                "-W",
                format!("{}/{}/witness_fuzz.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!("--bits={}", bits),
                &format!("--logrows={}", logrows),
                &format!("--scale={}", scale),
                &format!("--num-runs={}", 5),
                &format!("--transcript={}", transcript),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_evm_prove_and_verify(
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        num_runs: usize,
        logrows: usize,
    ) {
        let test_dir = TEST_DIR.path().to_str().unwrap();
        let anvil_url = ANVIL_URL.as_str();

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-settings",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                &format!("--input-visibility={}", input_visibility),
                &format!("--param-visibility={}", param_visibility),
                &format!("--output-visibility={}", output_visibility),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "calibrate-settings",
                "--data",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
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

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
                "-O",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
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
                "prove",
                "-W",
                format!("{}/{}/input.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
                "--transcript=evm",
                "--strategy=single",
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let circuit_settings = format!(
            "--settings-path={}/{}/settings.json",
            test_dir, example_name
        );
        let code_arg = format!("{}/{}/deployment.code", test_dir, example_name);
        let vk_arg = format!("{}/{}/key.vk", test_dir, example_name);
        let param_arg = format!("--srs-path={}/kzg{}.srs", test_dir, logrows);
        let opt_arg = format!("--optimizer-runs={}", num_runs);
        let rpc_arg = format!("--rpc-url={}", anvil_url);
        let addr_path_arg = format!("--addr-path={}/{}/addr.txt", test_dir, example_name);

        // create the verifier
        let mut args = vec![
            "create-evm-verifier",
            param_arg.as_str(),
            "--vk-path",
            vk_arg.as_str(),
            opt_arg.as_str(),
            circuit_settings.as_str(),
        ];

        let sol_arg = format!("{}/{}/kzg.sol", test_dir, example_name);
        let sol_bytecode_arg = format!("{}/{}/kzg.code", test_dir, example_name);

        // create everything to test the pipeline
        args.push("--sol-code-path");
        args.push(sol_arg.as_str());
        args.push("--sol-bytecode-path");
        args.push(sol_bytecode_arg.as_str());
        args.push("--deployment-code-path");
        args.push(code_arg.as_str());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // deploy the verifier
        let mut args = vec![
            "deploy-evm-verifier",
            rpc_arg.as_str(),
            addr_path_arg.as_str(),
        ];

        args.push("--sol-code-path");
        args.push(sol_arg.as_str());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // read in the address
        let addr = std::fs::read_to_string(format!("{}/{}/addr.txt", test_dir, example_name))
            .expect("failed to read address file");

        let deployed_addr_arg = format!("--addr={}", addr);

        // now verify the proof
        let pf_arg = format!("{}/{}/proof.pf", test_dir, example_name);
        let mut args = vec![
            "verify-evm",
            "--proof-path",
            pf_arg.as_str(),
            rpc_arg.as_str(),
            deployed_addr_arg.as_str(),
        ];

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        // As sanity check, add example that should fail.
        args[2] = PF_FAILURE;
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(!status.success());
    }

    fn kzg_evm_on_chain_input_prove_and_verify(
        example_name: String,
        num_runs: usize,
        input_source: &str,
        output_source: &str,
        logrows: usize,
    ) {
        let test_dir = TEST_DIR.path().to_str().unwrap();

        // set up the circuit
        let input_visbility = "public";
        let output_visbility = "public";
        let model_path = format!("{}/{}/network.onnx", test_dir, example_name);
        let circuit_settings = format!(
            "--settings-path={}/{}/settings.json",
            test_dir, example_name
        );

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-settings",
                "-M",
                model_path.as_str(),
                circuit_settings.as_str(),
                &format!("--input-visibility={}", input_visbility),
                &format!("--output-visibility={}", output_visbility),
                "--param-visibility=private",
                "--bits=16",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let data_path = format!("{}/{}/input.json", test_dir, example_name);
        let witness_path = format!("{}/{}/witness.json", test_dir, example_name);
        let test_on_chain_data_path = format!("{}/{}/on_chain_input.json", test_dir, example_name);
        let rpc_arg = format!("--rpc-url={}", ANVIL_URL.as_str());

        let test_input_source = format!("--input-source={}", input_source);
        let test_output_source = format!("--output-source={}", output_source);

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup-test-evm-data",
                "-D",
                data_path.as_str(),
                "-M",
                model_path.as_str(),
                circuit_settings.as_str(),
                "--test-data",
                test_on_chain_data_path.as_str(),
                rpc_arg.as_str(),
                test_input_source.as_str(),
                test_output_source.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                test_on_chain_data_path.as_str(),
                "-M",
                model_path.as_str(),
                circuit_settings.as_str(),
                "-O",
                witness_path.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                model_path.as_str(),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
                circuit_settings.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-W",
                witness_path.as_str(),
                "-M",
                model_path.as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                &format!("--srs-path={}/kzg{}.srs", test_dir, logrows),
                "--transcript=evm",
                "--strategy=single",
                circuit_settings.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let circuit_settings = format!(
            "--settings-path={}/{}/settings.json",
            test_dir, example_name
        );
        let vk_arg = format!("{}/{}/key.vk", test_dir, example_name);
        let param_arg = format!("--srs-path={}/kzg{}.srs", test_dir, logrows);

        let opt_arg = format!("--optimizer-runs={}", num_runs);

        let sol_arg = format!("{}/{}/kzg.sol", test_dir, example_name);
        let sol_bytecode_arg = format!("{}/{}/kzg.code", test_dir, example_name);

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "create-evm-da-verifier",
                circuit_settings.as_str(),
                "--sol-code-path",
                sol_arg.as_str(),
                "--sol-bytecode-path",
                sol_bytecode_arg.as_str(),
                param_arg.as_str(),
                "--vk-path",
                vk_arg.as_str(),
                "-D",
                test_on_chain_data_path.as_str(),
                opt_arg.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let addr_path_arg = format!("--addr-path={}/{}/addr.txt", test_dir, example_name);
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "deploy-evm-da-verifier",
                circuit_settings.as_str(),
                "-D",
                test_on_chain_data_path.as_str(),
                "--sol-code-path",
                sol_arg.as_str(),
                rpc_arg.as_str(),
                addr_path_arg.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let pf_arg = format!("{}/{}/proof.pf", test_dir, example_name);
        let uses_data_attestation = "--data-attestation".to_string();
        // read in the address
        let addr = std::fs::read_to_string(format!("{}/{}/addr.txt", test_dir, example_name))
            .expect("failed to read address file");

        let deployed_addr_arg = format!("--addr={}", addr);

        let mut args = vec![
            "verify-evm",
            "--proof-path",
            pf_arg.as_str(),
            deployed_addr_arg.as_str(),
            uses_data_attestation.as_str(),
            rpc_arg.as_str(),
        ];
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        // As sanity check, add example that should fail.
        args[2] = PF_FAILURE;
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(!status.success());
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
