#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod native_tests {

    use ezkl::circuit::Tolerance;
    use ezkl::fieldutils::{felt_to_integer_rep, integer_rep_to_felt, IntegerRep};
    // use ezkl::circuit::table::RESERVED_BLINDING_ROWS_PAD;
    use ezkl::graph::input::{FileSource, FileSourceInner, GraphData};
    use ezkl::graph::{DataSource, GraphSettings, GraphWitness};
    use ezkl::Commitments;
    use lazy_static::lazy_static;
    use rand::Rng;
    use std::env::var;
    use std::io::{Read, Write};
    use std::process::{Child, Command};
    use std::sync::Once;
    static COMPILE: Once = Once::new();
    #[allow(dead_code)]
    static COMPILE_WASM: Once = Once::new();
    static ENV_SETUP: Once = Once::new();

    //Sure to run this once
    #[derive(Debug)]
    #[allow(dead_code)]
    enum Hardfork {
        London,
        ArrowGlacier,
        GrayGlacier,
        Paris,
        Shanghai,
        Latest,
    }
    lazy_static! {
        static ref CARGO_TARGET_DIR: String =
            var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
        static ref ANVIL_URL: String = "http://localhost:3030".to_string();
        static ref LIMITLESS_ANVIL_URL: String = "http://localhost:8545".to_string();
        static ref ANVIL_DEFAULT_PRIVATE_KEY: String =
            "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80".to_string();
    }

    fn start_anvil(limitless: bool, hardfork: Hardfork) -> Child {
        let mut args = vec!["-p"];
        if limitless {
            args.push("8545");
            args.push("--code-size-limit=41943040");
            args.push("--disable-block-gas-limit");
        } else {
            args.push("3030");
        }
        match hardfork {
            Hardfork::Paris => args.push("--hardfork=paris"),
            Hardfork::London => args.push("--hardfork=london"),
            Hardfork::Latest => {}
            Hardfork::Shanghai => args.push("--hardfork=shanghai"),
            Hardfork::ArrowGlacier => args.push("--hardfork=arrowGlacier"),
            Hardfork::GrayGlacier => args.push("--hardfork=grayGlacier"),
        }
        let child = Command::new("anvil")
            .args(args)
            .spawn()
            .expect("failed to start anvil process");
        std::thread::sleep(std::time::Duration::from_secs(3));
        child
    }

    fn init_binary() {
        COMPILE.call_once(|| {
            println!("using cargo target dir: {}", *CARGO_TARGET_DIR);
            build_ezkl();
        });
    }

    ///
    #[allow(dead_code)]
    pub fn init_wasm() {
        COMPILE_WASM.call_once(|| {
            build_wasm_ezkl();
        });
    }

    fn setup_py_env() {
        ENV_SETUP.call_once(|| {
            // supposes that you have a virtualenv called .env and have run the following
            // equivalent of python -m venv .env
            // source .env/bin/activate
            // pip install -r requirements.txt
            // maturin develop --release --features python-bindings

            // now install torch, pandas, numpy, seaborn, jupyter
            let status = Command::new("pip")
                .args(["install", "numpy", "onnxruntime", "onnx"])
                .stdout(std::process::Stdio::null())
                .status()
                .expect("failed to execute process");

            assert!(status.success());
        });
    }

    fn download_srs(logrows: u32, commitment: Commitments) {
        // if does not exist, download it
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "get-srs",
                "--logrows",
                &format!("{}", logrows),
                "--commitment",
                &commitment.to_string(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    fn init_params(settings_path: std::path::PathBuf) {
        println!("using settings path: {}", settings_path.to_str().unwrap());
        // read in settings json
        let settings =
            std::fs::read_to_string(settings_path).expect("failed to read settings file");
        // read in to GraphSettings object
        let settings: GraphSettings = serde_json::from_str(&settings).unwrap();
        let logrows = settings.run_args.logrows;

        download_srs(logrows, settings.run_args.commitment.into());
    }

    fn mv_test_(test_dir: &str, test: &str) {
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

    fn mk_data_batches_(test_dir: &str, test: &str, output_dir: &str, num_batches: usize) {
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
                _ => panic!("Only File data sources support batching"),
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

    const PF_FAILURE: &str = "examples/test_failure_proof.json";

    const PF_FAILURE_AGGR: &str = "examples/test_failure_aggr_proof.json";

    const LARGE_TESTS: [&str; 6] = [
        "self_attention",
        "nanoGPT",
        "multihead_attention",
        "mobilenet",
        "mnist_gan",
        "smallworm",
    ];

    const ACCURACY_CAL_TESTS: [&str; 6] = [
        "accuracy",
        "1l_mlp",
        "4l_relu_conv_fc",
        "1l_elu",
        "1l_prelu",
        "1l_tiny_div",
    ];

    const TESTS: [&str; 94] = [
        "1l_mlp", //0
        "1l_slice",
        "1l_concat",
        "1l_flatten",
        // "1l_average",
        "1l_div",
        "1l_pad", // 5
        "1l_reshape",
        "1l_eltwise_div",
        "1l_sigmoid",
        "1l_sqrt",
        "1l_softmax", //10
        // "1l_instance_norm",
        "1l_batch_norm",
        "1l_prelu",
        "1l_leakyrelu",
        "1l_gelu_noappx",
        // "1l_gelu_tanh_appx",
        "1l_relu", //15
        "1l_downsample",
        "1l_tanh",
        "2l_relu_sigmoid_small",
        "2l_relu_fc",
        "2l_relu_small", //20
        "2l_relu_sigmoid",
        "1l_conv",
        "2l_sigmoid_small",
        "2l_relu_sigmoid_conv",
        "3l_relu_conv_fc", //25
        "4l_relu_conv_fc",
        "1l_erf",
        "1l_var",
        "1l_elu",
        "min", //30
        "max",
        "1l_max_pool",
        "1l_conv_transpose",
        "1l_upsample",
        "1l_identity", //35
        "idolmodel",   // too big evm
        "trig",        // too big evm
        "prelu_gmm",
        "lstm",
        "rnn", //40
        "quantize_dequantize",
        "1l_where",
        "boolean",
        "boolean_identity",
        "decision_tree", // 45
        "random_forest",
        "gradient_boosted_trees",
        "1l_topk",
        "xgboost",
        "lightgbm", //50
        "hummingbird_decision_tree",
        "oh_decision_tree",
        "linear_svc",
        "gather_elements",
        "less", //55
        "xgboost_reg",
        "1l_powf",
        "scatter_elements",
        "1l_linear",
        "linear_regression", //60
        "sklearn_mlp",
        "1l_mean",
        "rounding_ops",
        // "mean_as_constrain",
        "arange",
        "layernorm", //65
        "bitwise_ops",
        "blackman_window",
        "softsign", //68
        "softplus",
        "selu", //70
        "hard_sigmoid",
        "log_softmax",
        "eye",
        "ltsf",
        "remainder", //75
        "bitshift",
        "gather_nd",
        "scatter_nd",
        "celu",
        "gru",        // 80
        "hard_swish", // 81
        "hard_max",
        "tril",      // 83
        "triu",      // 84
        "logsumexp", // 85
        "clip",
        "mish",
        "reducel1",
        "reducel2", // 89
        "1l_lppool",
        "lstm_large",  // 91
        "lstm_medium", // 92
        "lenet_5",     // 93
    ];

    const WASM_TESTS: [&str; 46] = [
        "1l_mlp",
        "1l_slice",
        "1l_concat",
        "1l_flatten",
        // "1l_average",
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
        "1l_elu",
        "min",
        "max",
        "1l_max_pool",
        "1l_conv_transpose",
        "1l_upsample",
        "1l_identity",
        // "idolmodel",
        "trig",
        "prelu_gmm",
        "lstm",
        "rnn",
        "quantize_dequantize",
        "1l_where",
        "boolean",
        "boolean_identity",
        "gradient_boosted_trees",
        "1l_topk",
        // "xgboost",
        // "lightgbm",
        // "hummingbird_decision_tree",
    ];

    #[cfg(not(feature = "icicle"))]
    const TESTS_AGGR: [&str; 21] = [
        "1l_mlp",
        "1l_flatten",
        "1l_average",
        "1l_reshape",
        "1l_div",
        "1l_pad",
        "1l_sigmoid",
        "1l_gelu_noappx",
        "1l_sqrt",
        "1l_prelu",
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

    #[cfg(feature = "icicle")]
    const TESTS_AGGR: [&str; 3] = ["1l_mlp", "1l_flatten", "1l_average"];

    const TESTS_EVM: [&str; 23] = [
        "1l_mlp",
        "1l_flatten",
        "1l_average",
        "1l_reshape",
        "1l_sigmoid",
        "1l_div",
        "1l_sqrt",
        "1l_prelu",
        "1l_var",
        "1l_leakyrelu",
        "1l_gelu_noappx",
        "1l_relu",
        "1l_tanh",
        "2l_relu_sigmoid_small",
        "2l_relu_small",
        "min",
        "max",
        "1l_max_pool",
        "idolmodel",
        "1l_identity",
        "lstm",
        "rnn",
        "quantize_dequantize",
    ];

    const TESTS_EVM_AGGR: [&str; 18] = [
        "1l_mlp",
        "1l_reshape",
        "1l_sigmoid",
        "1l_div",
        "1l_sqrt",
        "1l_prelu",
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
        "idolmodel",
        "1l_identity",
    ];

    const EXAMPLES: [&str; 2] = ["mlp_4d_einsum", "conv2d_mnist"];

    macro_rules! test_func_aggr {
    () => {
        #[cfg(test)]
        mod tests_aggr {
            use seq_macro::seq;
            use crate::native_tests::TESTS_AGGR;
            use test_case::test_case;
            use crate::native_tests::aggr_prove_and_verify;
            use crate::native_tests::kzg_aggr_mock_prove_and_verify;
            use tempdir::TempDir;
            use ezkl::Commitments;

            #[cfg(not(feature="icicle"))]
            seq!(N in 0..=20 {

            #(#[test_case(TESTS_AGGR[N])])*
            fn kzg_aggr_mock_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                kzg_aggr_mock_prove_and_verify(path, test.to_string());
                test_dir.close().unwrap();
            }



            #(#[test_case(TESTS_AGGR[N])])*
            fn kzg_aggr_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                aggr_prove_and_verify(path, test.to_string(), "private", "private", "public", Commitments::KZG);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS_AGGR[N])])*
            fn ipa_aggr_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                aggr_prove_and_verify(path, test.to_string(), "private", "private", "public", Commitments::IPA);
                test_dir.close().unwrap();
            }

            });

            #[cfg(feature="icicle")]
            seq!(N in 0..=2 {
            #(#[test_case(TESTS_AGGR[N])])*
            fn aggr_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(test_dir.path().to_str().unwrap(), test);
                aggr_prove_and_verify(path, test.to_string(), "private", "private", "public", Commitments::KZG);
                test_dir.close().unwrap();
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
            use crate::native_tests::WASM_TESTS;
            use crate::native_tests::ACCURACY_CAL_TESTS;
            use crate::native_tests::LARGE_TESTS;
            use test_case::test_case;
            use crate::native_tests::mock;
            use crate::native_tests::accuracy_measurement;
            use crate::native_tests::prove_and_verify;
            use crate::native_tests::run_js_tests;
            use crate::native_tests::render_circuit;
            use crate::native_tests::model_serialization_different_binaries;
            use rand::Rng;
            use tempdir::TempDir;
            use ezkl::Commitments;

            #[test]
            fn model_serialization_different_binaries_() {
                let test = "1l_mlp";
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::native_tests::mv_test_(path, test);
                // percent tolerance test
                model_serialization_different_binaries(path, test.to_string());
                test_dir.close().unwrap();
            }

            seq!(N in 0..=5 {
            #(#[test_case(ACCURACY_CAL_TESTS[N])])*
            fn mock_accuracy_cal_tests(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "fixed", "public", 1, "accuracy", None, 0.0);
                test_dir.close().unwrap();
            }
        });

            seq!(N in 0..=93 {

            #(#[test_case(TESTS[N])])*
            #[ignore]
            fn render_circuit_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                render_circuit(path, test.to_string());
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn accuracy_measurement_div_rebase_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::setup_py_env();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                accuracy_measurement(path, test.to_string(), "private", "private", "public", 1, "accuracy", 2.6, true);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn accuracy_measurement_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::setup_py_env();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                accuracy_measurement(path, test.to_string(), "private", "private", "public", 1, "accuracy", 2.6, false);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn accuracy_measurement_fixed_params_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::setup_py_env();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                accuracy_measurement(path, test.to_string(), "private", "fixed", "private", 1, "accuracy", 2.6 , false);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn accuracy_measurement_public_inputs_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::setup_py_env();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                accuracy_measurement(path, test.to_string(), "public", "private", "private", 1, "accuracy", 2.6, false);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn resources_accuracy_measurement_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                crate::native_tests::setup_py_env();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                accuracy_measurement(path, test.to_string(), "private", "private", "public", 1, "resources", 3.1, false);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "private", "private", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_tolerance_public_outputs_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                // gen random number between 0.0 and 1.0
                let tolerance = rand::thread_rng().gen_range(0.0..1.0) * 100.0;
                mock(path, test.to_string(), "private", "private", "public", 1, "resources", None, tolerance);
                test_dir.close().unwrap();
            }



            #(#[test_case(TESTS[N])])*
            fn mock_large_batch_public_outputs_(test: &str) {
                // currently variable output rank is not supported in ONNX
                if test != "gather_nd" && test != "lstm_large"  && test != "lstm_medium" {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let large_batch_dir = &format!("large_batches_{}", test);
                    crate::native_tests::mk_data_batches_(path, test, &large_batch_dir, 10);
                    mock(path, large_batch_dir.to_string(), "private", "private", "public", 10, "resources", None, 0.0);
                    test_dir.close().unwrap();
                }
            }

            #(#[test_case(TESTS[N])])*
            fn mock_public_inputs_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "private", "private", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_params_public_inputs_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "hashed", "private", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_fixed_inputs_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "fixed", "private", "private", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_fixed_outputs_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "private", "private", "fixed", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_fixed_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "private", "fixed", "private", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_input_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "hashed", "private", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_kzg_input_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "polycommit", "private", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn mock_hashed_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "private", "hashed", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn mock_kzg_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "private", "polycommit", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_output_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "private", "hashed", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn mock_kzg_output_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "private", "polycommit", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_output_fixed_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "fixed", "hashed", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn mock_hashed_output_kzg_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "public", "polycommit", "hashed", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn mock_kzg_all_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "polycommit", "polycommit", "polycommit", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }


            #(#[test_case(TESTS[N])])*
            fn mock_hashed_input_output_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "hashed", "private", "hashed", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_input_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                // needs an extra row for the large model
                mock(path, test.to_string(),"hashed", "hashed", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn mock_hashed_all_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                // needs an extra row for the large model
                mock(path, test.to_string(),"hashed", "hashed", "hashed", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_single_col(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 1, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_triple_col(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 3, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_quadruple_col(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 4, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_octuple_col(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 8, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 1, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_tight_lookup_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 1, None, false, "single", Commitments::KZG, 1);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn ipa_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 1, None, false, "single", Commitments::IPA, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_public_input_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "public", "private", "public", 1, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_fixed_params_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "fixed", "public", 1, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_hashed_output(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "hashed", 1, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_kzg_output(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "polycommit", 1, None, false, "single", Commitments::KZG, 2);
               test_dir.close().unwrap();
            }

            #(#[test_case(TESTS[N])])*
            fn ipa_prove_and_verify_ipa_output(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
               prove_and_verify(path, test.to_string(), "safe", "private", "private", "polycommit", 1, None, false, "single", Commitments::IPA, 2);
               test_dir.close().unwrap();
            }

            });

            seq!(N in 0..=45 {

                #(#[test_case(WASM_TESTS[N])])*
                fn kzg_prove_and_verify_with_overflow_(test: &str) {
                    crate::native_tests::init_binary();
                    // crate::native_tests::init_wasm();
                    let test_dir = TempDir::new(test).unwrap();
                    env_logger::init();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    prove_and_verify(path, test.to_string(), "safe", "private", "private", "public", 1, None, true, "single", Commitments::KZG, 2);
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testWasm", false);
                    test_dir.close().unwrap();
                }

                #(#[test_case(WASM_TESTS[N])])*
                fn kzg_prove_and_verify_with_overflow_hashed_inputs_(test: &str) {
                    crate::native_tests::init_binary();
                    // crate::native_tests::init_wasm();
                    let test_dir = TempDir::new(test).unwrap();
                    env_logger::init();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    prove_and_verify(path, test.to_string(), "safe", "hashed", "private", "public", 1, None, true, "single", Commitments::KZG, 2);
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testWasm", false);
                    test_dir.close().unwrap();
                }

                #(#[test_case(WASM_TESTS[N])])*
                fn kzg_prove_and_verify_with_overflow_fixed_params_(test: &str) {
                    crate::native_tests::init_binary();
                    // crate::native_tests::init_wasm();
                    let test_dir = TempDir::new(test).unwrap();
                    env_logger::init();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    prove_and_verify(path, test.to_string(), "safe", "private", "fixed", "public", 1, None, true, "single", Commitments::KZG, 2);
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testWasm", false);
                    test_dir.close().unwrap();
                }

            });

            seq!(N in 0..=5 {

            #(#[test_case(LARGE_TESTS[N])])*
            #[ignore]
            fn large_prove_and_verify_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                prove_and_verify(path, test.to_string(), "unsafe", "private", "fixed", "public", 1, None, false, "single", Commitments::KZG, 2);
                test_dir.close().unwrap();
            }

            #(#[test_case(LARGE_TESTS[N])])*
            #[ignore]
            fn large_mock_(test: &str) {
                crate::native_tests::init_binary();
                let test_dir = TempDir::new(test).unwrap();
                let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                mock(path, test.to_string(), "private", "fixed", "public", 1, "resources", None, 0.0);
                test_dir.close().unwrap();
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
            use crate::native_tests::TESTS;
            use crate::native_tests::TESTS_EVM_AGGR;
            use test_case::test_case;
            use crate::native_tests::kzg_evm_prove_and_verify;
            use crate::native_tests::kzg_evm_prove_and_verify_reusable_verifier;

            use crate::native_tests::kzg_evm_on_chain_input_prove_and_verify;
            use crate::native_tests::kzg_evm_aggr_prove_and_verify;
            use tempdir::TempDir;
            use crate::native_tests::Hardfork;
            use crate::native_tests::run_js_tests;
            use ezkl::logger::init_logger;
            use crate::native_tests::lazy_static;

            // Global variables to store verifier hashes and identical verifiers
            lazy_static! {
                // create a new variable of type
                static ref REUSABLE_VERIFIER_ADDR: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
            }


            /// Currently only on chain inputs that return a non-negative value are supported.
            const TESTS_ON_CHAIN_INPUT: [&str; 17] = [
                "1l_mlp",
                "1l_average",
                "1l_reshape",
                "1l_sigmoid",
                "1l_div",
                "1l_sqrt",
                "1l_prelu",
                "1l_var",
                "1l_leakyrelu",
                "1l_gelu_noappx",
                "1l_relu",
                "1l_tanh",
                "2l_relu_sigmoid_small",
                "2l_relu_small",
                "2l_relu_fc",
                "min",
                "max"
            ];

            seq!(N in 0..=16 {
                #(#[test_case((TESTS_ON_CHAIN_INPUT[N],Hardfork::Latest))])*
                #(#[test_case((TESTS_ON_CHAIN_INPUT[N],Hardfork::Paris))])*
                #(#[test_case((TESTS_ON_CHAIN_INPUT[N],Hardfork::London))])*
                #(#[test_case((TESTS_ON_CHAIN_INPUT[N],Hardfork::Shanghai))])*
                fn kzg_evm_on_chain_input_prove_and_verify_(test: (&str,Hardfork)) {
                    let (test,hardfork) = test;
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, hardfork);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "on-chain", "file", "public", "private", "private");
                    // test_dir.close().unwrap();
                }

                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "file", "on-chain", "private", "public", "private");
                    // test_dir.close().unwrap();
                }

                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_input_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "on-chain", "on-chain", "public", "public", "private");
                    test_dir.close().unwrap();
                }

                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_input_output_hashed_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "on-chain", "on-chain", "hashed", "hashed", "private");
                    test_dir.close().unwrap();
                }
                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_input_kzg_output_kzg_params_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "on-chain", "file", "public", "polycommit", "polycommit");
                    test_dir.close().unwrap();
                }
                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_output_kzg_input_kzg_params_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "file", "on-chain", "polycommit", "public", "polycommit");
                    test_dir.close().unwrap();
                }
                #(#[test_case(TESTS_ON_CHAIN_INPUT[N])])*
                fn kzg_evm_on_chain_all_kzg_params_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    kzg_evm_on_chain_input_prove_and_verify(path, test.to_string(), "file", "file", "polycommit", "polycommit", "polycommit");
                    test_dir.close().unwrap();
                }
            });


            seq!(N in 0..=17 {
                // these take a particularly long time to run
                #(#[test_case(TESTS_EVM_AGGR[N])])*
                #[ignore]
                fn kzg_evm_aggr_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_aggr_prove_and_verify(path, test.to_string(), "private", "private", "public");
                    test_dir.close().unwrap();
                }

            });

            seq!(N in 0..=93 {
                #(#[test_case(TESTS[N])])*
                fn kzg_evm_prove_and_verify_reusable_verifier_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(true, Hardfork::Latest);
                    init_logger();
                    log::error!("Running kzg_evm_prove_and_verify_reusable_verifier_ for test: {}", test);
                    // default vis
                    let reusable_verifier_address: String = kzg_evm_prove_and_verify_reusable_verifier(2, path, test.to_string(), "private", "private", "public", &mut REUSABLE_VERIFIER_ADDR.lock().unwrap(), false);
                    // public/public vis
                    let reusable_verifier_address: String = kzg_evm_prove_and_verify_reusable_verifier(2, path, test.to_string(), "public", "private", "public", &mut Some(reusable_verifier_address), false);
                    // hashed input
                    let reusable_verifier_address: String = kzg_evm_prove_and_verify_reusable_verifier(2, path, test.to_string(), "hashed", "private", "public", &mut Some(reusable_verifier_address), false);

                    match REUSABLE_VERIFIER_ADDR.try_lock() {
                        Ok(mut addr) => {
                            *addr = Some(reusable_verifier_address.clone());
                            log::error!("Reusing the same verifeir deployed at address: {}", reusable_verifier_address);
                        }
                        Err(_) => {
                            log::error!("Failed to acquire lock on REUSABLE_VERIFIER_ADDR");
                        }
                    }

                    test_dir.close().unwrap();

                }

                #(#[test_case(TESTS[N])])*
                fn kzg_evm_prove_and_verify_reusable_verifier_with_overflow_(test: &str) {
                    // verifier too big to fit on chain
                    if test == "1l_eltwise_div" || test == "lenet_5" || test == "ltsf" {
                        return;
                    }
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    init_logger();
                    log::error!("Running kzg_evm_prove_and_verify_reusable_verifier_with_overflow_ for test: {}", test);
                    // default vis
                    let reusable_verifier_address: String = kzg_evm_prove_and_verify_reusable_verifier(2, path, test.to_string(), "private", "private", "public", &mut REUSABLE_VERIFIER_ADDR.lock().unwrap(), true);
                    // public/public vis
                    let reusable_verifier_address: String = kzg_evm_prove_and_verify_reusable_verifier(2, path, test.to_string(), "public", "private", "public", &mut Some(reusable_verifier_address), true);
                    // hashed input
                    let reusable_verifier_address: String = kzg_evm_prove_and_verify_reusable_verifier(2, path, test.to_string(), "hashed", "private", "public", &mut Some(reusable_verifier_address), true);

                    match REUSABLE_VERIFIER_ADDR.try_lock() {
                        Ok(mut addr) => {
                            *addr = Some(reusable_verifier_address.clone());
                            log::error!("Reusing the same verifeir deployed at address: {}", reusable_verifier_address);
                        }
                        Err(_) => {
                            log::error!("Failed to acquire lock on REUSABLE_VERIFIER_ADDR");
                        }
                    }

                    test_dir.close().unwrap();

                }
            });


            seq!(N in 0..=22 {

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "private", "private", "public");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();

                }


                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_hashed_input_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let mut _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "hashed", "private", "private");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();
                }


                #(#[test_case((TESTS_EVM[N], Hardfork::Latest))])*
                #(#[test_case((TESTS_EVM[N], Hardfork::Paris))])*
                #(#[test_case((TESTS_EVM[N], Hardfork::London))])*
                #(#[test_case((TESTS_EVM[N], Hardfork::Shanghai))])*
                fn kzg_evm_kzg_input_prove_and_verify_(test: (&str, Hardfork)) {
                    let (test,hardfork) = test;
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let mut _anvil_child = crate::native_tests::start_anvil(false, hardfork);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "polycommit", "private", "public");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();
                }


                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_hashed_params_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "private", "hashed", "public");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();

                }

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_hashed_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "private", "private", "hashed");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();
                }


                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_kzg_params_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "private", "polycommit", "public");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();
                }


                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_kzg_output_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "private", "private", "polycommit");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();
                }

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_kzg_all_prove_and_verify_(test: &str) {
                    crate::native_tests::init_binary();
                    let test_dir = TempDir::new(test).unwrap();
                    let path = test_dir.path().to_str().unwrap(); crate::native_tests::mv_test_(path, test);
                    let _anvil_child = crate::native_tests::start_anvil(false, Hardfork::Latest);
                    kzg_evm_prove_and_verify(2, path, test.to_string(), "polycommit", "polycommit", "polycommit");
                    #[cfg(not(feature = "icicle"))]
                    run_js_tests(path, test.to_string(), "testBrowserEvmVerify", false);
                    test_dir.close().unwrap();
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

    test_func!();
    test_func_aggr!();
    test_func_evm!();
    test_func_examples!();

    fn model_serialization_different_binaries(test_dir: &str, example_name: String) {
        let status = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "ezkl",
                "--",
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

        let status = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "ezkl",
                "--",
                "compile-circuit",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--compiled-circuit",
                format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // now alter binary slightly
        // create new temp cargo.toml with a different version
        // cpy old cargo.toml to cargo.toml.bak
        let status = Command::new("cp")
            .args(["Cargo.toml", "Cargo.toml.bak"])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let mut cargo_toml = std::fs::File::open("Cargo.toml").unwrap();
        let mut cargo_toml_contents = String::new();
        cargo_toml.read_to_string(&mut cargo_toml_contents).unwrap();
        let mut cargo_toml_contents = cargo_toml_contents.split('\n').collect::<Vec<_>>();

        // draw a random version number from 0.0.0 to 0.100.100
        let mut rng = rand::thread_rng();
        let version = &format!(
            "version = \"0.{}.{}-test\"",
            rng.gen_range(0..100),
            rng.gen_range(0..100)
        );
        let cargo_toml_contents = cargo_toml_contents
            .iter_mut()
            .map(|line| {
                if line.starts_with("version") {
                    *line = version;
                }
                *line
            })
            .collect::<Vec<_>>();
        let mut cargo_toml = std::fs::File::create("Cargo.toml").unwrap();
        cargo_toml
            .write_all(cargo_toml_contents.join("\n").as_bytes())
            .unwrap();

        let status = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "ezkl",
                "--",
                "gen-witness",
                "-D",
                &format!("{}/{}/input.json", test_dir, example_name),
                "-M",
                &format!("{}/{}/network.compiled", test_dir, example_name),
                "-O",
                &format!("{}/{}/witness.json", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // now delete cargo.toml and move cargo.toml.bak to cargo.toml
        let status = Command::new("rm")
            .args(["Cargo.toml"])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new("mv")
            .args(["Cargo.toml.bak", "Cargo.toml"])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
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
    #[allow(clippy::too_many_arguments)]
    fn mock(
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        batch_size: usize,
        cal_target: &str,
        scales_to_use: Option<Vec<u32>>,
        tolerance: f32,
    ) {
        let mut tolerance = tolerance;
        gen_circuit_settings_and_witness(
            test_dir,
            example_name.clone(),
            input_visibility,
            param_visibility,
            output_visibility,
            batch_size,
            cal_target,
            scales_to_use,
            2,
            false,
            &mut tolerance,
            Commitments::KZG,
            2,
        );

        if tolerance > 0.0 {
            // load witness and shift the output by a small amount that is less than tolerance percent
            let witness = GraphWitness::from_path(
                format!("{}/{}/witness.json", test_dir, example_name).into(),
            )
            .unwrap();
            let witness = witness.clone();
            let outputs = witness.outputs.clone();

            // get values as i64
            let output_perturbed_safe: Vec<Vec<halo2curves::bn256::Fr>> = outputs
                .iter()
                .map(|sv| {
                    sv.iter()
                        .map(|v| {
                            // randomly perturb by a small amount less than tolerance
                            let perturbation = if v == &halo2curves::bn256::Fr::zero() {
                                halo2curves::bn256::Fr::zero()
                            } else {
                                integer_rep_to_felt(
                                    (felt_to_integer_rep(*v) as f32
                                        * (rand::thread_rng().gen_range(-0.01..0.01) * tolerance))
                                        as IntegerRep,
                                )
                            };

                            *v + perturbation
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            // get values as i64
            let output_perturbed_bad: Vec<Vec<halo2curves::bn256::Fr>> = outputs
                .iter()
                .map(|sv| {
                    sv.iter()
                        .map(|v| {
                            // randomly perturb by a small amount less than tolerance
                            let perturbation = if v == &halo2curves::bn256::Fr::zero() {
                                halo2curves::bn256::Fr::from(2)
                            } else {
                                integer_rep_to_felt(
                                    (felt_to_integer_rep(*v) as f32
                                        * (rand::thread_rng().gen_range(0.02..0.1) * tolerance))
                                        as IntegerRep,
                                )
                            };
                            *v + perturbation
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let good_witness = GraphWitness {
                outputs: output_perturbed_safe,
                ..witness.clone()
            };

            // save
            good_witness
                .save(format!("{}/{}/witness_ok.json", test_dir, example_name).into())
                .unwrap();

            let bad_witness = GraphWitness {
                outputs: output_perturbed_bad,
                ..witness.clone()
            };

            // save
            bad_witness
                .save(format!("{}/{}/witness_bad.json", test_dir, example_name).into())
                .unwrap();

            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "mock",
                    "-W",
                    format!("{}/{}/witness.json", test_dir, example_name).as_str(),
                    "-M",
                    format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());

            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "mock",
                    "-W",
                    format!("{}/{}/witness_ok.json", test_dir, example_name).as_str(),
                    "-M",
                    format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());

            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "mock",
                    "-W",
                    format!("{}/{}/witness_bad.json", test_dir, example_name).as_str(),
                    "-M",
                    format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                ])
                .status()
                .expect("failed to execute process");
            assert!(!status.success());
        } else {
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "mock",
                    "-W",
                    format!("{}/{}/witness.json", test_dir, example_name).as_str(),
                    "-M",
                    format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gen_circuit_settings_and_witness(
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        batch_size: usize,
        cal_target: &str,
        scales_to_use: Option<Vec<u32>>,
        num_inner_columns: usize,
        div_rebasing: bool,
        tolerance: &mut f32,
        commitment: Commitments,
        lookup_safety_margin: usize,
    ) {
        let mut args = vec![
            "gen-settings".to_string(),
            "-M".to_string(),
            format!("{}/{}/network.onnx", test_dir, example_name),
            format!(
                "--settings-path={}/{}/settings.json",
                test_dir, example_name
            ),
            format!("--variables=batch_size->{}", batch_size),
            format!("--input-visibility={}", input_visibility),
            format!("--param-visibility={}", param_visibility),
            format!("--output-visibility={}", output_visibility),
            format!("--num-inner-cols={}", num_inner_columns),
            format!("--tolerance={}", tolerance),
            format!("--commitment={}", commitment),
        ];

        if div_rebasing {
            args.push("--div-rebasing".to_string());
        };

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .stdout(std::process::Stdio::null())
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let mut calibrate_args = vec![
            "calibrate-settings".to_string(),
            "--data".to_string(),
            format!("{}/{}/input.json", test_dir, example_name),
            "-M".to_string(),
            format!("{}/{}/network.onnx", test_dir, example_name),
            format!(
                "--settings-path={}/{}/settings.json",
                test_dir, example_name
            ),
            format!("--target={}", cal_target),
            format!("--lookup-safety-margin={}", lookup_safety_margin),
        ];

        if let Some(scales) = scales_to_use {
            let scales = scales
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            calibrate_args.push("--scales".to_string());
            calibrate_args.push(scales);
        }

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(calibrate_args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let mut settings =
            GraphSettings::load(&format!("{}/{}/settings.json", test_dir, example_name).into())
                .unwrap();

        let any_output_scales_smol = settings.model_output_scales.iter().any(|s| *s <= 0);

        if any_output_scales_smol {
            // set the tolerance to 0.0
            settings.run_args.tolerance = Tolerance {
                val: 0.0,
                scale: 0.0.into(),
            };
            settings
                .save(&format!("{}/{}/settings.json", test_dir, example_name).into())
                .unwrap();
            *tolerance = 0.0;
        }

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "compile-circuit",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "--compiled-circuit",
                format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                &format!(
                    "--settings-path={}/{}/settings.json",
                    test_dir, example_name
                ),
            ])
            .stdout(std::process::Stdio::null())
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                &format!("{}/{}/input.json", test_dir, example_name),
                "-M",
                &format!("{}/{}/network.compiled", test_dir, example_name),
                "-O",
                &format!("{}/{}/witness.json", test_dir, example_name),
            ])
            .stdout(std::process::Stdio::null())
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    #[allow(clippy::too_many_arguments)]
    fn accuracy_measurement(
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        batch_size: usize,
        cal_target: &str,
        target_perc: f32,
        div_rebasing: bool,
    ) {
        gen_circuit_settings_and_witness(
            test_dir,
            example_name.clone(),
            input_visibility,
            param_visibility,
            output_visibility,
            batch_size,
            cal_target,
            None,
            2,
            div_rebasing,
            &mut 0.0,
            Commitments::KZG,
            2,
        );

        println!(
            " ------------ running accuracy measurement for {}",
            example_name
        );
        // run python ./output_comparison.py in the test dir
        let status = Command::new("python")
            .args([
                "tests/output_comparison.py",
                &format!("{}/{}/network.onnx", test_dir, example_name),
                &format!("{}/{}/input.json", test_dir, example_name),
                &format!("{}/{}/witness.json", test_dir, example_name),
                &format!("{}/{}/settings.json", test_dir, example_name),
                &format!("{}", target_perc),
            ])
            .status()
            .expect("failed to execute process");

        assert!(status.success());
    }

    // Mock prove (fast, but does not cover some potential issues)
    fn render_circuit(test_dir: &str, example_name: String) {
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "render-circuit",
                "-M",
                format!("{}/{}/network.onnx", test_dir, example_name).as_str(),
                "-O",
                format!("{}/{}/render.png", test_dir, example_name).as_str(),
                "--lookup-range=-32768->32768",
                "-K=17",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_aggr_mock_prove_and_verify(test_dir: &str, example_name: String) {
        prove_and_verify(
            test_dir,
            example_name.clone(),
            "safe",
            "private",
            "private",
            "public",
            2,
            None,
            false,
            "for-aggr",
            Commitments::KZG,
            2,
        );
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "mock-aggregate",
                "--logrows=23",
                "--aggregation-snarks",
                &format!("{}/{}/proof.pf", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn aggr_prove_and_verify(
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        commitment: Commitments,
    ) {
        prove_and_verify(
            test_dir,
            example_name.clone(),
            "safe",
            input_visibility,
            param_visibility,
            output_visibility,
            2,
            None,
            false,
            "for-aggr",
            Commitments::KZG,
            2,
        );

        download_srs(23, commitment);
        // now setup-aggregate
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup-aggregate",
                "--sample-snarks",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--logrows=23",
                "--vk-path",
                &format!("{}/{}/aggr.vk", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/aggr.pk", test_dir, example_name),
                &format!("--commitment={}", commitment),
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
                "--pk-path",
                &format!("{}/{}/aggr.pk", test_dir, example_name),
                &format!("--commitment={}", commitment),
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
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_evm_aggr_prove_and_verify(
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
    ) {
        aggr_prove_and_verify(
            test_dir,
            example_name.clone(),
            input_visibility,
            param_visibility,
            output_visibility,
            Commitments::KZG,
        );

        download_srs(23, Commitments::KZG);

        let vk_arg = &format!("{}/{}/aggr.vk", test_dir, example_name);

        fn build_args<'a>(base_args: Vec<&'a str>, sol_arg: &'a str) -> Vec<&'a str> {
            let mut args = base_args;

            args.push("--sol-code-path");
            args.push(sol_arg);
            args
        }

        let sol_arg = format!("{}/{}/kzg_aggr.sol", test_dir, example_name);
        let addr_path_arg = format!("--addr-path={}/{}/addr.txt", test_dir, example_name);
        let rpc_arg = format!("--rpc-url={}", *ANVIL_URL);
        let settings_arg = format!("{}/{}/settings.json", test_dir, example_name);
        let private_key = format!("--private-key={}", *ANVIL_DEFAULT_PRIVATE_KEY);

        // create encoded calldata
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "encode-evm-calldata",
                "--proof-path",
                &format!("{}/{}/aggr.pf", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");

        assert!(status.success());

        let base_args = vec![
            "create-evm-verifier-aggr",
            "--vk-path",
            vk_arg.as_str(),
            "--aggregation-settings",
            settings_arg.as_str(),
            "--logrows=23",
        ];

        let args = build_args(base_args, &sol_arg);

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // deploy the verifier
        let args = vec![
            "deploy-evm",
            rpc_arg.as_str(),
            addr_path_arg.as_str(),
            "--sol-code-path",
            sol_arg.as_str(),
            private_key.as_str(),
        ];

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // read in the address
        let addr = std::fs::read_to_string(format!("{}/{}/addr.txt", test_dir, example_name))
            .expect("failed to read address file");

        let deployed_addr_arg = format!("--addr-verifier={}", addr);

        let pf_arg = format!("{}/{}/aggr.pf", test_dir, example_name);

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
    #[allow(clippy::too_many_arguments)]
    fn prove_and_verify(
        test_dir: &str,
        example_name: String,
        checkmode: &str,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        num_inner_columns: usize,
        scales_to_use: Option<Vec<u32>>,
        overflow: bool,
        proof_type: &str,
        commitment: Commitments,
        lookup_safety_margin: usize,
    ) {
        let target_str = if overflow {
            "resources/col-overflow"
        } else {
            "resources"
        };

        gen_circuit_settings_and_witness(
            test_dir,
            example_name.clone(),
            input_visibility,
            param_visibility,
            output_visibility,
            1,
            target_str,
            scales_to_use,
            num_inner_columns,
            false,
            &mut 0.0,
            commitment,
            lookup_safety_margin,
        );

        let settings_path = format!("{}/{}/settings.json", test_dir, example_name);

        init_params(settings_path.clone().into());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                &format!("{}/{}/network.compiled", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                "--disable-selector-compression",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-W",
                format!("{}/{}/witness.json", test_dir, example_name).as_str(),
                "-M",
                format!("{}/{}/network.compiled", test_dir, example_name).as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                &format!("--check-mode={}", checkmode),
                &format!("--proof-type={}", proof_type),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "swap-proof-commitments",
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--witness-path",
                format!("{}/{}/witness.json", test_dir, example_name).as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "verify",
                format!("--settings-path={}", settings_path).as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // load settings file
        let settings =
            std::fs::read_to_string(settings_path.clone()).expect("failed to read settings file");

        let graph_settings = serde_json::from_str::<GraphSettings>(&settings)
            .expect("failed to parse settings file");

        // get_srs for the graph_settings_num_instances
        download_srs(1, graph_settings.run_args.commitment.into());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "verify",
                format!("--settings-path={}", settings_path).as_str(),
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
                "--reduced-srs",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    // prove-serialize-verify, the usual full path
    fn kzg_evm_prove_and_verify(
        num_inner_columns: usize,
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
    ) {
        let anvil_url = ANVIL_URL.as_str();

        prove_and_verify(
            test_dir,
            example_name.clone(),
            "safe",
            input_visibility,
            param_visibility,
            output_visibility,
            num_inner_columns,
            None,
            false,
            "single",
            Commitments::KZG,
            2,
        );

        let settings_path = format!("{}/{}/settings.json", test_dir, example_name);
        init_params(settings_path.clone().into());

        let vk_arg = format!("{}/{}/key.vk", test_dir, example_name);
        let rpc_arg = format!("--rpc-url={}", anvil_url);
        let addr_path_arg = format!("--addr-path={}/{}/addr.txt", test_dir, example_name);
        let settings_arg = format!("--settings-path={}", settings_path);

        // create encoded calldata
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "encode-evm-calldata",
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");

        assert!(status.success());

        // create the verifier
        let mut args = vec!["create-evm-verifier", "--vk-path", &vk_arg, &settings_arg];

        let sol_arg = format!("{}/{}/kzg.sol", test_dir, example_name);

        // create everything to test the pipeline
        args.push("--sol-code-path");
        args.push(sol_arg.as_str());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // deploy the verifier
        let mut args = vec!["deploy-evm", rpc_arg.as_str(), addr_path_arg.as_str()];

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

        let deployed_addr_arg = format!("--addr-verifier={}", addr);

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

    // prove-serialize-verify, the usual full path
    fn kzg_evm_prove_and_verify_reusable_verifier(
        num_inner_columns: usize,
        test_dir: &str,
        example_name: String,
        input_visibility: &str,
        param_visibility: &str,
        output_visibility: &str,
        reusable_verifier_address: &mut Option<String>,
        overflow: bool,
    ) -> String {
        let anvil_url = ANVIL_URL.as_str();

        prove_and_verify(
            test_dir,
            example_name.clone(),
            "safe",
            input_visibility,
            param_visibility,
            output_visibility,
            num_inner_columns,
            None,
            overflow,
            "single",
            Commitments::KZG,
            2,
        );

        let settings_path = format!("{}/{}/settings.json", test_dir, example_name);
        init_params(settings_path.clone().into());

        let vk_arg = format!("{}/{}/key.vk", test_dir, example_name);
        let rpc_arg = format!("--rpc-url={}", anvil_url);
        let addr_path_arg = format!("--addr-path={}/{}/addr.txt", test_dir, example_name);
        let settings_arg = format!("--settings-path={}", settings_path);
        let sol_arg = format!("--sol-code-path={}/{}/kzg.sol", test_dir, example_name);

        // if the reusable verifier address is not set, create the verifier
        let deployed_addr_arg = match reusable_verifier_address {
            Some(addr) => addr.clone(),
            None => {
                // create the reusable verifier
                let args = vec![
                    "create-evm-verifier",
                    "--vk-path",
                    &vk_arg,
                    &settings_arg,
                    &sol_arg,
                    "--reusable",
                ];

                let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                    .args(&args)
                    .status()
                    .expect("failed to execute process");
                assert!(status.success());

                // deploy the verifier
                let args = vec![
                    "deploy-evm",
                    rpc_arg.as_str(),
                    addr_path_arg.as_str(),
                    sol_arg.as_str(),
                    "-C=verifier/reusable",
                ];

                let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                    .args(&args)
                    .status()
                    .expect("failed to execute process");
                assert!(status.success());

                // read in the address
                let addr =
                    std::fs::read_to_string(format!("{}/{}/addr.txt", test_dir, example_name))
                        .expect("failed to read address file");

                let deployed_addr_arg = format!("--addr-verifier={}", addr);
                // set the reusable verifier address
                *reusable_verifier_address = Some(addr);
                deployed_addr_arg
            }
        };

        let addr_path_arg_vk = format!("--addr-path={}/{}/addr_vk.txt", test_dir, example_name);
        let sol_arg_vk: String = format!("--sol-code-path={}/{}/vk.sol", test_dir, example_name);
        // create the verifier
        let args = vec![
            "create-evm-vka",
            "--vk-path",
            &vk_arg,
            &settings_arg,
            &sol_arg_vk,
        ];

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // deploy the vka
        let args = vec![
            "deploy-evm",
            rpc_arg.as_str(),
            addr_path_arg_vk.as_str(),
            sol_arg_vk.as_str(),
            "-C=vka",
        ];

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // read in the address
        let addr_vk = std::fs::read_to_string(format!("{}/{}/addr_vk.txt", test_dir, example_name))
            .expect("failed to read address file");

        let deployed_addr_arg_vk = format!("--addr-vk={}", addr_vk);

        // create encoded calldata
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "encode-evm-calldata",
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                &deployed_addr_arg_vk,
            ])
            .status()
            .expect("failed to execute process");

        assert!(status.success());

        // now verify the proof
        let pf_arg = format!("{}/{}/proof.pf", test_dir, example_name);
        let mut args = vec![
            "verify-evm",
            "--proof-path",
            pf_arg.as_str(),
            rpc_arg.as_str(),
            deployed_addr_arg.as_str(),
            deployed_addr_arg_vk.as_str(),
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
        // Returned deploy_addr_arg for reusable verifier
        deployed_addr_arg
    }

    // run js browser evm verify tests for a given example
    fn run_js_tests(test_dir: &str, example_name: String, js_test: &str, vk: bool) {
        let example = format!("--example={}", example_name);
        let dir = format!("--dir={}", test_dir);
        let mut args = vec!["run", "test", js_test, &example, &dir];
        let vk_string: String;
        if vk {
            vk_string = format!("--vk={}", vk);
            args.push(&vk_string);
        };
        let status = Command::new("pnpm")
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    fn kzg_evm_on_chain_input_prove_and_verify(
        test_dir: &str,
        example_name: String,
        input_source: &str,
        output_source: &str,
        input_visibility: &str,
        output_visibility: &str,
        param_visibility: &str,
    ) {
        gen_circuit_settings_and_witness(
            test_dir,
            example_name.clone(),
            input_visibility,
            param_visibility,
            output_visibility,
            1,
            "resources",
            // we need the accuracy
            Some(vec![4]),
            1,
            false,
            &mut 0.0,
            Commitments::KZG,
            2,
        );

        let model_path = format!("{}/{}/network.compiled", test_dir, example_name);
        let settings_path = format!("{}/{}/settings.json", test_dir, example_name);
        init_params(settings_path.clone().into());

        let data_path = format!("{}/{}/input.json", test_dir, example_name);
        let witness_path = format!("{}/{}/witness.json", test_dir, example_name);
        let test_on_chain_data_path = format!("{}/{}/on_chain_input.json", test_dir, example_name);
        let rpc_arg = format!("--rpc-url={}", LIMITLESS_ANVIL_URL.as_str());
        let private_key = format!("--private-key={}", *ANVIL_DEFAULT_PRIVATE_KEY);

        let test_input_source = format!("--input-source={}", input_source);
        let test_output_source = format!("--output-source={}", output_source);

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "setup",
                "-M",
                &model_path,
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // generate the witness, passing the vk path to generate the necessary kzg commits
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "gen-witness",
                "-D",
                &data_path,
                "-M",
                &model_path,
                "-O",
                &witness_path,
                "--vk-path",
                &format!("{}/{}/key.vk", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        // load witness
        let witness: GraphWitness = GraphWitness::from_path(witness_path.clone().into()).unwrap();
        // print out the witness
        println!("WITNESS: {:?}", witness);
        let mut input: GraphData = GraphData::from_path(data_path.clone().into()).unwrap();
        if input_source != "file" || output_source != "file" {
            println!("on chain input");
            if input_visibility == "hashed" {
                let hashes = witness.processed_inputs.unwrap().poseidon_hash.unwrap();
                input.input_data = DataSource::File(
                    hashes
                        .iter()
                        .map(|h| vec![FileSourceInner::Field(*h)])
                        .collect(),
                );
            }
            if output_visibility == "hashed" {
                let hashes = witness.processed_outputs.unwrap().poseidon_hash.unwrap();
                input.output_data = Some(DataSource::File(
                    hashes
                        .iter()
                        .map(|h| vec![FileSourceInner::Field(*h)])
                        .collect(),
                ));
            } else {
                input.output_data = Some(DataSource::File(
                    witness
                        .pretty_elements
                        .unwrap()
                        .rescaled_outputs
                        .iter()
                        .map(|o| {
                            o.iter()
                                .map(|f| FileSourceInner::Float(f.parse().unwrap()))
                                .collect()
                        })
                        .collect(),
                ));
            }
            input.save(data_path.clone().into()).unwrap();

            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "setup-test-evm-data",
                    "-D",
                    data_path.as_str(),
                    "-M",
                    &model_path,
                    "--test-data",
                    test_on_chain_data_path.as_str(),
                    rpc_arg.as_str(),
                    test_input_source.as_str(),
                    test_output_source.as_str(),
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        }

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "prove",
                "-W",
                &witness_path,
                "-M",
                &model_path,
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
                "--pk-path",
                &format!("{}/{}/key.pk", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let vk_arg = format!("{}/{}/key.vk", test_dir, example_name);

        let settings_arg = format!("--settings-path={}", settings_path);

        // create encoded calldata
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "encode-evm-calldata",
                "--proof-path",
                &format!("{}/{}/proof.pf", test_dir, example_name),
            ])
            .status()
            .expect("failed to execute process");

        assert!(status.success());

        // create the verifier
        let mut args = vec!["create-evm-verifier", "--vk-path", &vk_arg, &settings_arg];

        let sol_arg = format!("{}/{}/kzg.sol", test_dir, example_name);

        args.push("--sol-code-path");
        args.push(sol_arg.as_str());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let addr_path_verifier_arg = format!(
            "--addr-path={}/{}/addr_verifier.txt",
            test_dir, example_name
        );

        // deploy the verifier
        let mut args = vec![
            "deploy-evm",
            rpc_arg.as_str(),
            addr_path_verifier_arg.as_str(),
        ];

        args.push("--sol-code-path");
        args.push(sol_arg.as_str());

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let sol_arg = format!("{}/{}/kzg.sol", test_dir, example_name);

        let mut create_da_args = vec![
            "create-evm-da",
            &settings_arg,
            "--sol-code-path",
            sol_arg.as_str(),
            "-W",
            &witness_path,
        ];

        // if there is a on-chain source we add the data
        if input_source != "file" || output_source != "file" {
            create_da_args.push("-D");
            create_da_args.push(test_on_chain_data_path.as_str());
        }

        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&create_da_args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let deploy_evm_data_path = if input_source != "file" || output_source != "file" {
            test_on_chain_data_path.clone()
        } else {
            data_path.clone()
        };

        let addr_path_da_arg = format!("--addr-path={}/{}/addr_da.txt", test_dir, example_name);
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args([
                "deploy-evm-da",
                format!("--settings-path={}", settings_path).as_str(),
                "-D",
                deploy_evm_data_path.as_str(),
                "--sol-code-path",
                sol_arg.as_str(),
                rpc_arg.as_str(),
                addr_path_da_arg.as_str(),
                private_key.as_str(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let pf_arg = format!("{}/{}/proof.pf", test_dir, example_name);
        // read in the verifier address
        let addr_verifier =
            std::fs::read_to_string(format!("{}/{}/addr_verifier.txt", test_dir, example_name))
                .expect("failed to read address file");

        let deployed_addr_verifier_arg = format!("--addr-verifier={}", addr_verifier);

        // read in the da address
        let addr_da = std::fs::read_to_string(format!("{}/{}/addr_da.txt", test_dir, example_name))
            .expect("failed to read address file");

        let deployed_addr_da_arg = format!("--addr-da={}", addr_da);

        let args = vec![
            "verify-evm",
            "--proof-path",
            pf_arg.as_str(),
            deployed_addr_verifier_arg.as_str(),
            deployed_addr_da_arg.as_str(),
            rpc_arg.as_str(),
        ];
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(&args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        // Create a new set of test on chain data only for the on-chain input source
        if input_source != "file" || output_source != "file" {
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args([
                    "setup-test-evm-data",
                    "-D",
                    data_path.as_str(),
                    "-M",
                    &model_path,
                    "--test-data",
                    test_on_chain_data_path.as_str(),
                    rpc_arg.as_str(),
                    test_input_source.as_str(),
                    test_output_source.as_str(),
                ])
                .status()
                .expect("failed to execute process");

            assert!(status.success());

            let deployed_addr_arg = format!("--addr={}", addr_da);

            let args: Vec<&str> = vec![
                "test-update-account-calls",
                deployed_addr_arg.as_str(),
                "-D",
                test_on_chain_data_path.as_str(),
                rpc_arg.as_str(),
            ];
            let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
                .args(&args)
                .status()
                .expect("failed to execute process");

            assert!(status.success());
        }
        // As sanity check, add example that should fail.
        let args = vec![
            "verify-evm",
            "--proof-path",
            PF_FAILURE,
            deployed_addr_verifier_arg.as_str(),
            deployed_addr_da_arg.as_str(),
            rpc_arg.as_str(),
        ];
        let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(!status.success());
    }

    fn build_ezkl() {
        #[cfg(feature = "icicle")]
        let args = [
            "build",
            "--release",
            "--bin",
            "ezkl",
            "--features",
            "icicle",
        ];
        #[cfg(not(feature = "icicle"))]
        let args = ["build", "--release", "--bin", "ezkl"];
        #[cfg(not(feature = "mv-lookup"))]
        let args = [
            "build",
            "--release",
            "--bin",
            "ezkl",
            "--no-default-features",
            "--features",
            "ezkl",
        ];

        let status = Command::new("cargo")
            .args(args)
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    #[allow(dead_code)]
    fn build_wasm_ezkl() {
        // wasm-pack build --target nodejs --out-dir ./tests/wasm/nodejs . -- -Z build-std="panic_abort,std"
        let status = Command::new("wasm-pack")
            .args([
                "build",
                "--release",
                "--target",
                "nodejs",
                "--out-dir",
                "./tests/wasm/nodejs",
                ".",
                "--",
                "-Z",
                "build-std=panic_abort,std",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
        // fix the memory size
        //   sed -i "3s|.*|imports['env'] = {memory: new WebAssembly.Memory({initial:20,maximum:65536,shared:true})}|" tests/wasm/nodejs/ezkl.js
        let status = Command::new("sed")
            .args([
                "-i",
                // is required on macos
                // "\".js\"",
                "3s|.*|imports['env'] = {memory: new WebAssembly.Memory({initial:20,maximum:65536,shared:true})}|",
                "./tests/wasm/nodejs/ezkl.js",
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }
}
