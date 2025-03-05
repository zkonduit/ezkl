#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
#[cfg(test)]
mod py_tests {

    use lazy_static::lazy_static;
    use std::env::var;
    use std::process::{Child, Command};
    use std::sync::Once;
    use tempdir::TempDir;
    static COMPILE: Once = Once::new();
    static ENV_SETUP: Once = Once::new();
    static DOWNLOAD_VOICE_DATA: Once = Once::new();

    // Sure to run this once

    lazy_static! {
        static ref CARGO_TARGET_DIR: String =
            var("CARGO_TARGET_DIR").unwrap_or_else(|_| "./target".to_string());
        static ref ANVIL_URL: String = "http://localhost:3030".to_string();
    }

    fn start_anvil(limitless: bool) -> Child {
        let mut args = vec!["-p", "3030"];
        if limitless {
            args.push("--code-size-limit=41943040");
            args.push("--disable-block-gas-limit");
        }
        let child = Command::new("anvil")
            .args(args)
            // .stdout(Stdio::piped())
            .spawn()
            .expect("failed to start anvil process");

        std::thread::sleep(std::time::Duration::from_secs(3));
        child
    }

    fn download_voice_data() {
        let voice_data_dir = shellexpand::tilde("~/data/voice_data");

        DOWNLOAD_VOICE_DATA.call_once(|| {
            let status = Command::new("bash")
                .args(["examples/notebooks/voice_data.sh", &voice_data_dir])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        });
        // set VOICE_DATA_DIR environment variable
        unsafe {
            std::env::set_var("VOICE_DATA_DIR", format!("{}", voice_data_dir));
        }
    }

    fn download_catdog_data() {
        let cat_and_dog_data_dir = shellexpand::tilde("~/data/catdog_data");

        DOWNLOAD_VOICE_DATA.call_once(|| {
            let status = Command::new("bash")
                .args([
                    "examples/notebooks/cat_and_dog_data.sh",
                    &cat_and_dog_data_dir,
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        });
        // set VOICE_DATA_DIR environment variable
        unsafe {
            std::env::set_var("CATDOG_DATA_DIR", format!("{}", cat_and_dog_data_dir));
        }
    }

    fn setup_py_env() {
        ENV_SETUP.call_once(|| {
            // supposes that you have a virtualenv called .env and have run the following
            // equivalent of python -m venv .env
            // source .env/bin/activate
            // pip install -r requirements.txt
            // maturin develop --release --features python-bindings
            // first install tf2onnx as it has protobuf conflict with onnx
            let status = Command::new("pip")
                .args(["install", "tf2onnx==1.16.1"])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
            // now install torch, pandas, numpy, seaborn, jupyter
            let status = Command::new("pip")
                .args([
                    "install",
                    "torch-geometric==2.5.2",
                    "torch==2.2.2",
                    "datasets==3.2.0",
                    "torchtext==0.17.2",
                    "torchvision==0.17.2",
                    "pandas==2.2.1",
                    "seaborn==0.13.2",
                    "notebook==7.1.2",
                    "nbconvert==7.16.3",
                    "onnx==1.17.0",
                    "kaggle==1.6.8",
                    "py-solc-x==2.0.3",
                    "web3==7.5.0",
                    "librosa==0.10.1",
                    "keras==3.1.1",
                    "tensorflow==2.16.1",
                    "tensorflow-datasets==4.9.4",
                    "pytorch-lightning==2.2.1",
                    "sk2torch==1.2.0",
                    "scikit-learn==1.4.1.post1",
                    "xgboost==2.0.3",
                    "hummingbird-ml==0.4.11",
                    "lightgbm==4.3.0",
                    "numpy==1.26.4",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
            let status = Command::new("pip")
                .args(["install", "numpy==1.26.4"])
                .status()
                .expect("failed to execute process");

            assert!(status.success());
        });
    }

    fn init_binary() {
        COMPILE.call_once(|| {
            println!("using cargo target dir: {}", *CARGO_TARGET_DIR);
            setup_py_env();
        });
    }

    fn mv_test_(test_dir: &str, test: &str) {
        let path: std::path::PathBuf = format!("{}/{}", test_dir, test).into();
        if !path.exists() {
            let status = Command::new("cp")
                .args([
                    "-R",
                    &format!("./examples/notebooks/{}", test),
                    &format!("{}/{}", test_dir, test),
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
        }
    }

    const TESTS: [&str; 35] = [
        "mnist_gan.ipynb",                         // 0
        "ezkl_demo_batch.ipynb",                   // 1
        "proof_splitting.ipynb",                   // 2
        "variance.ipynb",                          // 3
        "keras_simple_demo.ipynb",                 // 4
        "mnist_gan_proof_splitting.ipynb",         // 5
        "hashed_vis.ipynb",                        // 6
        "simple_demo_all_public.ipynb",            // 7
        "data_attest.ipynb",                       // 8
        "little_transformer.ipynb",                // 9
        "simple_demo_aggregated_proofs.ipynb",     // 10
        "ezkl_demo.ipynb",                         // 11
        "lstm.ipynb",                              // 12
        "set_membership.ipynb",                    // 13
        "decision_tree.ipynb",                     // 14
        "random_forest.ipynb",                     // 15
        "gradient_boosted_trees.ipynb",            // 16
        "xgboost.ipynb",                           // 17
        "lightgbm.ipynb",                          // 18
        "svm.ipynb",                               // 19
        "simple_demo_public_input_output.ipynb",   // 20
        "simple_demo_public_network_output.ipynb", // 21
        "gcn.ipynb",                               // 22
        "linear_regression.ipynb",                 // 23
        "stacked_regression.ipynb",                // 24
        "data_attest_hashed.ipynb",                // 25
        "kzg_vis.ipynb",                           // 26
        "kmeans.ipynb",                            // 27
        "solvency.ipynb",                          // 28
        "sklearn_mlp.ipynb",                       // 29
        "generalized_inverse.ipynb",               // 30
        "mnist_classifier.ipynb",                  // 31
        "world_rotation.ipynb",                    // 32
        "logistic_regression.ipynb",               // 33
        "univ3-da.ipynb",                          // 34
    ];

    macro_rules! test_func {
    () => {
        #[cfg(test)]
        mod tests {
            use seq_macro::seq;
            use crate::py_tests::TESTS;
            use test_case::test_case;
            use super::*;


            seq!(N in 0..=32 {

            #(#[test_case(TESTS[N])])*
            fn run_notebook_(test: &str) {
                crate::py_tests::init_binary();
                let mut limitless = false;
                if test == TESTS[5] {
                    limitless = true;
                }
                let mut anvil_child = crate::py_tests::start_anvil(limitless);
                let test_dir: TempDir = TempDir::new("nb").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, test);
                run_notebook(path, test);
                test_dir.close().unwrap();
                anvil_child.kill().unwrap();
            }
            });

            #[test]
            fn neural_bag_of_words_notebook() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("neural_bow").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "neural_bow.ipynb");
                run_notebook(path, "neural_bow.ipynb");
                test_dir.close().unwrap();
            }

            #[test]
            fn felt_conversion_test_notebook() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("felt_conversion_test").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "felt_conversion_test.ipynb");
                run_notebook(path, "felt_conversion_test.ipynb");
                test_dir.close().unwrap();
            }

            #[test]
            fn voice_notebook_() {
                crate::py_tests::init_binary();
                let mut anvil_child = crate::py_tests::start_anvil(false);
                crate::py_tests::download_voice_data();
                let test_dir: TempDir = TempDir::new("voice_judge").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "voice_judge.ipynb");
                run_notebook(path, "voice_judge.ipynb");
                test_dir.close().unwrap();
                anvil_child.kill().unwrap();
            }


            #[test]
            fn cat_and_dog_notebook_() {
                crate::py_tests::init_binary();
                let mut anvil_child = crate::py_tests::start_anvil(false);
                crate::py_tests::download_catdog_data();
                let test_dir: TempDir = TempDir::new("cat_and_dog").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "cat_and_dog.ipynb");
                run_notebook(path, "cat_and_dog.ipynb");
                test_dir.close().unwrap();
                anvil_child.kill().unwrap();
            }

            #[test]
            fn reusable_verifier_notebook_() {
                crate::py_tests::init_binary();
                let mut anvil_child = crate::py_tests::start_anvil(false);
                let test_dir: TempDir = TempDir::new("reusable_verifier").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "reusable_verifier.ipynb");
                run_notebook(path, "reusable_verifier.ipynb");
                test_dir.close().unwrap();
                anvil_child.kill().unwrap();
            }

            #[test]
            fn postgres_notebook_() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("mean_postgres").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "mean_postgres.ipynb");
                run_notebook(path, "mean_postgres.ipynb");
                test_dir.close().unwrap();
            }

            #[test]
            fn tictactoe_autoencoder_notebook_() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("tictactoe_autoencoder").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "tictactoe_autoencoder.ipynb");
                run_notebook(path, "tictactoe_autoencoder.ipynb");
                test_dir.close().unwrap();
            }

            #[test]
            fn tictactoe_binary_classification_notebook_() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("tictactoe_binary_classification").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "tictactoe_binary_classification.ipynb");
                run_notebook(path, "tictactoe_binary_classification.ipynb");
                test_dir.close().unwrap();
            }

            #[test]
            fn nbeats_notebook_() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("nbeats").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "nbeats_timeseries_forecasting.ipynb");
                crate::py_tests::mv_test_(path, "eth_price.csv");
                run_notebook(path, "nbeats_timeseries_forecasting.ipynb");
                test_dir.close().unwrap();
            }


    }
    };
}

    fn run_notebook(test_dir: &str, test: &str) {
        // activate venv
        let status = Command::new("bash")
            .arg("-c")
            .arg("source .env/bin/activate")
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let path: std::path::PathBuf = format!("{}/{}", test_dir, test).into();
        let status = Command::new("jupyter")
            .args([
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                (path.to_str().unwrap()),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    test_func!();
}
