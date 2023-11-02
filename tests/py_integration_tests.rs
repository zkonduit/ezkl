#[cfg(not(target_arch = "wasm32"))]
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

    //Sure to run this once

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
        std::env::set_var("VOICE_DATA_DIR", format!("{}", voice_data_dir));
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
                .args([
                    "install",
                    "torch==2.0.1",
                    "pandas==2.0.3",
                    "numpy==1.23",
                    "seaborn==0.12.2",
                    "jupyter==1.0.0",
                    "onnx==1.14.0",
                    "kaggle==1.5.15",
                    "py-solc-x==1.1.1",
                    "web3==6.5.0",
                    "librosa==0.10.0.post2",
                    "keras==2.12.0",
                    "tensorflow==2.12.0",
                    "tf2onnx==1.14.0",
                    "pytorch-lightning==2.0.6",
                    "sk2torch==1.2.0",
                    "scikit-learn==1.3.1",
                    "xgboost==1.7.6",
                    "hummingbird-ml==0.4.9",
                    "lightgbm==4.0.0",
                ])
                .status()
                .expect("failed to execute process");
            assert!(status.success());
            let status = Command::new("pip")
                .args(["install", "numpy==1.23"])
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

    const TESTS: [&str; 33] = [
        "mnist_gan.ipynb",
        // "mnist_vae.ipynb",
        "keras_simple_demo.ipynb",
        "encrypted_vis.ipynb",
        "hashed_vis.ipynb",
        "simple_demo_all_public.ipynb",
        "data_attest.ipynb",
        "variance.ipynb",
        "mean_postgres.ipynb",
        "little_transformer.ipynb",
        "simple_demo_aggregated_proofs.ipynb",
        "ezkl_demo.ipynb",
        "lstm.ipynb",
        "set_membership.ipynb",
        "decision_tree.ipynb",
        "random_forest.ipynb",
        "gradient_boosted_trees.ipynb",
        "xgboost.ipynb",
        "lightgbm.ipynb",
        "svm.ipynb",
        "simple_demo_public_input_output.ipynb",
        "simple_demo_public_network_output.ipynb",
        "gcn.ipynb",
        "linear_regression.ipynb",
        "stacked_regression.ipynb",
        "data_attest_hashed.ipynb",
        "simple_hub_demo.ipynb",
        "kzg_vis.ipynb",
        "kmeans.ipynb",
        "solvency.ipynb",
        "proof_splitting.ipynb",
        "mnist_gan_proof_splitting.ipynb",
        "sklearn_mlp.ipynb",
        "generalized_inverse.ipynb",
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
            fn nbeats_notebook_() {
                crate::py_tests::init_binary();
                let test_dir: TempDir = TempDir::new("nbeats").unwrap();
                let path = test_dir.path().to_str().unwrap();
                crate::py_tests::mv_test_(path, "nbeats_timeseries_forecasting.ipynb");
                crate::py_tests::mv_test_(path, "eth_price.csv");
                run_notebook(path, "nbeats_timeseries_forecasting.ipynb");
                test_dir.close().unwrap();
            }
            });

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
