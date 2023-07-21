#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod py_tests {

    use lazy_static::lazy_static;
    use std::env::var;
    use std::process::Command;
    use std::sync::Once;
    use tempdir::TempDir;
    static COMPILE: Once = Once::new();
    static START_ANVIL: Once = Once::new();
    static ENV_SETUP: Once = Once::new();
    static DOWNLOAD_VOICE_DATA: Once = Once::new();

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
                    "torch",
                    "pandas",
                    "numpy",
                    "seaborn",
                    "jupyter",
                    "onnx",
                    "kaggle",
                    "py-solc-x",
                    "web3",
                    "librosa",
                    "keras",
                    "tensorflow",
                    "tf2onnx",
                ])
                .status()
                .expect("failed to execute process");

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

    fn mv_test_(test: &str) {
        let test_dir = TEST_DIR.path().to_str().unwrap();
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

    const TESTS: [&str; 6] = [
        "keras_simple_demo.ipynb",
        "encrypted_vis.ipynb",
        "hashed_vis.ipynb",
        "simple_demo.ipynb",
        "data_attest.ipynb",
        "variance.ipynb",
    ];

    macro_rules! test_func {
    () => {
        #[cfg(test)]
        mod tests {
            use seq_macro::seq;
            use crate::py_tests::TESTS;
            use test_case::test_case;
            use super::*;


            seq!(N in 0..=5 {

            #(#[test_case(TESTS[N])])*
            fn run_notebook_(test: &str) {
                crate::py_tests::init_binary();
                crate::py_tests::start_anvil();
                crate::py_tests::mv_test_(test);
                run_notebook(test);
            }
            #[test]
            fn voice_notebook_() {
                crate::py_tests::init_binary();
                crate::py_tests::start_anvil();
                crate::py_tests::download_voice_data();
                crate::py_tests::mv_test_("voice_judge.ipynb");
                run_notebook("voice_judge.ipynb");
            }

            });

    }
    };
}

    fn run_notebook(test: &str) {
        // activate venv
        let status = Command::new("bash")
            .arg("-c")
            .arg("source .env/bin/activate")
            .status()
            .expect("failed to execute process");
        assert!(status.success());

        let test_dir = TEST_DIR.path().to_str().unwrap();
        let path: std::path::PathBuf = format!("{}/{}", test_dir, test).into();
        let status = Command::new("jupyter")
            .args([
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                &path.to_str().unwrap(),
            ])
            .status()
            .expect("failed to execute process");
        assert!(status.success());
    }

    test_func!();
}
