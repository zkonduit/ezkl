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
    build_ezkl();
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "-K=23",
            "gen-srs",
            "--pfsys=kzg",
            "--params-path=kzg.params",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

const TESTS: [&str; 16] = [
    "1l_mlp",
    "1l_flatten",
    "1l_average",
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
    "2l_relu_sigmoid_conv",
    "3l_relu_conv_fc",
    "4l_relu_conv_fc",
];

const TESTS_AGGR: [&str; 12] = [
    "1l_mlp",
    "1l_flatten",
    "1l_average",
    "1l_reshape",
    "1l_sigmoid",
    "1l_sqrt",
    "1l_leakyrelu",
    "1l_relu",
    "2l_relu_fc",
    "2l_relu_sigmoid_small",
    "2l_relu_small",
    "1l_conv",
];

const NEG_TESTS: [(&str, &str); 2] = [
    ("2l_relu_sigmoid_small", "2l_relu_small"),
    ("2l_relu_small", "2l_relu_sigmoid_small"),
];

const TESTS_EVM: [&str; 11] = [
    "1l_mlp",
    "1l_flatten",
    "1l_average",
    "1l_reshape",
    "1l_sigmoid",
    "1l_sqrt",
    "1l_leakyrelu",
    "1l_relu",
    "2l_relu_sigmoid_small",
    "2l_relu_small",
    "2l_relu_fc",
];

const EXAMPLES: [&str; 2] = ["mlp_4d", "conv2d_mnist"];

macro_rules! test_func_aggr {
    () => {
        #[cfg(test)]
        mod tests_aggr {
            use seq_macro::seq;
            use crate::TESTS_AGGR;
            use test_case::test_case;
            use crate::kzg_aggr_prove_and_verify;
            seq!(N in 0..=11 {

            #(#[test_case(TESTS_AGGR[N])])*
            fn kzg_aggr_prove_and_verify_(test: &str) {
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
            use crate::TESTS;
            use test_case::test_case;
            use crate::mock;
            use crate::mock_public_inputs;
            use crate::mock_public_params;
            use crate::forward_pass;
            use crate::kzg_prove_and_verify;
            seq!(N in 0..=15 {
            #(#[test_case(TESTS[N])])*
            fn mock_public_outputs_(test: &str) {
                mock(test.to_string());
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

            #(#[test_case(TESTS[N])])*
            fn kzg_prove_and_verify_(test: &str) {
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
            use crate::TESTS_EVM;
            use test_case::test_case;
            use crate::kzg_evm_prove_and_verify;
            use crate::kzg_evm_aggr_prove_and_verify;

            /// Not all models will pass VerifyEVM because their contract size exceeds the limit, so we only
            /// specify a few that will
            const TESTS_SOLIDITY: [&str; 6] = [
                "1l_relu",
                "1l_leakyrelu",
                "1l_sqrt",
                "1l_sigmoid",
                "1l_reshape",
                "2l_relu_fc"
            ];

            seq!(N in 0..=10 {

                #(#[test_case(TESTS_EVM[N])])*
                fn kzg_evm_prove_and_verify_(test: &str) {
                    kzg_evm_prove_and_verify(test.to_string(), TESTS_SOLIDITY.contains(&test));
                }
                // these take a particularly long time to run
                #(#[test_case(TESTS_EVM[N])])*
                #[ignore]
                fn kzg_evm_aggr_prove_and_verify_(test: &str) {
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
            use crate::EXAMPLES;
            use test_case::test_case;
            use crate::run_example as run;
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
            use crate::NEG_TESTS;
            use test_case::test_case;
            use crate::neg_mock as run;
            seq!(N in 0..=1 {
            #(#[test_case(NEG_TESTS[N])])*
            fn neg_examples_(test: (&str, &str)) {
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
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
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

    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input_forward.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock(example_name: String) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_public_inputs(example_name: String) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--public-inputs",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// Mock prove (fast, but does not cover some potential issues)
fn mock_public_params(example_name: String) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--public-params",
            "--bits=16",
            "-K=17",
            "mock",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            // "-K",
            // "2",  //causes failure
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// prove-serialize-verify, the usual full path
fn kzg_aggr_prove_and_verify(example_name: String) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "prove",
            "--pfsys=kzg",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--proof-path",
            format!("kzg_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=poseidon",
            "--strategy=accum",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "aggregate",
            "--pfsys=kzg",
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--aggregation-snarks",
            format!("kzg_{}.pf", example_name).as_str(),
            "--aggregation-vk-paths",
            format!("kzg_{}.vk", example_name).as_str(),
            "--proof-path",
            format!("kzg_aggr_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_aggr_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=blake",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "verify-aggr",
            "--pfsys=kzg",
            "--proof-path",
            format!("kzg_aggr_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_aggr_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=blake",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// prove-serialize-verify, the usual full path
fn kzg_evm_aggr_prove_and_verify(example_name: String) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "prove",
            "--pfsys=kzg",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--proof-path",
            format!("kzg_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=poseidon",
            "--strategy=accum",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "aggregate",
            "--pfsys=kzg",
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--aggregation-snarks",
            format!("kzg_{}.pf", example_name).as_str(),
            "--aggregation-vk-paths",
            format!("kzg_{}.vk", example_name).as_str(),
            "--proof-path",
            format!("kzg_aggr_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_aggr_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=evm",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "create-evm-verifier-aggr",
            "--pfsys=kzg",
            "--deployment-code-path",
            format!("kzg_aggr_{}.code", example_name).as_str(),
            "--params-path=kzg.params",
            "--vk-path",
            format!("kzg_aggr_{}.vk", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "verify-evm",
            "--pfsys=kzg",
            "--proof-path",
            format!("kzg_aggr_{}.pf", example_name).as_str(),
            "--deployment-code-path",
            format!("kzg_aggr_{}.code", example_name).as_str(),
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// prove-serialize-verify, the usual full path
fn kzg_prove_and_verify(example_name: String) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "prove",
            "--pfsys=kzg",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--proof-path",
            format!("kzg_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=blake",
            "--strategy=single",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "verify",
            "--pfsys=kzg",
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--proof-path",
            format!("kzg_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=blake",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}

// prove-serialize-verify, the usual full path
fn kzg_evm_prove_and_verify(example_name: String, with_solidity: bool) {
    let status = Command::new(format!("{}/release/ezkl", *CARGO_TARGET_DIR))
        .args([
            "--bits=16",
            "-K=17",
            "prove",
            "--pfsys=kzg",
            "-D",
            format!("./examples/onnx/{}/input.json", example_name).as_str(),
            "-M",
            format!("./examples/onnx/{}/network.onnx", example_name).as_str(),
            "--proof-path",
            format!("kzg_{}.pf", example_name).as_str(),
            "--vk-path",
            format!("kzg_{}.vk", example_name).as_str(),
            "--params-path=kzg.params",
            "--transcript=evm",
            "--strategy=single",
        ])
        .status()
        .expect("failed to execute process");
    assert!(status.success());

    let input_arg = format!("./examples/onnx/{}/input.json", example_name);
    let network_arg = format!("./examples/onnx/{}/network.onnx", example_name);
    let code_arg = format!("kzg_{}.code", example_name);
    let vk_arg = format!("kzg_{}.vk", example_name);

    let mut args = vec![
        "--bits=16",
        "-K=17",
        "create-evm-verifier",
        "--pfsys=kzg",
        "-D",
        input_arg.as_str(),
        "-M",
        network_arg.as_str(),
        "--deployment-code-path",
        code_arg.as_str(),
        "--params-path=kzg.params",
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

    let pf_arg = format!("kzg_{}.pf", example_name);

    let mut args = vec![
        "--bits=16",
        "-K=17",
        "verify-evm",
        "--pfsys=kzg",
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
        .args(["build", "--release", "--bin", "ezkl"])
        .status()
        .expect("failed to execute process");
    assert!(status.success());
}
