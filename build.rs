use copy_to_output::copy_to_output;
use std::{env, process::Command};

fn main() {
    // Re-runs script if the python file has changed
    println!("cargo:rerun-if-changed=fix_verifier_sol.py");

    let build_type = &env::var("PROFILE").unwrap();
    let out_path = format!("target/{}", build_type);
    copy_to_output("fix_verifier_sol.py", build_type).expect("Could not copy");
    // make it executable
    let cmd_args = format!("{}/fix_verifier_sol.py", out_path);

    Command::new("chmod")
        .args(["+x", &cmd_args])
        .output()
        .unwrap();
}
