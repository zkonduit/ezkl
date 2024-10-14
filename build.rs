fn main() {

    if cfg!(feature = "ios-bindings") {
        println!("cargo::rustc-env=UNIFFI_CARGO_BUILD_EXTRA_ARGS=--features=ios-bindings --no-default-features");
    }

    println!("cargo::rerun-if-changed=build.rs");
}