# Halo2 Deep Learning

[![Test](https://github.com/jasonmorton/halo2deeplearning/workflows/Rust/badge.svg)](https://github.com/jasonmorton/halo2deeplearning/actions?query=workflow%3ARust)

This is a proof-of-concept implementation of inference for deep learning models in a zk-snark using Halo2. 2d convolution, fully connected (affine) layers, and nonlinearities such as ReLU and sigmoid are implemented.  The input image and model parameters are provided as private advice and the last layer is the public input (instance column). Other configurations are also possible.

We give an example of proving inference with a model that achieves 97.5% accuracy on MNIST in the examples.


Note that the library requires a nightly version of the rust toolchain. You can change the default toolchain by running:
```bash
rustup override set nightly         
```

## Running examples

The MNIST inference example is contained in `examples/conv2d_mnist`. To run it:
```bash
cargo run --release --example conv2d_mnist
```
We also provide an example which runs an MLP on input data with four dimensions. To run it:
```bash
cargo run --release --example mlp_4d
```

### Running onnx example

To run the example which loads parameters from an ONNX model file you need to enable the `onnx` build feature:

```bash
cargo run --release --example smallonnx --features onnx
```

## Running benchmarks

We include proof generation time benchmarks for some of the implemented layers including the affine, convolutional, and ReLu operations (more to come).

To run these benchmarks:
```bash
cargo bench
```

To run a specific benchmark append one of `affine, cnvrl, relu` to the command. You can then find benchmarks results and plots in `target/criterion`. Note that depending on the capabilities of your machine you may need to increase the target time on the Criterion config. For instance:

```rust
criterion_group! {
  name = benches;
  config = Criterion::default().measurement_time(Duration::from_secs(10));
  targets = runrelu
}
```

## Docs

Use `cargo doc --open` to compile and open the docs in your default browser. 
