# Halo2 Deep Learning

[![Test](https://github.com/jasonmorton/halo2deeplearning/workflows/Rust/badge.svg)](https://github.com/jasonmorton/halo2deeplearning/actions?query=workflow%3ARust)

This is a proof-of-concept implementation of inference for deep learning models in a zk-snark using Halo2. 2d convolution, fully connected (affine) layers, and nonlinearities such as ReLU and sigmoid are implemented.  The input image and model parameters are provided as private advice and the last layer is the public input (instance column). Other configurations are also possible.

See [here](https://github.com/jasonmorton/exampledl) for an example that uses this crate as a library.

We give an example of proving inference with a model that achieves 97.5% accuracy on MNIST in the tests.

## Running examples
The MNIST inference example (`test_prove_mnist_inference`) is by default ignored because making the proof uses a lot of memory and takes about three minutes. To run it, use
```bash
cargo test --release -- --ignored --nocapture
```
or ``--include-ignored` to run together with the rest.

Note that the library requires a nightly version of the rust toolchain. You can change the default toolchain by running:
```bash
rustup override set nightly         
```
