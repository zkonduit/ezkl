# Halo2 Deep Learning

[![Test](https://github.com/jasonmorton/halo2deeplearning/workflows/Rust/badge.svg)](https://github.com/jasonmorton/halo2deeplearning/actions?query=workflow%3ARust)

This is a proof-of-concept implementation of inference for deep learning models in a zk-snark using Halo2. 2d convolution, fully connected (affine) layers, and nonlinearities such as ReLU and sigmoid are implemented.  The input image and model parameters are provided as private advice and the last layer is the public input (instance column). Other configurations are also possible.

We give an example of proving inference with a model that achieves 97.5% accuracy on MNIST in the examples.


Note that the library requires a nightly version of the rust toolchain. You can change the default toolchain by running:
```bash
rustup override set nightly         
```

## `ezkl` command line interface

The `ezkl` cli provides a simple interface to load ONNX neural networks, convert them into a Halo2 circuit, then run a proof (given a public input).

Usage:

```bash
[*]
 |          ███████╗███████╗██╗  ██╗██╗
 |          ██╔════╝╚══███╔╝██║ ██╔╝██║
 |          █████╗    ███╔╝ █████╔╝ ██║
 |          ██╔══╝   ███╔╝  ██╔═██╗ ██║
 |          ███████╗███████╗██║  ██╗███████╗
 |          ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝
 |  
 |          -----------------------------------------------------------
 |          Easy Zero Knowledge for Neural Networks.
 |          -----------------------------------------------------------
 |          
Usage: ezkl [OPTIONS] --scale <SCALE> --bits <BITS>

Options:
  -D, --data <DATA>    The path to the .json data file [default: ]
  -M, --model <MODEL>  The path to the .onnx model file [default: ]
  -S, --scale <SCALE>  The denominator in the fixed point representation used when quantizing
  -B, --bits <BITS>    The number of bits used in lookup tables
  -h, --help           Print help information
  -V, --version        Print version information
```
If `-D` and `-M` are not provided the cli will query the user to manually enter the path(s). `.onnx` can be generated using pytorch or tensorflow. The data json file is structured as follows:

```json
{
    "input_data": [1, 5, 6 ...], // array of integers which represents the (private) inputs we run the proof on
    "input_shape": [3, 3, ...], // array of integers which represents the shape of model inputs (excluding batch size)
    "public_input": [1, 5, 6 ...], // array of integers which represents the public input (model output for now)
}
```
For examples of such files see `examples/onnx_models`.

To run a simple example using the cli:
```bash
cargo run --release --bin ezkl -- -D ./examples/onnx_models/ff_input.json -M ./examples/onnx_models/ff.onnx
```

To display a table of loaded Onnx nodes, and their associated parameters, set `RUST_LOG=DEBUG`.


## Running examples

The MNIST inference example is contained in `examples/conv2d_mnist`. To run it:
```bash
cargo run --release --example conv2d_mnist
```
We also provide an example which runs an MLP on input data with four dimensions. To run it:
```bash
cargo run --release --example mlp_4d
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

Use `cargo doc --open --feature onnx` to compile and open the docs in your default browser.
