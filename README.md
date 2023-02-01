# EZKL 

[![Test](https://github.com/zkonduit/ezkl/workflows/Rust/badge.svg)](https://github.com/zkonduit/ezkl/actions?query=workflow%3ARust)

`ezkl` is a library and command-line tool for doing inference for deep learning models and other computational graphs in a zk-snark. It enables the following workflow:

1. Define a computational graph, for instance a neural network (but really any arbitrary set of operations), as you would normally in pytorch or tensorflow. 
2. Export the final graph of operations as an [.onnx](https://onnx.ai/) file and some sample inputs to a `.json` file.
3. Point `ezkl` to the `.onnx` and `.json` files to generate a ZK-SNARK circuit with which you can prove statements such as: 
> "I ran this publicly available neural network on some private data and it produced this output"

> "I ran my private neural network on some public data and it produced this output"

> "I correctly ran this publicly available neural network on some public data and it produced this output"

The rust API is also sufficiently flexible to enable you to code up a computational graph, and resulting circuit, from scratch. For examples on how to do so see the **library examples** section below.

In the backend, we use [Halo2](https://github.com/privacy-scaling-explorations/halo2) as a proof system.  

For more details on how to use `ezkl`, see below ! 

----------------------

## Contributing 🌎

If you're interested in contributing and are unsure where to start, reach out to one of the maintainers:  

* dante (alexander-camuto)
* jason ( jasonmorton)

More broadly: 

- Feel free to open up a discussion topic to ask questions. 

- See currently open issues for ideas on how to contribute. 

- For PRs we use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) naming convention. 

----------------------

## Getting Started ⚙️

Note that the library requires a nightly version of the rust toolchain. You can change the default toolchain by running:

```bash
rustup override set nightly         
```

This repository includes onnx example files as a submodule for testing out the cli. Either pass the `--recurse-submodules` flag to the `clone` command or after cloning run `git submodule update --init --recursive`. 

### docs 📖

Use `cargo doc --open` to compile and open the docs in your default browser.


----------------------


## Command line interface 👾

The `ezkl` cli provides a simple interface to load `.onnx` files, which represent graphs of operations (such as neural networks), convert them into a Halo2 circuit, then run a proof.

### python and cli tutorial 🐍

You can easily create an `.onnx` file using `pytorch`. For samples of Onnx files see [here](https://github.com/zkonduit/onnx-examples). For a tutorial on how to quickly generate Onnx files using python, check out [pyezkl](https://github.com/zkonduit/pyezkl). 

These examples are also available as a submodule in `./examples/onnx`. To generate a proof on one of the examples, first build ezkl (`cargo build --release`) and add it to your favourite `PATH` variables.   

```bash
ezkl --bits=16 -K=17 prove -D ./examples/onnx/examples/1l_relu/input.json -M ./examples/onnx/examples/1l_relu/network.onnx -O 1l_relu.pf --vk-path 1l_relu.vk --params-path 1l_relu.params
``` 

This command generates a proof that the model was correctly run on private inputs (this is the default setting). It then outputs the resulting proof at the path specfifed by `-O`, parameters that can be used for subsequent verification at `--params-path` and the verifier key at `ipa_1l_relu.vk`. 
Luckily `ezkl` also provides command to verify the generated proofs: 

```bash 
ezkl --bits=16 -K=17 verify -M ./examples/onnx/examples/1l_relu/network.onnx -P 1l_relu.pf --vk-path 1l_relu.vk --params-path 1l_relu.params
``` 

The separate prove and verify steps can be combined into a single command, if you'd prefer to not write to your filesystem: 

```bash
ezkl --bits=16 -K=17 fullprove -D ./examples/onnx/examples/1l_relu/input.json -M ./examples/onnx/examples/1l_relu/network.onnx 
```
#### verifying with the EVM ◊

Note that `fullprove` can also be run with an EVM verifier.  We need to pass the `evm` feature flag to conditionally compile the requisite [foundry_evm](https://github.com/foundry-rs/foundry) dependencies. Using `foundry_evm` we spin up a local EVM executor and verify the generated proof. In future releases we'll create a simple pipeline for deploying to EVM based networks. Also note that this requires a local [solc](https://docs.soliditylang.org/en/v0.8.17/installing-solidity.html) installation. 


Example:

```bash
cargo run  --release --features evm --bin ezkl fullprove -D ./examples/onnx/examples/1l_relu/input.json -M ./examples/onnx/examples/1l_relu/network.onnx 
```


### usage 🔧

```bash
Usage: ezkl [OPTIONS] <COMMAND>

Commands:
  table      Loads model and prints model table
  mock       Loads model and input and runs mock prover (for testing)
  fullprove  Loads model and input and runs full prover (for testing)
  prove      Loads model and data, prepares vk and pk, and creates proof, saving proof in --output
  verify     Verifies a proof, returning accept or reject
  help       Print this message or the help of the given subcommand(s)

Options:
  -T, --tolerance <TOLERANCE>  The tolerance for error on model outputs [default: 0]
  -S, --scale <SCALE>          The denominator in the fixed point representation used when quantizing [default: 7]
  -B, --bits <BITS>            The number of bits used in lookup tables [default: 16]
  -K, --logrows <LOGROWS>      The log_2 number of rows [default: 17]
      --public-inputs          Flags whether inputs are public
      --public-outputs         Flags whether outputs are public
      --public-params          Flags whether params are public
  -h, --help                   Print help information
  -V, --version                Print version information
```

`bits`, `scale`, `tolerance`, and `logrows` have default values. You can use tolerance to express a tolerance to a certain amount of quantization error on the output eg. if set to 2 the circuit will verify even if the generated output deviates by an absolute value of 2 on any dimension from the expected output. `prove`, `mock`, `fullprove` all require `-D` and `-M` parameters, which if not provided, the cli will query the user to manually enter the path(s).

```bash

Usage: ezkl mock [OPTIONS]

Options:
  -D, --data <DATA>    The path to the .json data file [default: ]
  -M, --model <MODEL>  The path to the .onnx model file [default: ]

```

The `.onnx` file can be generated using pytorch or tensorflow. The data json file is structured as follows:

```javascript
{
    "input_data": [[1.0, 22.2, 0.12 ...]], // 2D arrays of floats which represents the (private) inputs we run the proof on
    "input_shapes": [[3, 3, ...]], // 2D array of integers which represents the shapes of model inputs (excluding batch size)
    "output_data": [[1.0, 5.0, 6.3 ...]], // 2D arrays of floats which represents the model outputs we want to constrain against (if any)
}
```

For examples of such files see `examples/onnx_models`.

To run a simple example using the cli:

```bash
cargo run --release --bin ezkl -- mock -D ./examples/onnx/examples/1l_relu/input.json -M ./examples/onnx/examples/1l_relu/network.onnx
```

To display a table of loaded Onnx nodes, and their associated parameters, set `RUST_LOG=DEBUG` or run:

```bash
cargo run --release --bin ezkl -- table -M ./examples/onnx/examples/1l_relu/network.onnx

```

## benchmarks ⏳

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

## library examples 🔍

Beyond the `.onnx` examples detailed above, we also include examples which directly use some of our rust API; allowing users to code up computational graphs and circuits from scratch in rust without having to go via python. 

The MNIST inference example using ezkl as a library is contained in `examples/conv2d_mnist`. To run it:

```bash
# download MNIST data
chmod +x data.sh
./data.sh
# test the model (takes 600-700 seconds)
cargo run --release --example conv2d_mnist
```

We also provide an example which runs an MLP on input data with four dimensions. To run it:

```bash
cargo run --release --example mlp_4d
```

----------------------
