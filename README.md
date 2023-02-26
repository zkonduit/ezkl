# EZKL 

[![Test](https://github.com/zkonduit/ezkl/workflows/Rust/badge.svg)](https://github.com/zkonduit/ezkl/actions?query=workflow%3ARust)

`ezkl` is a library and command-line tool for doing inference for deep learning models and other computational graphs in a zk-snark. It enables the following workflow:

1. Define a computational graph, for instance a neural network (but really any arbitrary set of operations), as you would normally in pytorch or tensorflow. 
2. Export the final graph of operations as an [.onnx](https://onnx.ai/) file and some sample inputs to a `.json` file.
3. Point `ezkl` to the `.onnx` and `.json` files to generate a ZK-SNARK circuit with which you can prove statements such as: 
> "I ran this publicly available neural network on some private data and it produced this output"

> "I ran my private neural network on some public data and it produced this output"

> "I correctly ran this publicly available neural network on some public data and it produced this output"

The rust API is also sufficiently flexible to enable you to code up a computational graph and resulting circuit from scratch. For examples on how to do so see the **library examples** section below.

In the backend we use [Halo2](https://github.com/privacy-scaling-explorations/halo2) as a proof system.  

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

### docs 📖

Use `cargo doc --open` to compile and open the docs in your default browser.


----------------------


## Command line interface 👾

The `ezkl` cli provides a simple interface to load `.onnx` files, which represent graphs of operations (such as neural networks), convert them into a Halo2 circuit, then run a proof.

### python and cli tutorial 🐍

You can easily create an `.onnx` file using `pytorch`. For samples of Onnx files see [here](https://github.com/zkonduit/onnx-examples). For a tutorial on how to quickly generate Onnx files using python, check out [pyezkl](https://github.com/zkonduit/pyezkl). 

Sample onnx files are also available in `./examples/onnx`. To generate a proof on one of the examples, first build ezkl (`cargo build --release`) and add it to your favourite `PATH` variables, then generate a structured reference string (SRS): 
```bash
ezkl -K=17 gen-srs --pfsys=kzg --params-path=kzg.params
``` 

```bash
ezkl --bits=16 -K=17 prove -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --vk-path 1l_relu.vk --params-path=kzg.params
``` 

This command generates a proof that the model was correctly run on private inputs (this is the default setting). It then outputs the resulting proof at the path specfifed by `--proof-path`, parameters that can be used for subsequent verification at `--params-path` and the verifier key at `--vk-path`. 
Luckily `ezkl` also provides command to verify the generated proofs: 

```bash 
ezkl --bits=16 -K=17 verify -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --vk-path 1l_relu.vk --params-path=kzg.params
``` 

To display a table of the loaded onnx nodes, their associated parameters, set `RUST_LOG=DEBUG` or run:

```bash
cargo run --release --bin ezkl -- table -M ./examples/onnx/1l_relu/network.onnx

```

#### verifying with the EVM ◊

Note that the above prove and verify stats can also be run with an EVM verifier. This can be done by generating a verifier smart contract after generating the proof 

```bash
# gen proof
ezkl --bits=16 -K=17 prove -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --vk-path 1l_relu.vk --params-path=kzg.params --transcript=evm
```
```bash
# gen evm verifier
ezkl -K=17 --bits=16 create-evm-verifier -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --pfsys=kzg --deployment-code-path 1l_relu.code --params-path=kzg.params --vk-path 1l_relu.vk --sol-code-path 1l_relu.sol
```
```bash
# Verify (EVM) 
ezkl -K=17 --bits=16 verify-evm --pfsys=kzg --proof-path 1l_relu.pf --deployment-code-path 1l_relu.code
```

Note that the `.sol` file above can be deployed and composed with other Solidity contracts, via a `verify()` function. Please read [this document](https://hackmd.io/QOHOPeryRsOraO7FUnG-tg) for more information about the interface of the contract, how to obtain the data needed for its function parameters, and its limitations.

The above pipeline can also be run using [proof aggregation](https://ethresear.ch/t/leveraging-snark-proof-aggregation-to-achieve-large-scale-pbft-based-consensus/11588) to reduce proof size and verifying times, so as to be more suitable for EVM deployment. A sample pipeline for doing so would be: 

```bash
# Generate a new 2^20 SRS
ezkl -K=20 gen-srs --pfsys=kzg --params-path=kzg.params
```

```bash 
# Single proof -> single proof we are going to feed into aggregation circuit. (Mock)-verifies + verifies natively as sanity check
ezkl -K=17 --bits=16 prove --pfsys=kzg --transcript=poseidon --strategy=accum -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --params-path=kzg.params  --vk-path=1l_relu.vk
```

```bash
# Aggregate -> generates aggregate proof and also (mock)-verifies + verifies natively as sanity check
ezkl -K=17 --bits=16 aggregate --transcript=evm -M ./examples/onnx/1l_relu/network.onnx --pfsys=kzg --aggregation-snarks=1l_relu.pf --aggregation-vk-paths 1l_relu.vk --vk-path aggr_1l_relu.vk --proof-path aggr_1l_relu.pf --params-path=kzg.params
``` 

```bash
# Generate verifier code -> create the EVM verifier code 
ezkl -K=17 --bits=16 aggregate create-evm-verifier-aggr --pfsys=kzg --deployment-code-path aggr_1l_relu.code --params-path=kzg.params --vk-path aggr_1l_relu.vk
```

```bash
# Verify (EVM) -> 
ezkl -K=17 --bits=16 verify-evm --pfsys=kzg --proof-path aggr_1l_relu.pf --deployment-code-path aggr_1l_relu.code
 
```

Also note that this may require a local [solc](https://docs.soliditylang.org/en/v0.8.17/installing-solidity.html) installation, and that aggregated proof verification in Solidity is not currently supported.

#### using pre-generated SRS 

Note that you can use pre-generated KZG SRS. These SRS can be converted to a format that is ingestable by the `pse/halo2` prover ezkl uses by leveraging [han0110/halo2-kzg-srs](https://github.com/han0110/halo2-kzg-srs). This repo also contains pre-converted SRS from large projects such as Hermez and the [perpetual powers of tau repo](https://github.com/privacy-scaling-explorations/perpetualpowersoftau). Simply download the pre-converted file locally and point `--params-path` to the file ! 


### general usage 🔧

```bash
Usage: ezkl [OPTIONS] <COMMAND>

Commands:
  table                     Loads model and prints model table
  gen-srs                   Generates a dummy SRS
  mock                      Loads model and input and runs mock prover (for testing)
  aggregate                 Aggregates proofs :)
  prove                     Loads model and data, prepares vk and pk, and creates proof
  create-evm-verifier       Creates an EVM verifier for a single proof
  create-evm-verifier-aggr  Creates an EVM verifier for an aggregate proof
  verify                    Verifies a proof, returning accept or reject
  verify-aggr               Verifies an aggregate proof, returning accept or reject
  verify-evm                Verifies a proof using a local EVM executor, returning accept or reject
  help                      Print this message or the help of the given subcommand(s)

Options:
  -T, --tolerance <TOLERANCE>          The tolerance for error on model outputs [default: 0]
  -S, --scale <SCALE>                  The denominator in the fixed point representation used when quantizing [default: 7]
  -B, --bits <BITS>                    The number of bits used in lookup tables [default: 16]
  -K, --logrows <LOGROWS>              The log_2 number of rows [default: 17]
      --public-inputs                  Flags whether inputs are public
      --public-outputs                 Flags whether outputs are public
      --public-params                  Flags whether params are public
  -M, --max-rotations <MAX_ROTATIONS>  Flags to set maximum rotations [default: 512]
  -h, --help                           Print help information
  -V, --version                        Print version information
```

`bits`, `scale`, `tolerance`, and `logrows` have default values. You can use tolerance to express a tolerance to a certain amount of quantization error on the output eg. if set to 2 the circuit will verify even if the generated output deviates by an absolute value of 2 on any dimension from the expected output. `prove` and `mock`, all require `-D` and `-M` parameters, which if not provided, the cli will query the user to manually enter the path(s).

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

To run a simple example using the cli see **python and cli tutorial** above.


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

## onnx examples 

This repository includes onnx example files as a submodule for testing out the cli.

If you want to add a model to `examples/onnx`, open a PR creating a new folder within `examples/onnx` with a descriptive model name. This folder should contain: 
- an `input.json` input file, with the fields expected by the  [ezkl](https://github.com/zkonduit/ezkl) cli. 
- a `network.onnx` file representing the trained model 
- a `gen.py` file for generating the `.json` and `.onnx` files following the general structure of `examples/tutorial/tutorial.py`.


TODO: add associated python files in the onnx model directories. 

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
