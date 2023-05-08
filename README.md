<h1 align="center">
	<br>
	 :thought_balloon:
	<br>
	<br>
EZKL
	<br>
	<br>
	<br>
</h1>

> Easy Zero-Knowledge Inference

[![Test](https://github.com/zkonduit/ezkl/workflows/Rust/badge.svg)](https://github.com/zkonduit/ezkl/actions?query=workflow%3ARust)

`ezkl` is a library and command-line tool for doing inference for deep learning models and other computational graphs in a zk-snark (ZKML). It enables the following workflow:

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

- Feel free to open up a discussion topic in [Discussions](https://github.com/zkonduit/ezkl/discussions) to ask questions. Alternatively, you may join the [EZKL Community Telegram Group](https://t.me/+76OjHb5CwJtkMTBh) to ask questions.

- See currently open issues for ideas on how to contribute.

- For PRs we use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) naming convention.

- To report bugs or request new features [create a new issue within Issues](https://github.com/zkonduit/ezkl/issues) to inform the greater community.


----------------------

## Getting Started ⚙️

### building the project 🔨
Note that the library requires a nightly version of the rust toolchain. You can change the default toolchain by running:

```bash
rustup override set nightly
```

After which you may build the library

```bash
cargo build --release
```

A folder `./target/release` will be generated. Add this folder to your PATH environment variable to call `ezkl` from the CLI.

```bash
# For UNIX like systems
# in .bashrc, .bash_profile, or some other path file
export PATH="<replace with where you cloned the repo>/ezkl/target/release:$PATH"
```

Restart your shell or reload your shell settings

```bash
# example, replace .bash_profile with the file you use to configure your shell
source ~/.bash_profile
```

You will need a functioning installation of `solc` in order to run `ezkl` properly.
[solc-select](https://github.com/crytic/solc-select) is recommended.
Follow the instructions on [solc-select](https://github.com/crytic/solc-select) to activate `solc` in your environment.


### docs 📖

Use `cargo doc --open` to compile and open the docs in your default browser.


----------------------


## Command line interface 👾

The `ezkl` cli provides a simple interface to load `.onnx` files, which represent graphs of operations (such as neural networks), convert them into a Halo2 circuit, then run a proof.

### python and cli tutorial 🐍

You can easily create an `.onnx` file using `pytorch`. For samples of Onnx files see [here](https://github.com/zkonduit/onnx-examples). For a tutorial on how to quickly generate Onnx files using python, check out [pyezkl](https://github.com/zkonduit/pyezkl).

Sample onnx files are also available in `./examples/onnx`. To generate a proof on one of the examples, first build ezkl (`cargo build --release`) and add it to your favourite `PATH` variables, then generate a structured reference string (SRS):
```bash
ezkl -K=17 gen-srs --params-path=kzg.params
```

Now setup the proving and verification keys:

```bash
ezkl --bits=16 -K=17 setup -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --pk-path 1l_relu.pk --vk-path 1l_relu.vk --params-path=kzg.params --circuit-params-path=circuit.params
```

Now generate a proof:

```bash
ezkl --bits=16 -K=17 prove -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --pk-path 1l_relu.pk --params-path=kzg.params
```

This command generates a proof that the model was correctly run on private inputs (this is the default setting). It then outputs the resulting proof at the path specfifed by `--proof-path`, parameters that can be used for subsequent verification at `--params-path` and the verifier key at `--vk-path`.
Luckily `ezkl` also provides command to verify the generated proofs:

```bash
ezkl --bits=16 -K=17 verify  --proof-path 1l_relu.pf --vk-path 1l_relu.vk --params-path=kzg.params --circuit-params-path=circuit.params
```

To display a table of the loaded onnx nodes, their associated parameters, set `RUST_LOG=DEBUG` or run:

```bash
cargo run --release --bin ezkl -- table -M ./examples/onnx/1l_relu/network.onnx

```


#### verifying with the EVM ◊

Note that the above prove and verify stats can also be run with an EVM verifier. This can be done by generating a verifier smart contract after generating the proof.

(run the following commands after calling `gen-params` and `setup`, as detailed above).

```bash
# gen proof
ezkl --bits=16 -K=17 prove -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --pk-path 1l_relu.pk --params-path=kzg.params --transcript=evm --circuit-params-path=circuit.params
```
```bash
# gen evm verifier
ezkl -K=17 --bits=16 create-evm-verifier --deployment-code-path 1l_relu.code --params-path=kzg.params --vk-path 1l_relu.vk --sol-code-path 1l_relu.sol --circuit-params-path=circuit.params
```
```bash
# Verify (EVM)
ezkl -K=17 --bits=16 verify-evm --proof-path 1l_relu.pf --deployment-code-path 1l_relu.code
```

Note that the `.sol` file above can be deployed and composed with other Solidity contracts, via a `verify()` function. Please read [this document](https://hackmd.io/QOHOPeryRsOraO7FUnG-tg) for more information about the interface of the contract, how to obtain the data needed for its function parameters, and its limitations.

The above pipeline can also be run using [proof aggregation](https://ethresear.ch/t/leveraging-snark-proof-aggregation-to-achieve-large-scale-pbft-based-consensus/11588) to reduce proof size and verifying times, so as to be more suitable for EVM deployment. A sample pipeline for doing so would be:

```bash
# Generate a new 2^20 SRS
ezkl -K=20 gen-srs --params-path=kzg.params
```

```bash
# setup
ezkl --bits=16 -K=17 setup -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --pk-path 1l_relu.pk --vk-path 1l_relu.vk --params-path=kzg.params --circuit-params-path=circuit.params
```


```bash
# Single proof -> single proof we are going to feed into aggregation circuit. (Mock)-verifies + verifies natively as sanity check
ezkl -K=17 --bits=16 prove --transcript=poseidon --strategy=accum -D ./examples/onnx/1l_relu/input.json -M ./examples/onnx/1l_relu/network.onnx --proof-path 1l_relu.pf --params-path=kzg.params  --vk-path=1l_relu.vk --circuit-params-path=circuit.params
```

```bash
# Aggregate -> generates aggregate proof and also (mock)-verifies + verifies natively as sanity check
ezkl -K=20 --bits=16 aggregate --app-logrows=17 --transcript=evm --circuit-params-paths=circuit.params --aggregation-snarks=1l_relu.pf --aggregation-vk-paths 1l_relu.vk --vk-path aggr_1l_relu.vk --proof-path aggr_1l_relu.pf --params-path=kzg.params
```

```bash
# Generate verifier code -> create the EVM verifier code
ezkl -K=17 --bits=16 create-evm-verifier-aggr --deployment-code-path aggr_1l_relu.code --params-path=kzg.params --vk-path aggr_1l_relu.vk
```

```bash
# Verify (EVM) ->
ezkl -K=17 --bits=16 verify-evm --proof-path aggr_1l_relu.pf --deployment-code-path aggr_1l_relu.code
```

Also note that this may require a local [solc](https://docs.soliditylang.org/en/v0.8.17/installing-solidity.html) installation, and that aggregated proof verification in Solidity is not currently supported.

For both pipelines the resulting verifier can be deployed to an EVM instance (mainnet or otherwise !) using the `deploy-verifier-evm` command:

```bash
Deploys an EVM verifier

Usage: ezkl deploy-verifier-evm [OPTIONS] --secret <SECRET> --rpc-url <RPC_URL>

Options:
  -S, --secret <SECRET>
          The path to the wallet mnemonic
  -U, --rpc-url <RPC_URL>
          RPC Url
      --deployment-code-path <DEPLOYMENT_CODE_PATH>
          The path to the desired EVM bytecode file (optional), either set this or sol_code_path
      --sol-code-path <SOL_CODE_PATH>
          The path to output the Solidity code (optional) supercedes deployment_code_path in priority
  -h, --help
          Print help

```

For instance:

```bash
ezkl deploy-verifier-evm -S ./mymnemonic.txt -U myethnode.xyz --deployment-code-path aggr_1l_relu.code
```

You can also send proofs to be verified on deployed contracts using `send-proof`:

```bash
Send a proof to be verified to an already deployed verifier

Usage: ezkl send-proof-evm --secret <SECRET> --rpc-url <RPC_URL> --addr <ADDR> --proof-path <PROOF_PATH>

Options:
  -S, --secret <SECRET>          The path to the wallet mnemonic
  -U, --rpc-url <RPC_URL>        RPC Url
      --addr <ADDR>              The deployed verifier address
      --proof-path <PROOF_PATH>  The path to the proof
  -h, --help                     Print help

```

For instance:

```bash
ezkl send-proof-evm -S ./mymnemonic.txt -U myethnode.xyz --addr 0xFFFF --proof-path my.snark
```


### using pre-generated SRS

Note that you can use pre-generated KZG SRS. These SRS can be converted to a format that is ingestable by the `pse/halo2` prover ezkl uses by leveraging [han0110/halo2-kzg-srs](https://github.com/han0110/halo2-kzg-srs). This repo also contains pre-converted SRS from large projects such as Hermez and the [perpetual powers of tau repo](https://github.com/privacy-scaling-explorations/perpetualpowersoftau). Simply download the pre-converted file locally and point `--params-path` to the file.

> Note: Ensure you download the files in raw format. As this will be more performant and is the serialization format `ezkl` assumes.



### general usage 🔧

> Note: to get the full suite of cli capabilities you'll need to compile `ezkl` with the `render` feature (`cargo build --features render --bin ezkl`). This enables the `render-circuit` command which can create `.png` representations of the compiled circuits. You'll also need to install the `libexpat1-dev` and `libfreetype6-dev` libraries on Debian systems (there are equivalents for MacOS as well).

```bash
Usage: ezkl [OPTIONS] <COMMAND>

Commands:
  table                     Loads model and prints model table
  render-circuit            Renders the model circuit to a .png file. For an overview of how to interpret these plots, see https://zcash.github.io/halo2/user/dev-tools.html
  forward                   Runs a vanilla forward pass, produces a quantized output, and saves it to a .json file
  gen-srs                   Generates a dummy SRS
  mock                      Loads model and input and runs mock prover (for testing)
  aggregate                 Aggregates proofs :)
  prove                     Loads model and data, prepares vk and pk, and creates proof
  create-evm-verifier       Creates an EVM verifier for a single proof
  create-evm-verifier-aggr  Creates an EVM verifier for an aggregate proof
  deploy-verifier           Deploys an EVM verifier
  verify                    Verifies a proof, returning accept or reject
  verify-aggr               Verifies an aggregate proof, returning accept or reject
  verify-evm                Verifies a proof using a local EVM executor, returning accept or reject
  print-proof-hex           Print the proof in hexadecimal
  help                      Print this message or the help of the given subcommand(s)

Options:
  -T, --tolerance <TOLERANCE>          The tolerance for error on model outputs. Set to a usize value
      for abs tolerance, or a float for percent error tolerance [default: 0]
  -S, --scale <SCALE>                  The denominator in the fixed point representation used when quantizing [default: 7]
  -B, --bits <BITS>                    The number of bits used in lookup tables [default: 16]
  -K, --logrows <LOGROWS>              The log_2 number of rows [default: 17]
      --public-inputs                  Flags whether inputs are public
      --public-outputs                 Flags whether outputs are public
      --public-params                  Flags whether params are public
      --pack-base <PACK_BASE>              Base used to pack the public-inputs to the circuit. set ( > 1) to pack instances as a single int. Useful when verifying on the EVM. Note that this will often break for very long inputs. Use with caution, still experimental.  [default: 1]
  -h, --help                           Print help
  -V, --version                        Print version
```

`bits`, `scale`, `tolerance`, and `logrows` have default values. Use the `tolerance` parameter to set the acceptable quantization error on the output. If set to a usize value, the circuit verifies even if the output deviates by the specified absolute value on any dimension. If set to a floating-point value, it represents a percentage error tolerance, and the circuit verifies if the output deviates within the specified percentage. For example, if set to 1.0, the circuit verifies even if the output deviates by 1 percent, and if set to 1, it verifies even if the output deviates by an absolute value of 1. The `prove` and `mock` commands require `-D` and `-M` parameters; if not provided, the CLI will prompt the user to manually enter the path(s).

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

If the above commands get too heavy and it becomes difficult to track parameters across commands; `ezkl` also supports loading global arguments (those specified before a subcommand) from a `.json` file. This can be done using the `RUNARGS` environment variable. For instance:

```bash
RUNARGS=/path/to/args.json ezkl subcommand --subcommand-params...
```
For an example of such a file see `examples/default_run_args.json`:
```json
{
    "tolerance": 0,
    "scale": 7,
    "bits": 11,
    "logrows": 17,
    "public_inputs": false,
    "public_outputs": true,
    "public_params": false,
}
```

Note that command-wide arguments can be specified using the `EZKLCONF` environment variable; which supercedes `RUNARGS` in priority !
This json includes both global level arguments _and_ subcommand specific arguments. Usage is thus as such:
```bash
EZKLCONF=/path/to/fullconfig.json ezkl
```


----------------------


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
----------------------


## onnx examples

This repository includes onnx example files as a submodule for testing out the cli.

If you want to add a model to `examples/onnx`, open a PR creating a new folder within `examples/onnx` with a descriptive model name. This folder should contain:
- an `input.json` input file, with the fields expected by the  [ezkl](https://github.com/zkonduit/ezkl) cli.
- a `network.onnx` file representing the trained model
- a `gen.py` file for generating the `.json` and `.onnx` files following the general structure of `examples/tutorial/tutorial.py`.


TODO: add associated python files in the onnx model directories.

----------------------


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


## Compiling to wasm 💾

The cli can also be compiled to for `wasm32-wasi` target (browser bindings with `wasm32-unknown-unknown` coming soon). To do so first ensure that [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) is installed.


You can then run:
```bash
rustup target add wasm32-wasi

wasm-pack build --bin ezkl --target wasm32-wasi
```
>Note: On Mac you may need to install llvm and clang using homebrew then explicitly set the `CC` and `AR` environment variables. For instance: `AR=/opt/homebrew/opt/llvm/bin/llvm-ar CC=/opt/homebrew/opt/llvm/bin/clang wasm-pack build --bin ezkl --target wasm32-wasi`

You can then run the compiled `.wasm` file as you would the normal cli detailed above (just not the `EVM` related commands), by using [wasmtime](https://docs.wasmtime.dev/cli-install.html).

```bash
wasmtime './target/wasm32-wasi/release/ezkl.wasm' -- --help
```
----------------------

## python bindings
Python bindings are built for `ezkl` using [PyO3](https://pyo3.rs) and [Maturin](https://github.com/PyO3/maturin). This is done so to allow users of `ezkl` to leverage on the rich Data Science ecosystem that Python has instead of using Rust only.

### production
Production Python bindings are made available via [pyezkl](https://github.com/zkonduit/pyezkl).


### development
To test the developmental Python bindings you will need to install [Python3](https://realpython.com/installing-python/). `ezkl` only supports version of python where `python >=3.7`.

Once python is installed setup a virtual environment and install `maturin`
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

You can now build the package for development and enable python bindings.
```bash
maturin develop --features python-bindings
```

Once done you will be able to access `ezkl_lib` as a python import as follows.
```python
import ezkl_lib
```

You may test if the existing build is working properly.
```
pytest
```

The list of python functions that can be accessed are found within `src/python.rs`

We also include a full run through and tutorial on using the bindings within a jupyter notebook in `examples/notebook`.
