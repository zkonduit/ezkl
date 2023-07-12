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

In the backend we use [Halo2](https://github.com/privacy-scaling-explorations/halo2) as a proof system.

The generated proofs can then be used on-chain to verify computation, only the Ethereum Virtual Machine (EVM) is supported at the moment.


- If you have any questions, we'd love for you to open up a discussion topic in [Discussions](https://github.com/zkonduit/ezkl/discussions). Alternatively, you can join the ✨[EZKL Community Telegram Group](https://t.me/+QRzaRvTPIthlYWMx)💫.


### resources 📖

|  |  |
| --- | --- |
| [docs](https://docs.ezkl.xyz ) | the official ezkl docs page |
| [colab notebook demo](https://colab.research.google.com/drive/1XuXNKqH7axOelZXyU3gpoTOCvFetIsKu?usp=sharing) | demo of ezkl python bindings on google's colab
| `cargo doc --open` | compile and open the docs in your default browser locally |

#### tutorials 

You can find a range of python based tutorials in the `examples/notebooks` section. These all assume you have the `ezkl` python library installed. If you want the bleeding edge version of the library, you can install it from the `main` branch with:

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin develop --release --features python-bindings
# dependencies specific to tutorials
pip install torch pandas numpy seaborn jupyter onnx kaggle py-solc-x web3 librosa
```


----------------------

## Getting Started ⚙️



https://user-images.githubusercontent.com/45801863/236771676-5bbbbfd1-ba6f-418a-902e-20738ce0e9f0.mp4


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



### Repos

The EZKL project has several libraries and repos. 

| Repo | Description |
| --- | --- |
| [@zkonduit/ezkl](https://github.com/zkonduit/ezkl) | the main ezkl repo in rust with wasm and python bindings |
| [@zkonduit/pyezkl](https://github.com/zkonduit/pyezkl) | additional functionality written in python to support data science and zero knowledge applications |

----------------------

## Contributing 🌎

If you're interested in contributing and are unsure where to start, reach out to one of the maintainers:

* dante (alexander-camuto)
* jason (jasonmorton)

More broadly:

- See currently open issues for ideas on how to contribute.

- For PRs we use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) naming convention.

- To report bugs or request new features [create a new issue within Issues](https://github.com/zkonduit/ezkl/issues) to inform the greater community.


Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be licensed to Zkonduit Inc. under the terms and conditions specified in the [CLA](https://github.com/zkonduit/ezkl/blob/main/cla.md), which you agree to by intentionally submitting a contribution. In particular, you have the right to submit the contribution and we can distribute it under the Apache 2.0 license, among other terms and conditions. 

## No security guarantees

Ezkl is unaudited, beta software undergoing rapid development. There may be bugs. No guarantees of security are made and it should not be relied on in production.

