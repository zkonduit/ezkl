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

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_public_network_output.ipynb) 

> "I ran my private neural network on some public data and it produced this output"

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_public_input_output.ipynb) 

> "I correctly ran this publicly available neural network on some public data and it produced this output"

[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/simple_demo_all_public.ipynb) 

In the backend we use [Halo2](https://github.com/privacy-scaling-explorations/halo2) as a proof system.

The generated proofs can then be used on-chain to verify computation, only the Ethereum Virtual Machine (EVM) is supported at the moment.

- If you have any questions, we'd love for you to open up a discussion topic in [Discussions](https://github.com/zkonduit/ezkl/discussions). Alternatively, you can join the ✨[EZKL Community Telegram Group](https://t.me/+QRzaRvTPIthlYWMx)💫.

- For more technical writeups and details check out our [blog](https://blog.ezkl.xyz/).

- To see what you can build with ezkl, check out [cryptoidol.tech](https://cryptoidol.tech/) where ezkl is used to create an AI that judges your singing ... forever.

----------------------

### getting started ⚙️

#### Python
Install the python bindings by calling.

```bash
pip install ezkl
```
Google Colab Example to learn how you can train a neural net and deploy an inference verifier onchain for use in other smart contracts. [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/ezkl_demo.ipynb) 


More notebook tutorials can be found within `examples/notebooks`.

#### CLI
Install the CLI
```bash
curl https://hub.ezkl.xyz/install_ezkl_cli.sh | bash
```

https://user-images.githubusercontent.com/45801863/236771676-5bbbbfd1-ba6f-418a-902e-20738ce0e9f0.mp4

For more details visit the [docs](https://docs.ezkl.xyz).

Build the auto-generated rust documentation and open the docs in your browser locally. `cargo doc --open`


### building the project 🔨

#### Rust CLI

You can install the library from source

```bash
cargo install --locked --path .
```

You will need a functioning installation of `solc` in order to run `ezkl` properly.
[solc-select](https://github.com/crytic/solc-select) is recommended.
Follow the instructions on [solc-select](https://github.com/crytic/solc-select) to activate `solc` in your environment.


#### building python bindings
Python bindings exists and can be built using `maturin`. You will need `rust` and `cargo` to be installed.

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin develop --release --features python-bindings
# dependencies specific to tutorials
pip install torch pandas numpy seaborn jupyter onnx kaggle py-solc-x web3 librosa tensorflow keras tf2onnx
```

### repos

The EZKL project has several libraries and repos. 

| Repo | Description |
| --- | --- |
| [@zkonduit/ezkl](https://github.com/zkonduit/ezkl) | the main ezkl repo in rust with wasm and python bindings |
| [@zkonduit/ezkljs](https://github.com/zkonduit/ezkljs) | typescript and javascript tooling to help integrate ezkl into web apps |

----------------------

### contributing 🌎

If you're interested in contributing and are unsure where to start, reach out to one of the maintainers:

* dante (alexander-camuto)
* jason (jasonmorton)

More broadly:

- See currently open issues for ideas on how to contribute.

- For PRs we use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) naming convention.

- To report bugs or request new features [create a new issue within Issues](https://github.com/zkonduit/ezkl/issues) to inform the greater community.


Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you shall be licensed to Zkonduit Inc. under the terms and conditions specified in the [CLA](https://github.com/zkonduit/ezkl/blob/main/cla.md), which you agree to by intentionally submitting a contribution. In particular, you have the right to submit the contribution and we can distribute it under the Apache 2.0 license, among other terms and conditions. 

### no security guarantees

Ezkl is unaudited, beta software undergoing rapid development. There may be bugs. No guarantees of security are made and it should not be relied on in production.

> NOTE: Because operations are quantized when they are converted from an onnx file to a zk-circuit, outputs in python and ezkl may differ slightly. 


