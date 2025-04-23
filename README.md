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

In the backend we use the collaboratively-developed [Halo2](https://github.com/privacy-scaling-explorations/halo2) as a proof system.

The generated proofs can then be verified with much less computational resources, including on-chain (with the Ethereum Virtual Machine), in a browser, or on a device. 

## Competition on Subnet 2: Age Verification Optimization

In collaboration with Inference Labs, we're launching a competition to optimize the age verification application on their Subnet 2 Accelerator Program, part of the Bittensor network. This implementation includes Metal GPU acceleration for Apple Silicon devices.

### Features and Optimizations

- **Metal GPU Acceleration**: Leverages Metal to significantly improve MSM operations on Apple Silicon
- **Memory Optimization**: Reduces memory usage by up to 40% through custom data structures and optimized algorithms
- **Performance Tuning**: Circuit-specific optimizations for the age verification model
- **iOS Support**: Full integration with iOS for on-device private age verification

### Getting Started with the Competition

To participate in the competition:

```bash
# Clone the repository
git clone https://github.com/zkonduit/ezkl.git
cd ezkl

# Build with Metal acceleration support
cargo build --release --features macos-metal

# Run the age verification optimizer
cargo run --release --example age_verification_optimize -- path/to/age.onnx path/to/input.json
```

For detailed instructions on the competition, check out [QUICKSTART.md](https://github.com/inference-labs-inc/omron-subnet/blob/main/docs/competitions/QUICKSTART.md).

## General Information

- If you have any questions, we'd love for you to open up a discussion topic in [Discussions](https://github.com/zkonduit/ezkl/discussions). Alternatively, you can join the ✨[EZKL Community Telegram Group](https://t.me/+QRzaRvTPIthlYWMx)💫.

- For more technical writeups and details check out our [blog](https://blog.ezkl.xyz/).

- To see what you can build with ezkl, check out [cryptoidol.tech](https://cryptoidol.tech/) where ezkl is used to create an AI that judges your singing ... forever.

----------------------

### Getting Started ⚙️

The easiest way to get started is to try out a notebook. 

#### Python
Install the python bindings by calling.

```bash
pip install ezkl
```
Or for the GPU:

```bash
pip install ezkl-gpu
```

Google Colab Example to learn how you can train a neural net and deploy an inference verifier onchain for use in other smart contracts. [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zkonduit/ezkl/blob/main/examples/notebooks/ezkl_demo.ipynb) 


More notebook tutorials can be found within `examples/notebooks`.

#### CLI
Install the CLI
``` shell
curl https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash
```

https://user-images.githubusercontent.com/45801863/236771676-5bbbbfd1-ba6f-418a-902e-20738ce0e9f0.mp4

For more details visit the [docs](https://docs.ezkl.xyz). The CLI is faster than Python, as it has less overhead. For even more speed and convenience, check out the [remote proving service](https://ei40vx5x6j0.typeform.com/to/sFv1oxvb), which feels like the CLI but is backed by a tuned cluster.

Build the auto-generated rust documentation and open the docs in your browser locally. `cargo doc --open`

#### In-browser EVM Verifier

As an alternative to running the native Halo2 verifier as a WASM binding in the browser, you can use the in-browser EVM verifier. The source code of which you can find in the `in-browser-evm-verifier` directory and a README with instructions on how to use it.


### Building the Project 🔨

#### Rust CLI

You can install the library from source

```bash
cargo install --locked --path .
```

`ezkl` now auto-manages solc installation for you.

#### Building Python Bindings
Python bindings exists and can be built using `maturin`. You will need `rust` and `cargo` to be installed.

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
maturin develop --release --features python-bindings
# dependencies specific to tutorials
pip install torch pandas numpy seaborn jupyter onnx kaggle py-solc-x web3 librosa tensorflow keras tf2onnx
```

### GPU Acceleration

If you have access to NVIDIA GPUs, you can enable acceleration by building with the feature `icicle` and setting the following environment variable:

```sh
export ENABLE_ICICLE_GPU=true
```

GPU acceleration is provided by [Icicle](https://github.com/ingonyama-zk/icicle)

To go back to running with CPU, the previous environment variable must be **unset** instead of being switch to a value of false:

```sh
unset ENABLE_ICICLE_GPU
```

**NOTE:** Even with the above environment variable set, icicle is disabled for circuits where k <= 8. To change the value of `k` where icicle is enabled, you can set the environment variable `ICICLE_SMALL_K`.

### Apple Silicon (Metal) Acceleration

For Apple Silicon devices (M1, M2, M3), you can enable Metal GPU acceleration by building with:

```sh
cargo build --release --features macos-metal
```

For iOS devices, use:

```sh
cargo build --target aarch64-apple-ios --release --features ios-metal
```

Metal acceleration significantly improves performance for MSM operations, particularly important for the age verification circuit in the Subnet 2 competition.

### Contributing 🌎

If you're interested in contributing and are unsure where to start, reach out to one of the maintainers:

* dante (alexander-camuto)
* jason (jasonmorton)

More broadly:

- See currently open issues for ideas on how to contribute.

- For PRs we use the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) naming convention.

- To report bugs or request new features [create a new issue within Issues](https://github.com/zkonduit/ezkl/issues) to inform the greater community.


Any contribution intentionally submitted for inclusion in the work by you shall be licensed to Zkonduit Inc. under the terms and conditions specified in the [CLA](https://github.com/zkonduit/ezkl/blob/main/cla.md), which you agree to by intentionally submitting a contribution. In particular, you have the right to submit the contribution and we can distribute it, among other terms and conditions. 


### Audits & Security

[v21.0.0](https://github.com/zkonduit/ezkl/releases/tag/v21.0.0) has been audited by Trail of Bits, the report can be found [here](https://github.com/trailofbits/publications/blob/master/reviews/2025-03-zkonduit-ezkl-securityreview.pdf).

> NOTE: Because operations are quantized when they are converted from an onnx file to a zk-circuit, outputs in python and ezkl may differ slightly.


Check out `docs/advanced_security` for more advanced information on potential threat vectors that are specific to zero-knowledge inference, quantization, and to machine learning models generally.


### No Warranty

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c) 2025 Zkonduit Inc. 

