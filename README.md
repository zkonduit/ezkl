# EZKL

[![Test](https://github.com/zkonduit/ezkl/workflows/Rust/badge.svg)](https://github.com/zkonduit/ezkl/actions?query=workflow%3ARust)

`ezkl` is a library and command-line tool for doing inference for deep learning models and other computational graphs in a zk-snark. The backend uses Halo2.  Typically the input image is private advice, the model parameters are public or private, and the last layer is the public input (instance column) which will be sent to the verifier along with the proof. Other configurations are also possible.

Note that the library requires a nightly version of the rust toolchain. You can change the default toolchain by running:

```bash
rustup override set nightly         
```

## docs

Use `cargo doc --open` to compile and open the docs in your default browser.

## command line interface

The `ezkl` cli provides a simple interface to load Onnx files, which represent graphs of operations (such as neural networks), and convert them into a Halo2 circuit, then run a proof (given a public input).

Usage:

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
  -S, --scale <SCALE>      The denominator in the fixed point representation used when quantizing [default: 7]
  -B, --bits <BITS>        The number of bits used in lookup tables [default: 14]
  -K, --logrows <LOGROWS>  The log_2 number of rows [default: 16]
  -h, --help               Print help information
  -V, --version            Print version information
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
    "input_shape": [[3, 3, ...]], // 2D array of integers which represents the shapes of model inputs (excluding batch size)
    "public_inputs": [[1.0, 5.0, 6.3 ...]], // 2D arrays of floats which represents the public inputs (model outputs for now)
}
```

For examples of such files see `examples/onnx_models`.

To run a simple example using the cli:

```bash
cargo run --release --bin ezkl -- mock -D ./examples/onnx_models/ff_input.json -M ./examples/onnx_models/ff.onnx
```

To display a table of loaded Onnx nodes, and their associated parameters, set `RUST_LOG=DEBUG` or run:

```bash
cargo run --release --bin ezkl -- table -M ./examples/onnx_models/ff.onnx

```
### verifying with the EVM

Note that `fullprove` can also be run with an EVM verifier. In this case we use KZG commitments, rather than the default IPA commitments, and we need to pass the `evm` feature flag to conditionally compile the requisite [foundry_evm](https://github.com/foundry-rs/foundry) dependencies. Using `foundry_evm` we spin up a local EVM executor and verify the generated proof. In future releases we'll create a simple pipeline for deploying to EVM based networks.
Example:

```bash
cargo run  --release --features evm --bin ezkl fullprove  -D ./examples/onnx_models/ff_input.json -M ./examples/onnx_models/ff.onnx --pfsys kzg
```

## benchmarks

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

## examples

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

We also provide onnx model files and their corresponding input json files in `examples/onnx_models`. These can be run using the cli commands listed above.

## python tutorial

You can easily create an Onnx file using `pytorch`.
To get started install [miniconda](https://docs.conda.io/en/latest/miniconda.html) for your system. From there create an new evironment:

```bash
conda create -n ezkl python=3.9
```

Activate your newly created environment and install the requisite dependencies:

```bash
conda activate ezkl; pip install torch numpy;             
```

We're gonna to create a (relatively) complex Onnx graph that takes in 3 inputs `x`, `y`, and `z` and produces two outputs that we can verify against public inputs.

To do so create a `onnx_graph.py` file and load the following depenencies:

```python
import io
import numpy as np
from torch import nn
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import json

```

We can now define our computational graph as a pytorch `nn.Module` which will be as follows:

```python
class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(3, 3, (2, 2), 1, 2)

        self._initialize_weights()

    def forward(self, x, y, z):
        x =  self.sigmoid(self.conv(y@x**2 + (x) - (self.relu(z)))) + 2
        return (x, self.relu(z) / 3)


    def _initialize_weights(self):
        init.orthogonal_(self.conv.weight)
```

As noted above this graph takes in 3 inputs and produces 2 outputs. We can now define our main function which instantiates an instance of circuit and saves it to an Onnx file.

```python
def main():
    torch_model = Circuit()
    # Input to the model
    shape = [3, 2, 2]
    x = 0.1*torch.rand(1,*shape, requires_grad=True)
    y = 0.1*torch.rand(1,*shape, requires_grad=True)
    z = 0.1*torch.rand(1,*shape, requires_grad=True)
    torch_out = torch_model(x, y, z)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      (x,y,z),                   # model input (or a tuple for multiple inputs)
                      "network.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    d = ((x).detach().numpy()).reshape([-1]).tolist()
    dy = ((y).detach().numpy()).reshape([-1]).tolist()
    dz = ((z).detach().numpy()).reshape([-1]).tolist()


    data = dict(input_shapes = [shape, shape, shape],
                input_data = [d, dy, dz],
                public_inputs = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump( data, open( "input.json", 'w' ) )

if __name__ == "__main__":
    main()
```

You can now run the file to generate a `.onnx` file. Note that this also create the required input json file, whereby we use the outputs of the pytorch model as the public inputs to the circuit.
If you run the following command on the generated files:

```bash
cargo run --bin ezkl -- --scale 4 --bits 16 -K 17 table  -M ./network.onnx
```

you should see the following table being displayed. This is a tabular representation of the Onnx graph, with some additional information required for circuit construction (like the number of advices to use, the fixed point representation denominator at the operation's input and output). You should see all the operations we created in `Circuit(nn.Module)` represented. Nodes 14 and 17 correspond to the output nodes here.

```bash
| node           | output_max | min_cols | in_scale | out_scale | is_output | const_value | inputs     | in_dims   | out_dims     | idx  | Bucket |
| -------------- | ---------- | -------- | -------- | --------- | --------- | ----------- | ---------- | --------- | ------------ | ---- | ------ |
| Source         | 256        | 1        | 4        | 4         | false     |             |            |           | [3, 2, 2]    | 0    | 0      |
| Source         | 256        | 1        | 4        | 4         | false     |             |            |           | [3, 2, 2]    | 1    | 0      |
| Source         | 256        | 1        | 4        | 4         | false     |             |            |           | [3, 2, 2]    | 2    | 0      |
| conv.weight    | 5          | 1        | 4        | 4         | false     | [4...]      |            |           | [3, 3, 2, 2] | 3    |        |
| conv.bias      | 1024       | 1        | 4        | 12        | false     | [-1024...]  |            |           | [3]          | 4    |        |
| power.exp      | 32         | 1        | 4        | 4         | false     | [32...]     |            |           | [1]          | 5    |        |
| Add            | 262144     | 31       | 8        | 8         | false     |             | [7, 0]     | [3, 2, 2] | [3, 2, 2]    | 8    | 0      |
| Relu           | 256        | 12       | 4        | 4         | false     |             | [2]        | [3, 2, 2] | [3, 2, 2]    | 9    | 1      |
| Sub            | 524288     | 31       | 8        | 8         | false     |             | [8, 9]     | [3, 2, 2] | [3, 2, 2]    | 10   | 1      |
| ConvHir        | 10485760   | 67       | 8        | 12        | false     |             | [10, 3, 4] | [3, 2, 2] | [3, 5, 5]    | 11   | 1      |
| Sigmoid        | 16         | 75       | 12       | 4         | false     |             | [11]       | [3, 5, 5] | [3, 5, 5]    | 12   | 2      |
| add.const      | 32         | 1        | 4        | 4         | false     | [32...]     |            |           | [1]          | 13   |        |
| Add            | 64         | 92       | 4        | 4         | true      |             | [12, 13]   | [3, 5, 5] | [3, 5, 5]    | 14   | 2      |
| Relu           | 256        | 12       | 4        | 4         | false     |             | [2]        | [3, 2, 2] | [3, 2, 2]    | 15   | 1      |
| div.const      | 48         | 1        | 4        | 4         | false     | [48...]     |            |           | [1]          | 16   |        |
| Div            | 85.333336  | 12       | 4        | 4         | true      |             | [15]       | [3, 2, 2] | [3, 2, 2]    | 17   | 2      |
```

From there we can run proofs on the generated files, but note that because of quantization errors the public inputs may need to be tweaked to match the output of the circuit and generate a valid proof. You can also express a tolerance to such errors using the `tolerance` flag (which we use below). The types of claims we can make with the setup of this tutorial are ones such as: "I ran my private model on data and produced the expected outputs (as dictated by the public inputs to the circuit)".

``` bash
 RUST_LOG=debug cargo run --bin ezkl -- --tolerance 2 --scale 4 --bits 16 -K 17 mock  -D ./input.json -M ./network.onnx
```
