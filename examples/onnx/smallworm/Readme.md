## The worm 

This is an onnx file for a [WormVAE](https://github.com/TuragaLab/wormvae?tab=readme-ov-file) model, which is a VAE / latent-space representation of the C. elegans connectome.

The model "is a large-scale latent variable model with a very high-dimensional latent space
consisting of voltage dynamics of 300 neurons over 5 minutes of time at the simulation frequency
of 160 Hz. The generative model for these latent variables is described by stochastic differential
equations modeling the nonlinear dynamics of the network activity." (see [here](https://openreview.net/pdf?id=CJzi3dRlJE-)). 

In effect this is a generative model for a worm's voltage dynamics, which can be used to generate new worm-like voltage dynamics given previous connectome state.

Using ezkl you can create a zk circuit equivalent to the wormvae model, allowing you to "prove" execution of the worm model. If you're feeling particularly adventurous, you can also use the zk circuit to generate new worm-state that can be verified on chain. 

To do so you'll first want to fetch the files using git-lfs (as the onnx file is too large to be stored in git). 

```bash
git lfs fetch --all
```

You'll then want to use the usual ezkl loop to generate the zk circuit. We recommend using fixed visibility for the model parameters, as the model is quite large and this will prune the circuit significantly. 

```bash
ezkl gen-settings --param-visibility=fixed
cp input.json calibration.json
ezkl calibrate-settings
ezkl compile-circuit
ezkl gen-witness
ezkl prove
```

You might also need to aggregate the proof to get it to fit on chain.

```bash
ezkl aggregate
```

You can then create a smart contract that verifies this aggregate proof

```bash
ezkl create-evm-verifier-aggr
```

This can then be deployed on the chain of your choice.


> Note: the model is large and thus we recommend a machine with at least 512GB of RAM to run the above commands. If you're ever compute constrained you can always use the lilith service to generate the zk circuit. Message us on discord or telegram for more details :) 

