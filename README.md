# Halo2 DL [![Test](https://github.com/jasonmorton/halo2deeplearning/workflows/Test/badge.svg)](https://github.com/jasonmorton/halommrl/actions?query=workflow%3ATest)

MNIST in a zk-snark

This is a proof-of-concept implementation of inference for deep learning models in a zk-snark using Halo2. 2d convolution, fully connected (affine) layers, and nonlinearities such as ReLu and sigmoid are implemented.  The input image and model parameters are provided as private advice and the last layer is the public input (instance column). Other configurations are also possible.

We give an example of proving inference with a model that achieves 97.5% accuracy on MNIST.



