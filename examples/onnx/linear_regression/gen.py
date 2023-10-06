import os
import torch
import ezkl
import json
from hummingbird.ml import convert


# here we create and (potentially train a model)

# make sure you have the dependencies required here already installed
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)

circuit = convert(reg, "torch", X[:1]).model


# export to onnx format
# !!!!!!!!!!!!!!!!! This will flash a warning but it is fine !!!!!!!!!!!!!!!!!!!!!

# Input to the model
shape = X.shape[1:]
x = torch.rand(1, *shape, requires_grad=True)
torch_out = circuit(x)
# Export the model
torch.onnx.export(circuit,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_shapes=[shape],
            input_data=[d],
            output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
