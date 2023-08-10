import random
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import json


model = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
x = torch.randn(1, 3)  # make a sequence of length 5

print(x)

# Flips the neural net into inference mode
model.eval()
model.to('cpu')

# Export the model
torch.onnx.export(model,               # model being run
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

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data_json = dict(input_data=[data_array])

print(data_json)

# Serialize data into file:
json.dump(data_json, open("input.json", 'w'))
