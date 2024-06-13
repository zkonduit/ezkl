from torch import nn
import torch
import json
import numpy as np


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self):
        return torch.arange(0, 10, 2)


circuit = MyModel()



torch.onnx.export(circuit, (), "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  output_names=['output'],  # the model's output names
                )




data = dict(
    input_data=[[]],
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
