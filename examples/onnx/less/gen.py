from torch import nn
import torch
import json
import numpy as np


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, w, x):
        return torch.less(w, x)


circuit = MyModel()


w = torch.rand(1, 4)
x = torch.rand(1, 4)

torch.onnx.export(circuit, (w, x), "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input', 'input1'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'input1': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})


d = ((w).detach().numpy()).reshape([-1]).tolist()
d1 = ((x).detach().numpy()).reshape([-1]).tolist()


data = dict(
    input_data=[d, d1],
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
