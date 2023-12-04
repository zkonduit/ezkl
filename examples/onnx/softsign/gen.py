from torch import nn
import torch
import json
import numpy as np


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):          
        m = nn.Softsign()(x)
        
        return m 


circuit = MyModel()

x = torch.empty(1, 8).uniform_(0, 1)

out = circuit(x)

print(out)

torch.onnx.export(circuit, x, "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})


d1 = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(
    input_data=[d1],
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
