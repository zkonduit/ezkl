from torch import nn
import torch
import json


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return [torch.where(x >= 0.0, 3.0, 5.0)]


circuit = MyModel()


x = torch.randint(1, (1, 64))

torch.onnx.export(circuit, x, "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(
    input_data=[d],
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
