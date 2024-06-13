import io
import numpy as np
from torch import nn
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import json


class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(3, 3, (2, 2), 1, 2)

        self._initialize_weights()

    def forward(self, x, y, z):
        x = self.sigmoid(self.conv(y@x**2 + (x) - (self.relu(z)))) + 2
        return (x, self.relu(z) / 3)

    def _initialize_weights(self):
        init.orthogonal_(self.conv.weight)


def main():
    torch_model = Circuit()
    # Input to the model
    shape = [3, 2, 2]
    x = 0.1*torch.rand(1, *shape, requires_grad=True)
    y = 0.1*torch.rand(1, *shape, requires_grad=True)
    z = 0.1*torch.rand(1, *shape, requires_grad=True)
    torch_out = torch_model(x, y, z)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      # model input (or a tuple for multiple inputs)
                      (x, y, z),
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
    dy = ((y).detach().numpy()).reshape([-1]).tolist()
    dz = ((z).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_shapes=[shape, shape, shape],
                input_data=[d, dy, dz],
                output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump(data, open("input.json", 'w'))


if __name__ == "__main__":
    main()
