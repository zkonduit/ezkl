import io
import numpy as np
from torch import nn
import torch.onnx
import torch
import torch.nn as nn
import torch.nn.init as init
import json


class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()

    def forward(self, w, x, y):

        return torch.round(w), torch.floor(x), torch.ceil(y)


def main():
    torch_model = Circuit()
    # Input to the model
    shape = [3, 2, 3]
    w = 0.1*torch.rand(1, *shape, requires_grad=True)
    x = 0.1*torch.rand(1, *shape, requires_grad=True)
    y = 0.1*torch.rand(1, *shape, requires_grad=True)
    torch_out = torch_model(w, x, y)
    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      # model input (or a tuple for multiple inputs)
                      (w, x, y),
                      # where to save the model (can be a file or file-like object)
                      "network.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=16,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['w', 'x', 'y'],   # the model's input names
                      # the model's output names
                      output_names=['output_w', 'output_x', 'output_y'],
                      dynamic_axes={'x': {0: 'batch_size'},    # variable length axes
                                    'y': {0: 'batch_size'},
                                    'w': {0: 'batch_size'},
                                    'output_w': {0: 'batch_size'},
                                    'output_x': {0: 'batch_size'},
                                    'output_y': {0: 'batch_size'}
                                    })

    dw = ((w).detach().numpy()).reshape([-1]).tolist()
    dx = ((x).detach().numpy()).reshape([-1]).tolist()
    dy = ((y).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_shapes=[shape, shape, shape, shape],
                input_data=[dw, dx, dy],
                output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

    # Serialize data into file:
    json.dump(data, open("input.json", 'w'))


if __name__ == "__main__":
    main()
