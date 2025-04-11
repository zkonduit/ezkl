from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return torch.pow(x, -0.1)


circuit = MyModel()
export(circuit, input_shape = [4], opset_version=15, include_output=False)
