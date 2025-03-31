from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self):
        return torch.arange(0, 10, 2)


circuit = MyModel()
export(circuit, include_input=False, opset_version=15)
