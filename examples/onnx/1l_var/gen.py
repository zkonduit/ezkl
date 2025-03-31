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
        return [torch.var(x, unbiased=False, dim=[1,2])]


circuit = MyModel()
export(circuit, input_shape = [1,3,3])
