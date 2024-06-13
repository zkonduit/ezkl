import torch
from torch import nn
from ezkl import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = x.reshape([-1, 6])
        return x


circuit = MyModel()
export(circuit, input_shape=[1, 3, 2])
