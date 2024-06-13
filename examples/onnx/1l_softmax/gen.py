import torch
from torch import nn
from ezkl import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.layer = nn.Softmax(dim=1)

    def forward(self, x):
        return self.layer(x)


circuit = MyModel()
export(circuit, input_shape=[3])
