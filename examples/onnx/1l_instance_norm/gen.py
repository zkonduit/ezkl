from torch import nn
from ezkl import export
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.layer = nn.InstanceNorm2d(3).eval()

    def forward(self, x):
        return [self.layer(x)]


circuit = MyModel()
export(circuit, input_shape=[3, 2, 2])
