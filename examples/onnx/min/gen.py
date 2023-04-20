from torch import nn
from ezkl import export
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return [torch.min(x)]


circuit = Model()
export(circuit, input_shape=[3, 2, 2])
