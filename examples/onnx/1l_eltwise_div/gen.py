from torch import nn
from ezkl import export


class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

    def forward(self, x):
        return x/(2*x)


circuit = Circuit()
export(circuit, input_shape=[1])
