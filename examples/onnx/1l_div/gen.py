from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Model(nn.Module):
    def __init__(self, inplace=False):
        super(Model, self).__init__()

    def forward(self, x):
        return x/ 10


circuit = Model()
export(circuit, input_shape = [1])
