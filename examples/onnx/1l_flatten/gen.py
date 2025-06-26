from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)


circuit = Circuit()
export(circuit, input_shape=[3, 2, 3])
