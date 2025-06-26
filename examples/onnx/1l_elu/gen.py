from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()
        self.layer = nn.ELU()

    def forward(self, x):
        return self.layer(x)


circuit = Circuit()
export(circuit, input_shape=[3, 6, 6])
