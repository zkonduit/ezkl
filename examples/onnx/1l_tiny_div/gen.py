from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

    def forward(self, x):
        return x/ 10000


circuit = Circuit()
export(circuit, input_shape=[8], opset_version=17, include_output=False)
