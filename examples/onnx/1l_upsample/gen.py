from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        y = self.upsample(x)
        return y


circuit = Circuit()
export(circuit, input_shape = [3, 5, 5], opset_version=11)
