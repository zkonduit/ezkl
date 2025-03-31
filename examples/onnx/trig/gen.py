from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Circuit(nn.Module):
    def __init__(self):
        super(Circuit, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.softplus(x)
        x = torch.cos(x)
        x = torch.sin(x)
        x = torch.tan(x)
        x = torch.acos(x)
        x = torch.asin(x)
        x = torch.atan(x)
        # x = torch.cosh(x)
        # x = torch.sinh(x)
        x = torch.tanh(x)
        # x = torch.acosh(x)
        # x = torch.asinh(x)
        # x = torch.atanh(x)
        return (-x).abs().sign()


circuit = Circuit()
export(circuit, input_shape=[3, 2, 3])
