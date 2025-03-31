from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()
        self.convtranspose = nn.ConvTranspose2d(3, 3, (5, 5), stride=2, padding=2, output_padding=1)

    def forward(self, x):
        y = self.convtranspose(x)
        return y


circuit = Circuit()
export(circuit, input_shape=[3, 5, 5])
