from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.BatchNorm2d(3)

    def forward(self, x):
        return self.layer(x)


circuit = Model()
export(circuit, input_shape=[3, 2, 2])
