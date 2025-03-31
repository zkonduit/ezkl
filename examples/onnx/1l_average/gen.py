from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.AvgPool2d(2, 1, (1, 1))

    def forward(self, x):
        return self.layer(x)[0]


circuit = Model()
export(circuit, input_shape=[3, 2, 2])
