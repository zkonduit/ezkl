from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = self.sigmoid(x)
        x = self.sigmoid(x)

        return x


circuit = MyModel()
export(circuit, input_shape = [1])
