from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.ReLU()

    def forward(self, x):
        return self.layer(x)


circuit = MyModel()
export(circuit, input_shape = [3])


    
