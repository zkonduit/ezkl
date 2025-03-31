from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):          
        m = torch.norm(x, p=1, dim=1)
        
        return m 


circuit = MyModel()
export(circuit, input_shape=[1, 2, 2, 8], opset_version=17, include_output=False)
