import torch 
from torch import nn
from ezkl import export

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return torch.sqrt(x)

circuit = MyModel()
export(circuit, input_shape = [3])


    
