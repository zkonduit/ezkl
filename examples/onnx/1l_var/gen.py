from torch import nn
from ezkl import export
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return [torch.var(x, unbiased=False)]

circuit = MyModel()
export(circuit, input_shape = [3,3])


    
