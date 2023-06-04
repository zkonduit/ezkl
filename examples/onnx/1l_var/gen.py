from torch import nn
from ezkl import export
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return [torch.var(x, unbiased=False, dim=[1,2])]

circuit = MyModel()
export(circuit, input_shape = [1,3,3])


    
