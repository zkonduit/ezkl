from torch import nn
from ezkl import export

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return nn.functional.pad(x, (1,1))



circuit = Model()
export(circuit, input_shape = [3, 2, 2])


    
