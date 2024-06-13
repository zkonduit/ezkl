from torch import nn
from ezkl import export

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.PReLU(num_parameters=3, init=0.25)

    def forward(self, x):
        return self.layer(x)

circuit = Model()
export(circuit, input_shape = [3, 2, 2])


    
