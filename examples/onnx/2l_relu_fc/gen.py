from torch import nn
import torch.nn.init as init
from ezkl import export

class Model(nn.Module):
    def __init__(self, inplace=False):
        super(Model, self).__init__()

        self.aff1 = nn.Linear(3,1)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x =  self.aff1(x)
        x =  self.relu(x)
        return (x)


    def _initialize_weights(self):
        init.orthogonal_(self.aff1.weight)

circuit = Model()
export(circuit, input_shape = [3])