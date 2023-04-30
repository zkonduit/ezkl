from torch import nn
from ezkl import export


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 0)

    def forward(self, x):
        return self.layer2(self.layer(x))


circuit = Model()
export(circuit, input_shape=[10])
