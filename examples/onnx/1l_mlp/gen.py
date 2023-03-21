from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(3,4,bias=True,)
        self.rellayer = nn.ReLU()

    def forward(self, x):
        return self.rellayer(self.layer(x))

circuit = Model()
