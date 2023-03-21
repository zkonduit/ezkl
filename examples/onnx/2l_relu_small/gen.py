from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rellayer = nn.ReLU()

    def forward(self, x):
        x = self.rellayer(x)
        return self.rellayer(x)

circuit = Model()
