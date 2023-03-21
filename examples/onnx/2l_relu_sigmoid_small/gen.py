from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.siglayer = nn.Sigmoid()
        self.rellayer = nn.ReLU()

    def forward(self, x):
        x = self.rellayer(x)
        x = self.siglayer(x)
        return x

circuit = Model()