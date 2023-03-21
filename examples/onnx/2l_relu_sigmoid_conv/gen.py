from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cl1 = nn.Conv2d(3,3,(5,5), padding = (2,2))
        self.cl2 = nn.Conv2d(3,1,(1,1), padding = (1,1))
        self.siglayer = nn.Sigmoid()
        self.rellayer = nn.ReLU()

    def forward(self, x):
        x = self.rellayer(self.cl1(x))
        x = self.rellayer(self.cl2(x))
        return self.siglayer(x)

circuit = Model()