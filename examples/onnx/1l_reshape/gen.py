from torch import nn

class Model(nn.Module):
    # def __init__(self):
    #     super(Model, self).__init__()
    #     self.layer = nn.Flatten()

    def forward(self, x):
        return x.view(6)

circuit = Model()