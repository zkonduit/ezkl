
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=(0,0), count_include_pad=True)

    def forward(self, x):
        x = nn.functional.pad(x, pad = (0,0,0,0,0,0,0,0), mode='constant', value=0)
        return self.layer(x)

circuit = Model()