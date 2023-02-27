from torch import nn
from ezkl import export

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)
        
        self.relu = nn.ReLU()

        self.d1 = nn.Linear(48, 48)
        self.d2 = nn.Linear(48, 10)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)
    #    x = x.flatten()

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = self.relu(x)

        # logits => 32x10
        logits = self.d2(x)
       
        return logits

circuit = MyModel()
export(circuit, input_shape = [1,28,28])


    
