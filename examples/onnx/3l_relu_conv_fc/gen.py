from torch import nn
from ezkl import export

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1,4, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(4,4, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4*4*4, 10)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1,4*4*4)
        x = self.fc(x)
        return x

circuit = MyModel()
export(circuit, input_shape = [1,28,28])


    
