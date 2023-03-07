from torch import nn
from ezkl import export

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.relu(self.relu(x))
        x = self.relu(self.fc(x))
        return x

circuit = MyModel()
export(circuit, input_shape = [1])


    
