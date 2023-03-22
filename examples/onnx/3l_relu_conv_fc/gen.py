from torch import nn
import importlib.util as imputil
spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

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
exp.export(circuit, input_shape = [1,28,28])


    
