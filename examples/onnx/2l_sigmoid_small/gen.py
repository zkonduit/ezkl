from torch import nn
import importlib.util as imputil
spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = self.sigmoid(x)
        x = self.sigmoid(x)

        return x

circuit = MyModel()
exp.export(circuit, input_shape = [1])


    
