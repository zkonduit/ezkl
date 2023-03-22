from torch import nn
import importlib.util as imputil
spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.PReLU(num_parameters=3, init=0.25)

    def forward(self, x):
        return self.layer(x)

circuit = Model()
exp.export(circuit, input_shape = [3, 3, 2])


    
