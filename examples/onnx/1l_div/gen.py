from torch import nn
import importlib.util as imputil
spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

class Circuit(nn.Module):
    def __init__(self, inplace=False):
        super(Circuit, self).__init__()

    def forward(self, x):
        return x/ 10

circuit = Circuit()
exp.export(circuit, input_shape = [1])


    
