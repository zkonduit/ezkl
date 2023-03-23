from torch import nn
import importlib.util as imputil
spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

class Model(nn.Module):
    # def __init__(self):
    #     super(Model, self).__init__()
    #     self.layer = nn.Flatten()

    def forward(self, x):
        return x.view(6)

circuit = Model()

exp.export(circuit, input_shape = [3,2])
