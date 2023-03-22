
from torch import nn
import importlib.util as imputil
 

spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=(0,0), count_include_pad=True)

    def forward(self, x):
        return self.layer(x)

circuit = Model()

exp.export(circuit, [1,5,5])