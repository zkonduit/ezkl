from torch import nn
import torch.nn.init as init
import importlib.util as imputil
spec = imputil.spec_from_file_location("exporter", "/Users/siddharthaalluri/Desktop/0xPARC/ezkl/examples/onnx/exporter.py")   
exp = imputil.module_from_spec(spec)       
spec.loader.exec_module(exp)

class Model(nn.Module):
    def __init__(self, inplace=False):
        super(Model, self).__init__()

        self.aff1 = nn.Linear(3,1)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x =  self.aff1(x)
        x =  self.relu(x)
        return (x)


    def _initialize_weights(self):
        init.orthogonal_(self.aff1.weight)

circuit = Model()
exp.export(circuit, input_shape = [3])