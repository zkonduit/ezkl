from torch import nn
from ezkl import export
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.MaxPool2d(2, 1, (1, 1), 1, 1)

    def forward(self, x):
        return self.layer(x)[0]


circuit = Model()


# Input to the model
shape = [3, 2, 2]
x = torch.rand(1, *shape, requires_grad=False)
torch_out = circuit(x)
# Export the model
torch.onnx.export(circuit,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_shapes=[shape],
            input_data=[d],
            output_data=[(o).reshape([-1]).tolist() for o in torch_out])
