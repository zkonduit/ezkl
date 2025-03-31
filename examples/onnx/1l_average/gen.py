from torch import nn
import torch
import json


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.AvgPool2d(2, 1, (1, 1))

    def forward(self, x):
        return self.layer(x)[0]

circuit = Model()
circuit.eval()

input_shape = [3, 2, 2]
x = 0.1 * torch.rand(1,*input_shape, requires_grad=True)
torch_out = circuit(x)

data = {
    "input_data": [((x).detach().numpy()).reshape([-1]).tolist()],
    "output_data": [((torch_out).detach().numpy()).reshape([-1]).tolist()]
}

# Export input.json
with open("input.json", "w") as f:
    json.dump(data, f)

# Export network.onnx
torch.onnx.export(
    circuit,
    x,
    "network.onnx",
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names = ['input'],
    output_names = ['output'],
    dynamic_axes={
        'input' : {0 : 'batch_size'},
        'output' : {0 : 'batch_size'}
    }
)
