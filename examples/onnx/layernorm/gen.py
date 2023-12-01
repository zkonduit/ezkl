import torch
import torch.nn as nn
import json


# A single model that only does layernorm
class LayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.ln(x)
    

x = torch.randn(1, 10, 10)
model = LayerNorm(10)
out = model(x)

torch.onnx.export(model, x, "network.onnx", export_params=True, do_constant_folding=True, input_names = ['input'],output_names = ['output'],dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})
data_array = ((x).detach().numpy()).reshape([-1]).tolist()
data = dict(input_data = [data_array], output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in out])
json.dump( data, open( "input.json", 'w' ) )

