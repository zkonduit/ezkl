import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import QuantStub, DeQuantStub

# define NN architecture


class PredictLiquidationsV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.layer_1 = nn.Linear(in_features=41, out_features=1)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer_1(x)
        x = self.dequant(x)
        return x


# instantiate the model
model_0 = PredictLiquidationsV0()

# for QAT
# model_0.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
torch.ao.quantization.prepare_qat(model_0, inplace=True)


# convert to a QAT model
quantized_model_0 = torch.ao.quantization.convert(
    model_0.eval(), inplace=False)

# evaluate quantized_model_0
# ...

x = torch.randn((1, 41), requires_grad=True)

# export as onnx
quantized_model_0.eval()
torch.onnx.export(quantized_model_0,
                  torch.randn((1, 41), requires_grad=True),
                  'network.onnx',
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})


d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data=[d],)

# save to input.json
json.dump(data, open("input.json", 'w'))
