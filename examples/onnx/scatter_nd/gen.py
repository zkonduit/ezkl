import torch
import torch.nn as nn
import sys
import json

sys.path.append("..")

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.enc_in
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]

class Configs:
    def __init__(self, seq_len, pred_len, enc_in=321, individual=True):
      self.seq_len = seq_len
      self.pred_len = pred_len
      self.enc_in = enc_in
      self.individual = individual

model = 'Linear'
seq_len = 10
pred_len = 4
enc_in = 3

configs = Configs(seq_len, pred_len, enc_in, True)
circuit = Model(configs)

x = torch.randn(1, seq_len, pred_len)


torch.onnx.export(circuit, x, "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  # the model's input names
                  input_names=['input'],
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})


d1 = ((x).detach().numpy()).reshape([-1]).tolist()


data = dict(
    input_data=[d1],
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
