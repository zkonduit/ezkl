from torch import nn
import torch.nn.init as init
import torch
import json

N = 100

class Model(nn.Module):
    def __init__(self, inplace=False):
        super(Model, self).__init__()

        self.aff1 = nn.Linear(N,N)
        self.aff2 = nn.Linear(N,N)
        self.aff3 = nn.Linear(N,N)
        self.aff4 = nn.Linear(N,N)
        self.aff5 = nn.Linear(N,N)
        self.aff6 = nn.Linear(N,N)
        self.aff7 = nn.Linear(N,N)
        self.aff8 = nn.Linear(N,N)
        self.aff9 = nn.Linear(N,N)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.aff1(x)
        x = self.relu(x)
        x = self.aff2(x)
        x = self.relu(x)
        x = self.aff3(x)
        x = self.relu(x)
        x = self.aff4(x)
        x = self.relu(x)
        x = self.aff5(x)
        x = self.relu(x)
        x = self.aff6(x)
        x = self.relu(x)
        x = self.aff7(x)
        x = self.relu(x)
        x = self.aff8(x)
        x = self.relu(x)
        x = self.aff9(x)
        return x


    def _initialize_weights(self):
        init.orthogonal_(self.aff1.weight)

model = Model()

# Flips the neural net into inference mode
model.eval()
model.to('cpu')


x = torch.randn(1, N)
# Export the model
torch.onnx.export(model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data_json = dict(input_data=[data_array])

print(data_json)

# Serialize data into file:
json.dump(data_json, open("input.json", 'w'))
