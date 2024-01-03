from torch import nn
import torch
import json


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        topk_largest = torch.topk(x, 4)
        topk_smallest = torch.topk(x, 4, largest=False)
        print(topk_largest)
        print(topk_smallest)
        return [topk_largest.values, topk_smallest.values]


circuit = MyModel()


x = torch.randint(10, (1, 6))
y = circuit(x)

torch.onnx.export(circuit, x, "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=14,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(
    input_data=[d],
    output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in y]
)

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
