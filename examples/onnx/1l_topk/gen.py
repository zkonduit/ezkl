from torch import nn
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from export import export


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        topk_largest = torch.topk(x, 4)
        topk_smallest = torch.topk(x, 4, largest=False)
        print(topk_largest)
        print(topk_smallest)
        return topk_largest.values + topk_smallest.values


circuit = MyModel()
export(circuit, input_shape = [1, 6], int_input=True, opset_version=14)
