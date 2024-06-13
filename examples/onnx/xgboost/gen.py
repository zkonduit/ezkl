# make sure you have the dependencies required here already installed
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as Gbc
import torch
import ezkl
import os
from torch import nn
import xgboost as xgb
from hummingbird.ml import convert

NUM_CLASSES = 3

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = Gbc(n_estimators=12)
clr.fit(X_train, y_train)

# convert to torch


torch_gbt = convert(clr, 'torch')

print(torch_gbt)
# assert predictions from torch are = to sklearn
diffs = []

for i in range(len(X_test)):
    torch_pred = torch_gbt.predict(torch.tensor(X_test[i].reshape(1, -1)))
    sk_pred = clr.predict(X_test[i].reshape(1, -1))
    diffs.append(torch_pred != sk_pred[0])

print("num diff: ", sum(diffs))


# Input to the model
shape = X_train.shape[1:]
x = torch.rand(1, *shape, requires_grad=False)
torch_out = torch_gbt.predict(x)
# Export the model
torch.onnx.export(torch_gbt.model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  "network.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})

d = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_shapes=[shape],
            input_data=[d],
            output_data=[(o).reshape([-1]).tolist() for o in torch_out])

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
