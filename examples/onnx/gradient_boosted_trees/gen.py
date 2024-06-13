# make sure you have the dependencies required here already installed
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as Gbc
import sk2torch
import torch
import ezkl
import os
from torch import nn

NUM_CLASSES = 3

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = Gbc(n_estimators=10)
clr.fit(X_train, y_train)


trees = []
for bundle in clr.estimators_:
    tree_bundle = []
    for tree in bundle:
        tree_bundle.append(sk2torch.wrap(tree))
    trees.append(tree_bundle)


class GradientBoostedTrees(nn.Module):
    def __init__(self, trees):
        super(GradientBoostedTrees, self).__init__()
        bundle_modules = []
        for bundle in trees:
            module = nn.ModuleList(bundle)
            bundle_modules.append(module)
        self.trees = nn.ModuleList(bundle_modules)
        self.num_classifiers = torch.tensor(
            [len(self.trees) for _ in range(NUM_CLASSES)])

    def forward(self, x):
        # first bundle
        local_pred = self.trees[0][0](x)
        local_pred = local_pred.reshape(-1, 1)
        for tree in self.trees[0][1:]:
            tree_out = tree(x)
            tree_out = tree_out.reshape(-1, 1)
            local_pred = torch.cat((local_pred, tree_out), 1)
        local_pred = local_pred.reshape(-1, NUM_CLASSES)
        out = local_pred

        for bundle in self.trees[1:]:
            local_pred = bundle[0](x)
            local_pred = local_pred.reshape(-1, 1)
            for tree in bundle[1:]:
                tree_out = tree(x)
                tree_out = tree_out.reshape(-1, 1)
                local_pred = torch.cat((local_pred, tree_out), 1)
            # local_pred = local_pred.reshape(x.shape[0], 3)
            local_pred = local_pred.reshape(-1, NUM_CLASSES)
            out = out + local_pred
        output = out / self.num_classifiers
        return out.reshape(-1, NUM_CLASSES)


torch_rf = GradientBoostedTrees(trees)
# assert predictions from torch are = to sklearn

for i in range(len(X_test)):
    torch_pred = torch_rf(torch.tensor(X_test[i].reshape(1, -1)))
    sk_pred = clr.predict(X_test[i].reshape(1, -1))
    print(torch_pred, sk_pred[0])
    assert torch_pred.argmax() == sk_pred[0]


torch_rf.eval()

# Input to the model
shape = X_train.shape[1:]
x = torch.rand(1, *shape, requires_grad=False)
torch_out = torch_rf(x)
# Export the model
torch.onnx.export(torch_rf,               # model being run
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
            output_data=[((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out])

# Serialize data into file:
json.dump(data, open("input.json", 'w'))
