"""
Export ONNX model that uses Piecewise Linear (PLA) Sigmoid instead of Lookup.

This example shows how to avoid expensive Sigmoid lookup in ezkl by replacing it
with a PLA that uses only mul/add/compare — expressible with copy/arithmetic constraints.
Run from this directory: python gen.py
"""
import os
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
import json

from pwl_sigmoid import PWLSigmoid, calibrate_and_save, fit_pwl_sigmoid


def _load_pwl_sigmoid():
    path = os.path.join(os.path.dirname(__file__), "pwl_params.npz")
    if os.path.exists(path):
        d = np.load(path)
        return PWLSigmoid(d["breakpoints"], d["slopes"], d["intercepts"])
    breakpoints, slopes, intercepts = fit_pwl_sigmoid(n_segments=16, target_max_rel_error=0.005)
    np.savez(path, breakpoints=breakpoints, slopes=slopes, intercepts=intercepts)
    return PWLSigmoid(breakpoints, slopes, intercepts)


class Circuit(nn.Module):
    """Small circuit: conv + PLA Sigmoid (no lookup)."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.act = _load_pwl_sigmoid()
        self.conv = nn.Conv2d(3, 3, (2, 2), 1, 2)
        init.orthogonal_(self.conv.weight)

    def forward(self, x, y, z):
        x = self.act(self.conv(y @ x**2 + (x) - (self.relu(z)))) + 2
        return (x, self.relu(z) / 3)


def main():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "pwl_params.npz")):
        calibrate_and_save(target_accuracy=0.995, out_path=os.path.join(os.path.dirname(__file__), "pwl_params.npz"))
    torch_model = Circuit()
    shape = [3, 2, 2]
    x = 0.1 * torch.rand(1, *shape, requires_grad=True)
    y = 0.1 * torch.rand(1, *shape, requires_grad=True)
    z = 0.1 * torch.rand(1, *shape, requires_grad=True)
    torch_out = torch_model(x, y, z)

    out_dir = os.path.dirname(__file__)
    onnx_path = os.path.join(out_dir, "network.onnx")
    input_path = os.path.join(out_dir, "input.json")

    torch.onnx.export(
        torch_model,
        (x, y, z),
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    data = dict(
        input_shapes=[shape, shape, shape],
        input_data=[
            x.detach().numpy().reshape(-1).tolist(),
            y.detach().numpy().reshape(-1).tolist(),
            z.detach().numpy().reshape(-1).tolist(),
        ],
        output_data=[o.detach().numpy().reshape(-1).tolist() for o in torch_out],
    )
    with open(input_path, "w") as f:
        json.dump(data, f)
    print("Exported network.onnx and input.json")


if __name__ == "__main__":
    main()
