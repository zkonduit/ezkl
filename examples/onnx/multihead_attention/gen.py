# 1. We define a simple transformer model with MultiHeadAttention layers
import ezkl
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.temperature = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        n_batches = q.size(0)

        q = self.w_qs(q).view(n_batches, -1, self.n_heads,
                              self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(n_batches, -1, self.n_heads,
                              self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(n_batches, -1, self.n_heads,
                              self.d_k).transpose(1, 2)

        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model)
        q = self.dropout(self.fc(q))
        return self.layer_norm(q)


class SimpleTransformer(nn.Module):
    def __init__(self, nlayer, d_model=512, n_heads=8):
        super().__init__()

        self.layers = nn.ModuleList(
            [MultiHeadAttention(n_heads, d_model) for _ in range(nlayer)])
        # self.layers = nn.ModuleList([MultiHeadAttention(n_heads, d_model)])
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # assuming x is of shape [batch_size, sequence_length, d_model]
        for layer in self.layers:
            x = layer(x, x, x)
        x = x.mean(dim=1)  # taking mean over the sequence length
        x = self.fc(x)
        return x


# 2. We export it
model = SimpleTransformer(2, d_model=128)
input_shape = [1, 16, 128]
x = 0.1*torch.rand(1, *input_shape, requires_grad=True)

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

data = dict(input_data=[data_array])
json.dump(data, open("input.json", 'w'))


# 3. We do our ezkl work


# ezkl.gen_settings("network.onnx", "settings.json")

# !RUST_LOG = full
# res = await ezkl.calibrate_settings(
#     "input.json", "network.onnx", "settings.json", "resources")
