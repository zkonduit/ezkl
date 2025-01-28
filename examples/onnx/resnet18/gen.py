import ezkl
import torch
from torchvision.models import resnet18
import json
import asyncio


def single_channel_resnet18() -> torch.nn.Module:
    model = resnet18()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

model = single_channel_resnet18()

shape = [1, 32, 32]
model_path = "..."
data_path = "..."
settings_path = "..."
cal_path = "..."

x = 0.1*torch.rand(1,*shape, requires_grad=True)
torch.onnx.export(
    model,
    x,
    model_path,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }
)

data_array = ((x).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data = [data_array])
json.dump(data, open(data_path, 'w'))

data_array = (torch.rand(20, *shape, requires_grad=True).detach().numpy()).reshape([-1]).tolist()

data = dict(input_data = [data_array])

# Serialize data into file:
json.dump(data, open(cal_path, 'w'))

py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "private"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed" # private by default

res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)

assert res == True

async def foo():
    return await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
res = asyncio.run(foo())
print(res)