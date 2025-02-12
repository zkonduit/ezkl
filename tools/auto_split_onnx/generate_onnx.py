import torch
import torchvision.models as models

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with the shape (1, 3, 224, 224)
# 1 means batch size, 3 means RGB channels, 224x224 is the input image size
dummy_input = torch.randn(1, 3, 224, 224)

# Specify the output ONNX file path
onnx_file_path = "resnet18.onnx"

# Export the model to ONNX format
torch.onnx.export(model,               # The model to be exported
                  dummy_input,        # The input to the model
                  onnx_file_path,     # The path to save the ONNX file
                  export_params=True, # Whether to export the trained parameters
                  opset_version=11,   # ONNX version
                  do_constant_folding=True, # Whether to perform constant folding optimization
                  input_names=['input'],   # Input name
                  output_names=['output'],  # Output name
                  dynamic_axes={'input': {0: 'batch_size'},    # Dynamic batch size
                                'output': {0: 'batch_size'}})  # Dynamic batch size

print(f"ResNet-18 ONNX model exported to {onnx_file_path}")

