import json
import torch.onnx

def export(circuit, input_shape, opset =11 ):
    output = {}
    input_vec = torch.rand(input_shape)
    torch.onnx.export(circuit, input_vec,
                     "network.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opset,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    output_vec = circuit.forward(input_vec)
    output["input_shapes"] = [input_shape]
    output["input_data"] = input_vec.numpy().tolist()
    output["output_data"] = output_vec.detach().numpy().tolist()
    with open("input.json", "w") as outfile:
        json.dump(output, outfile)
