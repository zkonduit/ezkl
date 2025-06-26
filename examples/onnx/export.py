import json
import torch

def export(
    torch_model,
    input_shape=None,
    include_input=True,
    include_output=True,
    int_input=False,
    opset_version=10,
    onnx_filename="network.onnx",
    input_filename="input.json",
):
    """Export a PyTorch model.
    Arguments:
    torch_model: a PyTorch model class, such as Network(torch.nn.Module)
    input_shape: Shape of input
    int_input: set random input to int type
    opset_version: opset version to use
    onnx_filename: Default "network.onnx", the name of the onnx file to be generated
    input_filename: Default "input.json", the name of the json input file to be generated for ezkl
    """
    if include_input:
        if int_input:
            x = torch.randint(10, input_shape)
        else:
            x = 0.1*torch.rand(1,*input_shape, requires_grad=True)

        # Flips the neural net into inference mode
        torch_model.eval()

        torch_out = torch_model(x)

        # Export the model
        torch.onnx.export(torch_model,
                        x,
                        onnx_filename,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names = ['input'],
                        output_names = ['output'],
                        dynamic_axes={
                            'input' : {0 : 'batch_size'},
                            'output' : {0 : 'batch_size'}
                        })
        # convert output_data
        if include_output:
            try:
                output_data = [((torch_out).detach().numpy()).reshape([-1]).tolist()]

            # if the output is a list of tensors
            except AttributeError:
                output_data = [((o).detach().numpy()).reshape([-1]).tolist() for o in torch_out]

            data = {
                "input_data": [((x).detach().numpy()).reshape([-1]).tolist()],
                "output_data": output_data
            }
        else:
            data = {
                "input_data": [((x).detach().numpy()).reshape([-1]).tolist()],
            }

        # Serialize data into file:
        with open(input_filename, "w") as f:
            json.dump(data, f, indent=4)

    else:
        # Export the model
        torch.onnx.export(torch_model,
                        (),
                        onnx_filename,
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names = [],
                        output_names = ['output'])

        data = dict(
            input_data=[[]],
        )

        # Serialize data into file:
        with open(input_filename, "w") as f:
            json.dump(data, f, indent=4)