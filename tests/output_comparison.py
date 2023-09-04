import ezkl
import json
import onnx
import onnxruntime
import numpy as np
import sys


def get_ezkl_output(witness_file, settings_file):
    # convert the quantized ezkl output to float value
    witness_output = json.load(open(witness_file))
    outputs = witness_output['outputs']
    with open(settings_file) as f:
        settings = json.load(f)
    ezkl_outputs = [[ezkl.vecu64_to_float(
        outputs[i][j], settings['model_output_scales'][i]) for j in range(len(outputs[i]))] for i in range(len(outputs))]
    return ezkl_outputs


def get_onnx_output(model_file, input_file):
    # generate the ML model output from the ONNX file
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)
    with open(input_file) as f:
        inputs = json.load(f)
    # reshape the input to the model
    num_inputs = len(inputs['input_data'])

    onnx_input = dict()
    for i in range(num_inputs):
        input_node = onnx_model.graph.input[i]
        dims = []
        elem_type = input_node.type.tensor_type.elem_type
        for dim in input_node.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dims.append(1)
            else:
                dims.append(dim.dim_value)
        if elem_type == 7:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                np.int64).reshape(dims)
        elif elem_type == 9:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                bool).reshape(dims)
        else:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                np.float32).reshape(dims)
        onnx_session = onnxruntime.InferenceSession(model_file)
        onnx_input[input_node.name] = inputs_onnx
    onnx_output = onnx_session.run(None, onnx_input)
    return onnx_output[0]


def compare_outputs(zk_output, onnx_output):
    # calculate percentage difference between the 2 outputs (which are lists)

    res = []

    zip_object = zip(np.array(zk_output).flatten(),
                     np.array(onnx_output).flatten())
    for list1_i, list2_i in zip_object:
        if list1_i == 0.0 and list2_i == 0.0:
            res.append(0)
        else:
            diff = list1_i - list2_i
            res.append(100 * (diff) / (list2_i))

    print("zk_output: ", zk_output)
    print("onnx_output: ", onnx_output)
    print("res: ", res)

    return np.mean(np.abs(res))


if __name__ == '__main__':
    # model file is first argument to script
    model_file = sys.argv[1]
    # input file is second argument to script
    input_file = sys.argv[2]
    # witness file is third argument to script
    witness_file = sys.argv[3]
    # settings file is fourth argument to script
    settings_file = sys.argv[4]
    # get the ezkl output
    ezkl_output = get_ezkl_output(witness_file, settings_file)
    # get the onnx output
    onnx_output = get_onnx_output(model_file, input_file)
    # compare the outputs
    percentage_difference = compare_outputs(ezkl_output, onnx_output)
    # print the percentage difference
    print("mean percent diff: ", percentage_difference)
    assert percentage_difference < 1.1, "Percentage difference is too high"
