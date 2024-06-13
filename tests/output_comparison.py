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
    ezkl_outputs = [[ezkl.felt_to_float(
        outputs[i][j], settings['model_output_scales'][i]) for j in range(len(outputs[i]))] for i in range(len(outputs))]
    return ezkl_outputs


def get_onnx_output(model_file, input_file):
    # generate the ML model output from the ONNX file
    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)

    with open(input_file) as f:
        inputs = json.load(f)
    # reshape the input to the model
    num_inputs = len(onnx_model.graph.input)

    onnx_input = dict()
    for i in range(num_inputs):
        input_node = onnx_model.graph.input[i]
        dims = []
        elem_type = input_node.type.tensor_type.elem_type
        print("elem_type: ", elem_type)
        for dim in input_node.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                dims.append(1)
            else:
                dims.append(dim.dim_value)
        if elem_type == 6:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                np.int32).reshape(dims)
        elif elem_type == 7:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                np.int64).reshape(dims)
        elif elem_type == 9:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                bool).reshape(dims)
        else:
            inputs_onnx = np.array(inputs['input_data'][i]).astype(
                np.float32).reshape(dims)
        onnx_input[input_node.name] = inputs_onnx
    try:
        onnx_session = onnxruntime.InferenceSession(model_file)
        onnx_output = onnx_session.run(None, onnx_input)
    except Exception as e:
        print("error: ", e)
        onnx_output = inputs['output_data']
    print("onnx ", onnx_output)
    return onnx_output[0]


def compare_outputs(zk_output, onnx_output):
    # calculate percentage difference between the 2 outputs (which are lists)

    res = []

    contains_sublist = any(isinstance(sub, list) for sub in zk_output)
    print("zk ", zk_output)
    if contains_sublist:
        try:
            if len(onnx_output) == 1:
                zk_output = zk_output[0]
        except Exception as e:
            zk_output = zk_output[0]
    print("zk ", zk_output)

    zip_object = zip(np.array(zk_output).flatten(),
                     np.array(onnx_output).flatten())
    for (i, (list1_i, list2_i)) in enumerate(zip_object):
        if list1_i == 0.0 and list2_i == 0.0:
            res.append(0)
        else:
            diff = list1_i - list2_i
            res.append(100 * (diff) / (list2_i))
            # iterate and print the diffs  if they are greater than 0.0
            if abs(diff) > 0.0:
                print("------- index: ", i)
                print("------- diff: ", diff)
                print("------- zk_output: ", list1_i)
                print("------- onnx_output: ", list2_i)

    return res


if __name__ == '__main__':
    # model file is first argument to script
    model_file = sys.argv[1]
    # input file is second argument to script
    input_file = sys.argv[2]
    # witness file is third argument to script
    witness_file = sys.argv[3]
    # settings file is fourth argument to script
    settings_file = sys.argv[4]
    # target
    target = float(sys.argv[5])
    # get the ezkl output
    ezkl_output = get_ezkl_output(witness_file, settings_file)
    # get the onnx output
    onnx_output = get_onnx_output(model_file, input_file)
    # compare the outputs
    percentage_difference = compare_outputs(ezkl_output, onnx_output)
    mean_percentage_difference = np.mean(np.abs(percentage_difference))
    max_percentage_difference = np.max(np.abs(percentage_difference))
    # print the percentage difference
    print("mean percent diff: ", mean_percentage_difference)
    print("max percent diff: ", max_percentage_difference)
    assert mean_percentage_difference < target, "Percentage difference is too high"
