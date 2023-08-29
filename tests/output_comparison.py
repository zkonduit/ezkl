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
    inputs_onnx = np.array(inputs['input_data']).astype(np.float32)
    onnx_session = onnxruntime.InferenceSession(model_file)
    onnx_input = {onnx_session.get_inputs()[0].name: inputs_onnx}
    onnx_output = onnx_session.run(None, onnx_input)
    return onnx_output[0]


def compare_outputs(zk_output, onnx_output):
    # calculate percentage difference between the 2 outputs (which are lists)

    res = []

    zip_object = zip(zk_output, onnx_output)
    for list1_i, list2_i in zip_object:
        for second_list1_i, second_list2_i in zip(list1_i, list2_i):
            if second_list1_i == 0.0 and second_list2_i == 0.0:
                res.append(0)
            else:
                res.append(100*(second_list1_i - second_list2_i) /
                           (second_list2_i))

    return np.mean(res)


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
    assert percentage_difference < 2.0, "Percentage difference is too high"
