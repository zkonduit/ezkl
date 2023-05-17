import ezkl_lib
import os
import pytest
import json

folder_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '.',
    )
)

examples_path = os.path.abspath(
    os.path.join(
        folder_path,
        '..',
        '..',
        'examples',
    )
)

params_path = os.path.join(folder_path, 'kzg_test.params')


def test_table_1l_average():
    """
    Test for table() with 1l_average.onnx
    """
    path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    expected_table = \
        """ 
┌─────────┬───────────┬────────┬──────────────┬─────┐
│ opkind  │ out_scale │ inputs │ out_dims     │ idx │
├─────────┼───────────┼────────┼──────────────┼─────┤
│ Input   │ 7         │        │ [1, 1, 5, 5] │ 0   │
├─────────┼───────────┼────────┼──────────────┼─────┤
│ SUMPOOL │ 7         │ [0]    │ [1, 1, 3, 3] │ 1   │
└─────────┴───────────┴────────┴──────────────┴─────┘"""
    assert ezkl_lib.table(path) == expected_table


def test_gen_srs():
    """
    Test for gen_srs() with 17 logrows.
    You may want to comment this test as it takes a long time to run
    """
    ezkl_lib.gen_srs(params_path, 17)
    assert os.path.isfile(params_path)


def test_forward():
    """
    Test for vanilla forward pass
    """
    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'input.json'
    )
    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )
    output_path = os.path.join(
        folder_path,
        'output.json'
    )
    # TODO: Dictionary outputs
    res = ezkl_lib.forward(data_path, model_path, output_path)
    # assert res == {"input_data":[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]],"input_shapes":[[1,5,5]],"output_data":[[0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625]]}

    with open(output_path, "r") as f:
        data = json.load(f)

    assert data == {"input_data": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], "input_shapes": [
        [1, 1, 5, 5]], "output_data": [[0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.9140625]]}

    os.remove(output_path)


def test_mock():
    """
    Test for mock
    """

    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'input.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    res = ezkl_lib.mock(data_path, model_path)
    assert res == True


def test_setup():
    """
    Test for setup
    """

    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'input.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    pk_path = os.path.join(folder_path, 'test.pk')
    vk_path = os.path.join(folder_path, 'test.vk')
    circuit_params_path = os.path.join(folder_path, 'circuit.params')

    res = ezkl_lib.setup(
        data_path,
        model_path,
        vk_path,
        pk_path,
        params_path,
        circuit_params_path,
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(circuit_params_path)


def test_prove():
    """
    Test for prove
    """

    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'input.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    pk_path = os.path.join(folder_path, 'test.pk')
    proof_path = os.path.join(folder_path, 'test.pf')
    circuit_params_path = os.path.join(folder_path, 'circuit.params')

    res = ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        params_path,
        "poseidon",
        "single",
        circuit_params_path,
    )
    assert res == True
    assert os.path.isfile(proof_path)


def test_verify():
    """
    Test for verify
    """

    vk_path = os.path.join(folder_path, 'test.vk')
    proof_path = os.path.join(folder_path, 'test.pf')
    circuit_params_path = os.path.join(folder_path, 'circuit.params')

    res = ezkl_lib.verify(
        proof_path,
        circuit_params_path,
        vk_path,
        params_path,
    )
    assert res == True
