import ezkl_lib
from ezkl_lib import PyCalArgs
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
params_k20_path = os.path.join(folder_path, 'kzg_test_k20.params')


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

    expected_table = (
        " \n"
        "┌─────────┬───────────┬────────┬──────────────┬─────┐\n"
        "│ opkind  │ out_scale │ inputs │ out_dims     │ idx │\n"
        "├─────────┼───────────┼────────┼──────────────┼─────┤\n"
        "│ Input   │ 7         │        │ [1, 3, 2, 2] │ 0   │\n"
        "├─────────┼───────────┼────────┼──────────────┼─────┤\n"
        "│ PAD     │ 7         │ [0]    │ [1, 3, 4, 4] │ 1   │\n"
        "├─────────┼───────────┼────────┼──────────────┼─────┤\n"
        "│ SUMPOOL │ 7         │ [1]    │ [1, 3, 3, 3] │ 2   │\n"
        "├─────────┼───────────┼────────┼──────────────┼─────┤\n"
        "│ RESHAPE │ 7         │ [2]    │ [3, 3, 3]    │ 3   │\n"
        "└─────────┴───────────┴────────┴──────────────┴─────┘"
    )
    assert ezkl_lib.table(path) == expected_table


def test_gen_srs():
    """
    test for gen_srs() with 17 logrows and 20 logrows.
    You may want to comment this test as it takes a long time to run
    """
    ezkl_lib.gen_srs(params_path, 17)
    assert os.path.isfile(params_path)

    ezkl_lib.gen_srs(params_k20_path, 20)
    assert os.path.isfile(params_k20_path)


def test_calibrate():
    """
    Test for calibration pass
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
        'circuit.params'
    )

    cal_args = PyCalArgs()
    cal_args.data = data_path
    # TODO: Dictionary outputs
    res = ezkl_lib.gen_circuit_params(
        model_path, output_path,
        py_cal_args=cal_args)

    assert os.path.isfile(output_path)
    assert res == True


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
        'input_forward.json'
    )
    circuit_params_path = os.path.join(
        folder_path,
        'circuit.params'
    )

    # TODO: Dictionary outputs
    res = ezkl_lib.forward(data_path, model_path,
                           output_path, circuit_params_path=circuit_params_path)
    # assert res == {"input_data":[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]],"input_shapes":[[1,5,5]],"output_data":[[0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625,0.9140625]]}

    with open(output_path, "r") as f:
        data = json.load(f)
    assert data == {"input_data": [[0.053262424, 0.074970566, 0.052355476, 0.028825462, 0.058487028, 0.008225823, 0.07530029, 0.0821458, 0.06227987, 0.024306035, 0.05793174, 0.04044203]], "output_data": [[0.053222656, 0.12841797, 0.07519531, 0.10546875, 0.20947266, 0.104003906, 0.052246094, 0.08105469, 0.028808594, 0.05859375, 0.06689453, 0.008300781, 0.13378906, 0.2241211,
                                                                                                                                                                                                             0.09033203, 0.07519531, 0.15722656, 0.08203125, 0.0625, 0.08691406, 0.024414062, 0.12060547, 0.18554688, 0.064941406, 0.05810547, 0.09863281, 0.040527344]], "input_hashes": [[8270957937025516140, 11801026918842104328, 2203849898884507041, 140307258138425306]], "output_hashes": [[4554067273356176515, 2525802612124249168, 5413776662459769622, 1194961624936436872]]}


def test_mock():
    """
    Test for mock
    """

    data_path = os.path.join(
        folder_path,
        'input_forward.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    circuit_params_path = os.path.join(folder_path, 'circuit.params')

    res = ezkl_lib.mock(data_path, model_path,
                        circuit_params_path)
    assert res == True


def test_setup():
    """
    Test for setup
    """

    data_path = os.path.join(
        folder_path,
        'input_forward.json'
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


def test_setup_evm():
    """
    Test for setup
    """

    data_path = os.path.join(
        folder_path,
        'input_forward.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    pk_path = os.path.join(folder_path, 'test_evm.pk')
    vk_path = os.path.join(folder_path, 'test_evm.vk')
    circuit_params_path = os.path.join(folder_path, 'circuit.params')

    res = ezkl_lib.setup(
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


def test_prove_and_verify():
    """
    Test for prove and verify
    """

    data_path = os.path.join(
        folder_path,
        'input_forward.json'
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

    vk_path = os.path.join(folder_path, 'test.vk')
    res = ezkl_lib.verify(proof_path, circuit_params_path,
                          vk_path, params_path)
    assert res == True
    assert os.path.isfile(vk_path)


def test_prove_evm():
    """
    Test for prove using evm transcript
    """

    data_path = os.path.join(
        folder_path,
        'input_forward.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )

    pk_path = os.path.join(folder_path, 'test_evm.pk')
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    circuit_params_path = os.path.join(folder_path, 'circuit.params')

    res = ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        params_path,
        "evm",
        "single",
        circuit_params_path,
    )
    assert res == True
    assert os.path.isfile(proof_path)

    res = ezkl_lib.print_proof_hex(proof_path)
    # to figure out a better way of testing print_proof_hex
    assert type(res) == str


def test_create_evm_verifier():
    """
    Create EVM verifier with solidity code
    In order to run this test you will need to install solc in your environment
    """
    vk_path = os.path.join(folder_path, 'test_evm.vk')
    circuit_params_path = os.path.join(folder_path, 'circuit.params')
    deployment_code_path = os.path.join(folder_path, 'deploy.code')
    sol_code_path = os.path.join(folder_path, 'test.sol')

    res = ezkl_lib.create_evm_verifier(
        vk_path,
        params_path,
        circuit_params_path,
        deployment_code_path,
        sol_code_path
    )

    assert res == True
    assert os.path.isfile(deployment_code_path)
    assert os.path.isfile(sol_code_path)


def test_verify_evm():
    """
    Verifies an evm proof
    In order to run this you will need to install solc in your environment
    """
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    deployment_code_path = os.path.join(folder_path, 'deploy.code')

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    res = ezkl_lib.verify_evm(
        proof_path,
        deployment_code_path,
        # sol_code_path
        # optimizer_runs
    )

    assert res == True


def test_aggregate_and_verify_aggr():
    """
    Tests for aggregated proof and verifying aggregate proof
    """
    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_relu',
        'input.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_relu',
        'network.onnx'
    )

    pk_path = os.path.join(folder_path, '1l_relu.pk')
    vk_path = os.path.join(folder_path, '1l_relu.vk')
    circuit_params_path = os.path.join(folder_path, '1l_relu_circuit.params')

    cal_args = PyCalArgs()
    cal_args.data = data_path

    ezkl_lib.gen_circuit_params(
        model_path,
        circuit_params_path,
        py_cal_args=cal_args
    )

    ezkl_lib.setup(
        model_path,
        vk_path,
        pk_path,
        params_path,
        circuit_params_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        params_path,
        "poseidon",
        "accum",
        circuit_params_path,
    )

    aggregate_proof_path = os.path.join(folder_path, 'aggr_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_1l_relu.vk')

    res = ezkl_lib.aggregate(
        aggregate_proof_path,
        [proof_path],
        [circuit_params_path],
        [vk_path],
        aggregate_vk_path,
        params_k20_path,
        "poseidon",
        20,
        "unsafe"
    )

    assert res == True
    assert os.path.isfile(aggregate_proof_path)
    assert os.path.isfile(aggregate_vk_path)

    res = ezkl_lib.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        params_k20_path,
        20,
    )
    assert res == True


def test_evm_aggregate_and_verify_aggr():
    """
    Tests for EVM aggregated proof and verifying aggregate proof
    """
    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_relu',
        'input.json'
    )

    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_relu',
        'network.onnx'
    )

    pk_path = os.path.join(folder_path, '1l_relu.pk')
    vk_path = os.path.join(folder_path, '1l_relu.vk')
    circuit_params_path = os.path.join(folder_path, '1l_relu_circuit.params')

    cal_args = PyCalArgs()
    cal_args.data = data_path

    ezkl_lib.gen_circuit_params(
        model_path,
        circuit_params_path,
        py_cal_args=cal_args
    )

    ezkl_lib.setup(
        model_path,
        vk_path,
        pk_path,
        params_path,
        circuit_params_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        params_path,
        "poseidon",
        "accum",
        circuit_params_path,
    )

    aggregate_proof_path = os.path.join(folder_path, 'aggr_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_1l_relu.vk')

    res = ezkl_lib.aggregate(
        aggregate_proof_path,
        [proof_path],
        [circuit_params_path],
        [vk_path],
        aggregate_vk_path,
        params_k20_path,
        "evm",
        20,
        "unsafe"
    )

    assert res == True
    assert os.path.isfile(aggregate_proof_path)
    assert os.path.isfile(aggregate_vk_path)

    aggregate_deploy_path = os.path.join(folder_path, 'aggr_1l_relu.code')
    sol_code_path = os.path.join(folder_path, 'aggr_1l_relu.sol')

    res = ezkl_lib.create_evm_verifier_aggr(
        aggregate_vk_path,
        params_k20_path,
        aggregate_deploy_path,
        sol_code_path
    )

    assert res == True
    assert os.path.isfile(aggregate_deploy_path)

    res = ezkl_lib.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        params_k20_path,
        20,
    )
    assert res == True
