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

srs_path = os.path.join(folder_path, 'kzg_test.params')
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
    ezkl_lib.gen_srs(srs_path, 17)
    assert os.path.isfile(srs_path)

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
        'settings.json'
    )

    # TODO: Dictionary outputs
    res = ezkl_lib.gen_settings(model_path, output_path)
    assert res == True

    res = ezkl_lib.calibrate_settings(
        data_path, model_path, output_path, "resources")
    assert res == True
    assert os.path.isfile(output_path)


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
    settings_path = os.path.join(
        folder_path,
        'settings.json'
    )

    res = ezkl_lib.forward(data_path, model_path,
                           output_path, settings_path=settings_path)

    with open(output_path, "r") as f:
        data = json.load(f)

    assert data == res

    assert data["input_data"] == [[0.05326242372393608, 0.07497056573629379, 0.05235547572374344, 0.028825461864471436, 0.05848702788352966,
                                   0.008225822821259499, 0.07530029118061066, 0.0821458026766777, 0.06227986887097359, 0.024306034669280052, 0.05793173983693123, 0.040442030876874924]]
    assert data["output_data"] == [[0.05322265625, 0.12841796875, 0.0751953125, 0.10546875, 0.20947265625, 0.10400390625, 0.05224609375, 0.0810546875, 0.02880859375, 0.05859375, 0.06689453125, 0.00830078125,
                                    0.1337890625, 0.22412109375, 0.09033203125, 0.0751953125, 0.1572265625, 0.08203125, 0.0625, 0.0869140625, 0.0244140625, 0.12060546875, 0.185546875, 0.06494140625, 0.05810546875, 0.0986328125, 0.04052734375]]

    assert data["processed_inputs"]["poseidon_hash"] == [[
        8270957937025516140, 11801026918842104328, 2203849898884507041, 140307258138425306]]
    assert data["processed_params"]["poseidon_hash"] == []
    assert data["processed_outputs"]["poseidon_hash"] == [[4554067273356176515, 2525802612124249168,
                                                           5413776662459769622, 1194961624936436872]]


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

    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl_lib.mock(data_path, model_path,
                        settings_path)
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
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl_lib.setup(
        model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)


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
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl_lib.setup(
        model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)


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
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        srs_path,
        "poseidon",
        "single",
        settings_path,
        False
    )
    assert res == True
    assert os.path.isfile(proof_path)

    vk_path = os.path.join(folder_path, 'test.vk')
    res = ezkl_lib.verify(proof_path, settings_path,
                          vk_path, srs_path)
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
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        srs_path,
        "evm",
        "single",
        settings_path,
        False
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
    settings_path = os.path.join(folder_path, 'settings.json')
    deployment_code_path = os.path.join(folder_path, 'deploy.code')
    sol_code_path = os.path.join(folder_path, 'test.sol')

    res = ezkl_lib.create_evm_verifier(
        vk_path,
        srs_path,
        settings_path,
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
    settings_path = os.path.join(
        folder_path, '1l_relu_settings.json')

   # TODO: Dictionary outputs
    res = ezkl_lib.gen_settings(model_path, settings_path)
    assert res == True

    res = ezkl_lib.calibrate_settings(
        data_path, model_path, settings_path, "resources")
    assert res == True
    assert os.path.isfile(settings_path)

    ezkl_lib.setup(
        model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        srs_path,
        "poseidon",
        "accum",
        settings_path,
        False
    )

    aggregate_proof_path = os.path.join(folder_path, 'aggr_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_1l_relu.vk')

    res = ezkl_lib.aggregate(
        aggregate_proof_path,
        [proof_path],
        [settings_path],
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
    settings_path = os.path.join(
        folder_path, '1l_relu_settings.json')

    ezkl_lib.gen_settings(
        model_path,
        settings_path,
    )

    ezkl_lib.calibrate_settings(
        data_path,
        model_path,
        settings_path,
        "resources",
    )

    ezkl_lib.setup(
        model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    ezkl_lib.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        srs_path,
        "poseidon",
        "accum",
        settings_path,
        False
    )

    aggregate_proof_path = os.path.join(folder_path, 'aggr_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_1l_relu.vk')

    res = ezkl_lib.aggregate(
        aggregate_proof_path,
        [proof_path],
        [settings_path],
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
    sol_bytecode_path = os.path.join(folder_path, 'aggr_1l_relu.bytecode')

    res = ezkl_lib.create_evm_verifier_aggr(
        aggregate_vk_path,
        params_k20_path,
        aggregate_deploy_path,
        sol_code_path,
        sol_bytecode_path
    )

    assert res == True
    assert os.path.isfile(aggregate_deploy_path)
    assert os.path.isfile(sol_code_path)
    assert os.path.isfile(sol_bytecode_path)

    res = ezkl_lib.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        params_k20_path,
        20,
    )
    assert res == True
