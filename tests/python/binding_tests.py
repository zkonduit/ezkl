import ezkl
import os
import pytest
import json
import asyncio
import subprocess
import time

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
params_k17_path = os.path.join(folder_path, 'kzg_test_k17.params')
params_k20_path = os.path.join(folder_path, 'kzg_test_k20.params')
anvil_url = "http://localhost:3030"


def setup_module(module):
    """setup anvil."""
    global proc
    # requries an anvil install
    proc = subprocess.Popen(["anvil", "-p", "3030"])
    time.sleep(1)


def teardown_module(module):
    """teardown anvil.
    """
    proc.terminate()


def test_field_serialization():
    """
    Test field element serialization
    """
    expected = "0x0000000000000000000000000000000000000000000000000000000000000001"
    assert expected == ezkl.vecu64_to_felt([1, 0, 0, 0])

    expected = "0x0000000000000000000000000000000000000000000000000000000000000019"
    assert expected == ezkl.vecu64_to_felt([25, 0, 0, 0])

    expected = "0x0000000000000005000000000000000100000000000000020000000000000002"
    assert expected == ezkl.vecu64_to_felt([2, 2, 1, 5])


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
    assert ezkl.table(path) == expected_table


def test_gen_srs():
    """
    test for gen_srs() with 17 logrows and 20 logrows.
    You may want to comment this test as it takes a long time to run
    """
    ezkl.gen_srs(params_k17_path, 17)
    assert os.path.isfile(params_k17_path)

    ezkl.gen_srs(params_k20_path, 20)
    assert os.path.isfile(params_k20_path)


async def calibrate():
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

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "hashed"
    run_args.output_visibility = "hashed"

    # TODO: Dictionary outputs
    res = ezkl.gen_settings(
        model_path, output_path, py_run_args=run_args)
    assert res == True

    res = await ezkl.calibrate_settings(
        data_path, model_path, output_path, "resources")
    assert res == True
    assert os.path.isfile(output_path)


def test_calibrate():
    """
    Test for calibrate
    """
    asyncio.run(calibrate())


def test_model_compile():
    """
   Test for model compilation/serialization
   """
    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_average',
        'network.onnx'
    )
    compiled_model_path = os.path.join(
        folder_path,
        'model.compiled'
    )
    settings_path = os.path.join(
        folder_path,
        'settings.json'
    )
    res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
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
        folder_path,
        'model.compiled'
    )
    output_path = os.path.join(
        folder_path,
        'witness.json'
    )
    settings_path = os.path.join(
        folder_path,
        'settings.json'
    )

    res = ezkl.gen_witness(data_path, model_path,
                           output_path, settings_path=settings_path)

    with open(output_path, "r") as f:
        data = json.load(f)

    assert data["inputs"] == res["inputs"]
    assert data["outputs"] == res["outputs"]

    assert data["processed_inputs"]["poseidon_hash"] == res["processed_inputs"]["poseidon_hash"]
    assert data["processed_outputs"]["poseidon_hash"] == res["processed_outputs"]["poseidon_hash"]


def test_get_srs():
    """
    Test for get_srs
    """
    settings_path = os.path.join(folder_path, 'settings.json')
    res = ezkl.get_srs(srs_path, settings_path)

    assert res == True

    assert os.path.isfile(srs_path)


def test_mock():
    """
    Test for mock
    """

    data_path = os.path.join(
        folder_path,
        'witness.json'
    )

    model_path = os.path.join(
        folder_path,
        'model.compiled'
    )

    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl.mock(data_path, model_path,
                    settings_path)
    assert res == True


def test_setup():
    """
    Test for setup
    """

    data_path = os.path.join(
        folder_path,
        'witness.json'
    )

    model_path = os.path.join(
        folder_path,
        'model.compiled'
    )

    pk_path = os.path.join(folder_path, 'test.pk')
    vk_path = os.path.join(folder_path, 'test.vk')
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl.setup(
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

    model_path = os.path.join(
        folder_path,
        'model.compiled'
    )

    pk_path = os.path.join(folder_path, 'test_evm.pk')
    vk_path = os.path.join(folder_path, 'test_evm.vk')
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl.setup(
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
        'witness.json'
    )

    model_path = os.path.join(
        folder_path,
        'model.compiled'
    )

    pk_path = os.path.join(folder_path, 'test.pk')
    proof_path = os.path.join(folder_path, 'test.pf')
    settings_path = os.path.join(folder_path, 'settings.json')

    res = ezkl.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        srs_path,
        "poseidon",
        "single",
        settings_path,
    )
    assert res['transcript_type'] == 'Poseidon'
    assert os.path.isfile(proof_path)

    vk_path = os.path.join(folder_path, 'test.vk')
    res = ezkl.verify(proof_path, settings_path,
                      vk_path, srs_path)
    assert res == True
    assert os.path.isfile(vk_path)


def test_prove_evm():
    """
    Test for prove using evm transcript
    """

    data_path = os.path.join(
        folder_path,
        'witness.json'
    )

    model_path = os.path.join(
        folder_path,
        'model.compiled'
    )

    pk_path = os.path.join(folder_path, 'test_evm.pk')
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    settings_path = os.path.join(folder_path, 'settings.json')
    res = ezkl.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        srs_path,
        "evm",
        "single",
        settings_path,
    )
    assert res['transcript_type'] == 'EVM'
    assert os.path.isfile(proof_path)

    res = ezkl.print_proof_hex(proof_path)
    # to figure out a better way of testing print_proof_hex
    assert type(res) == str


def test_create_evm_verifier():
    """
    Create EVM verifier with solidity code
    In order to run this test you will need to install solc in your environment
    """
    vk_path = os.path.join(folder_path, 'test_evm.vk')
    settings_path = os.path.join(folder_path, 'settings.json')
    sol_code_path = os.path.join(folder_path, 'test.sol')
    abi_path = os.path.join(folder_path, 'test.abi')

    res = ezkl.create_evm_verifier(
        vk_path,
        srs_path,
        settings_path,
        sol_code_path,
        abi_path
    )

    assert res == True
    assert os.path.isfile(sol_code_path)


def test_deploy_evm():
    """
    Verifies an evm proof
    In order to run this you will need to install solc in your environment
    """
    addr_path = os.path.join(folder_path, 'address.json')
    sol_code_path = os.path.join(folder_path, 'test.sol')

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    res = ezkl.deploy_evm(
        addr_path,
        sol_code_path,
        rpc_url=anvil_url,
    )

    assert res == True


def test_verify_evm():
    """
    Verifies an evm proof
    In order to run this you will need to install solc in your environment
    """
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    addr_path = os.path.join(folder_path, 'address.json')

    with open(addr_path, 'r') as file:
        addr = file.read().rstrip()

    print(addr)

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    res = ezkl.verify_evm(
        proof_path,
        addr,
        rpc_url=anvil_url,
        # sol_code_path
        # optimizer_runs
    )

    assert res == True


async def aggregate_and_verify_aggr():
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

    compiled_model_path = os.path.join(
        folder_path,
        'compiled_relu.onnx'
    )

    pk_path = os.path.join(folder_path, '1l_relu.pk')
    vk_path = os.path.join(folder_path, '1l_relu.vk')
    settings_path = os.path.join(
        folder_path, '1l_relu_aggr_settings.json')

   # TODO: Dictionary outputs
    res = ezkl.gen_settings(model_path, settings_path)
    assert res == True

    res = await ezkl.calibrate_settings(
        data_path, model_path, settings_path, "resources")
    assert res == True
    assert os.path.isfile(settings_path)

    res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
    assert res == True

    ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    output_path = os.path.join(
        folder_path,
        '1l_relu_aggr_witness.json'
    )

    res = ezkl.gen_witness(data_path, compiled_model_path,
                           output_path, settings_path=settings_path)

    ezkl.prove(
        output_path,
        compiled_model_path,
        pk_path,
        proof_path,
        srs_path,
        "poseidon",
        "accum",
        settings_path,
    )

    # mock aggregate
    res = ezkl.mock_aggregate([proof_path], 20)
    assert res == True

    aggregate_proof_path = os.path.join(folder_path, 'aggr_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_1l_relu.vk')
    aggregate_pk_path = os.path.join(folder_path, 'aggr_1l_relu.pk')

    res = ezkl.setup_aggregate(
        [proof_path],
        aggregate_vk_path,
        aggregate_pk_path,
        params_k20_path,
        20,
    )

    res = ezkl.aggregate(
        aggregate_proof_path,
        [proof_path],
        aggregate_pk_path,
        params_k20_path,
        "poseidon",
        20,
        "unsafe"
    )

    assert res == True
    assert os.path.isfile(aggregate_proof_path)
    assert os.path.isfile(aggregate_vk_path)

    res = ezkl.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        params_k20_path,
        20,
    )
    assert res == True


def test_aggregate_and_verify_aggr():
    """
    Tests for aggregated proof and verifying aggregate proof
    """
    asyncio.run(aggregate_and_verify_aggr())


async def evm_aggregate_and_verify_aggr():
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
        folder_path, '1l_relu_evm_aggr_settings.json')

    ezkl.gen_settings(
        model_path,
        settings_path,
    )

    await ezkl.calibrate_settings(
        data_path,
        model_path,
        settings_path,
        "resources",
    )

    compiled_model_path = os.path.join(
        folder_path,
        'compiled_relu.onnx'
    )

    res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
    assert res == True

    ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    output_path = os.path.join(
        folder_path,
        '1l_relu_aggr_evm_witness.json'
    )

    res = ezkl.gen_witness(data_path, compiled_model_path,
                           output_path, settings_path=settings_path)

    ezkl.prove(
        output_path,
        compiled_model_path,
        pk_path,
        proof_path,
        srs_path,
        "poseidon",
        "accum",
        settings_path,
    )

    aggregate_proof_path = os.path.join(folder_path, 'aggr_evm_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_evm_1l_relu.vk')
    aggregate_pk_path = os.path.join(folder_path, 'aggr_evm_1l_relu.pk')

    res = ezkl.setup_aggregate(
        [proof_path],
        aggregate_vk_path,
        aggregate_pk_path,
        params_k20_path,
        20,
    )

    res = ezkl.aggregate(
        aggregate_proof_path,
        [proof_path],
        aggregate_pk_path,
        params_k20_path,
        "evm",
        20,
        "unsafe"
    )

    assert res == True
    assert os.path.isfile(aggregate_proof_path)
    assert os.path.isfile(aggregate_vk_path)

    sol_code_path = os.path.join(folder_path, 'aggr_evm_1l_relu.sol')
    abi_path = os.path.join(folder_path, 'aggr_evm_1l_relu.abi')

    res = ezkl.create_evm_verifier_aggr(
        aggregate_vk_path,
        params_k20_path,
        sol_code_path,
        abi_path,
        [settings_path]
    )

    assert res == True
    assert os.path.isfile(sol_code_path)

    addr_path = os.path.join(folder_path, 'address_aggr.json')

    res = ezkl.deploy_evm(
        addr_path,
        sol_code_path,
        rpc_url=anvil_url,
    )

    # as a sanity check
    res = ezkl.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        params_k20_path,
        20,
    )
    assert res == True

    # with open(addr_path, 'r') as file:
    #     addr_aggr = file.read().rstrip()

    # res = ezkl.verify_evm(
    #     aggregate_proof_path,
    #     addr_aggr,
    #     rpc_url=anvil_url,
    # )

    # assert res == True


def test_evm_aggregate_and_verify_aggr():
    """
    Tests for aggregated proof and verifying aggregate proof
    """
    asyncio.run(evm_aggregate_and_verify_aggr())
