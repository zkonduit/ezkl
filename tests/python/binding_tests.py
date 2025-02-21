import ezkl
import os
import pytest
import json
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
params_k21_path = os.path.join(folder_path, 'kzg_test_k21.params')
anvil_url = "http://localhost:3030"


def setup_module(module):
    """setup anvil."""
    global proc
    # requires an anvil install
    proc = subprocess.Popen(["anvil", "-p", "3030"])
    time.sleep(1)


def teardown_module(module):
    """teardown anvil.
    """
    proc.terminate()


def test_py_run_args():
    """
    Test for PyRunArgs
    """
    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "hashed"
    run_args.output_visibility = "hashed"


def test_poseidon_hash():
    """
    Test for poseidon_hash
    """
    message = [1.0, 2.0, 3.0, 4.0]
    message = [ezkl.float_to_felt(x, 7) for x in message]
    res = ezkl.poseidon_hash(message)
    assert ezkl.felt_to_big_endian(
        res[0]) == "0x2369898875588bf49b6539376b09705ea69aee318a58e6fcc1e68fc3e7ad81ab"



def test_field_serialization():
    """
    Test field element serialization
    """

    input = 890
    scale = 7
    felt = ezkl.float_to_felt(input, scale)
    roundtrip_input = ezkl.felt_to_float(felt, scale)
    assert input == roundtrip_input

    input = -700
    scale = 7
    felt = ezkl.float_to_felt(input, scale)
    roundtrip_input = ezkl.felt_to_float(felt, scale)
    assert input == roundtrip_input


def test_buffer_to_felts():
    """
    Test buffer_to_felt
    """
    buffer = bytearray("a sample string!", 'utf-8')
    felts = ezkl.buffer_to_felts(buffer)
    ref_felt_1 = "0x0000000000000000000000000000000021676e6972747320656c706d61732061"
    assert ezkl.felt_to_big_endian(felts[0]) == ref_felt_1

    buffer = bytearray("a sample string!"+"high", 'utf-8')
    felts = ezkl.buffer_to_felts(buffer)
    ref_felt_2 = "0x0000000000000000000000000000000000000000000000000000000068676968"
    assert [ezkl.felt_to_big_endian(felts[0]),  ezkl.felt_to_big_endian(felts[1])] == [ref_felt_1, ref_felt_2]


def test_gen_srs():
    """
    test for gen_srs() with 17 logrows and 20 logrows.
    You may want to comment this test as it takes a long time to run
    """
    ezkl.gen_srs(params_k17_path, 17)
    assert os.path.isfile(params_k17_path)

    ezkl.gen_srs(params_k21_path, 21)
    assert os.path.isfile(params_k21_path)



async def test_calibrate_over_user_range():
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
        data_path, model_path, output_path, "resources", 1, [0, 1, 2])
    assert res == True
    assert os.path.isfile(output_path)



async def test_calibrate():
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


def test_model_compile():
    """
   Test for model compilation/serialization
   """
    model_path = os.path.join(
        examples_path,
        'onnx',
        '1l_relu',
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
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True


async def test_forward():
    """
    Test for vanilla forward pass
    """
    data_path = os.path.join(
        examples_path,
        'onnx',
        '1l_relu',
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

    res = await ezkl.gen_witness(data_path, model_path, output_path)

    with open(output_path, "r") as f:
        data = json.load(f)

    assert data["inputs"] == res["inputs"]
    assert data["outputs"] == res["outputs"]

    assert data["processed_inputs"]["poseidon_hash"] == res["processed_inputs"]["poseidon_hash"]
    assert data["processed_outputs"]["poseidon_hash"] == res["processed_outputs"]["poseidon_hash"]


async def test_get_srs():
    """
    Test for get_srs
    """
    settings_path = os.path.join(folder_path, 'settings.json')
    res = await ezkl.get_srs(settings_path, srs_path=srs_path)

    assert res == True

    assert os.path.isfile(srs_path)

    another_srs_path = os.path.join(folder_path, "kzg_test_k8.params")

    res = await ezkl.get_srs(logrows=8, srs_path=another_srs_path, commitment=ezkl.PyCommitments.KZG)

    assert os.path.isfile(another_srs_path)


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

    res = ezkl.mock(data_path, model_path)
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
        srs_path=srs_path,
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    res = ezkl.gen_vk_from_pk_single(pk_path, settings_path, vk_path)
    assert res == True
    assert os.path.isfile(vk_path)


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

    res = ezkl.setup(
        model_path,
        vk_path,
        pk_path,
        srs_path=srs_path,
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)


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

    res = ezkl.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        "for-aggr",
        srs_path=srs_path,
    )
    assert res['transcript_type'] == 'Poseidon'
    assert os.path.isfile(proof_path)

    settings_path = os.path.join(folder_path, 'settings.json')
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
    res = ezkl.prove(
        data_path,
        model_path,
        pk_path,
        proof_path,
        "single",
        srs_path=srs_path,
    )
    assert res['transcript_type'] == 'EVM'
    assert os.path.isfile(proof_path)


async def test_create_evm_verifier():
    """
    Create EVM verifier with solidity code
    In order to run this test you will need to install solc in your environment
    """
    vk_path = os.path.join(folder_path, 'test_evm.vk')
    settings_path = os.path.join(folder_path, 'settings.json')
    sol_code_path = os.path.join(folder_path, 'test.sol')
    abi_path = os.path.join(folder_path, 'test.abi')
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    calldata_path = os.path.join(folder_path, 'calldata.bytes')

    # res is now a vector of bytes
    res = ezkl.encode_evm_calldata(proof_path, calldata_path)

    assert os.path.isfile(calldata_path)
    assert len(res) > 0


    res = await ezkl.create_evm_verifier(
        vk_path,
        settings_path,
        sol_code_path,
        abi_path,
        srs_path=srs_path,
    )

    assert res == True
    assert os.path.isfile(sol_code_path)

async def test_create_evm_verifier_separate_vk():
    """
    Create EVM a verifier with solidity code and separate vk
    In order to run this test you will need to install solc in your environment
    """
    vk_path = os.path.join(folder_path, 'test_evm.vk')
    settings_path = os.path.join(folder_path, 'settings.json')
    sol_code_path = os.path.join(folder_path, 'test_separate.sol')
    vk_code_path = os.path.join(folder_path, 'test_vk.sol')
    abi_path = os.path.join(folder_path, 'test_separate.abi')
    abi_vk_path = os.path.join(folder_path, 'test_vk_separate.abi')
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    calldata_path = os.path.join(folder_path, 'calldata.bytes')

    # # res is now a vector of bytes
    # res = ezkl.encode_evm_calldata(proof_path, calldata_path)

    # assert os.path.isfile(calldata_path)
    # assert len(res) > 0


    res = await ezkl.create_evm_verifier(
        vk_path,
        settings_path,
        sol_code_path,
        abi_path,
        srs_path=srs_path,
        reusable=True
    )

    res = await ezkl.create_evm_vka(
        vk_path,
        settings_path,
        vk_code_path,
        abi_vk_path,
        srs_path=srs_path,
    )

    assert res == True
    assert os.path.isfile(sol_code_path)


async def test_deploy_evm_reusable_and_vka():
    """
    Test deployment of the reusable verifier smart contract + vka
    In order to run this you will need to install solc in your environment
    """
    addr_path_verifier = os.path.join(folder_path, 'address_separate.json')
    addr_path_vk = os.path.join(folder_path, 'address_vk.json')
    sol_code_path = os.path.join(folder_path, 'test_separate.sol')
    vk_code_path = os.path.join(folder_path, 'test_vk.sol')

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    res = await ezkl.deploy_evm(
        addr_path_verifier,
        sol_code_path,
        anvil_url,
        "verifier/reusable",
    )

    res = await ezkl.deploy_evm(
        addr_path_vk,
        vk_code_path,
        anvil_url,
        "vka",
    )

    assert res == True

async def test_deploy_evm():
    """
    Test deployment of the verifier smart contract
    In order to run this you will need to install solc in your environment
    """
    addr_path = os.path.join(folder_path, 'address.json')
    sol_code_path = os.path.join(folder_path, 'test.sol')

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    res = await ezkl.deploy_evm(
        addr_path,
        sol_code_path,
        anvil_url,
    )

    assert res == True


async def test_deploy_evm_with_private_key():
    """
    Test deployment of the verifier smart contract using a custom private key
    In order to run this you will need to install solc in your environment
    """
    addr_path = os.path.join(folder_path, 'address.json')
    sol_code_path = os.path.join(folder_path, 'test.sol')

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    anvil_default_private_key = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

    res = await ezkl.deploy_evm(
        addr_path,
        sol_code_path,
        rpc_url=anvil_url,
        private_key=anvil_default_private_key
    )

    assert res == True

    custom_zero_balance_private_key = "ff9dfe0b6d31e93ba13460a4d6f63b5e31dd9532b1304f1cbccea7092a042aa4"

    with pytest.raises(RuntimeError, match="Failed to run deploy_evm"):
        res = await ezkl.deploy_evm(
            addr_path,
            sol_code_path,
            rpc_url=anvil_url,
            private_key=custom_zero_balance_private_key
        )


async def test_verify_evm():
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

    res = await ezkl.verify_evm(
        addr,
        proof_path,
        rpc_url=anvil_url,
        # sol_code_path
        # optimizer_runs
    )

    assert res == True

async def test_verify_evm_separate_vk():
    """
    Verifies an evm proof
    In order to run this you will need to install solc in your environment
    """
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    addr_path_verifier = os.path.join(folder_path, 'address_separate.json')
    addr_path_vk = os.path.join(folder_path, 'address_vk.json')
    proof_path = os.path.join(folder_path, 'test_evm.pf')
    calldata_path = os.path.join(folder_path, 'calldata_separate.bytes')

    with open(addr_path_verifier, 'r') as file:
        addr_verifier = file.read().rstrip()

    print(addr_verifier)

    with open(addr_path_vk, 'r') as file:
        addr_vk = file.read().rstrip()

    print(addr_vk)

    # res is now a vector of bytes
    res = ezkl.encode_evm_calldata(proof_path, calldata_path, addr_vk=addr_vk)

    assert os.path.isfile(calldata_path)
    assert len(res) > 0

    # TODO: without optimization there will be out of gas errors
    # sol_code_path = os.path.join(folder_path, 'test.sol')

    res = await ezkl.verify_evm(
        addr_verifier,
        proof_path,
        rpc_url=anvil_url,
        addr_vk=addr_vk,
        # sol_code_path
        # optimizer_runs
    )

    assert res == True


async def test_aggregate_and_verify_aggr():
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

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path=srs_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    output_path = os.path.join(
        folder_path,
        '1l_relu_aggr_witness.json'
    )

    res = await ezkl.gen_witness(data_path, compiled_model_path,
                           output_path)

    ezkl.prove(
        output_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "for-aggr",
        srs_path=srs_path,
    )

    # mock aggregate
    res = ezkl.mock_aggregate([proof_path], 21)
    assert res == True

    aggregate_proof_path = os.path.join(folder_path, 'aggr_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_1l_relu.vk')
    aggregate_pk_path = os.path.join(folder_path, 'aggr_1l_relu.pk')

    res = ezkl.setup_aggregate(
        [proof_path],
        aggregate_vk_path,
        aggregate_pk_path,
        21,
        srs_path=params_k21_path,
    )

    res = ezkl.gen_vk_from_pk_aggr(aggregate_pk_path, aggregate_vk_path)
    assert res == True
    assert os.path.isfile(vk_path)

    res = ezkl.aggregate(
        [proof_path],
        aggregate_proof_path,
        aggregate_pk_path,
        "poseidon",
        21,
        "unsafe",
        srs_path=params_k21_path,
    )

    assert res == True
    assert os.path.isfile(aggregate_proof_path)
    assert os.path.isfile(aggregate_vk_path)

    res = ezkl.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        21,
        srs_path=params_k21_path,
    )
    assert res == True


async def test_evm_aggregate_and_verify_aggr():
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

    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    assert res == True

    ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path=srs_path,
    )

    proof_path = os.path.join(folder_path, '1l_relu.pf')

    output_path = os.path.join(
        folder_path,
        '1l_relu_aggr_evm_witness.json'
    )

    res = await ezkl.gen_witness(data_path, compiled_model_path,
                           output_path)

    ezkl.prove(
        output_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "for-aggr",
        srs_path=srs_path,
    )

    aggregate_proof_path = os.path.join(folder_path, 'aggr_evm_1l_relu.pf')
    aggregate_vk_path = os.path.join(folder_path, 'aggr_evm_1l_relu.vk')
    aggregate_pk_path = os.path.join(folder_path, 'aggr_evm_1l_relu.pk')

    res = ezkl.setup_aggregate(
        [proof_path],
        aggregate_vk_path,
        aggregate_pk_path,
        21,
        srs_path=params_k21_path,
    )

    res = ezkl.aggregate(
        [proof_path],
        aggregate_proof_path,
        aggregate_pk_path,
        "evm",
        21,
        "unsafe",
        srs_path=params_k21_path,
    )

    assert res == True
    assert os.path.isfile(aggregate_proof_path)
    assert os.path.isfile(aggregate_vk_path)

    sol_code_path = os.path.join(folder_path, 'aggr_evm_1l_relu.sol')
    abi_path = os.path.join(folder_path, 'aggr_evm_1l_relu.abi')

    res = await ezkl.create_evm_verifier_aggr(
        [settings_path],
        aggregate_vk_path,
        sol_code_path,
        abi_path,
        logrows=21,
        srs_path=params_k21_path,
    )

    assert res == True
    assert os.path.isfile(sol_code_path)

    addr_path = os.path.join(folder_path, 'address_aggr.json')

    res = await ezkl.deploy_evm(
        addr_path,
        sol_code_path,
        rpc_url=anvil_url,
    )

    # as a sanity check
    res = ezkl.verify_aggr(
        aggregate_proof_path,
        aggregate_vk_path,
        21,
        srs_path=params_k21_path,
    )
    assert res == True

    # with open(addr_path, 'r') as file:
    #     addr_aggr = file.read().rstrip()

    # res = await ezkl.verify_evm(
    #     aggregate_proof_path,
    #     addr_aggr,
    #     rpc_url=anvil_url,
    # )

    # assert res == True


def get_examples():
    EXAMPLES_OMIT = [
        # these are too large
        'mobilenet_large',
        'mobilenet',
        'doodles',
        'nanoGPT',
        "self_attention",
        'multihead_attention',
        'large_op_graph',
        '1l_instance_norm',
        'variable_cnn',
        'accuracy',
        'linear_regression',
        "mnist_gan",
        "smallworm",
        "fr_age",
        "1d_conv",
    ]
    examples = []
    for subdir, _, _ in os.walk(os.path.join(examples_path, "onnx")):
        name = subdir.split('/')[-1]
        if name in EXAMPLES_OMIT or name == "onnx":
            continue
        else:
            examples.append((
                os.path.join(subdir, "network.onnx"),
                os.path.join(subdir, "input.json"),
            ))
    return examples


@pytest.mark.parametrize("model_file, input_file", get_examples())
async def test_all_examples(model_file, input_file):
    """Tests all examples in the examples folder"""
    # gen settings
    settings_path = os.path.join(folder_path, "settings.json")
    compiled_model_path = os.path.join(folder_path, 'network.ezkl')
    pk_path = os.path.join(folder_path, 'test.pk')
    vk_path = os.path.join(folder_path, 'test.vk')
    witness_path = os.path.join(folder_path, 'witness.json')
    proof_path = os.path.join(folder_path, 'proof.json')

    print("Testing example: ", model_file)

    run_args = ezkl.PyRunArgs()
    run_args.variables = [("batch_size", 1), ("sequence_length", 100), ("<Sym1>", 1)]
    run_args.logrows = 22

    res = ezkl.gen_settings(model_file, settings_path, py_run_args=run_args)
    assert res

    res = await ezkl.calibrate_settings(
        input_file, model_file, settings_path, "resources")
    assert res

    print("Compiling example: ", model_file)
    res = ezkl.compile_circuit(model_file, compiled_model_path, settings_path)
    assert res

    with open(settings_path, 'r') as f:
        data = json.load(f)

    logrows = data["run_args"]["logrows"]
    srs_path = os.path.join(folder_path, f"srs_{logrows}")

    # generate the srs file if the path does not exist
    if not os.path.exists(srs_path):
        print("Generating srs file: ", srs_path)
        ezkl.gen_srs(os.path.join(folder_path, srs_path), logrows)

    print("Setting up example: ", model_file)
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)

    print("Generating witness for example: ", model_file)
    res = await ezkl.gen_witness(input_file, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    print("Proving example: ", model_file)
    ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
        srs_path=srs_path,
    )

    assert os.path.isfile(proof_path)

    print("Verifying example: ", model_file)
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path=srs_path,
    )

    assert res == True
