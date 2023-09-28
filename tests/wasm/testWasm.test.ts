import * as wasmFunctions from './nodejs/ezkl'
import {
    readEzklArtifactsFile,
    readEzklSrsFile,
    serialize,
    deserialize
} from './utils';
import fs from 'fs';
exports.USER_NAME = require("minimist")(process.argv.slice(2))["example"];
exports.PATH = require("minimist")(process.argv.slice(2))["dir"];

const timingData: {
    example: string,
    proveTime: number,
    verifyTime: number,
    verifyResult: boolean | undefined
}[] = [];

describe('Generate witness, prove and verify', () => {

    let proof_ser: Uint8ClampedArray
    let circuit_settings_ser: Uint8ClampedArray;
    let params_ser: Uint8ClampedArray;

    let proveTime = 0;
    let verifyTime = 0;
    let verifyResult: boolean | undefined = false;

    const example = exports.USER_NAME || "1l_mlp"
    const path = exports.PATH || "../ezkl/examples/onnx"

    it('prove', async () => {
        let result
        let witness = await readEzklArtifactsFile(path, example, 'input.json');
        let pk = await readEzklArtifactsFile(path, example, 'key.pk');
        let circuit_ser = await readEzklArtifactsFile(path, example, 'network.onnx');
        circuit_settings_ser = await readEzklArtifactsFile(path, example, 'settings.json');
        params_ser = await readEzklSrsFile(path, example);
        const startTimeProve = Date.now();
        result = wasmFunctions.prove(witness, pk, circuit_ser, params_ser);
        const endTimeProve = Date.now();
        proof_ser = new Uint8ClampedArray(result.buffer);
        // test serialization/deserialization methods
        const proof_ser_ref = serialize(deserialize(proof_ser));
        const test = serialize("text");
        console.log(test);
        expect(proof_ser_ref).toEqual(proof_ser);
        proveTime = endTimeProve - startTimeProve;
        expect(result).toBeInstanceOf(Uint8Array);
    });

    it('verify', async () => {
        let result
        const vk = await readEzklArtifactsFile(path, example, 'key.vk');
        const startTimeVerify = Date.now();
        result = wasmFunctions.verify(proof_ser, vk, circuit_settings_ser, params_ser);
        const endTimeVerify = Date.now();
        verifyTime = endTimeVerify - startTimeVerify;
        verifyResult = result;
        // test serialization/deserialization methods
        expect(typeof result).toBe('boolean');
        expect(result).toBe(true);
    });

    afterAll(() => {
        fs.writeFileSync('timingData.json', JSON.stringify(timingData, null, 2));
    });
});
