import {
  serialize,
  deserialize
} from './utils';
import * as wasmFunctions from './nodejs/ezkl'
import { compileContracts } from './utils'
import * as fs from 'fs'

exports.EXAMPLE = require("minimist")(process.argv.slice(2))["example"];
exports.PATH = require("minimist")(process.argv.slice(2))["dir"];
exports.VK = require("minimist")(process.argv.slice(2))["vk"];

describe('localEVMVerify', () => {

  let bytecode_verifier_buffer: Uint8Array

  let bytecode_vk_buffer: Uint8Array | undefined = undefined

  let proof: any

  const example = exports.EXAMPLE || "1l_mlp"
  const path = exports.PATH || "../ezkl/examples/onnx"
  const vk = exports.VK || false

  beforeEach(() => {
    const solcOutput = compileContracts(path, example, 'kzg')

    let bytecode_verifier =
      solcOutput.contracts['artifacts/Verifier.sol']['Halo2Verifier'].evm.bytecode
        .object
    bytecode_verifier_buffer = new TextEncoder().encode(bytecode_verifier)


    if (vk) {
      const solcOutput_vk = compileContracts(path, example, 'vk')

      let bytecode_vk =
        solcOutput_vk.contracts['artifacts/Verifier.sol']['Halo2VerifyingKey'].evm.bytecode
          .object
      bytecode_vk_buffer = new TextEncoder().encode(bytecode_vk)


      console.log('size of verifier bytecode', bytecode_verifier.length)
    }
    console.log('verifier bytecode', bytecode_verifier)
  })

  it('should return true when verification succeeds', async () => {
    const proofFileBuffer = fs.readFileSync(`${path}/${example}/proof.pf`)
    const proofSer = new Uint8ClampedArray(proofFileBuffer.buffer)

    proof = deserialize(proofSer)

    const result = wasmFunctions.verifyEVM(proofSer, bytecode_verifier_buffer, bytecode_vk_buffer)

    console.log('result', result)

    expect(result).toBe(true)
  })

  it('should fail to verify faulty proofs', async () => {
    let result: boolean = true
    console.log(proof.proof)
    try {
      let index = Math.round((Math.random() * (proof.proof.length))) % proof.proof.length
      console.log('index', index)
      console.log('index', proof.proof[index])
      let number = (proof.proof[index] + 1) % 256
      console.log('index', index)
      console.log('new number', number)
      proof.proof[index] = number
      console.log('index post', proof.proof[index])
      const proofModified = serialize(proof)
      result = wasmFunctions.verifyEVM(proofModified, bytecode_verifier_buffer, bytecode_vk_buffer)
    } catch (error) {
      // Check if the error thrown is the "out of gas" error.
      expect(error).toEqual(
        expect.objectContaining({
          error: 'revert',
          errorType: 'EvmError',
        }),
      )
      result = false
    }
    // If localEVMVerify doesn't throw, check the results
    expect(result).toBe(false)
  })
})
