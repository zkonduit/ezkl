import localEVMVerify from '../../in-browser-evm-verifier/src/index'
import { serialize, deserialize } from '@ezkljs/engine/nodejs'
import { compileContracts } from './utils'
import * as fs from 'fs'

exports.EXAMPLE = require("minimist")(process.argv.slice(2))["example"];
exports.PATH = require("minimist")(process.argv.slice(2))["dir"];
exports.VK = require("minimist")(process.argv.slice(2))["vk"];

describe('localEVMVerify', () => {

  let bytecode_verifier: string

  let bytecode_vk: string | undefined = undefined

  let proof: any

  const example = exports.EXAMPLE || "1l_mlp"
  const path = exports.PATH || "../ezkl/examples/onnx"
  const vk = exports.VK || false

  beforeEach(() => {
    const solcOutput = compileContracts(path, example, 'kzg')

    bytecode_verifier =
      solcOutput.contracts['artifacts/Verifier.sol']['Halo2Verifier'].evm.bytecode
        .object

    if (vk) {
      const solcOutput_vk = compileContracts(path, example, 'vk')

      bytecode_vk =
        solcOutput_vk.contracts['artifacts/Verifier.sol']['Halo2VerifyingKey'].evm.bytecode
          .object


      console.log('size of verifier bytecode', bytecode_verifier.length)
    }
    console.log('verifier bytecode', bytecode_verifier)
  })

  it('should return true when verification succeeds', async () => {
    const proofFileBuffer = fs.readFileSync(`${path}/${example}/proof.pf`)

    proof = deserialize(proofFileBuffer)

    const result = await localEVMVerify(proofFileBuffer, bytecode_verifier, bytecode_vk)

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
      result = await localEVMVerify(proofModified, bytecode_verifier, bytecode_vk)
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
