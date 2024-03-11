import localEVMVerify from '../../in-browser-evm-verifier/src/index'
import { serialize, deserialize } from '@ezkljs/engine/nodejs'
import { compileContracts } from './utils'
import * as fs from 'fs'

exports.USER_NAME = require("minimist")(process.argv.slice(2))["example"];
exports.PATH = require("minimist")(process.argv.slice(2))["dir"];

describe('localEVMVerify', () => {

  let bytecode: string

  let proof: any

  const example = exports.USER_NAME || "1l_mlp"
  const path = exports.PATH || "../ezkl/examples/onnx"

  beforeEach(() => {
    let solcOutput = compileContracts(path, example)

    bytecode =
      solcOutput.contracts['artifacts/Verifier.sol']['Halo2Verifier'].evm.bytecode
        .object

    console.log('size', bytecode.length)
  })

  it('should return true when verification succeeds', async () => {
    const proofFileBuffer = fs.readFileSync(`${path}/${example}/proof.pf`)

    proof = deserialize(proofFileBuffer)

    const result = await localEVMVerify(proofFileBuffer, bytecode)

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
      result = await localEVMVerify(proofModified, bytecode)
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
