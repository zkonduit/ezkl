import localEVMVerify, { Hardfork } from '@ezkljs/verify'
import { parseProof, compileContracts } from './utils'

exports.USER_NAME = require("minimist")(process.argv.slice(2))["example"];
exports.PATH = require("minimist")(process.argv.slice(2))["dir"];

describe('localEVMVerify', () => {

  let bytecode: string

  let instances: string[]

  let proof: string

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
    ;[instances, proof] = parseProof(path, example)
    console.log('instances', instances)
    console.log('proof', proof)

    const result = await localEVMVerify(
      proof,
      instances,
      bytecode,
      Hardfork['London'],
    )

    expect(result).toBe(true)
  })

  it('should fail to verify faulty proofs', async () => {
    let result: boolean = true
    try {
      let index = Math.floor(Math.random() * (proof.length - 2)) + 2
      let number = (parseInt(proof[index] || '0', 16) + 1) % 16
      let newChar = number.toString(16)
      console.log('index', index)
      console.log('newChar', newChar)
      const proofModified =
        proof.slice(0, index) + newChar + proof.slice(index + 1)
      console.log('proofModified', proofModified)
      result = await localEVMVerify(proofModified, instances, bytecode)
    } catch (error) {
      // Check if the error thrown is the "out of gas" error.
      expect(error).toEqual(
        expect.objectContaining({
          error: 'out of gas',
          errorType: 'EvmError',
        }),
      )
      result = false
    }
    // If localEVMVerify doesn't throw, check the results
    expect(result).toBe(false)
  })
})
