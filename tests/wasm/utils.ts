import * as fs from 'fs/promises';
import * as fsSync from 'fs'
import JSONBig from 'json-bigint';
import { vecU64ToFelt } from '@ezkljs/engine/nodejs'
const solc = require('solc');

export async function readEzklArtifactsFile(path: string, example: string, filename: string): Promise<Uint8ClampedArray> {
    //const filePath = path.join(__dirname, '..', '..', 'ezkl', 'examples', 'onnx', example, filename);
    const filePath = `${path}/${example}/${filename}`
    const buffer = await fs.readFile(filePath);
    return new Uint8ClampedArray(buffer.buffer);
}

export async function readEzklSrsFile(path: string, example: string): Promise<Uint8ClampedArray> {
    // const settingsPath = path.join(__dirname, '..', '..', 'ezkl', 'examples', 'onnx', example, 'settings.json');
    const settingsPath = `${path}/${example}/settings.json`
    const settingsBuffer = await fs.readFile(settingsPath, { encoding: 'utf-8' });
    const settings = JSONBig.parse(settingsBuffer);
    const logrows = settings.run_args.logrows;
    // const filePath = path.join(__dirname, '..', '..', 'ezkl', 'examples', 'onnx', `kzg${logrows}.srs`);
    const filePath = `${path}/kzg${logrows}.srs`
    const buffer = await fs.readFile(filePath);
    return new Uint8ClampedArray(buffer.buffer);
}

export function deserialize(buffer: Uint8Array | Uint8ClampedArray): object { // buffer is a Uint8ClampedArray | Uint8Array // return a JSON object
    if (buffer instanceof Uint8ClampedArray) {
        buffer = new Uint8Array(buffer.buffer);
    }
    const string = new TextDecoder().decode(buffer);
    const jsonObject = JSONBig.parse(string);
    return jsonObject;
}

export function serialize(data: object | string): Uint8ClampedArray { // data is an object // return a Uint8ClampedArray
    // Step 1: Stringify the Object with BigInt support
    if (typeof data === "object") {
        data = JSONBig.stringify(data);
    }
    // Step 2: Encode the JSON String
    const uint8Array = new TextEncoder().encode(data as string);

    // Step 3: Convert to Uint8ClampedArray
    return new Uint8ClampedArray(uint8Array.buffer);
}

export function getSolcInput(path: string, example: string) {
    return {
      language: 'Solidity',
      sources: {
        'artifacts/Verifier.sol': {
          content: fsSync.readFileSync(`${path}/${example}/kzg.sol`, 'utf-8'),
        },
        // If more contracts were to be compiled, they should have their own entries here
      },
      settings: {
        optimizer: {
          enabled: true,
          runs: 200,
        },
        evmVersion: 'london',
        outputSelection: {
          '*': {
            '*': ['abi', 'evm.bytecode'],
          },
        },
      },
    }
  }
  
  export function compileContracts(path: string, example: string) {
    const input = getSolcInput(path, example)
    const output = JSON.parse(solc.compile(JSON.stringify(input)))
  
    let compilationFailed = false
  
    if (output.errors) {
      for (const error of output.errors) {
        if (error.severity === 'error') {
          console.error(error.formattedMessage)
          compilationFailed = true
        } else {
          console.warn(error.formattedMessage)
        }
      }
    }
  
    if (compilationFailed) {
      return undefined
    }
  
    return output
  }
  interface Snark {
    instances: Array<Array<Array<BigInt>>>
    proof: Uint8Array
  }
  
  function byteToHex(byte: number) {
    // convert the possibly signed byte (-128 to 127) to an unsigned byte (0 to 255).
    // if you know, that you only deal with unsigned bytes (Uint8Array), you can omit this line
    const unsignedByte = byte & 0xff
  
    // If the number can be represented with only 4 bits (0-15),
    // the hexadecimal representation of this number is only one char (0-9, a-f).
    if (unsignedByte < 16) {
      return '0' + unsignedByte.toString(16)
    } else {
      return unsignedByte.toString(16)
    }
  }
  
  // bytes is an typed array (Int8Array or Uint8Array)
  function toHexString(bytes: Uint8Array | Int8Array): string {
    // Since the .map() method is not available for typed arrays,
    // we will convert the typed array to an array using Array.from().
    return Array.from(bytes)
      .map((byte) => byteToHex(byte))
      .join('')
  }
  
  export function parseProof(path: string, example: string): [string[], string] {
    // Read the proof file
    const proofFileContent: string = fsSync.readFileSync(`${path}/${example}/proof.pf`, 'utf-8')
    // Parse it into Snark object using JSONBig
    const proof: Snark = JSONBig.parse(proofFileContent)
    console.log(proof.instances)
    // Parse instances to public inputs
    const instances: string[][] = []
  
    for (const val of proof.instances) {
      const inner_array: string[] = []
      for (const inner of val) {
        const u64sString = JSONBig.stringify(inner)
        console.log('u64String', u64sString)
        const u64sSer = new TextEncoder().encode(u64sString)
        const u64sSerClamped = new Uint8ClampedArray(u64sSer.buffer)
        let hexFieldElement = vecU64ToFelt(u64sSerClamped)
        let uint = BigInt(hexFieldElement)
        inner_array.push(uint.toString())
      }
      instances.push(inner_array)
    }
  
    const publicInputs = instances.flat()
    const proofString = toHexString(proof.proof)
    return [publicInputs, '0x' + proofString]
  }
  
