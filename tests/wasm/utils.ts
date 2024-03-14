import * as fs from 'fs/promises';
import * as fsSync from 'fs'
import JSONBig from 'json-bigint';
const solc = require('solc');

// import os module
const os = require('os');

// check the available memory
const userHomeDir = os.homedir();

export async function readEzklArtifactsFile(path: string, example: string, filename: string): Promise<Uint8ClampedArray> {
  //const filePath = path.join(__dirname, '..', '..', 'ezkl', 'examples', 'onnx', example, filename);
  const filePath = `${path}/${example}/${filename}`
  const buffer = await fs.readFile(filePath);
  return new Uint8ClampedArray(buffer.buffer);
}

export async function readEzklSrsFile(logrows: string): Promise<Uint8ClampedArray> {
  const filePath = `${userHomeDir}/.ezkl/srs/kzg${logrows}.srs`
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

export function getSolcInput(path: string, example: string, name: string) {
  return {
    language: 'Solidity',
    sources: {
      'artifacts/Verifier.sol': {
        content: fsSync.readFileSync(`${path}/${example}/${name}.sol`, 'utf-8'),
      },
      // If more contracts were to be compiled, they should have their own entries here
    },
    settings: {
      optimizer: {
        enabled: true,
        runs: 1,
      },
      evmVersion: 'shanghai',
      outputSelection: {
        '*': {
          '*': ['abi', 'evm.bytecode'],
        },
      },
    },
  }
}

export function compileContracts(path: string, example: string, name: string) {
  const input = getSolcInput(path, example, name)
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


