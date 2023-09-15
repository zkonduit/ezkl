import * as fs from 'fs/promises';
import * as path from 'path';
import JSONBig from 'json-bigint';

// HELPERS FOR ALL TESTS

// Function to convert the return buffer elgamalGenRandom into a JSON object
export function uint8ArrayToJsonObject(uint8Array: Uint8Array) {
    let string = new TextDecoder().decode(uint8Array);
    let jsonObject = JSONBig.parse(string);
    return jsonObject;
}

// HELPERS FOR PROVE AND VERIFY TESTS

export async function readDataFile(filename: string): Promise<Uint8ClampedArray> {
    const filePath = path.join(__dirname, '..', 'public', 'data', filename);
    const buffer = await fs.readFile(filePath);
    return new Uint8ClampedArray(buffer.buffer);
}

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

export function deserialize(filename: string): any {
    return readDataFile(filename).then((uint8Array) => {
        const string = new TextDecoder().decode(uint8Array);
        const jsonObject = JSONBig.parse(string);
        return jsonObject;
    });
}

export function serialize(data: any): Uint8ClampedArray {
    // Step 1: Stringify the Object with BigInt support
    const jsonString = JSONBig.stringify(data);

    // Step 2: Encode the JSON String
    const uint8Array = new TextEncoder().encode(jsonString);

    // Step 3: Convert to Uint8ClampedArray
    return new Uint8ClampedArray(uint8Array.buffer);
}

/// HELPERS FOR FIELD ELEMENT UTILS TESTS

// Convert an integer value in a field element
export const intToFieldElement = (int: bigint): bigint => {
    const primeFieldHex = "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";
    const prime = BigInt(primeFieldHex);
    return (BigInt(int) + BigInt(prime)) % BigInt(prime);
};

// Meant to simulate random input tensor values. 
export function randomZScore() {
    let u1, u2;
    do {
        u1 = Math.random();  // Uniform random number between 0 and 1
        u2 = Math.random();  // Uniform random number between 0 and 1
    } while (u1 <= Number.EPSILON);  // Avoid getting u1 as 0

    // Box-Muller transform
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    // const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);  // Another independent value if needed

    return z0;  // z0 is a random value from a standard normal distribution (z-score)
}