import * as fs from 'fs/promises';
import JSONBig from 'json-bigint';

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
