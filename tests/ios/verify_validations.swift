// Swift version of verify_validations test
import ezkl
import Foundation

func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let pathToFile = "../../../../tests/assets/"
    let compiledCircuitPath = pathToFile + "model.compiled"
    let networkPath = pathToFile + "network.onnx"
    let witnessPath = pathToFile + "witness.json"
    let inputPath = pathToFile + "input.json"
    let proofPath = pathToFile + "proof.json"
    let vkPath = pathToFile + "vk.key"
    let pkPath = pathToFile + "pk.key"
    let settingsPath = pathToFile + "settings.json"
    let srsPath = pathToFile + "kzg"

    // Load necessary files
    guard let compiledCircuit = loadFileAsBytes(from: compiledCircuitPath) else {
        fatalError("Failed to load network compiled file")
    }
    guard let network = loadFileAsBytes(from: networkPath) else {
        fatalError("Failed to load network file")
    }
    guard let witness = loadFileAsBytes(from: witnessPath) else {
        fatalError("Failed to load witness file")
    }
    guard let input = loadFileAsBytes(from: inputPath) else {
        fatalError("Failed to load input file")
    }
    guard let proof = loadFileAsBytes(from: proofPath) else {
        fatalError("Failed to load proof file")
    }
    guard let vk = loadFileAsBytes(from: vkPath) else {
        fatalError("Failed to load vk file")
    }
    guard let pk = loadFileAsBytes(from: pkPath) else {
        fatalError("Failed to load pk file")
    }
    guard let settings = loadFileAsBytes(from: settingsPath) else {
        fatalError("Failed to load settings file")
    }
    guard let srs = loadFileAsBytes(from: srsPath) else {
        fatalError("Failed to load srs file")
    }

    // Witness validation (should fail for network compiled)
    let witnessValidationResult1 = try? witnessValidation(witness:compiledCircuit)
    assert(witnessValidationResult1 == nil, "Witness validation should fail for network compiled")

    // Witness validation (should pass for witness)
    let witnessValidationResult2 = try? witnessValidation(witness:witness)
    assert(witnessValidationResult2 != nil, "Witness validation should pass for witness")

    // Compiled circuit validation (should fail for onnx network)
    let circuitValidationResult1 = try? compiledCircuitValidation(compiledCircuit:network)
    assert(circuitValidationResult1 == nil, "Compiled circuit validation should fail for onnx network")

    // Compiled circuit validation (should pass for compiled network)
    let circuitValidationResult2 = try? compiledCircuitValidation(compiledCircuit:compiledCircuit)
    assert(circuitValidationResult2 != nil, "Compiled circuit validation should pass for compiled network")

    // Input validation (should fail for witness)
    let inputValidationResult1 = try? inputValidation(input:witness)
    assert(inputValidationResult1 == nil, "Input validation should fail for witness")

    // Input validation (should pass for input)
    let inputValidationResult2 = try? inputValidation(input:input)
    assert(inputValidationResult2 != nil, "Input validation should pass for input")

    // Proof validation (should fail for witness)
    let proofValidationResult1 = try? proofValidation(proof:witness)
    assert(proofValidationResult1 == nil, "Proof validation should fail for witness")

    // Proof validation (should pass for proof)
    let proofValidationResult2 = try? proofValidation(proof:proof)
    assert(proofValidationResult2 != nil, "Proof validation should pass for proof")

    // Verifying key (vk) validation (should pass)
    let vkValidationResult = try? vkValidation(vk:vk, settings:settings)
    assert(vkValidationResult != nil, "VK validation should pass for vk")

    // Proving key (pk) validation (should pass)
    let pkValidationResult = try? pkValidation(pk:pk, settings:settings)
    assert(pkValidationResult != nil, "PK validation should pass for pk")

    // Settings validation (should fail for proof)
    let settingsValidationResult1 = try? settingsValidation(settings:proof)
    assert(settingsValidationResult1 == nil, "Settings validation should fail for proof")

    // Settings validation (should pass for settings)
    let settingsValidationResult2 = try? settingsValidation(settings:settings)
    assert(settingsValidationResult2 != nil, "Settings validation should pass for settings")

    // SRS validation (should pass)
    let srsValidationResult = try? srsValidation(srs:srs)
    assert(srsValidationResult != nil, "SRS validation should pass for srs")

} catch {
    fatalError("Test failed with error: \(error)")
}