// Swift version of pk_is_valid_test
import ezkl
import Foundation

func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let pathToFile = "../../../../tests/wasm/"
    let networkCompiledPath = pathToFile + "model.compiled"
    let srsPath = pathToFile + "kzg"
    let witnessPath = pathToFile + "witness.json"
    let settingsPath = pathToFile + "settings.json"

    // Load necessary files
    guard let compiledCircuit = loadFileAsBytes(from: networkCompiledPath) else {
        fatalError("Failed to load network compiled file")
    }
    guard let srs = loadFileAsBytes(from: srsPath) else {
        fatalError("Failed to load SRS file")
    }
    guard let witness = loadFileAsBytes(from: witnessPath) else {
        fatalError("Failed to load witness file")
    }
    guard let settings = loadFileAsBytes(from: settingsPath) else {
        fatalError("Failed to load settings file")
    }

    // Generate the vk (Verifying Key)
    let vk = try genVk(
        compiledCircuit: compiledCircuit,
        srs: srs,
        compressSelectors: true // Corresponds to the `true` boolean in the Rust code
    )

    // Generate the pk (Proving Key)
    let pk = try genPk(
        vk: vk,
        compiledCircuit: compiledCircuit,
        srs: srs
    )

    // Prove using the witness and proving key
    let proof = try prove(
        witness: witness,
        pk: pk,
        compiledCircuit: compiledCircuit,
        srs: srs
    )

    // Ensure that the proof is not empty
    assert(proof.count > 0, "Proof generation failed, proof is empty")

    // Verify the proof
    let value = try verify(
        proof: proof,
        vk: vk,
        settings: settings,
        srs: srs
    )

    // Ensure that the verification passed
    assert(value == true, "Verification failed")

} catch {
    fatalError("Test failed with error: \(error)")
}