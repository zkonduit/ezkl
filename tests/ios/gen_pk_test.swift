// Swift version of gen_pk_test
import ezkl
import Foundation

func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let pathToFile = "../../../../tests/assets/"
    let networkCompiledPath = pathToFile + "model.compiled"
    let srsPath = pathToFile + "kzg"

    // Load necessary files
    guard let compiledCircuit = loadFileAsBytes(from: networkCompiledPath) else {
        fatalError("Failed to load network compiled file")
    }
    guard let srs = loadFileAsBytes(from: srsPath) else {
        fatalError("Failed to load SRS file")
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

    // Ensure that the proving key is not empty
    assert(pk.count > 0, "Proving key generation failed, pk is empty")

} catch {
    fatalError("Test failed with error: \(error)")
}