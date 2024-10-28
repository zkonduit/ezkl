// Swift version of gen_vk_test
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

    // Ensure that the verifying key is not empty
    assert(vk.count > 0, "Verifying key generation failed, vk is empty")

} catch {
    fatalError("Test failed with error: \(error)")
}