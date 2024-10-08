// Swift version of verify_kzg_commit test
import ezkl
import Foundation

func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let pathToFile = "../../../../tests/wasm/"
    let vkPath = pathToFile + "vk.key"
    let srsPath = pathToFile + "kzg"
    let settingsPath = pathToFile + "settings.json"

    guard let vk = loadFileAsBytes(from: vkPath) else {
        fatalError("Failed to load vk file")
    }
    guard let srs = loadFileAsBytes(from: srsPath) else {
        fatalError("Failed to load srs file")
    }
    guard let settings = loadFileAsBytes(from: settingsPath) else {
        fatalError("Failed to load settings file")
    }

    // Create a vector of field elements
    var message: [UInt64] = []
    for i in 0..<32 {
        message.append(UInt64(i))
    }

    // Serialize the message array
    let messageData = try JSONEncoder().encode(message)

    // Deserialize settings
    struct GraphSettings: Decodable {}
    let settingsDecoded = try JSONDecoder().decode(GraphSettings.self, from: settings)

    // Generate commitment
    let commitmentData = try kzgCommit(
        message: messageData,
        vk: vk,
        settings: settings,
        srs: srs
    )

    // Deserialize the resulting commitment
    struct G1Affine: Decodable {}
    let commitment = try JSONDecoder().decode([G1Affine].self, from: commitmentData)

    // Reference commitment using params and vk
    // For Swift, you'd need to implement or link the corresponding methods like in Rust
    let referenceCommitment = try polyCommit(
        message: message,
        vk: vk,
        srs: srs
    )

    // Check if the commitment matches the reference
    assert(commitment == referenceCommitment, "Commitments do not match")

} catch {
    fatalError("Test failed with error: \(error)")
}