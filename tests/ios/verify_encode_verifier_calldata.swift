// Swift version of verify_encode_verifier_calldata test
import ezkl
import Foundation

func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let pathToFile = "../../../../tests/wasm/"
    let proofPath = pathToFile + "proof.json"

    guard let proof = loadFileAsBytes(from: proofPath) else {
        fatalError("Failed to load proof file")
    }

    // Test without vk address
    let calldataNoVk = try encodeVerifierCalldata(
        proof: proof,
        vkAddress: nil
    )

    // Deserialize the proof data
    struct Snark: Decodable {
        let proof: Data
        let instances: Data
    }

    let snark = try JSONDecoder().decode(Snark.self, from: proof)

    let flattenedInstances = snark.instances.flatMap { $0 }
    let referenceCalldataNoVk = try encodeCalldata(
        vk: nil,
        proof: snark.proof,
        instances: flattenedInstances
    )

    // Check if the encoded calldata matches the reference
    assert(calldataNoVk == referenceCalldataNoVk, "Calldata without vk does not match")

    // Test with vk address
    let vkAddressString = "0000000000000000000000000000000000000000"
    let vkAddressData = Data(hexString: vkAddressString)

    guard vkAddressData.count == 20 else {
        fatalError("Invalid VK address length")
    }

    let vkAddressArray = [UInt8](vkAddressData)

    // Serialize vkAddress to match JSON serialization in Rust
    let serializedVkAddress = try JSONEncoder().encode(vkAddressArray)

    let calldataWithVk = try encodeVerifierCalldata(
        proof: proof,
        vk: serializedVkAddress
    )

    let referenceCalldataWithVk = try encodeCalldata(
        vk: vkAddressArray,
        proof: snark.proof,
        instances: flattenedInstances
    )

    // Check if the encoded calldata matches the reference
    assert(calldataWithVk == referenceCalldataWithVk, "Calldata with vk does not match")

} catch {
    fatalError("Test failed with error: \(error)")
}