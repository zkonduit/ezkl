// Write a simple swift test
import ezkl
import Foundation

let pathToFile = "../../../../tests/wasm/"


func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let proofAggrPath = pathToFile + "proof_aggr.json"
    let vkAggrPath = pathToFile + "vk_aggr.key"
    let srs1Path = pathToFile + "kzg1.srs"

    guard let proofAggr = loadFileAsBytes(from: proofAggrPath) else {
        fatalError("Failed to load proofAggr file")
    }
    guard let vkAggr = loadFileAsBytes(from: vkAggrPath) else {
        fatalError("Failed to load vkAggr file")
    }
    guard let srs1 = loadFileAsBytes(from: srs1Path) else {
        fatalError("Failed to load srs1 file")
    }

    let value = try verifyAggr(
        proof: proofAggr,
        vk: vkAggr,
        logrows: 21,
        srs: srs1,
        commitment: "kzg"
    )

    // should not fail
    assert(value == true, "Failed the test")

}