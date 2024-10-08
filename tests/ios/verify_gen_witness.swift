// Swift version of verify_gen_witness test
import ezkl
import Foundation

func loadFileAsBytes(from path: String) -> Data? {
    let url = URL(fileURLWithPath: path)
    return try? Data(contentsOf: url)
}

do {
    let pathToFile = "../../../../tests/wasm/"
    let networkCompiledPath = pathToFile + "model.compiled"
    let inputPath = pathToFile + "input.json"
    let witnessPath = pathToFile + "witness.json"

    // Load necessary files
    guard let networkCompiled = loadFileAsBytes(from: networkCompiledPath) else {
        fatalError("Failed to load network compiled file")
    }
    guard let input = loadFileAsBytes(from: inputPath) else {
        fatalError("Failed to load input file")
    }
    guard let referenceWitnessData = loadFileAsBytes(from: witnessPath) else {
        fatalError("Failed to load witness file")
    }

    // Generate witness using genWitness function
    let witnessData = try genWitness(
        compiledCircuit: networkCompiled,
        input: input
    )

    // Deserialize the witness
    struct GraphWitness: Decodable, Equatable {}
    let witness = try JSONDecoder().decode(GraphWitness.self, from: witnessData)

    // Deserialize the reference witness
    let referenceWitness = try JSONDecoder().decode(GraphWitness.self, from: referenceWitnessData)

    // Check if the witness matches the reference witness
    assert(witness == referenceWitness, "Witnesses do not match")

} catch {
    fatalError("Test failed with error: \(error)")
}