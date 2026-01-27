# ONNX Model Auto Split Script Documentation

This script can automatically split an ONNX model into multiple sub-models according to a given upper threshold. It utilizes various libraries such as `onnx`, `networkx`, `onnxsim`, `json`, and `ezkl` for its operations.

## Script Explanation

This script addresses the challenge of generating correctness proofs for large ONNX models on machines with limited hardware capabilities. For instance, if a machine has a processing constraint of \(2^{24}\) but the model requires significantly more, it becomes difficult to handle.

To overcome this, the script automatically partitions a large model into multiple smaller sub-models based on a given upper threshold. It ensures that the intermediate results between sub-models are protected through hashing for privacy.

By splitting the large model, this approach enables verification of larger models on machines with average hardware. Additionally, it facilitates parallel validation of the models by allowing multiple sub-models to be validated simultaneously using multithreading or multiple machines, thus improving overall efficiency.



## Command Line Arguments

The script can be executed with the following command line arguments:

- `--onnx_model_path`: (str) Path to the ONNX model. Default is `'./resnet18.onnx'`.
- `--simplified_model_path`: (str) Path to save the simplified ONNX model. Default is `'./resnet18_simplified.onnx'`.
- `--upper_bound_per_subgraph`: (int) Upper bound per subgraph in the form of \(2^n\). Default is `23`.
- `--input_shape`: (str) Input shape for the ONNX model in JSON format. Default is `'{"input": [1, 3, 224, 224]}'`.
- `--simplify`: (flag) Indicates if the model should be simplified. Default is `False`.

## Example Usage

Need to simplify the original onnx model:
```bash
python auto_split_onnx.py --onnx_model_path './model.onnx' --simplified_model_path './model_sim.onnx' --upper_bound_per_subgraph 20 --simplify
```



## Functions

### 1. `find_closest_power(total_value)`

Calculates the closest power of 2 that is greater than or equal to the given `total_value`.

- **Parameters**:
  - `total_value` (int): The total value to find the closest power of 2 for.

- **Returns**:
  - (int): The closest power of 2 in the form of \(2^n\).

---

### 2. `generate_random_input_data(onnx_model_path, data_path)`

Generates random input data matching the input shapes of the ONNX model and saves it as a JSON file.

- **Parameters**:
  - `onnx_model_path` (str): Path to the ONNX model file.
  - `data_path` (str): Path where the generated JSON input data will be saved.

---

### 3. `load_onnx_model_to_graph(onnx_model_path)`

Loads the ONNX model and constructs a directed graph representation of the model's operations.

- **Parameters**:
  - `onnx_model_path` (str): Path to the ONNX model file.

- **Returns**:
  - `(G, ops)`:
    - `G` (nx.DiGraph): The directed graph representing the model.
    - `ops` (list): A list of operation names in the model.

---

### 4. `process_graph(simplified_model_path, G, ops, upper_bound_per_subgraph)`

Processes the graph to split it into subgraphs based on the defined upper bounds and assesses the power requirements for each subgraph.

- **Parameters**:
  - `simplified_model_path` (str): Path to the simplified ONNX model file.
  - `G` (nx.DiGraph): The directed graph representing the model.
  - `ops` (list): A list of operation names in the model.
  - `upper_bound_per_subgraph` (int): The upper bound for the number of assignments per subgraph in the form of \(2^n\).

---

### 5. `simplify_onnx_model(onnx_model_path, input_shape, output_path)`

Simplifies the ONNX model using `onnxsim` and saves the simplified model.

- **Parameters**:
  - `onnx_model_path` (str): Path to the ONNX model file.
  - `input_shape` (dict): A dictionary defining the input shape for the model.
  - `output_path` (str): Path to save the simplified ONNX model.

- **Returns**:
  - (bool): A flag indicating whether the simplification was successful.

---

### 6. `check_total_assignments(json_file, n)`

Checks if the total number of assignments is within the acceptable limits defined by \(2^n\).

- **Parameters**:
  - `json_file` (str): Path to the JSON file containing total assignments.
  - `n` (int): The upper limit in the form of \(2^n\).

- **Returns**:
  - `(is_pass, pow)`:
    - `is_pass` (bool): Whether the total assignments are within bounds.
    - `pow` (int): The closest power of 2 to the total assignments.

---

### 7. `judge_model_upper_bound(original_model_path, subgraph_index, subgraph_inputs, subgraph_outputs, upper_bound)`

Evaluates whether a subgraph of the model meets the upper bound requirements by extracting the subgraph and calibrating its settings.

- **Parameters**:
  - `original_model_path` (str): Path to the original ONNX model.
  - `subgraph_index` (int): Index of the subgraph being evaluated.
  - `subgraph_inputs` (set): Set of input names for the subgraph.
  - `subgraph_outputs` (set): Set of output names for the subgraph.
  - `upper_bound` (int): The upper bound for the number of assignments in the form of \(2^n\).

- **Returns**:
  - `(is_pass, pow)`:
    - `is_pass` (bool): Whether the subgraph meets the requirements.
    - `pow` (int): The closest power of 2 related to the total assignments.

---

### 8. `save_graph(G, file_name)`

Saves the directed graph as an image file.

- **Parameters**:
  - `G` (nx.DiGraph): The directed graph to be saved.
  - `file_name` (str): The name of the file where the graph will be saved.

---

### 9. `print_node_names_and_types(onnx_model_path)`

Prints the names and types of all nodes in the ONNX model.

- **Parameters**:
  - `onnx_model_path` (str): Path to the ONNX model file.

---

## Disclaimer
This software is experimental and un-audited. We do not provide any warranties, express or implied, including but not limited to warranties of merchantability or fitness for a particular purpose. We will not be liable for any losses, damages, or issues arising from the use of this software, whether direct or indirect.

Users are encouraged to exercise caution and conduct their own independent assessments and testing. By using this software, you acknowledge and accept the risks associated with its experimental nature and agree that the developers and contributors are not responsible for any consequences resulting from its use.

