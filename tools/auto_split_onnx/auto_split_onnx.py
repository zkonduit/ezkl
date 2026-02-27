import onnx
import networkx as nx
import numpy as np
import json
from onnxsim import simplify
import os
import ezkl
import math
from collections import deque
import matplotlib.pyplot as plt
import time
from collections import Counter
import argparse

def find_closest_power(total_value):
    n = math.ceil(math.log2(total_value))
    return n

def generate_random_input_data(onnx_model_path, data_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Get the input information of the model
    input_info = []
    for input_tensor in graph.input:
        # Get the dimensions of the input; assume all dimensions are known
        dims = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_info.append(dims)
        
    # Generate random data and store it as JSON
    data = {'input_data': []}
    for dims in input_info:
        # Generate random data matching the input dimensions
        random_input = np.random.rand(*dims).reshape([-1]).tolist()
        data['input_data'].append(random_input)
    
    # Serialize the data to a file
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file)


def load_onnx_model_to_graph(onnx_model_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph
    ops = [node.name for node in graph.node]
    # print(ops)
    G = nx.DiGraph()
    for node in graph.node:
        # print(node.name)
        inputs = [node.input[0]] if node.op_type != 'Concat' and node.op_type != "Add" and len(node.input) > 0 else node.input
        G.add_node(node.name, op_type=node.op_type, inputs=inputs, outputs=[o for o in node.output])
        for input_name in inputs:
            G.add_edge(input_name, node.name)
        for output_name in node.output:
            G.add_edge(node.name, output_name)
            
    return G, ops


def process_graph(simplified_model_path, G, ops, upper_bound_per_subgraph):
    # Record the input and output lines of the operations
    ops_io = {op: {"inputs": [], "outputs": []} for op in ops}
    for op in ops:
        ops_io[op]["inputs"] = list(G.predecessors(op))
        ops_io[op]["outputs"] = list(G.successors(op))
    
    # Topological sorting
    topo_sorted_nodes = list(nx.topological_sort(G))
    # print("Topological sort:", topo_sorted_nodes)
    
    # Split subgraphs
    subgraphs = []
    subgraphs_pow = []
    current_pow = []
    current_subgraph = []
    subgraph_index = 0
    
    loop_index = 0
    append_flag = True
    
    while True:
        if loop_index >= len(topo_sorted_nodes):
            break
        op = topo_sorted_nodes[loop_index]
        if op not in ops:
            loop_index += 1
            continue
        
        if append_flag:
            current_subgraph.append(op)
        else:
            append_flag = True
            
        subgraph_inputs = set()
        subgraph_outputs = set()
        for op in current_subgraph:
            subgraph_inputs.update(ops_io[op]["inputs"])
            subgraph_outputs.update(ops_io[op]["outputs"])
        internal_lines = subgraph_inputs & subgraph_outputs
        subgraph_inputs -= internal_lines
        subgraph_outputs -= internal_lines
        
        is_pass, pow = judge_model_upper_bound(simplified_model_path, subgraph_index, subgraph_inputs, subgraph_outputs, upper_bound_per_subgraph)
        current_pow.append(pow)
        is_final_sub_model = False
        if "output" in subgraph_outputs:
            is_final_sub_model = True
        if not is_pass:
            # If current_subgraph has only one node, then save it as a subgraph and issue a warning
            if len(current_subgraph[:-1]) == 0:
                print(f"Warning: Subgraph {subgraph_index} only has one node: {current_subgraph[0]} and needs 2^{pow} assignments.")
                subgraphs.append(current_subgraph)
                subgraphs_pow.append(current_pow[-1])
                print(f"Subgraph {subgraph_index}: Nodes: {current_subgraph}")
            else:
                subgraphs.append(current_subgraph[:-1])  # Save the current subgraph, excluding the current node
                subgraphs_pow.append(current_pow[-2])
                print(f"Subgraph {subgraph_index}: Nodes: {current_subgraph[:-1]}")
            
            # Set temporary model name
            temp_model_name = f"temp_sub_model_{subgraph_index}.onnx"    
            onnx.utils.extract_model(simplified_model_path, temp_model_name, subgraph_inputs, subgraph_outputs)
            
            if len(current_subgraph[:-1]) == 0:
                current_subgraph = []
                append_flag = True
            else:
                current_subgraph = [op]  # Start a new subgraph with the current node
                append_flag = False  # Do not add new nodes in the next round
                loop_index += 1
            subgraph_index += 1
            current_pow = []
        elif is_final_sub_model:
            subgraphs.append(current_subgraph)
            subgraphs_pow.append(current_pow[-2])
            break
            # loop_index += 1
        else:
            loop_index += 1
            continue
        
    # Print subgraph information and corresponding power
    for i, subgraph in enumerate(subgraphs):
        print(f"Subgraph {i + 1}: Nodes: {subgraph}")
    
    print("Subgraph Pow:", subgraphs_pow)
    print("Total Subgraph_Pow:", len(subgraphs_pow))
    count_dict = Counter(subgraphs_pow)
    # Sort count_dict by keys
    sorted_items = sorted(count_dict.items())

    for num, count in sorted_items:
        if count > 0:
            print(f"Number {num} appears {count} times")



def simplify_onnx_model(onnx_model_path, input_shape, output_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    
    # Simplify the model
    model_simp, check = simplify(model, overwrite_input_shapes=input_shape)
    
    # Save the simplified model
    onnx.save(model_simp, output_path)
    
    return check


def check_total_assignments(json_file, n):
    is_pass = False
    with open(json_file, 'r') as f:
        data = json.load(f)
        total_assignments = data.get("total_assignments", 0)

    # find the closest 2^n of total_assignments, total_assignments must less than 2^n
    pow = find_closest_power(total_assignments)
    
    if pow <= n:
        is_pass = True
    else:
        is_pass = False
    return is_pass, pow

## upper_bound: n in 2^n
def judge_model_upper_bound(original_model_path, subgraph_index, subgraph_inputs, subgraph_outputs, upper_bound):
    ## convert set to list
    subgraph_inputs = list(subgraph_inputs)
    subgraph_outputs = list(subgraph_outputs)
    # print(f"Subgraph {subgraph_index}: Inputs: {subgraph_inputs}, Outputs: {subgraph_outputs}")
    
    is_final_sub_model = False
    ## if subgraph_outputs contains "output" node, then it is the final model
    if "output" in subgraph_outputs:
        is_final_sub_model = True
    
    
    ## set temp model name
    temp_model_name = f"temp_sub_model_{subgraph_index}.onnx"
    
    onnx.utils.extract_model(original_model_path, temp_model_name, subgraph_inputs, subgraph_outputs)
    
    
    run_args = ezkl.PyRunArgs()
    # settings_path = 
    if subgraph_index == 1 & is_final_sub_model == False:
        run_args.input_visibility = "private"
        run_args.param_visibility = "private"
        run_args.output_visibility = "hashed"
        
    elif subgraph_index == 1 & is_final_sub_model == True:
        run_args.input_visibility = "private"
        run_args.param_visibility = "private"
        run_args.output_visibility = "private"
        
    elif subgraph_index != 1 & is_final_sub_model == False:
        run_args.input_visibility = "hashed"
        run_args.param_visibility = "private"
        run_args.output_visibility = "hashed"
        
    elif subgraph_index != 1 & is_final_sub_model == True:
        run_args.input_visibility = "hashed"
        run_args.param_visibility = "private"
        run_args.output_visibility = "public"
        
    else:
        raise ValueError("Invalid subgraph index and final model flag.")
    
    res = ezkl.gen_settings(temp_model_name, py_run_args=run_args)
    assert res == True
    
    data_path = f"input_data_{subgraph_index}.json"
    generate_random_input_data(temp_model_name, data_path)
    
    res = ezkl.calibrate_settings(data_path, temp_model_name)
    
    setting_file_path = "settings.json"
    
    is_pass, pow = check_total_assignments(setting_file_path, upper_bound)
    print(f"Subgraph {subgraph_index}: Inputs: {subgraph_inputs}, Outputs: {subgraph_outputs}, Pow: {pow}, Is Pass: {is_pass}")
    
    ## remove temp model and data
    if not is_final_sub_model:
        os.remove(temp_model_name)
    # os.remove(data_path)
    
    return is_pass, pow

def save_graph(G, file_name):
    nx.draw(G, with_labels=True, font_size=5)
    plt.savefig(file_name)
    
def print_node_names_and_types(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    # Iterate through each node in the graph
    for node in graph.node:
        # Get the node's name and type
        node_name = node.name if node.name else "Unnamed"
        node_type = node.op_type
        
        # Print the node's name and type
        print(f"Node Name: {node_name}, Node Type: {node_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ONNX model and generate subgraphs.")
    parser.add_argument("--onnx_model_path", type=str, default='./resnet18.onnx', help="Path to the ONNX model. Default is './resnet18.onnx'.")
    parser.add_argument("--simplified_model_path", type=str, default='./resnet18_simplified.onnx', help="Path to save the simplified ONNX model. Default is './resnet18_simplified.onnx'.")
    parser.add_argument("--upper_bound_per_subgraph", type=int, default=23, help="Upper bound per subgraph in the form of 2^n. Default is 23.")
    parser.add_argument("--input_shape", type=str, default='{"input": [1, 3, 224, 224]}', 
                        help="Input shape for the ONNX model in JSON format. Default is '{\"input\": [1, 3, 224, 224]}'.")
    parser.add_argument("--simplify", action='store_true', 
                        help="Flag to indicate if the model should be simplified. Default is False.")
    
    args = parser.parse_args()
    
    # If simplifying the model is required
    if args.simplify:
        input_shape = json.loads(args.input_shape) if args.input_shape else {}
        simplify_onnx_model(args.onnx_model_path, input_shape, args.simplified_model_path)
    else:
        # If not simplifying, use the original model path
        args.simplified_model_path = args.onnx_model_path

    # Load the ONNX model into a graph
    G, ops = load_onnx_model_to_graph(args.simplified_model_path)
    
    # Process the graph
    process_graph(args.simplified_model_path, G, ops, args.upper_bound_per_subgraph)


