import json
import re
import os

def process_json_item(item):
    """Processes a single JSON item and extracts information from the 'output' field.

    Args:
        item: A dictionary representing a single JSON item.

    Returns:
        A modified dictionary with a 'Nodes' key containing the extracted information,
        or the original item if 'output' is missing or invalid.
    """
    if 'output' not in item:
        return item

    output_text = item['output']
    nodes = []

    # Regular expression to find all [[n]] blocks and their content
    matches = re.findall(r'\[\[(\d+)\]\]\n(.*?)(?=\[\[\d+\]\]|\Z)', output_text, re.DOTALL)

    for match in matches:
        node_num = int(match[0])  # Convert the node number to an integer
        content = match[1].strip()  # Remove leading/trailing whitespace

        node = {}
        # Extract Code_Line, CWE_ID, and Description
        lines = content.split('\n')
        
        code_line = None
        cwe_id = None
        description = None
        
        for line in lines:
            if line.startswith("[Code_Line]"):
                code_line = line.replace("[Code_Line]", "").strip()
            elif line.startswith("[CWE_ID]"):
                cwe_id = line.replace("[CWE_ID]", "").strip()
            elif line.startswith("[Description]"):
                description = line.replace("[Description]", "").strip()

        if code_line is not None and cwe_id is not None and description is not None:
            node = {
                "Node_Number": node_num,
                "Code_Line": code_line,
                "CWE_ID": cwe_id,
                "Description": description
            }
            nodes.append(node)
        else:  # Handle the case where any required field is missing
            print(f"Warning: Incomplete data for node [[{node_num}]]. Skipping this node.")
            continue


    if nodes:
        item['Nodes'] = nodes
    else:
        print(f"Warning: No valid nodes found in output.  'Nodes' key not added.")

    return item


def process_json_file(input_file, output_file):
    """Processes a JSON file, extracts information, and writes the modified data to a new file.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
        return

    processed_data = []
    for item in data:
        processed_item = process_json_item(item)
        processed_data.append(processed_item)

    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(processed_data, f_out, indent=4)  # Use indent for readability
    except Exception as e:
        print(f"Error writing to output file: {e}")


import json
import argparse

def process_data(input_file, output_file):
    # Read original data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build preference pairs list
    preference_pairs = []
    
    for item in data:
        # Extract original code
        original_code = item['func_src_after']
        prompt = item['description']
        
        # Traverse all nodes
        for node in item['Nodes']:
            # Build preference pair
            pair = {
                "prompt": prompt,
                "original_code": original_code,
                "generated_code": node['Generated_Code']
            }
            preference_pairs.append(pair)
    
    print(f"Processed {len(preference_pairs)} preference pairs.")
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preference_pairs, f, indent=2, ensure_ascii=False)

def make_preference_pair(input_file, output_file):
    process_data(input_file, output_file)

def convert_preference_pairs_to_dpo(preference_pairs):
    """Convert preference pairs format to DPO training format"""
    dpo_data = []
    
    for item in preference_pairs:
        # Extract fields
        prompt = item.get('prompt', '')
        original_code = item.get('original_code', '')
        generated_code = item.get('generated_code', '')
        
        # Convert to DPO format
        dpo_item = {
            "instruction": f'"{prompt}"',  # Add quotes to match existing format
            "input": "",     # DPO format requires input field, usually empty
            "chosen": original_code,      # Original code as preferred (chosen)
            "rejected": generated_code    # Generated code as rejected (may have security issues)
        }
        
        dpo_data.append(dpo_item)
    
    return dpo_data

def convert_to_dpo_format(input_file, output_file):
    """Process single file format conversion"""
    try:
        # Read input file
        print(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            preference_pairs = json.load(f)
        
        print(f"Successfully read {len(preference_pairs)} preference pairs")
        
        # Format conversion
        print("Converting format...")
        dpo_data = convert_preference_pairs_to_dpo(preference_pairs)
        
        # Save converted data
        print(f"Saving converted data to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Format conversion complete! Generated {len(dpo_data)} DPO training samples")
        
    except Exception as e:
        print(f"Error: Exception during processing - {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Define base directories relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # Get TSP root directory
    
    # Example paths using relative paths
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Example file paths
    input_file = os.path.join(data_dir, "nodes_generation_rd1.json")
    output_file = os.path.join(output_dir, "generated_data_rd1.json")
    
    # Process model generations and create preference pairs
    make_preference_pair(input_file, output_file)
    print(f"Processing complete. Results written to '{output_file}'.")
    
    # Process another round
    input_file = os.path.join(data_dir, "nodes_generation_rd2.json")
    output_file = os.path.join(output_dir, "generated_data_rd2.json")
    make_preference_pair(input_file, output_file)
    print(f"Processing complete. Results written to '{output_file}'.")