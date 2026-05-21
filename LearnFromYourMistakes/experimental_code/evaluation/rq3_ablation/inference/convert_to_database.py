import os
import json
import argparse
from tqdm import tqdm

def convert_to_database_for_eval(model: str):

    data_to_eval_path = f"./output/{model}_inference_output.json"

    with open(data_to_eval_path, 'r') as f:
        json_data = json.load(f)

    base_path = f"./Testcases_{model}/"

    # Iterate through JSON data and create files
    for item in tqdm(json_data):
        # Get ID and generated code
        file_id = item["ID"]
        generation_code = item["Generation"]
        
        parts = file_id.split('_')

        if len(parts) >= 2:
            file_name = parts[1]
        else:
            # If no underscore, use the entire ID as filename
            file_name = file_id
    
        # Create target path
        subfolder_path = os.path.join(base_path)
        
        # Ensure directory exists
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Define file path
        file_path = os.path.join(subfolder_path, f"{file_name}.c")
        
        # Write generated code to corresponding .c file
        with open(file_path, 'w') as f:
            f.write(generation_code)

    # Check created directories and files
    os.listdir(base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert inference output to database format")
    parser.add_argument("model", nargs="?", help="Model name")
    args = parser.parse_args()

    if args.model:
        # If command line argument provided, use it
        convert_to_database_for_eval(args.model)
    else:
        # If no argument provided, use default hardcoded behavior
        model = "qwen2.5-coder-7b-instruct"
        convert_to_database_for_eval(model)
        model = "qwen2.5-coder-7b-dpo-security-4-ablation"
        convert_to_database_for_eval(model)
        model = "qwen2.5-coder-7b-dpo-security-4-ablation-sft"
        convert_to_database_for_eval(model)