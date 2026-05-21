import os
import json
from tqdm import tqdm

def convert_to_database_for_eval(model: str):

    data_to_eval_path = f"./SecurityEval/inference/output/{model}_inference_output.json"

    with open(data_to_eval_path, 'r') as f:
        json_data = json.load(f)

    base_path = f"./SecurityEval/Testcases_{model}/"

    # Iterate through JSON data and create files
    for item in tqdm(json_data):
        # Get ID and generated code
        file_id = item["ID"]
        generation_code = item["Generation"]
        
        # Extract CWE-xxx part
        folder_name = file_id.split('_')[0]
        
        # Create target path
        subfolder_path = os.path.join(base_path, folder_name)
        
        # Ensure directory exists
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Get filename from ID second part
        file_name = file_id.split('_')[1]
        # print(file_name)
        
        # Define file path
        file_path = os.path.join(subfolder_path, file_name)
        
        # Write generated code to corresponding .py file
        with open(file_path, 'w') as f:
            f.write(generation_code)

    # Check created directories and files
    os.listdir(base_path)


if __name__ == "__main__":
    import sys
    
    # Check if model name provided via command line
    if len(sys.argv) > 1:
        model = sys.argv[1]
        print(f"Using model specified via command line: {model}")
    else:
        # Default model name
        model = "c_llama_rd_4_1_correct"
        print(f"Using default model: {model}")
    
    convert_to_database_for_eval(model)