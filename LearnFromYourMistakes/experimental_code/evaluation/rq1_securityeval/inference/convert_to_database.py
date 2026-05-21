import os
import json
from tqdm import tqdm

def convert_to_database_for_eval(model: str):

    data_to_eval_path = f"./SecurityEval/inference/output/{model}_inference_output.json"

    with open(data_to_eval_path, 'r') as f:
        json_data = json.load(f)

    base_path = f"./SecurityEval/Testcases_{model}/"

    
    for item in tqdm(json_data):

        file_id = item["ID"]
        generation_code = item["Generation"]
        
        folder_name = file_id.split('_')[0]
        
        subfolder_path = os.path.join(base_path, folder_name)

        os.makedirs(subfolder_path, exist_ok=True)

        file_name = file_id.split('_')[1]
        file_suffix = file_id.split('_')[2]
        file_name = f"{file_name}_{file_suffix}"
        # print(file_name)

        file_path = os.path.join(subfolder_path, file_name)
        
        with open(file_path, 'w') as f:
            f.write(generation_code)

    os.listdir(base_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model = sys.argv[1]
        print(f"Converting database for model: {model}")
    else:
        print("Usage: python convert_to_database.py <model_name>")
        print("Example: python convert_to_database.py codellama-7b-self-play-dpo-4-2")
        sys.exit(1)

    convert_to_database_for_eval(model)