from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import argparse
import os

def create_llama_prompt(question: str, code_to_gen: str) -> str:
    """Create a prompt using the exact template format from training."""
    return (
        f'<s>[INST] {question} [/INST] {code_to_gen}'
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to input data file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output file')
    parser.add_argument('--model', type=str, default="CodeLlama-7b-Instruct-hf", help='Model name or path to the model')

    args = parser.parse_args()

    model = os.environ.get("TSP_MODEL_PATH", args.model)
    
    # Read data
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data = data[:]
    
    # Set parameters
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        max_tokens=4096
    )

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=4,
        max_model_len=16024,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        trust_remote_code=True
    )

    # Collect all prompt information
    prompt_info_list = []
    for data_idx, d in enumerate(data):
        for node_idx, node in enumerate(d['Nodes']):
            node_code = node['Code_Line'].strip()
            code_before_node = d['func_src_after'].split(node_code)[0]
            
            # Create prompt
            tokenized_prompt = create_llama_prompt(d['description'], code_before_node)
            # print(tokenized_prompt.length)
            prompt_info_list.append({
                'prompt': tokenized_prompt,
                'code_before': code_before_node,
                'data_idx': data_idx,
                'node_idx': node_idx
            })

    # Batch generation
    prompts = [pi['prompt'] for pi in prompt_info_list]
    outputs = llm.generate(prompts, sampling_params)

    # Assign results to each node
    for output, pi in zip(outputs, prompt_info_list):
        generated_text = output.outputs[0].text.strip()
        data_idx = pi['data_idx']
        node_idx = pi['node_idx']
        code_before = pi['code_before']
        
        # Update generated code for current node
        data[data_idx]['Nodes'][node_idx]['Generated_Code'] = f"{code_before}{generated_text}"

    # Save final generated results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
if __name__ == "__main__":
    main()
