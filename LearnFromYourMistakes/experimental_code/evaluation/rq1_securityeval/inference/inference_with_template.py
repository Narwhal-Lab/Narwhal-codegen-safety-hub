import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import os


def create_cllama_prompt(ID: str, Prompt: str) -> str:
    user_message = f"Please directly complete the {ID} without explanation. {Prompt}"
    return f"<s>[INST] {user_message} [/INST]"
    
    
def main():
    parser = argparse.ArgumentParser(description='Run inference using vLLM')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--temperature', type=float, default=0.4, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum number of tokens')
    parser.add_argument('--gpu_mem_util', type=float, default=0.85, help='GPU memory utilization')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7', help='Comma-separated GPU IDs to use (e.g. "0,1,2,3")')
    parser.add_argument('--tensor_parallel_size', type=int, default=8, help='Number of GPUs for tensor parallelism')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(f"GPU IDs: {args.gpu_ids}")

    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # data = data[:10]

    # Create prompts using the template
    prompts = [create_cllama_prompt(d['ID'], d['Prompt']) for d in data]

    # Initialize model and tokenizer
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size, 
        max_model_len=args.max_tokens,
        max_num_seqs=16,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True
    )
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    final_output = []

    for i in range(len(outputs)):
        item = {}
        generated_text = outputs[i].outputs[0].text
        item = data[i]
        item['Generation'] = generated_text.strip()  # Remove any leading/trailing whitespace
        final_output.append(item)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    main()