#!/usr/bin/env python3
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import os


def create_cllama_prompt(ID: str, Prompt: str) -> str:
    return (
        f'<s>[INST] Please directly complete the code generation task without explanation.\n\n'
        f'{Prompt} [/INST]'
    )

def create_starcoder_prompt(ID: str, Prompt: str) -> str:
    """
    Create appropriate prompt format for StarCoder model
    StarCoder is a code generation model, no chat template needed
    """
    return f"// Complete the following code task:\n// {Prompt}\n\n"

def create_chat_template_prompt(tokenizer, ID: str, Prompt: str) -> str:
    """
    Create prompt using tokenizer's apply_chat_template method
    For models that support chat template
    """
    messages = [
        {"role": "user", "content": Prompt}
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        print(f"Warning: apply_chat_template failed for ID {ID}: {e}")
        # Fallback to simple format if apply_chat_template fails
        return f"User: {Prompt}\nAssistant:"

def detect_model_type(model_path: str) -> str:
    """
    Detect model type based on model path
    """
    model_path_lower = model_path.lower()
    if 'starcoder' in model_path_lower:
        return 'starcoder'
    elif 'qwen' in model_path_lower or 'deepseek' in model_path_lower or 'mixtral' in model_path_lower or 'gemma' in model_path_lower or 'phi' in model_path_lower:
        return 'chat_template'
    elif 'codellama' in model_path_lower or 'llama' in model_path_lower:
        return 'codellama'
    else:
        # Default to chat_template format (more suitable for modern models)
        return 'chat_template'
    
def main():
    parser = argparse.ArgumentParser(description='Run inference using vLLM')
    parser.add_argument('--model', type=str, required=True, help='Path to the model')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum number of tokens')
    parser.add_argument('--gpu_mem_util', type=float, default=0.85, help='GPU memory utilization')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='Comma-separated GPU IDs to use (e.g. "0,1,2,3")')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--force_template', type=str, choices=['starcoder', 'codellama', 'chat_template'], 
                        help='Force use specific template type instead of auto-detection')
    
    args = parser.parse_args()
    
    # Detect model type
    if args.force_template:
        model_type = args.force_template
        print(f"Forced template type: {model_type}")
    else:
        model_type = detect_model_type(args.model)
        print(f"Detected model type: {model_type}")
    
    # Set GPU devices to use
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(f"Using GPU IDs: {args.gpu_ids}")

    # Load input data
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # data = data[:10]

    # Initialize tokenizer first (needed for chat_template)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create prompts using the appropriate template
    if model_type == 'starcoder':
        prompts = [create_starcoder_prompt(d['ID'], d['Prompt']) for d in data]
        print("Using StarCoder prompt format")
    elif model_type == 'codellama':
        prompts = [create_cllama_prompt(d['ID'], d['Prompt']) for d in data]
        print("Using CodeLlama prompt format")
    elif model_type == 'chat_template':
        prompts = [create_chat_template_prompt(tokenizer, d['ID'], d['Prompt']) for d in data]
        print("Using Chat Template prompt format")
        # Print first prompt example
        if prompts:
            print(f"Chat Template example:\n{prompts[0][:300]}...")
    
    # Initialize model and tokenizer
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,  # Use specified number of GPUs for parallelism
        max_model_len=args.max_tokens,
        max_num_seqs=16,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True
    )

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    final_output = []

    for i in range(len(outputs)):
        # Get original output object
        output = outputs[i]
        # Get text of first generation result
        generated_text = output.outputs[0].text
        
        # Print detailed info of model's direct output
        print(f"\n=== Sample {i+1}/{len(outputs)} Model Raw Output ===")
        print(f"Request ID: {output.request_id}")
        print(f"Model type: {model_type}")
        print(f"Prompt: {output.prompt[:100]}...")
        print(f"Number of outputs: {len(output.outputs)}")
        print(f"First output text: {generated_text[:300]}...")
        print(f"Finish reason: {output.outputs[0].finish_reason}")
        
        print(f"\nOriginal prompt:\n{prompts[i][:200]}...")
        print(f"\nComplete model output:\n{generated_text}")
        print("-" * 80)
        
        item = data[i].copy()  # Create copy of original data item
        # Save only model generated text, not including original prompt
        item['Generation'] = generated_text.strip()  # Only include model generated text
        item['ModelType'] = model_type  # Record model type used
        final_output.append(item)

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"\nInference complete, results saved to: {args.output}")
    print(f"Model type used: {model_type}")

if __name__ == "__main__":
    main()