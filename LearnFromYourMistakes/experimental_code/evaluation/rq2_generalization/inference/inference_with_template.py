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

def create_qwen_coder_prompt(ID: str, Prompt: str) -> str:
    """
    Create more suitable prompt format for Qwen2.5-Coder model
    Use simple instruction format, avoid complexity of chat template
    """
    return f"Please complete the following code generation task:\n\n{Prompt}\n\nCode:"

def create_chat_template_prompt(tokenizer, ID: str, Prompt: str) -> str:
    """
    Create prompt using tokenizer's apply_chat_template method
    Suitable for models supporting chat template
    """
    enhanced_prompt = f"Please complete the following code generation task. Only provide the code without explanation:\n\n{Prompt}"
    
    messages = [
        {"role": "user", "content": enhanced_prompt}
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
        return f"Please complete the following code generation task. Only provide the code without explanation:\n\n{Prompt}\n\nCode:"

def detect_model_type(model_path: str) -> str:
    """
    Detect model type based on model path
    """
    model_path_lower = model_path.lower()
    if 'starcoder' in model_path_lower:
        return 'starcoder'
    elif any(x in model_path_lower for x in ['qwen', 'deepseek', 'mixtral', 'gemma', 'phi']):
        return 'chat_template'
    elif 'codellama' in model_path_lower or 'llama' in model_path_lower:
        return 'codellama'
    else:
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
    
    if args.force_template:
        model_type = args.force_template
        print(f"Forcing template type: {model_type}")
    else:
        model_type = detect_model_type(args.model)
        print(f"Detected model type: {model_type}")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print(f"Using GPU IDs: {args.gpu_ids}")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    prompts = []
    valid_data_indices = []
    
    for i, d in enumerate(data):
        prompt_text = None
        item_id = None
        
        if 'Prompt' in d:
            prompt_text = d['Prompt']
        elif 'prompt' in d:
            prompt_text = d['prompt']
        else:
            print(f"Warning: Cannot find prompt field, skipping item {i}")
            continue
            
        if 'ID' in d:
            item_id = d['ID']
        elif 'id' in d:
            item_id = d['id']
        else:
            item_id = f"sample_{i+1:04d}"
        
        if model_type == 'starcoder':
            prompt = create_starcoder_prompt(item_id, prompt_text)
        elif model_type == 'codellama':
            prompt = create_cllama_prompt(item_id, prompt_text)
        elif model_type == 'chat_template':
            prompt = create_chat_template_prompt(tokenizer, item_id, prompt_text)
        else:
            prompt = create_chat_template_prompt(tokenizer, item_id, prompt_text)
        
        prompts.append(prompt)
        valid_data_indices.append(i)
    
    print(f"Using {model_type} prompt format, processed {len(prompts)} valid samples")
    
    if prompts:
        print(f"Prompt example:\n{prompts[0][:500]}...")
    
    if 'qwen' in args.model.lower() and 'coder' in args.model.lower():
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            stop=["<|endoftext|>", "<|im_end|>", "\n\n\n"]
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
    
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_tokens,
        max_num_seqs=16,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True
    )

    if not prompts:
        print("No valid prompts found, exiting")
        return

    outputs = llm.generate(prompts, sampling_params)
    final_output = []

    for i in range(len(outputs)):
        output = outputs[i]
        generated_text = output.outputs[0].text
        
        print(f"\n=== Sample {i+1}/{len(outputs)} Model Raw Output ===")
        print(f"Request ID: {output.request_id}")
        print(f"Model Type: {model_type}")
        print(f"Prompt: {output.prompt[:200]}...")
        print(f"Number of outputs: {len(output.outputs)}")
        print(f"First output text: {generated_text[:500]}...")
        print(f"Finish reason: {output.outputs[0].finish_reason}")
        
        if len(generated_text.strip()) < 10:
            print(f"Warning: Generated text too short (length: {len(generated_text.strip())})")
            print(f"Raw generated text: '{generated_text}'")
        
        print(f"\nOriginal prompt:\n{prompts[i][:300]}...")
        print(f"\nComplete model output:\n{generated_text}")
        print("-" * 80)
        
        data_idx = valid_data_indices[i] if i < len(valid_data_indices) else i
        if data_idx < len(data):
            item = data[data_idx].copy()
            item['Generation'] = generated_text.strip()
            item['ModelType'] = model_type
            
            if 'ID' not in item and 'id' not in item:
                item['ID'] = f"sample_{i+1:04d}"
            
            final_output.append(item)
        else:
            print(f"Warning: Data index {data_idx} out of range")

    output_dir = os.path.join(os.path.dirname(args.output), "generated_code")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    
    for i, item in enumerate(final_output):
        if 'Generation' in item:
            code_filename = f"code_{i+1:04d}.txt"
            code_path = os.path.join(output_dir, code_filename)
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(item['Generation'])
    
    print(f"\nInference completed, results saved to: {args.output}")
    print(f"Generated code saved in directory: {output_dir}")
    print(f"Model type used: {model_type}")
    print(f"Number of samples with short code: {sum(1 for item in final_output if len(item.get('Generation', '').strip()) < 10)}")

if __name__ == "__main__":
    main()