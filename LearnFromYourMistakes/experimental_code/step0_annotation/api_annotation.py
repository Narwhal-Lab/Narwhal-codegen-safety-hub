
"""Evaluate Assistant's response harmlessness and helpfulness by GPT4"""

from __future__ import annotations
import os
import hashlib
import re
import logging
import time
import json
from typing import Any, Callable

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

import ray

from tqdm import tqdm
from api_prompt import user_prompt
from api_prompt import system_prompt

@ray.remote(num_cpus=1)
def Wrap_gpt_api(
    system_content: str,
    user_content: str,
    image_url: str | None = None,
    post_process: Callable = lambda x: x,
) -> Any:
    from urllib3.util.retry import Retry 
    import urllib3

   

    retry_strategy = Retry(
        total=5,  # Maximum retry count
        backoff_factor=0.1,  # Wait factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
        allowed_methods=['POST'],  # Retry only for POST request
        raise_on_redirect=False,  # Don't raise exception
        raise_on_status=False,  # Don't raise exception
    )
    http = urllib3.PoolManager(
        retries=retry_strategy,
    )   
    
    messages = [
        {'role': 'system', 'content': system_content},
        {
            'role': 'user',
            "content": [
                {"type": "text", "text": user_content},

            ],
        },
    ]

    openai_api = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    params_gpt = {
        'model': 'gpt-4o',
        'messages': messages,
        'temperature': 0.05,
    }
    url = openai_api + '/v1/chat/completions'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': API_KEY,
        'Connection':'close',
        }


    encoded_data = json.dumps(params_gpt).encode('utf-8')
    max_try = 1000
    while max_try > 0:
        try:
            response = http.request('POST', url, body=encoded_data, headers=headers)
            if response.status == 200:
                    response = json.loads(response.data.decode('utf-8'))['choices'][0]['message']['content']
                    logging.info(response)
                    break
            else:
                err_msg = f'Access openai error, status code: {response.status} response: {response.data.decode("utf-8")}'
                logging.error(err_msg)
                time.sleep(3)
                max_try -= 1
                continue
        except:
            err_msg = f'Access openai error, status code: {response.status} response: {response.data.decode("utf-8")}'
            logging.error(err_msg)
            time.sleep(3)
            max_try -= 1
            continue
    else:
        print('Wrap Proxy API Failed...')
        # print('Using OpenAI API...')
        # response = ray.get(gpt_api.remote(system_content, user_content))
        response = 'Wrap Proxy API Failed...'

    return post_process(response)

def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid

def api(
    system_contents: list[str],
    user_contents: list[str],
    image_urls: list[str] | None = None,
    num_workers: int = 50,
    post_process: Callable = lambda x: x,
    hash_checker: Callable = lambda x: True,
    cache_dir: str = None,
):
    """API"""
    if cache_dir is None:
        cache_dir = os.environ.get("TSP_CACHE_DIR", "./cache")
    if len(system_contents) != len(user_contents):
        raise ValueError('Length of system_contents and user_contents should be equal.')
    server = Wrap_gpt_api

    api_interaction_count = 0
    ray.init()

    contents = list(enumerate(zip(system_contents, user_contents)))
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
    uids = [generate_hash_uid(content) for content in contents]
    not_finished = []
    while True:

        if len(not_finished) == 0 and len(contents) == 0:
            break

        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            uid = uids[index]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    try:
                        result = json.load(f)
                    except:
                        os.remove(cache_path)
                        continue
                results[index] = result
                bar.update(1)
                continue

            future = server.remote(content[0], content[1], post_process)
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue

        indices, futures = zip(*not_finished)

        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)

        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results[finished_indices[i]], f, ensure_ascii=False, indent=4)

        not_finished = [(index, future) for index, future in not_finished if future not in finished]

        bar.update(len(finished))
    bar.close()

    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return results



def get_datasets(input_file):
    
    datasets_path = input_file
    
    with open(datasets_path, 'r') as f:
        data = json.load(f)

    final_dataset = []
    
    for item in tqdm(data, desc="Processing Entries"):
        
        final_item = item
    
        final_dataset.append(final_item)
    
    return final_dataset
            


def extract_results_output(input_string):
    # Find all keys (inside [[ ]])
    keys = re.findall(r'\[\[(.*?)\]\]', input_string)
    
    # Split the string by [[ ]] and trim the results
    values = re.split(r'\[\[(?:.*?)\]\]', input_string)[1:]
    values = [value.strip() for value in values]
    
    # Zip keys and values together
    result = dict(zip(keys, values))
    
    return result

def preference_generation(input_file, output_file):
    

    final_dataset = get_datasets(input_file)

    
    
    
    def post_process(response: str):
        return response
    
    final_data = []
    system_prompts = []
    
    user_prompts = [user_prompt.format(description=item['description'], code=item['func_src_after']) for item in final_dataset]
    
    system_prompts = [system_prompt] * len(final_dataset)
        
    results = api(system_prompts, user_prompts, post_process=post_process)
    
    for i, item in enumerate(final_dataset):
        
        source = final_dataset[i]
        
        final_item = item
        final_item['output'] = results[i]
        final_data.append(final_item)
        
        
        
    output_dir = output_file
    with open(output_dir, 'w', encoding='utf-8') as outfile:
        json.dump(final_data, outfile, ensure_ascii=False, indent=4)
    
    
def main(input_file, output_file) -> None:
    
    preference_generation(input_file, output_file)

# Useful for debug
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', type=str, default='./data/preference_images_datasets.json')
    
    parser.add_argument('--output_file', type=str, default='./data/preference_images_datasets_output.json')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    output_file = args.output_file
    
    
    main(input_file, output_file)
