import json
import logging
import os
import re
import threading
import time

import requests

MODEL_API_FAILED = 'Model API failed. Please try again later'
class custom_model:

    def __init__(self, model, url, token):
        self.url = url
        self.token = token
        self.model = model

    def send(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [  # Used for chat
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ],
            # "prompt": f"{prompt}", # Relay mode does not use this
            # "stream": True
        }

        if "https" not in self.url:
            payload["prompt"] = f"{prompt}"
            # payload["stream"] = False
            # vLLM does not require the model to be specified
            del payload["model"]

        if 'starcoder2-15b' in self.model:
            del payload["messages"]
            payload["max_tokens"] = 2048
            # payload["temperature"] = 0.2

        if 'CodeQwen1.5-7B' in self.model:
            # Add a stop token, otherwise the response may continue indefinitely and block subsequent requests
            payload["messages"][0]['stop'] = ["<|im_end|>"]
        if 'codegemma-1.1-7b-it' in self.model:
            # Add a stop token, otherwise the response may continue indefinitely and block subsequent requests
            # payload["messages"][0]['stop'] = ["<|im_end|>"]
            # payload['max_tokens'] = 8192
            pass

        if "claude" in self.model:
            pass
            # payload["stream"] = True

        # Send request
        try:
            response = requests.request("POST", self.url, json=payload, headers=headers, timeout=60 * 3)
            response.raise_for_status()  # Check for HTTP errors


            response_json = response.json()
            if "chat" not in self.url:
                return response_json["choices"][0]["text"]

            if "message" in response_json and "content" in response_json["message"]:
                return response_json["message"]["content"]
            elif "response" in response_json:  # Some Ollama versions or specific models may use the 'response' field
                return response_json["response"]
            elif "choices" in response_json and "message" in response_json["choices"][0]:
                return response_json["choices"][0]["message"]["content"]
            else:
                # If content cannot be found, log the full response for debugging
                logging.error(f"Error: Unexpected response format from Ollama: {response_json}")
                return "error: Unexpected response format"
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Server response status: {e.response.status_code}")
                logging.error(f"Server response body: {e.response.text}")
            return MODEL_API_FAILED
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
            logging.error(f"Raw response text: {response.text}")
            return MODEL_API_FAILED
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return MODEL_API_FAILED
        # try:
        #     response = requests.request("POST", self.url, json=payload, headers=headers)
        #     return response.json()["choices"][0]["message"]["content"]
        # except:
        #     return "error"

def failed_re_do_factory(wait_func):
    def failed_re_do_decorator(func):
        def wrapper(*args, **kwargs):
            # Track retry attempts
            num_of_attempts = 0
            # Call the function
            generate_ori_text = func(*args, **kwargs)
            # Check whether the API call failed
            is_failed = MODEL_API_FAILED in generate_ori_text
            # Retry if the API call failed and fewer than 3 attempts have been made
            while is_failed and num_of_attempts < 3:
                # Record the current thread name
                logging.warning(f"{threading.current_thread().name}: API call failed, retry attempt {num_of_attempts + 1}")
                # Wait for a while
                time.sleep(wait_func(num_of_attempts))
                # Call the function again
                generate_ori_text = func(*args, **kwargs)
                # Check whether the API call failed
                is_failed = MODEL_API_FAILED in generate_ori_text
                # Increase the retry count
                num_of_attempts += 1
            # Return the result
            return generate_ori_text
        return wrapper
    return failed_re_do_decorator


def remove_think_section(input_string: str) -> str:
    """
    Remove all <think>...</think> sections from the input string, including the tags themselves.

    Args:
        input_string (str): The original string containing <think>...</think> sections.

    Returns:
        str: The string with all <think>...</think> sections removed.
    """
    # Match <think>...</think> tags and everything inside them.
    # The re.DOTALL flag allows '.' to match newlines because the <think> block may span multiple lines.
    # '.*?' is non-greedy so it matches the nearest </think> instead of the last one in the string.
    pattern = r"<think>.*?</think>"

    # Replace all matches with an empty string.
    cleaned_string = re.sub(pattern, "", input_string, flags=re.DOTALL)

    return cleaned_string


def config_model():
    """
    Configure the set of model endpoints used in the experiments.

    All hosts, paths, and credentials are read from environment variables
    so that nothing sensitive is committed to the repository. See
    `.env.example` for the full list of supported variables.
    """
    # Local vLLM / Ollama server hosting the open-source models.
    vllm_host = os.environ.get("VLLM_HOST", "localhost")
    # Filesystem prefix where the open-source model weights live, as
    # passed to vLLM's `--model` flag.
    vllm_model_base = os.environ.get("VLLM_MODEL_BASE", "/path/to/your/vllm")
    # Bearer token used by the relay/proxy in front of the local vLLM
    # service (leave empty if your local server does not require one).
    vllm_relay_key = os.environ.get("VLLM_RELAY_API_KEY", "")

    # Closed-source / hosted models go through an OpenAI-compatible
    # endpoint. Configure both URL and key per provider, or share one
    # multi-model relay by pointing them all at the same URL+key.
    kimi_api_url = os.environ.get(
        "KIMI_API_URL", "https://api.moonshot.cn/v1/chat/completions"
    )
    kimi_api_key = os.environ.get("KIMI_API_KEY", "")

    proxy_url = os.environ.get(
        "LLM_PROXY_URL", "https://api.openai.com/v1/chat/completions"
    )
    proxy_key = os.environ.get("LLM_PROXY_API_KEY", "")

    model_map = {}
    model_map["qwen3:32b"] = custom_model(
        "qwen3:32b",
        f"http://{vllm_host}:11435/v1/chat/completions",
        "nil",
    )
    model_map["DeepSeek-Coder-V2-Instruct"] = custom_model(
        "DeepSeek-Coder-V2-Instruct",
        f"http://{vllm_host}:11436/v1/chat/completions",
        "nil",
    )
    model_map["CodeQwen1.5-7B"] = custom_model(
        "CodeQwen1.5-7B",
        f"http://{vllm_host}:11437/v1/chat/completions",
        "nil",
    )

    model_map["mixtral:8x7b"] = custom_model(
        "mixtral:8x7b",
        f"http://{vllm_host}:11438/v1/chat/completions",
        vllm_relay_key,
    )
    model_map["codegemma-1.1-7b-it"] = custom_model(
        f"{vllm_model_base}/codegemma-1.1-7b-it",
        f"http://{vllm_host}:11439/v1/chat/completions",
        vllm_relay_key,
    )
    model_map["Llama-4-Scout-17B-16E-Instruct"] = custom_model(
        f"{vllm_model_base}/Llama-4-Scout-17B-16E-Instruct",
        f"http://{vllm_host}:11439/v1/chat/completions",
        vllm_relay_key,
    )

    model_map["starcoder2-15b"] = custom_model(
        f"{vllm_model_base}/starcoder2-15b",
        f"http://{vllm_host}:11435/v1/completions",
        vllm_relay_key,
    )
    model_map["Qwen3-14B"] = custom_model(
        f"{vllm_model_base}/Qwen3-14B",
        f"http://{vllm_host}:11436/v1/chat/completions",
        vllm_relay_key,
    )

    model_map["kimi-k2"] = custom_model(
        "kimi-k2-0711-preview",
        kimi_api_url,
        kimi_api_key,
    )

    # =============== Closed-source ===============
    model_map["gemini-2.5-flash"] = custom_model(
        "gemini-2.5-flash",
        proxy_url,
        proxy_key,
    )
    model_map["gpt-4o"] = custom_model(
        "gpt-4o",
        proxy_url,
        proxy_key,
    )
    model_map["grok-3"] = custom_model(
        "grok-3",
        proxy_url,
        proxy_key,
    )
    model_map["claude-sonnet-4-20250514"] = custom_model(
        "claude-sonnet-4-20250514",
        proxy_url,
        proxy_key,
    )
    return model_map
