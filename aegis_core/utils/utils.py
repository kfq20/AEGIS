import os
import json
import time
from functools import wraps

def load_model_api_config(model_api_config, model_name):
    with open(model_api_config, "r") as f:
        model_api_config = json.load(f)
    for model_name in model_api_config:
        actural_max_workers = model_api_config[model_name]["max_workers_per_model"] * len(model_api_config[model_name]["model_list"])
        model_api_config[model_name]["max_workers"] = actural_max_workers
        
        # 确保每个 model_list 项都有正确的 provider 配置
        for model_config in model_api_config[model_name]["model_list"]:
            # 如果没有指定 provider，根据配置推断
            if "provider" not in model_config:
                if "azure_endpoint" in model_config:
                    model_config["provider"] = "azure"
                elif "model_url" in model_config and "generativelanguage.googleapis.com" in model_config["model_url"]:
                    model_config["provider"] = "gemini"
                else:
                    model_config["provider"] = "openai"
    return model_api_config

def write_to_jsonl(lock, file_name, data):
    with lock:
        with open(file_name, 'a') as f:
            json.dump(data, f)
            f.write('\n')

def reserve_unprocessed_queries(output_path, test_dataset):
    processed_queries = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                infered_sample = json.loads(line)
                processed_queries.add(infered_sample["query"])

    test_dataset = [sample for sample in test_dataset if sample["query"] not in processed_queries]
    return test_dataset

def retry_on_exception(retries=3, delay=2, backoff=2, exceptions_to_catch=None):
    """
    A decorator to synchronously retry a function call upon specific exceptions.

    Args:
        retries (int): The maximum number of retries.
        delay (int): The initial delay between retries in seconds.
        backoff (int): The factor by which the delay should increase after each retry.
        exceptions_to_catch (tuple): A tuple of Exception classes to catch and retry on.
    """
    if exceptions_to_catch is None:
        # A general catch-all, but specific exceptions are better.
        exceptions_to_catch = (Exception,)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Make copies of the mutable parameters
            mtries, mdelay = retries, delay

            while mtries > 1:
                try:
                    # Try to execute the function
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    # If a specified error occurs, print a message, wait, and retry
                    print(f"Call to '{func.__name__}' failed: {e}. Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            
            # This is the final attempt
            return func(*args, **kwargs)
        return wrapper
    return decorator