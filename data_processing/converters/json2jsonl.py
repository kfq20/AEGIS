# json to jsonl
import json
import os

def json_to_jsonl(json_file, jsonl_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    with open(jsonl_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

input_file = 'data_processing/unified_dataset/unified_training_dataset_who_when.json'
output_file = 'data_processing/unified_dataset/unified_training_dataset_who_when.jsonl'
json_to_jsonl(json_file=input_file, jsonl_file=output_file)