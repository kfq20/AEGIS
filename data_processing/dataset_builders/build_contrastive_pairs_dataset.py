#!/usr/bin/env python3
"""
Create contrastive pairs dataset from positive and negative samples.
Match samples based on query+framework+benchmark as keys.
"""

import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return parsed data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_match_key(sample: Dict) -> Tuple[str, str, str]:
    """
    Create matching key from sample based on query+framework+benchmark.
    Returns tuple: (query_hash, framework, benchmark)
    """
    query = sample["input"]["query"].strip()
    framework = sample["metadata"]["framework"].lower()
    benchmark = sample["metadata"]["benchmark"].lower()
    
    # Create a simple hash of the query to handle long queries
    query_hash = str(hash(query))
    
    return (query_hash, framework, benchmark)

def create_match_key_readable(sample: Dict) -> Tuple[str, str, str]:
    """
    Create matching key with readable query prefix for debugging.
    """
    query = sample["input"]["query"].strip()
    framework = sample["metadata"]["framework"].lower()
    benchmark = sample["metadata"]["benchmark"].lower()
    
    # Use first 50 characters of query for readability
    query_prefix = query[:50].replace('\n', ' ').strip()
    
    return (query_prefix, framework, benchmark)

def group_samples_by_key(samples: List[Dict]) -> Dict[Tuple, List[Dict]]:
    """Group samples by their matching key."""
    grouped = defaultdict(list)
    
    for sample in samples:
        key = create_match_key(sample)
        grouped[key].append(sample)
    
    return dict(grouped)

def create_contrastive_pairs(positive_samples: List[Dict], negative_samples: List[Dict]) -> List[Dict]:
    """
    Create contrastive pairs from positive and negative samples.
    Each pair contains one positive sample and multiple negative samples for the same task.
    """
    # Group positive and negative samples by their keys
    positive_groups = group_samples_by_key(positive_samples)
    negative_groups = group_samples_by_key(negative_samples)
    
    contrastive_pairs = []
    matched_keys = 0
    unmatched_positive = 0
    unmatched_negative = 0
    
    print(f"Positive sample groups: {len(positive_groups)}")
    print(f"Negative sample groups: {len(negative_groups)}")
    
    # Count samples for better statistics
    matched_positive_samples = 0
    matched_negative_samples = 0
    unmatched_positive_samples = 0
    unmatched_negative_samples = 0
    
    # Find matches between positive and negative samples
    for key, pos_samples in positive_groups.items():
        if key in negative_groups:
            matched_keys += 1
            neg_samples = negative_groups[key]
            matched_positive_samples += len(pos_samples)
            matched_negative_samples += len(neg_samples)
            
            # For debugging, get readable key
            readable_key = create_match_key_readable(pos_samples[0])
            
            # Create a contrastive pair for each positive sample
            for pos_sample in pos_samples:
                pair = {
                    "query": pos_sample["input"]["query"],
                    "key_info": {
                        "framework": key[1],
                        "benchmark": key[2],
                        "query_hash": key[0],
                        "readable_query": readable_key[0]
                    },
                    "positive_sample": pos_sample,
                    "negative_samples": neg_samples
                }
                contrastive_pairs.append(pair)
        else:
            unmatched_positive += 1
            unmatched_positive_samples += len(pos_samples)
    
    # Count unmatched negative samples
    for key, neg_samples in negative_groups.items():
        if key not in positive_groups:
            unmatched_negative += 1
            unmatched_negative_samples += len(neg_samples)
    
    print(f"\nMatching results:")
    print(f"  Matched keys: {matched_keys}")
    print(f"  Unmatched positive groups: {unmatched_positive}")
    print(f"  Unmatched negative groups: {unmatched_negative}")
    print(f"  ")
    print(f"  Matched positive samples: {matched_positive_samples}")
    print(f"  Matched negative samples: {matched_negative_samples}")
    print(f"  Unmatched positive samples: {unmatched_positive_samples}")
    print(f"  Unmatched negative samples: {unmatched_negative_samples}")
    print(f"  Total contrastive pairs created: {len(contrastive_pairs)}")
    
    return contrastive_pairs

def save_contrastive_pairs(pairs: List[Dict], output_path: str):
    """Save contrastive pairs to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Contrastive pairs saved to: {output_path}")

def main():
    # File paths
    positive_path = "data_processing/unified_dataset_with_normal/unified_training_dataset.json"
    negative_path = "data_processing/unified_dataset/unified_training_dataset_easy.json"
    output_path = "data_processing/contrastive_pairs_dataset.json"
    
    print("Loading data files...")
    
    # Load positive and negative samples
    try:
        positive_samples = load_json_file(positive_path)
        negative_samples = load_json_file(negative_path)
        
        print(f"Loaded {len(positive_samples)} positive samples")
        print(f"Loaded {len(negative_samples)} negative samples")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create contrastive pairs
    print("\nCreating contrastive pairs...")
    pairs = create_contrastive_pairs(positive_samples, negative_samples)
    
    # Save results
    save_contrastive_pairs(pairs, output_path)
    
    # Print some statistics
    if pairs:
        print(f"\nSample statistics:")
        neg_counts = [len(pair["negative_samples"]) for pair in pairs]
        print(f"  Average negative samples per positive: {sum(neg_counts) / len(neg_counts):.2f}")
        print(f"  Min negative samples: {min(neg_counts)}")
        print(f"  Max negative samples: {max(neg_counts)}")
        
        # Show some examples
        print(f"\nFirst few pairs:")
        for i, pair in enumerate(pairs[:3]):
            print(f"  Pair {i+1}: {pair['key_info']['framework']}+{pair['key_info']['benchmark']}")
            print(f"    Query: {pair['key_info']['readable_query']}...")
            print(f"    Negative samples: {len(pair['negative_samples'])}")

if __name__ == "__main__":
    main()