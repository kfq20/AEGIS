#!/usr/bin/env python3
import json
import os
import uuid
from typing import Dict, List, Any
from pathlib import Path

INPUT_FILE = "datasets/who_when_classified_dataset.json"
OUTPUT_FILE = "data_processing/unified_dataset/unified_training_dataset_who_when.jsonl"

def convert_history_to_conversation_history(history: List[Dict]) -> List[Dict]:
    """Convert who_when history format to conversation_history format"""
    conversation_history = []
    
    for i, step in enumerate(history, 1):
        # Skip Computer_terminal steps as they're system outputs
        assistant_name = step.get("name", step.get("role", "Unknown"))
        if assistant_name == "Assistant" or assistant_name == "Computer_terminal":
            continue
            
        conversation_step = {
            "step": i,
            "agent_name": step.get("name", step.get("role", "Unknown")),
            "agent_role": step.get("role", "Unknown"),
            "content": step.get("content", ""),
            "phase": "reasoning"  # Default phase for who_when data
        }
        if conversation_step["agent_name"] == "Assistant":
            print("warning: assistant name is not standard")
        conversation_history.append(conversation_step)
    
    return conversation_history

def extract_final_output(history: List[Dict]) -> str:
    """Extract the final output from history"""
    # Look for the last non-terminate content
    for step in reversed(history):
        if step.get("content") and step.get("content") != "TERMINATE":
            return step.get("content", "")
    return ""

def convert_sample(sample: Dict, sample_id: str) -> Dict[str, Any]:
    """Convert a who_when sample to unified format"""
    
    # Extract conversation history
    history = sample.get("history", [])
    conversation_history = convert_history_to_conversation_history(history)
    
    # Extract final output
    final_output = extract_final_output(history)
    
    # Create metadata
    metadata = {
        "framework": "who_when",  # Custom framework name
        "benchmark": "who_when",  # Custom benchmark name
        "model": "unknown",  # Not specified in who_when
        "num_agents": len(set(step.get("name") for step in history if step.get("name") != "Computer_terminal")),
        "num_injected_agents": 0,  # Not applicable for who_when
        "task_type": "general"  # Default task type
    }
    
    # Create input
    input_obj = {
        "query": sample.get("question", ""),
        "conversation_history": conversation_history,
        "final_output": final_output
    }
    
    # Create output (faulty agents info)
    mistake_agent = sample.get("mistake_agent", "")
    fm_error_type = sample.get("fm_error_type", "")
    
    output_obj = {
        "faulty_agents": [{
            "agent_name": mistake_agent,
            "error_type": fm_error_type,
            "injection_strategy": "unknown"  # Not applicable for who_when
        }] if mistake_agent and fm_error_type else []
    }
    
    # Create ground truth
    ground_truth_obj = {
        "correct_answer": sample.get("ground_truth", ""),
        "injected_agents": [],  # Not applicable for who_when
        "is_injection_successful": False  # Not injection-based
    }
    
    # Create final unified format
    unified_sample = {
        "id": sample_id,
        "metadata": metadata,
        "input": input_obj,
        "output": output_obj,
        "ground_truth": ground_truth_obj
    }
    
    return unified_sample

def main():
    # Load who_when dataset
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        who_when_data = json.load(f)
    
    # Extract actual data samples
    samples = who_when_data.get("data", [])
    print(f"Loaded {len(samples)} samples from who_when dataset")
    
    # Convert samples
    converted_samples = []
    for i, sample in enumerate(samples):
        # Generate unique ID
        sample_id = f"who_when_{i}_{uuid.uuid4().hex[:8]}"
        
        # Convert sample
        converted_sample = convert_sample(sample, sample_id)
        converted_samples.append(converted_sample)
    
    # Write to JSONL file
    output_dir = Path(OUTPUT_FILE).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_samples)} samples")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    # Print sample statistics
    frameworks = set(sample["metadata"]["framework"] for sample in converted_samples)
    benchmarks = set(sample["metadata"]["benchmark"] for sample in converted_samples)
    task_types = set(sample["metadata"]["task_type"] for sample in converted_samples)
    
    print(f"\nStatistics:")
    print(f"  Frameworks: {frameworks}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Task types: {task_types}")
    
    # Check conversation history
    non_empty_history = sum(1 for sample in converted_samples 
                           if sample["input"]["conversation_history"])
    print(f"  Samples with conversation history: {non_empty_history}/{len(converted_samples)}")
    
    # Check faulty agents
    with_faulty_agents = sum(1 for sample in converted_samples 
                            if sample["output"]["faulty_agents"])
    print(f"  Samples with faulty agents: {with_faulty_agents}/{len(converted_samples)}")

if __name__ == "__main__":
    main() 