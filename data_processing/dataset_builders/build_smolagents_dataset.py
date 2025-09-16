#!/usr/bin/env python3
"""
Convert results from results_inj/smoagents_logs into the unified training dataset format.

Evaluation policy:
- Use strict metric (normalized string equality) to judge correctness
- Only include incorrect samples (strict == False) as negative samples

Outputs:
- data_processing/unified_dataset/unified_training_dataset_smol+gaia.jsonl
- data_processing/unified_dataset/unified_training_dataset_smol+gaia.json
- data_processing/unified_dataset/dataset_statistics_smol+gaia.json

You can change input/output via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class UnifiedTrainingData:
    id: str
    metadata: Dict[str, Any]
    input: Dict[str, Any]
    output: Dict[str, Any]
    ground_truth: Dict[str, Any]


# ------------------------------
# Utility functions
# ------------------------------

def normalize_text_for_strict(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def is_strict_correct(prediction: str | None, true_answer: str | None) -> bool:
    return normalize_text_for_strict(prediction) == normalize_text_for_strict(true_answer)


def standardize_agent_name(original_name: str) -> str:
    if not isinstance(original_name, str) or not original_name:
        return "Unknown"
    name = original_name.strip()
    
    # Convert assistant to manager
    if name.lower() == "assistant":
        return "Manager"
    
    # Convert tool_call, tool_response, and search_agent to searcher
    if name.lower() in ["tool-call", "tool-response", "search_agent"]:
        return "Searcher"
    
    # Simple normalization consistent with existing processors' spirit
    return name.title()


def infer_task_type_from_question(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["math", "calculate", "solve", "equation", "number"]):
        return "math"
    if any(k in q for k in ["code", "program", "function", "implement", "algorithm"]):
        return "code_generation"
    return "reasoning"


def generate_id(benchmark: str, model: str, framework: str, line_number: int) -> str:
    source_str = f"{benchmark}_{model}_{framework}_{line_number}"
    hash_obj = hashlib.md5(source_str.encode())
    return f"{source_str}_{hash_obj.hexdigest()[:8]}"


def to_unified_sample(sample: Dict[str, Any], line_number: int, benchmark_name: str) -> UnifiedTrainingData:
    # Extract key fields from smoagents log schema
    question: str = sample.get("question") or sample.get("augmented_question") or ""
    prediction_clean: str | None = sample.get("clean_prediction_before_corruption")
    prediction_raw: str | None = sample.get("prediction")
    prediction: str | None = prediction_clean if (isinstance(prediction_clean, str) and prediction_clean.strip()) else (
        prediction_raw if isinstance(prediction_raw, str) else None
    )
    true_answer: str | None = sample.get("true_answer")

    framework = "smoagents"
    model = sample.get("agent_name") or "unknown-model"

    # Metadata
    agents_field = sample.get("agents")
    num_agents = len(agents_field) if isinstance(agents_field, list) else (agents_field if isinstance(agents_field, int) else 1)
    prompt_injection_applied = bool(sample.get("prompt_injection_applied", False))

    metadata: Dict[str, Any] = {
        "framework": framework,
        "benchmark": benchmark_name,  # Use unique dataset name as benchmark label
        "model": model,
        "num_agents": num_agents,
        "num_injected_agents": 1 if prompt_injection_applied else 0,
        "task_type": infer_task_type_from_question(question),
    }

    # Input
    conversation_history: List[Dict[str, Any]] = []
    # Prefer detailed interaction history if available
    interm = sample.get("intermediate_steps")
    if isinstance(interm, list) and interm:
        for si, step in enumerate(interm, 1):
            if not isinstance(step, dict):
                continue
            role = step.get("role", "Assistant")
            content = step.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            conversation_history.append({
                "step": si,
                "agent_name": standardize_agent_name(role),
                "agent_role": role,
                "content": content,
                "phase": "reasoning",
            })
    else:
        # Fallback: synthesize a minimal single turn with the question
        if question:
            conversation_history.append({
                "step": 1,
                "agent_name": "User",
                "agent_role": "User",
                "content": question,
                "phase": "reasoning",
            })

    input_data: Dict[str, Any] = {
        "query": question,
        "conversation_history": conversation_history,
        "final_output": prediction or "",
    }

    # Output (faulty agents)
    faulty_agent_name = sample.get("injection_target_agent") or sample.get("injection_target_agent_index") or model
    error_type = sample.get("injection_fm_type") or sample.get("fm_error_type") or ""
    injection_strategy = sample.get("injection_strategy") or sample.get("injection_instruction") or "prompt_injection"

    output_data: Dict[str, Any] = {
        "faulty_agents": [
            {
                "agent_name": standardize_agent_name(str(faulty_agent_name)),
                "error_type": str(error_type),
                "injection_strategy": str(injection_strategy),
            }
        ]
    }

    # Ground truth
    ground_truth: Dict[str, Any] = {
        "correct_answer": true_answer or "",
        "injected_agents": [
            {
                "agent_name": standardize_agent_name(str(faulty_agent_name)),
                "error_type": str(error_type),
                "injection_strategy": str(injection_strategy),
                "malicious_action_description": sample.get("injection_instruction", ""),
            }
        ] if prompt_injection_applied else [],
        # Mark success when injection was applied and strict metric deems answer incorrect
        "is_injection_successful": prompt_injection_applied and (not is_strict_correct(prediction, true_answer)),
    }

    return UnifiedTrainingData(
        id=generate_id(benchmark_name, model, framework, line_number),
        metadata=metadata,
        input=input_data,
        output=output_data,
        ground_truth=ground_truth,
    )


def save_dataset(unified_data: List[UnifiedTrainingData], output_dir: str, suffix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"unified_training_dataset_{suffix}" if suffix else "unified_training_dataset"

    # JSONL
    jsonl_path = os.path.join(output_dir, f"{base_name}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in unified_data:
            f.write(json.dumps({
                "id": item.id,
                "metadata": item.metadata,
                "input": item.input,
                "output": item.output,
                "ground_truth": item.ground_truth,
            }, ensure_ascii=False) + "\n")

    # JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "id": item.id,
                "metadata": item.metadata,
                "input": item.input,
                "output": item.output,
                "ground_truth": item.ground_truth,
            } for item in unified_data
        ], f, ensure_ascii=False, indent=2)

    # Stats
    stats = generate_statistics(unified_data)
    stats_path = os.path.join(output_dir, f"dataset_statistics_{suffix}.json" if suffix else "dataset_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return jsonl_path


def generate_statistics(unified_data: List[UnifiedTrainingData]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "总样本数": len(unified_data),
        "按框架分布": {},
        "按数据集分布": {},
        "按任务类型分布": {},
        "按错误类型分布": {},
        "按注入策略分布": {},
        "多智能体注入分布": {},
    }

    for item in unified_data:
        metadata = item.metadata
        output = item.output

        framework = metadata.get("framework", "unknown")
        stats["按框架分布"][framework] = stats["按框架分布"].get(framework, 0) + 1

        benchmark = metadata.get("benchmark", "unknown")
        stats["按数据集分布"][benchmark] = stats["按数据集分布"].get(benchmark, 0) + 1

        task_type = metadata.get("task_type", "unknown")
        stats["按任务类型分布"][task_type] = stats["按任务类型分布"].get(task_type, 0) + 1

        for agent in output.get("faulty_agents", []):
            et = agent.get("error_type", "")
            stats["按错误类型分布"][et] = stats["按错误类型分布"].get(et, 0) + 1
            stg = agent.get("injection_strategy", "")
            stats["按注入策略分布"][stg] = stats["按注入策略分布"].get(stg, 0) + 1

        num_injected = metadata.get("num_injected_agents", 0)
        stats["多智能体注入分布"][str(num_injected)] = stats["多智能体注入分布"].get(str(num_injected), 0) + 1

    return stats


# ------------------------------
# Main processing
# ------------------------------

def collect_negative_samples(input_dir: str, benchmark_name: str) -> List[UnifiedTrainingData]:
    negatives: List[UnifiedTrainingData] = []
    input_path = Path(input_dir)
    jsonl_files = sorted(p for p in input_path.glob("*.jsonl") if p.is_file())

    total = 0
    kept = 0
    for file_path in jsonl_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total += 1

                    # Choose prediction string
                    pred_clean = obj.get("clean_prediction_before_corruption")
                    pred_raw = obj.get("prediction")
                    pred = pred_clean if (isinstance(pred_clean, str) and pred_clean.strip()) else (
                        pred_raw if isinstance(pred_raw, str) else None
                    )
                    true_ans = obj.get("true_answer")

                    if is_strict_correct(pred, true_ans):
                        continue  # only collect incorrect ones

                    negatives.append(to_unified_sample(obj, line_no, benchmark_name))
                    kept += 1
        except Exception:
            # Skip unreadable files
            continue

    print(f"Processed lines: {total}, kept negatives (strict wrong): {kept}")
    return negatives


def main():
    parser = argparse.ArgumentParser(description="Build unified dataset from smoagents logs using strict metric (collect negatives only)")
    parser.add_argument("--input_dir", default="results_inj/smoagents_logs", help="Directory with smoagents JSONL logs")
    parser.add_argument("--output_dir", default="data_processing/unified_dataset", help="Output directory")
    parser.add_argument("--benchmark_name", default="smol+gaia", help="Benchmark/dataset name label to embed in metadata")

    args = parser.parse_args()

    negatives = collect_negative_samples(args.input_dir, args.benchmark_name)
    if not negatives:
        print("No negative samples found. Nothing to write.")
        return

    suffix = args.benchmark_name
    path = save_dataset(negatives, args.output_dir, suffix)
    print("Saved:")
    print(f"  JSONL: {os.path.join(args.output_dir, f'unified_training_dataset_{suffix}.jsonl')}")
    print(f"  JSON:  {os.path.join(args.output_dir, f'unified_training_dataset_{suffix}.json')}")
    print(f"  Stats: {os.path.join(args.output_dir, f'dataset_statistics_{suffix}.json')}")


if __name__ == "__main__":
    main()

