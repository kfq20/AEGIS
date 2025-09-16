#!/usr/bin/env python3
"""
Convert results from owl/magentic_one/injection_logs into the unified training dataset format.

Policy:
- Only read files ending with _incorrect_only.json (valid injection negatives)
- Parse conversation history from plain-text logs by splitting on
  headers like: ---------- TextMessage (Role) ----------, ---------- MultiModalMessage (Role) ----------
- Use strict metric (normalized string equality) to judge correctness if prediction present
- If no explicit prediction field exists, extract the final answer as the content of the last
  TextMessage (MagenticOneOrchestrator)
- Strictly keep MagenticOne role names as-is (do not map to generic names). For filename agent
  "Orchestrator", normalize to "MagenticOneOrchestrator" to match logs.
- Drop any sample whose faulty agent does not appear in conversation_history.

Outputs:
- data_processing/unified_dataset/unified_training_dataset_magentic+gaia.jsonl
- data_processing/unified_dataset/unified_training_dataset_magentic+gaia.json
- data_processing/unified_dataset/dataset_statistics_magentic+gaia.json

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
from typing import Any, Dict, List, Optional, Tuple

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


def is_strict_correct(prediction: Optional[str], true_answer: Optional[str]) -> bool:
    return normalize_text_for_strict(prediction) == normalize_text_for_strict(true_answer)


def standardize_agent_name(original_name: str) -> str:
    """Keep MagenticOne role names exactly as they appear in logs."""
    if not isinstance(original_name, str) or not original_name:
        return "Unknown"
    return original_name.strip()


def map_filename_agent_to_log_role(agent_from_filename: str) -> str:
    """Normalize filename agent label to actual role label used in logs."""
    if not isinstance(agent_from_filename, str):
        return "Unknown"
    name = agent_from_filename.strip()
    if name == "Orchestrator":
        return "MagenticOneOrchestrator"
    return name


def infer_task_type_from_question(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["math", "calculate", "solve", "equation", "number"]):
        return "math"
    if any(k in q for k in ["code", "program", "function", "implement", "algorithm"]):
        return "code_generation"
    return "reasoning"


def generate_id(benchmark: str, model: str, framework: str, source_key: str) -> str:
    source_str = f"{benchmark}_{model}_{framework}_{source_key}"
    hash_obj = hashlib.md5(source_str.encode())
    return f"{source_str}_{hash_obj.hexdigest()[:8]}"


# ------------------------------
# Log parsing (plain text)
# ------------------------------

_HEADER_RE = re.compile(r"^-{6,}\s*(TextMessage|MultiModalMessage)\s*\(([^)]+)\)\s*-{6,}\s*$")


def parse_conversation_from_logs(log_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Parse conversation history blocks from raw log text.

    Returns:
        - conversation history list of steps
        - roles_to_contents: mapping role -> list of message contents (for convenience)
    """
    if not isinstance(log_text, str) or not log_text:
        return [], {}

    lines = log_text.splitlines()
    blocks: List[Tuple[str, str, List[str]]] = []  # (type, role, content_lines)

    current_type: Optional[str] = None
    current_role: Optional[str] = None
    current_buf: List[str] = []

    def flush():
        nonlocal current_type, current_role, current_buf
        if current_type and current_role is not None:
            # Trim trailing blank lines
            while current_buf and not current_buf[-1].strip():
                current_buf.pop()
            blocks.append((current_type, current_role, current_buf[:]))
        current_type, current_role, current_buf = None, None, []

    for raw_line in lines:
        m = _HEADER_RE.match(raw_line)
        if m:
            # New header starts; flush previous
            flush()
            current_type = m.group(1)
            current_role = m.group(2)
            current_buf = []
        else:
            if current_type is not None:
                current_buf.append(raw_line)
            # else: ignore text outside known blocks

    # Flush final
    flush()

    # Build conversation history
    history: List[Dict[str, Any]] = []
    roles_to_contents: Dict[str, List[str]] = {}
    for idx, (btype, role, content_lines) in enumerate(blocks, start=1):
        content = "\n".join(content_lines).strip()
        std_role = standardize_agent_name(role)
        history.append({
            "step": idx,
            "agent_name": std_role,  # identical to log role
            "agent_role": role,      # raw log role
            "content": content,
            "phase": "reasoning",
        })
        roles_to_contents.setdefault(role, []).append(content)

    return history, roles_to_contents


def extract_final_answer(log_text: str, fallback: Optional[str] = None) -> str:
    """Extract final answer from the last TextMessage (MagenticOneOrchestrator).
    If not present, return fallback or empty string.
    """
    if not isinstance(log_text, str):
        return fallback or ""

    # Find all blocks and keep last orchestrator
    history, _ = parse_conversation_from_logs(log_text)
    last_orch: Optional[str] = None
    for step in history:
        role = (step.get("agent_role") or "").strip()
        if role == "MagenticOneOrchestrator":
            last_orch = step.get("content") or ""
    if last_orch:
        return last_orch.strip()

    return (fallback or "").strip()


# ------------------------------
# File name parsing
# ------------------------------

_FILENAME_RE = re.compile(
    r"^level_(?P<level>[^_]+)_(?P<dataset>[^_]+)_(?P<agent>[^_]+)_(?P<fm>FM-[\d.]+)_(?P<strategy>[^_.]+)"
)


def parse_filename_params(filename: str) -> Dict[str, str]:
    m = _FILENAME_RE.match(filename)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "gaia_level": d.get("level", ""),
        "dataset_type": d.get("dataset", ""),
        "injection_target_agent": d.get("agent", ""),
        "fm_error_type": d.get("fm", ""),
        "injection_strategy": d.get("strategy", ""),
    }


# ------------------------------
# Core conversion
# ------------------------------

def faulty_agent_present_in_history(faulty_agent: str, conversation_history: List[Dict[str, Any]]) -> bool:
    if not faulty_agent:
        return False
    for step in conversation_history:
        role = step.get("agent_role")
        name = step.get("agent_name")
        if role == faulty_agent or name == faulty_agent:
            return True
    return False


def to_unified_sample(sample_key: str, sample: Dict[str, Any], file_params: Dict[str, str], benchmark_name: str) -> Optional[UnifiedTrainingData]:
    question: str = sample.get("question") or ""
    true_answer: Optional[str] = sample.get("correct_answer")
    logs_text: str = sample.get("logs") or ""

    # Parse conversation history from logs
    conversation_history, _roles_map = parse_conversation_from_logs(logs_text)

    # Extract prediction
    model_answer: Optional[str] = sample.get("model_answer")
    final_output: str = model_answer or extract_final_answer(logs_text)

    framework = "magentic_one"
    model = "unknown-model"

    # Metadata
    agent_roles = sorted({step.get("agent_role", "") for step in conversation_history if isinstance(step, dict)})
    num_agents = len([r for r in agent_roles if r]) or 1

    metadata: Dict[str, Any] = {
        "framework": framework,
        "benchmark": benchmark_name,
        "model": model,
        "num_agents": num_agents,
        "num_injected_agents": 1,
        "task_type": infer_task_type_from_question(question),
    }

    # Input
    input_data: Dict[str, Any] = {
        "query": question,
        "conversation_history": conversation_history if conversation_history else ([{
            "step": 1,
            "agent_name": "user",
            "agent_role": "user",
            "content": question,
            "phase": "reasoning",
        }] if question else []),
        "final_output": final_output,
    }

    # Output and ground truth using filename params
    target_agent_raw = file_params.get("injection_target_agent") or "Unknown"
    faulty_agent_role = map_filename_agent_to_log_role(target_agent_raw)
    error_type = file_params.get("fm_error_type") or ""
    injection_strategy = file_params.get("injection_strategy") or "prompt_injection"

    # Validate: faulty agent must appear in conversation history
    # if not faulty_agent_present_in_history(faulty_agent_role, input_data["conversation_history"]):
    #     return None

    output_data: Dict[str, Any] = {
        "faulty_agents": [
            {
                "agent_name": faulty_agent_role,
                "error_type": str(error_type),
                "injection_strategy": str(injection_strategy),
            }
        ]
    }

    ground_truth: Dict[str, Any] = {
        "correct_answer": true_answer or "",
        "injected_agents": [
            {
                "agent_name": faulty_agent_role,
                "error_type": str(error_type),
                "injection_strategy": str(injection_strategy),
                "malicious_action_description": "",  # not available in these logs
            }
        ],
        "is_injection_successful": not is_strict_correct(final_output, true_answer),
    }

    uid = generate_id(benchmark_name, model, framework, sample_key)

    return UnifiedTrainingData(
        id=uid,
        metadata=metadata,
        input=input_data,
        output=output_data,
        ground_truth=ground_truth,
    )


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


def save_dataset(unified_data: List[UnifiedTrainingData], output_dir: str, suffix: str) -> None:
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


# ------------------------------
# Main
# ------------------------------

def collect_from_incorrect_only(input_dir: str, benchmark_name: str) -> List[UnifiedTrainingData]:
    results: List[UnifiedTrainingData] = []
    base = Path(input_dir)
    # files = sorted(p for p in base.glob("*_incorrect_only.json") if p.is_file())
    files = sorted(p for p in base.glob("*_valid.json") if p.is_file())

    for fp in files:
        file_params = parse_filename_params(fp.name)
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        for sample_key, sample in data.items():
            if not isinstance(sample, dict):
                continue
            try:
                unified = to_unified_sample(sample_key, sample, file_params, benchmark_name)
                if unified is not None:
                    results.append(unified)
            except Exception:
                # Skip malformed entries
                continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Build unified dataset from magentic_one logs (incorrect_only)")
    parser.add_argument("--input_dir", default="owl/magentic_one/logs", help="Directory with magentic_one logs")
    parser.add_argument("--output_dir", default="data_processing/unified_dataset_with_normal", help="Output directory")
    parser.add_argument("--benchmark_name", default="magentic+gaia", help="Benchmark/dataset name label to embed in metadata")

    args = parser.parse_args()

    samples = collect_from_incorrect_only(args.input_dir, args.benchmark_name)
    if not samples:
        print("No samples collected from incorrect_only files.")
        return

    save_dataset(samples, args.output_dir, args.benchmark_name)
    print("Saved:")
    print(f"  JSONL: {os.path.join(args.output_dir, f'unified_training_dataset_{args.benchmark_name}.jsonl')}")
    print(f"  JSON:  {os.path.join(args.output_dir, f'unified_training_dataset_{args.benchmark_name}.json')}")
    print(f"  Stats: {os.path.join(args.output_dir, f'dataset_statistics_{args.benchmark_name}.json')}")


if __name__ == "__main__":
    main()