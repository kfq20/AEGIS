import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def try_import_factory():
    """Best-effort import of FMMaliciousFactory utilities."""
    try:
        from gaia_agents.magentic_one.malicious_factory.fm_malicious_system import (
            FMMaliciousFactory,
        )
        return FMMaliciousFactory
    except Exception:
        # Fallback if sys.path not set up when imported elsewhere
        try:
            import sys
            from pathlib import Path as _Path
            project_root = _Path(__file__).resolve().parents[5]
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from gaia_agents.magentic_one.malicious_factory.fm_malicious_system import (
                FMMaliciousFactory,
            )
            return FMMaliciousFactory
        except Exception:
            return None


def load_expected_methods() -> Set[str]:
    """Return expected method_ids like 'FM-2.1_prompt_injection'.

    Uses FMMaliciousFactory.get_all_injection_methods() if available; otherwise, a conservative fallback.
    """
    factory_cls = try_import_factory()
    if factory_cls is not None:
        try:
            methods = factory_cls().get_all_injection_methods()
            return {m["method_id"] for m in methods}
        except Exception:
            pass

    # Fallback: derive from known FM types and the factory's constraints (26 combos)
    fm_all = [
        "FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5",
        "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
        "FM-3.1", "FM-3.2", "FM-3.3",
    ]
    prompt = {f"{fm}_prompt_injection" for fm in fm_all}
    # response_corruption allowed FM types (per factory): exclude FM-1.1, FM-1.2
    resp_fms = [
        "FM-1.3", "FM-1.4", "FM-1.5",
        "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
        "FM-3.1", "FM-3.2", "FM-3.3",
    ]
    resp = {f"{fm}_response_corruption" for fm in resp_fms}
    return prompt.union(resp)


def parse_method_from_filename(path: Path) -> Optional[str]:
    """Try to infer method_id from filename like inj-on-38-FM-2.1_prompt_injection-....jsonl"""
    m = re.search(r"(FM-\d+\.\d+)_(prompt_injection|response_corruption)", path.name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def load_baseline_task_ids(baseline_file: Optional[str], metric: str) -> Set[str]:
    if not baseline_file:
        return set()
    try:
        # Reuse helper to ensure consistency with prior filtering
        from scripts.run_injection_on_correct import load_correct_questions
        ids, _ = load_correct_questions(baseline_file, metric)
        return ids
    except Exception:
        # Fallback: accept any task_id present in baseline file
        task_ids: Set[str] = set()
        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        tid = str(obj.get("task_id", "") or "")
                        if tid:
                            task_ids.add(tid)
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        return task_ids


def summarize_directory(
    directory: str,
    baseline_file: Optional[str] = None,
    metric: str = "strict",
) -> Dict[str, any]:
    """Summarize coverage and trace stats from JSONL files in directory."""
    expected_methods = load_expected_methods()

    dir_path = Path(directory)
    files = sorted([p for p in dir_path.glob("*.jsonl") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No JSONL files found in: {dir_path}")

    # Coverage and stats containers
    coverage: Dict[str, Set[str]] = defaultdict(set)  # task_id -> {method_id}
    methods_to_tasks: Dict[str, Set[str]] = defaultdict(set)
    method_seen_files: Dict[str, Set[str]] = defaultdict(set)
    task_id_to_question: Dict[str, str] = {}

    # Trace stats
    manager_steps_counter: Counter = Counter()
    search_steps_counter: Counter = Counter()
    injection_target_counter: Counter = Counter()

    # Helper for task key
    def _task_key(obj: dict) -> str:
        tid = str(obj.get("task_id", "") or "")
        if tid:
            return tid
        q = str(obj.get("question", "") or "")
        return f"Q::{q[:64]}" if q else "UNKNOWN"

    # Iterate files and entries
    for file_path in files:
        fallback_method = parse_method_from_filename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not obj or not obj.get("injection_enabled"):
                    continue

                method_id = None
                fm = obj.get("injection_fm_type")
                strat = obj.get("injection_strategy")
                if fm and strat:
                    method_id = f"{fm}_{strat}"
                else:
                    method_id = fallback_method
                if not method_id:
                    continue

                task_key = _task_key(obj)
                if not task_key or task_key == "UNKNOWN":
                    continue

                # Map task_id to question (for readability)
                if task_key not in task_id_to_question:
                    task_id_to_question[task_key] = str(obj.get("question", ""))

                # Coverage
                coverage[task_key].add(method_id)
                methods_to_tasks[method_id].add(task_key)
                method_seen_files[method_id].add(file_path.name)

                # Trace stats
                agents = obj.get("agents", {}) or {}
                mgr_steps = agents.get("manager", {}).get("steps_full") or []
                web_steps = agents.get("search_agent", {}).get("steps_full") or []
                manager_steps_counter[method_id] += int(len(mgr_steps))
                search_steps_counter[method_id] += int(len(web_steps))

                # Target agent
                target = obj.get("injection_target_agent") or "(missing)"
                injection_target_counter[target] += 1

    # Establish task universe
    baseline_ids = load_baseline_task_ids(baseline_file, metric)
    if baseline_ids:
        task_universe: Set[str] = set(baseline_ids)
        # Some records may fall back to question-based keys; include them as well
        task_universe.update([k for k in coverage.keys() if k.startswith("Q::")])
    else:
        task_universe = set(coverage.keys())

    # Compute coverage and missing
    methods_expected: Set[str] = set(expected_methods)
    methods_observed: Set[str] = set(methods_to_tasks.keys())
    missing_methods_globally = sorted(list(methods_expected - methods_observed))

    per_task_missing: Dict[str, List[str]] = {}
    for task in task_universe:
        covered = coverage.get(task, set())
        missing = sorted(list(methods_expected - covered))
        if missing:
            per_task_missing[task] = missing

    # Aggregate per-method coverage counts
    method_coverage_counts = {
        m: len(methods_to_tasks.get(m, set())) for m in sorted(methods_expected)
    }

    # Average steps per method (avoid division by zero)
    avg_manager_steps = {}
    avg_search_steps = {}
    for m in methods_expected:
        denom = max(1, method_coverage_counts.get(m, 0))
        avg_manager_steps[m] = manager_steps_counter.get(m, 0) / denom
        avg_search_steps[m] = search_steps_counter.get(m, 0) / denom

    # Final decision: can we exhaustively cover all combinations for all tasks?
    exhaustive = len(per_task_missing) == 0 and not missing_methods_globally

    summary = {
        "directory": str(dir_path.resolve()),
        "num_jsonl_files": len(files),
        "expected_methods_count": len(methods_expected),
        "task_universe_count": len(task_universe),
        "methods_observed_count": len(methods_observed),
        "exhaustive": exhaustive,
        "missing_methods_globally": missing_methods_globally,
        "method_coverage_counts": method_coverage_counts,
        "num_tasks_with_any_coverage": len(coverage),
        "num_tasks_missing_any_methods": len(per_task_missing),
        "per_task_missing": per_task_missing,
        "injection_target_distribution": dict(injection_target_counter),
        "avg_manager_steps_per_method": avg_manager_steps,
        "avg_search_steps_per_method": avg_search_steps,
    }

    return summary


def save_summary(summary: Dict[str, any], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Summarize injection coverage and traces")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing JSONL outputs")
    parser.add_argument("--baseline-file", type=str, default=None, help="Baseline no-injection JSONL to get 38 task_ids")
    parser.add_argument("--metric", type=str, choices=["strict", "numeric"], default="strict")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path for the summary")
    args = parser.parse_args()

    summary = summarize_directory(args.dir, args.baseline_file, args.metric)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        out_json = Path(args.out)
        save_summary(summary, out_json)
        print(f"Saved summary to: {out_json.resolve()}")


if __name__ == "__main__":
    main()

