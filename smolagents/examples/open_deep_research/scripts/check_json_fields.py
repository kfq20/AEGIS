import json
import sys
from pathlib import Path


def safe_len(v):
    try:
        return len(v) if v is not None else 0
    except Exception:
        return 0


def trunc(s: str, n: int = 140):
    if isinstance(s, str):
        return s[:n] + ("..." if len(s) > n else "")
    return s


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_json_fields.py <jsonl-file>")
        sys.exit(2)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"File not found: {p}")
        sys.exit(1)
    last = None
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                last = line
    if not last:
        print("Empty file")
        sys.exit(0)
    obj = json.loads(last)

    agents = obj.get("agents", {})
    manager = agents.get("manager", {})
    search = agents.get("search_agent", {})
    steps_m = manager.get("steps_full") or []
    steps_s = search.get("steps_full") or []

    fields = [
        "agents",
        "intermediate_steps",
        "injection_enabled",
        "injection_fm_type",
        "injection_strategy",
        "injection_instruction",
        "clean_prediction_before_corruption",
        "injected_question",
        "token_counts",
    ]
    print("question_snippet:", trunc(obj.get("question", "")))
    for k in fields:
        print(k, "=", ("present" if k in obj and obj[k] is not None else "missing_or_null"))
    print("manager.system_prompt_len:", safe_len(manager.get("system_prompt")))
    print("manager.steps_full.len:", len(steps_m))
    print("search.system_prompt_len:", safe_len(search.get("system_prompt")))
    print("search.steps_full.len:", len(steps_s))
    if steps_m:
        print("sample_manager_step_keys:", sorted([k for k in steps_m[0].keys() if k != "observations_images"]))
    if steps_s:
        print("sample_search_step_keys:", sorted([k for k in steps_s[0].keys() if k != "observations_images"]))


if __name__ == "__main__":
    main()

