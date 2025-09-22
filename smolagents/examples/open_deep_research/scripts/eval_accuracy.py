#!/usr/bin/env python3
import argparse
import json
import re
from typing import Tuple


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip("\"'` ")


def evaluate_jsonl(path: str, limit: int | None = None) -> Tuple[int, int, int]:
    total = 0
    strict_correct = 0
    numeric_correct = 0

    with open(path, "r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp, start=1):
            if limit is not None and idx > limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            pred = normalize(rec.get("prediction"))
            truth = normalize(rec.get("true_answer"))

            # Strict: truth substring appears in prediction (case/space-insensitive)
            if truth and truth in pred:
                strict_correct += 1

            # Numeric: all numbers from truth appear in prediction (as tokens)
            nums_pred = set(re.findall(r"-?\d+(?:\.\d+)?", pred))
            nums_truth = re.findall(r"-?\d+(?:\.\d+)?", truth)
            if nums_truth and all(n in nums_pred for n in nums_truth):
                numeric_correct += 1

    return total, strict_correct, numeric_correct


def main():
    parser = argparse.ArgumentParser(description="Evaluate GAIA JSONL accuracy")
    parser.add_argument("--file", required=True, help="Path to the results JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N lines")
    args = parser.parse_args()

    total, strict_correct, numeric_correct = evaluate_jsonl(args.file, args.limit)
    print(f"TOTAL={total}")
    if total > 0:
        print(f"STRICT_CORRECT={strict_correct}")
        print(f"STRICT_ACC={strict_correct/total:.4f}")
        print(f"NUMERIC_CORRECT={numeric_correct}")
        print(f"NUMERIC_ACC={numeric_correct/total:.4f}")
    else:
        print("No records found.")


if __name__ == "__main__":
    main()

