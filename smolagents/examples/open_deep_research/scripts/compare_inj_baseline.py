import argparse
import csv
import json
import re
import time
from pathlib import Path


def is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and injected results on common questions")
    parser.add_argument("--inj", required=True, help="Injected results jsonl path")
    parser.add_argument("--base", required=True, help="Baseline results jsonl path")
    parser.add_argument("--outdir", default="output/validation", help="Directory to write CSV")
    args = parser.parse_args()

    inj_by_q = {}
    for e in load_jsonl(args.inj):
        q = str(e.get("question", ""))
        inj_by_q[q] = {
            "prediction": str(e.get("prediction", "")),
            "true": str(e.get("true_answer", "")),
            "fm": e.get("injection_fm_type"),
            "strategy": e.get("injection_strategy"),
        }

    base_by_q = {}
    inj_qs = set(inj_by_q.keys())
    for e in load_jsonl(args.base):
        q = str(e.get("question", ""))
        if q in inj_qs:
            base_by_q[q] = {
                "prediction": str(e.get("prediction", "")),
                "true": str(e.get("true_answer", "")),
            }

    rows = []
    bs_strict = bs_num = inj_strict = inj_num = total = 0
    for q in inj_qs:
        inj = inj_by_q[q]
        base = base_by_q.get(q, {"prediction": "", "true": inj["true"]})
        bpred = base["prediction"].strip()
        ipred = inj["prediction"].strip()
        gold = inj["true"].strip()

        b_s = 1 if normalize_answer(bpred) == normalize_answer(gold) else 0
        i_s = 1 if normalize_answer(ipred) == normalize_answer(gold) else 0

        b_n = 0
        i_n = 0
        if is_numeric(bpred) and is_numeric(gold):
            try:
                b_n = 1 if abs(float(bpred) - float(gold)) < 1e-6 else 0
            except Exception:
                b_n = 0
        if is_numeric(ipred) and is_numeric(gold):
            try:
                i_n = 1 if abs(float(ipred) - float(gold)) < 1e-6 else 0
            except Exception:
                i_n = 0

        rows.append(
            [
                q,
                gold,
                bpred,
                ipred,
                b_s,
                i_s,
                b_n,
                i_n,
                inj["fm"],
                inj["strategy"],
            ]
        )
        total += 1
        bs_strict += b_s
        inj_strict += i_s
        bs_num += b_n
        inj_num += i_n

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"inj_on_correct_compare_{time.strftime('%m%d-%H%M%S')}.csv"
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "question",
                "true_answer",
                "baseline_prediction",
                "injected_prediction",
                "baseline_strict",
                "injected_strict",
                "baseline_numeric",
                "injected_numeric",
                "fm_type",
                "strategy",
            ]
        )
        w.writerows(rows)

    print(f"TOTAL={total}")
    print(f"BASELINE_STRICT={bs_strict} ACC={bs_strict/total if total else 0:.4f}")
    print(f"INJECTED_STRICT={inj_strict} ACC={inj_strict/total if total else 0:.4f}")
    print(f"BASELINE_NUMERIC={bs_num} ACC={bs_num/total if total else 0:.4f}")
    print(f"INJECTED_NUMERIC={inj_num} ACC={inj_num/total if total else 0:.4f}")
    print(f"CSV={outfile}")


if __name__ == "__main__":
    main()

