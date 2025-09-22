import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Set, Tuple

from dotenv import load_dotenv

# 复用主评测模块与可视化工具
from scripts.visual_qa import visualizer
import scripts.reformulator as _unused  # noqa: F401 触发依赖加载
import run_gaia_full as gaia


def is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def load_correct_questions(baseline_file: str, metric: str) -> Tuple[Set[str], Set[str]]:
    """Return sets of (task_ids, questions) that are correct in baseline."""
    correct_task_ids: Set[str] = set()
    correct_questions: Set[str] = set()
    with open(baseline_file, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            q = str(entry.get("question", ""))
            tid = str(entry.get("task_id", ""))
            pred = str(entry.get("prediction", "")).strip()
            gold = str(entry.get("true_answer", "")).strip()
            if metric == "strict":
                if normalize_answer(pred) == normalize_answer(gold):
                    correct_questions.add(q)
                    if tid:
                        correct_task_ids.add(tid)
            elif metric == "numeric":
                if is_numeric(pred) and is_numeric(gold):
                    try:
                        if abs(float(pred) - float(gold)) < 1e-6:
                            correct_questions.add(q)
                            if tid:
                                correct_task_ids.add(tid)
                    except Exception:
                        pass
            else:
                # 默认严格匹配
                if normalize_answer(pred) == normalize_answer(gold):
                    correct_questions.add(q)
                    if tid:
                        correct_task_ids.add(tid)
    return correct_task_ids, correct_questions


def main():
    parser = argparse.ArgumentParser(description="对基线正确任务进行二次注入评测")
    parser.add_argument("--baseline-file", type=str, required=True, help="不注入基线输出 JSONL 文件路径")
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--metric", type=str, choices=["strict", "numeric"], default="strict")
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["union", "ids_only", "questions_only"],
        default="union",
        help="如何从基线正确集中选择样本：仅按task_id、仅按question或两者并集",
    )
    parser.add_argument("--fm-type", type=str, default="FM-1.1")
    parser.add_argument(
        "--injection-strategy",
        type=str,
        default="prompt_injection",
        choices=["prompt_injection", "response_corruption"],
    )
    parser.add_argument(
        "--injection-target-agent",
        type=str,
        default="manager",
        choices=["manager", "search_agent"],
        help="选择注入目标 agent（manager 或 search_agent）",
    )
    parser.add_argument(
        "--disable-injection",
        action="store_true",
        help="禁用注入，按基线方式运行并生成带 steps_full 的无注入轨迹",
    )
    args = parser.parse_args()

    load_dotenv(override=True)

    # 读取基线正确题目（优先按 task_id 匹配，回退按 question 匹配）
    correct_task_ids, correct_questions = load_correct_questions(args.baseline_file, args.metric)
    if not correct_task_ids and not correct_questions:
        print("No correct questions found from baseline. Exit.")
        return
    print(f"Found baseline-correct items -> task_ids={len(correct_task_ids)}, questions={len(correct_questions)} ({args.metric}).")

    # 加载数据集并过滤（支持 ids_only / questions_only / union）
    eval_ds = gaia.load_gaia_dataset(False, args.set_to_run)
    ds_list = eval_ds.to_list()
    by_id = [row for row in ds_list if str(row.get("task_id", "")) in correct_task_ids]
    by_q = [row for row in ds_list if row.get("question") in correct_questions]

    if args.selection_mode == "ids_only":
        examples = by_id
        print(f"Selected mode=ids_only, by_id={len(by_id)}")
    elif args.selection_mode == "questions_only":
        examples = by_q
        print(f"Selected mode=questions_only, by_question={len(by_q)}")
    else:
        # union
        seen_ids = set()
        seen_qs = set()
        examples = []
        for row in by_id + by_q:
            tid = str(row.get("task_id", ""))
            q = row.get("question")
            if tid:
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                examples.append(row)
            else:
                if q in seen_qs:
                    continue
                seen_qs.add(q)
                examples.append(row)
        print(f"Selected mode=union, by_id={len(by_id)}, by_question={len(by_q)}, merged_unique={len(examples)}")
    if args.max_tasks is not None and args.max_tasks > 0:
        examples = examples[: args.max_tasks]
    if not examples:
        print("No examples matched baseline correct questions. Exit.")
        return
    print(f"Selected {len(examples)} examples for injection run.")

    # 输出文件
    answers_file = f"output/{args.set_to_run}/{args.run_name}.jsonl"
    Path(f"output/{args.set_to_run}").mkdir(parents=True, exist_ok=True)

    # 并发执行注入评测
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = []
        for ex in examples:
            futures.append(
                exe.submit(
                    gaia.answer_single_question,
                    ex,
                    args.model_id,
                    answers_file,
                    visualizer,
                    False,  # use_open_models
                    None,   # open_provider
                    None,   # open_model_id
                    not args.disable_injection,   # injection_enabled
                    args.fm_type,
                    args.injection_strategy,
                    args.injection_target_agent,
                )
            )
        for f in as_completed(futures):
            f.result()

    print("Injection run finished.")


if __name__ == "__main__":
    main()

