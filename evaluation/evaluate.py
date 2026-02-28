import json
from collections import defaultdict
import pandas as pd
import argparse

def calculate_metrics(true_labels_list, pred_labels_list):
    all_classes = set()
    for labels in true_labels_list:
        all_classes.update(labels)
    for labels in pred_labels_list:
        all_classes.update(labels)
    all_classes = list(all_classes)
    
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    per_class_stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_classes}

    for true_set, pred_set in zip(true_labels_list, pred_labels_list):
        tp_set = true_set.intersection(pred_set)
        fp_set = pred_set.difference(true_set)
        fn_set = true_set.difference(pred_set)

        micro_tp += len(tp_set)
        micro_fp += len(fp_set)
        micro_fn += len(fn_set)

        for cls in tp_set:
            per_class_stats[cls]['tp'] += 1
        for cls in fp_set:
            per_class_stats[cls]['fp'] += 1
        for cls in fn_set:
            per_class_stats[cls]['fn'] += 1
            
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    macro_precision_list, macro_recall_list, macro_f1_list = [], [], []
    for cls in all_classes:
        tp = per_class_stats[cls]['tp']
        fp = per_class_stats[cls]['fp']
        fn = per_class_stats[cls]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        macro_precision_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    macro_precision = sum(macro_precision_list) / len(macro_precision_list) if len(macro_precision_list) > 0 else 0.0
    macro_recall = sum(macro_recall_list) / len(macro_recall_list) if len(macro_recall_list) > 0 else 0.0
    macro_f1 = sum(macro_f1_list) / len(macro_f1_list) if len(macro_f1_list) > 0 else 0.0

    return {
        "precision_micro": micro_precision, "recall_micro": micro_recall, "f1_micro": micro_f1,
        "precision_macro": macro_precision, "recall_macro": macro_recall, "f1_macro": macro_f1,
    }


def evaluate_results(filepath):
    total_samples = 0
    valid_samples = 0
    skipped_samples = 0
    exact_matches = 0

    all_true_globals, all_pred_globals = [], []
    all_true_agents, all_pred_agents = [], []
    all_true_errors, all_pred_errors = [], []
    
    analysis_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_samples += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {total_samples}: {e}")
                print(f"Line content (first 200 chars): {repr(line[:200])}")
                continue
            
            model_raw_output = data.get("model_output", "") 
            
            pred_data = []
            
            if "response_text" in data and data["response_text"]:
                try:
                    response_text = data["response_text"].strip()
                    if response_text.startswith('{') and response_text.endswith('}'):
                        response_json = json.loads(response_text)
                        final_answer_keys = ["final_answer", "Final Answer"]
                        for key in final_answer_keys:
                            if key in response_json and isinstance(response_json[key], dict):
                                if "faulty_agents" in response_json[key]:
                                    pred_data = response_json[key]["faulty_agents"]
                                    break
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
            
            if not pred_data:
                model_detection = data.get("model_detection", {})
                if isinstance(model_detection, dict):
                    final_answer_keys = ["Final Answer", "final_answer"]
                    for key in final_answer_keys:
                        if key in model_detection and isinstance(model_detection[key], dict):
                            if "faulty_agents" in model_detection[key]:
                                pred_data = model_detection[key]["faulty_agents"]
                                break
                    
                    if not pred_data and "faulty_agents" in model_detection:
                        pred_data = model_detection["faulty_agents"]
                        
                elif isinstance(model_detection, list):
                    pred_data = model_detection
                elif isinstance(model_detection, int):
                    pred_data = model_detection
                else:
                    if model_detection is None:
                        pred_data = data.get("qwen_detection", {}).get("faulty_agents", [])
                    else:
                        pred_data = model_detection.get("faulty_agents", [])
            
            gt_data = data.get("ground_truth", {}).get("faulty_agents", [])
            
            if not isinstance(pred_data, list):
                skipped_samples += 1
                continue
            
            valid_samples += 1
            
            true_global_set = set((d['agent_name'], d['error_type']) for d in gt_data)
            true_agent_set = set(d['agent_name'] for d in gt_data)
            true_error_set = set(d['error_type'] for d in gt_data)
            
            pred_global_set = set((d.get('agent_name'), d.get('error_type')) for d in pred_data if isinstance(d, dict)) 
            pred_agent_set = set(d.get('agent_name') for d in pred_data if isinstance(d, dict))
            pred_error_set = set(d.get('error_type') for d in pred_data if isinstance(d, dict))

            all_true_globals.append(true_global_set)
            all_pred_globals.append(pred_global_set)
            all_true_agents.append(true_agent_set)
            all_pred_agents.append(pred_agent_set)
            all_true_errors.append(true_error_set)
            all_pred_errors.append(pred_error_set)

            if true_global_set == pred_global_set:
                exact_matches += 1
            
            analysis_entry = {
                "sample_id": total_samples,
                "model_raw_output": model_raw_output,
                "parsed_prediction": pred_data,
                "ground_truth": gt_data,
                "parsing_success": len(pred_data) > 0 or len(gt_data) == 0,  # 是否成功解析出结果
                "exact_match": true_global_set == pred_global_set
            }
            analysis_data.append(analysis_entry)

    emr = exact_matches / valid_samples if valid_samples > 0 else 0.0
    
    global_metrics = calculate_metrics(all_true_globals, all_pred_globals)
    agent_metrics = calculate_metrics(all_true_agents, all_pred_agents)
    error_metrics = calculate_metrics(all_true_errors, all_pred_errors)

    results = {
        "Overall": {
            "Total Samples": total_samples, 
            # "Valid Samples": valid_samples,
            "Skipped Samples (OOM failures)": skipped_samples,
            "Exact Match Ratio": emr
        },
        "Global Level (Exact Pair Matching)": global_metrics,
        "Agent Level": agent_metrics,
        "Error Type Level": error_metrics,
    }
    
    df = pd.DataFrame.from_dict(results, orient='index')
    print("--- Model Performance Evaluation ---")
    # print(f"Note: Skipped {skipped_samples} samples due to OOM failures")
    print(df.round(4))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection results')
    parser.add_argument('--file_path', type=str, default='baseline/results/ours_qwen25_14b_sft_v2.jsonl', help='Path to the JSONL file')
    args = parser.parse_args()
    
    file_path = args.file_path

    evaluate_results(file_path)