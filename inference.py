import os
import json
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
import traceback

from methods import get_method_class
from utils import reserve_unprocessed_queries, load_model_api_config, write_to_jsonl

def load_existing_results(results_file_path):
    """从已有的结果文件中加载已处理的数据"""
    processed_samples = {}
    if os.path.exists(results_file_path):
        print(f"📖 加载已有结果文件: {results_file_path}")
        with open(results_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data = json.loads(line.strip())
                        if 'query' in data:
                            processed_samples[data['query']] = data
                except json.JSONDecodeError as e:
                    print(f"⚠️ 警告: 第 {line_num} 行 JSON 解析失败: {e}")
                    continue
        
        print(f"✅ 已加载 {len(processed_samples)} 个已处理样本")
    else:
        print(f"📝 结果文件不存在，将创建新文件: {results_file_path}")
    
    return processed_samples

def filter_processed_samples(test_dataset, processed_samples):
    """过滤掉已处理的样本"""
    filtered_dataset = []
    skipped_count = 0
    
    for sample in test_dataset:
        query = sample.get('query', '')
        if query in processed_samples:
            skipped_count += 1
        else:
            filtered_dataset.append(sample)
    
    print(f"📊 数据集统计:")
    print(f"  总样本数: {len(test_dataset)}")
    print(f"  已处理: {skipped_count}")
    print(f"  待处理: {len(filtered_dataset)}")
    
    return filtered_dataset

def collect_conversation_history(mas_instance, framework_name: str) -> dict:
    """收集各个框架的对话历史"""
    history_data = {
        "framework": framework_name,
        "full_history": [],
        "agent_contexts": None,
        "conversation_history": []
    }
    
    try:
        if framework_name == "llm_debate":
            agent_contexts = getattr(mas_instance, 'agent_contexts', None)
            if agent_contexts:
                history_data["agent_contexts"] = agent_contexts
                for agent_index, context in enumerate(agent_contexts):
                    for msg_index, message in enumerate(context):
                        history_data["full_history"].append({
                            "role": f"Assistant {agent_index+1}" if message["role"] == "assistant" else "User",
                            "role_index": agent_index,
                            "content": message["content"],
                            "msg_index": msg_index,
                            "agent_name": f"Assistant {agent_index+1}"
                        })
        
        elif framework_name in ["dylan", "dylan_math", "dylan_humaneval", "dylan_mmlu"]:
            if hasattr(mas_instance, 'nodes'):
                for i, node in enumerate(mas_instance.nodes):
                    if node and node.get('reply'):
                        role = node.get('role', f'agent_{i % getattr(mas_instance, "num_agents", 4)}')
                        role_index = i % getattr(mas_instance, "num_agents", 4)
                        round_num = i // getattr(mas_instance, "num_agents", 4)
                        history_data["full_history"].append({
                            "role": role,
                            "role_index": role_index,
                            "content": node.get('reply', ''),
                            "node_id": i,
                            "round": round_num,
                            "agent_name": role,
                            "active": node.get('active', False)
                        })

            elif hasattr(mas_instance, 'agent_contexts') and mas_instance.agent_contexts:
                for agent_idx, context in enumerate(mas_instance.agent_contexts):
                    for msg_idx, msg in enumerate(context):
                        # if msg['role'] == 'assistant':
                            history_data["full_history"].append({
                                "role": f"Assistant {agent_idx+1}" if msg["role"] == "assistant" else msg["role"],
                                "role_index": agent_idx,
                                "content": msg['content'],
                                "msg_index": msg_idx,
                                "agent_name": msg['role']
                            })

        elif framework_name in ["agentverse", "agentverse_humaneval", "agentverse_mgsm"]:
            if hasattr(mas_instance, 'history'):
                history_data["conversation_history"] = mas_instance.history
                for i, entry in enumerate(mas_instance.history):
                    if isinstance(entry, dict):
                        history_data["full_history"].append({
                            "role": entry.get("role", "Unknown"),
                            "content": entry.get("content", ""),
                            "step": i,
                            "agent_name": entry.get("agent_name", entry.get("role", "Unknown"))
                        })
        
        elif framework_name in ["macnet", "macnet_srdd"]:
            if hasattr(mas_instance, 'execution_history'):
                history_data["full_history"] = mas_instance.execution_history
            elif hasattr(mas_instance, 'nodes'):
                for node_id, node in mas_instance.nodes.items():
                    if hasattr(node, 'generated_answer') and node.generated_answer:
                        history_data["full_history"].append({
                            "role": f"Node{node_id}",
                            "role_index": node_id,
                            "content": node.generated_answer,
                            "node_id": node_id,
                            "agent_name": f"Node{node_id}",
                            "suggestions": getattr(node, 'suggestions', None)
                        })
        
        else:
            for attr_name in ['history', 'conversation_history', 'messages', 'agent_contexts']:
                if hasattr(mas_instance, attr_name):
                    attr_value = getattr(mas_instance, attr_name)
                    if attr_value:
                        history_data[attr_name] = attr_value
                        break
    
    except Exception as e:
        print(f"⚠️ 警告: 收集{framework_name}历史时出错: {e}")
    
    return history_data


def process_sample(args, general_config, sample, output_path, lock):
    MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
    mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
    save_data = sample.copy()
    try:
        mas_output = mas.inference(sample)
        
        response_content = ""
        if isinstance(mas_output, dict):
            if "response" in mas_output:
                response_content = mas_output["response"]
                save_data.update(mas_output)
            else:
                for key, value in mas_output.items():
                    if isinstance(value, str) and value.strip():
                        response_content = value
                        break
                if not response_content:
                    response_content = str(mas_output)
                save_data.update(mas_output)
        else:
            if mas_output is None:
                response_content = ""
            else:
                response_content = str(mas_output)
        
        save_data["response"] = response_content
        
        history_data = collect_conversation_history(mas, args.method_name)
        
        history_data["final_output"] = response_content
        save_data["injection_log"] = history_data
        
        if hasattr(mas, 'history'):
            save_data["history"] = mas.history
        
        save_data["fm_error_type"] = ""
        save_data["injection_strategy"] = ""
        
        save_data["status"] = "success"
        
    except Exception as e:
        save_data["error"] = f"Inference Error: {traceback.format_exc()}"
        save_data["status"] = "error"
    
    try:
        save_data.update({"token_stats": mas.get_token_stats()})
    except:
        save_data["token_stats"] = {}
    
    write_to_jsonl(lock, output_path, save_data)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args related to the method
    parser.add_argument("--method_name", type=str, default="agentverse", help="MAS name.")
    parser.add_argument("--method_config_name", type=str, default=None, help="The config file name. If None, the default config file will be used.")

    # args related to the model
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="The agent backend to be used for inference.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_temperature", type=float, default=0, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    
    # args related to dataset
    parser.add_argument("--test_dataset_name", type=str, default="math", help="The dataset to be used for testing.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to the output file.")
    parser.add_argument("--require_val", action="store_true")
    parser.add_argument("--use_subset", action="store_true", help="Use subset data from datasets/data/subset/ (smaller, framework-specific datasets)")
    parser.add_argument("--subset_framework", type=str, default=None, 
                       help="Framework name for subset data (agentverse, dylan, llm_debate, macnet). "
                            "If not specified, will auto-detect from method_name. "
                            "Available files: {DATASET}+{FRAMEWORK}.json")
    
    parser.add_argument("--resume_from", type=str, default=None, help="从指定的结果文件继续处理")
    parser.add_argument("--resume", action="store_true", help="自动检测并继续处理最新的结果文件")
    
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    
    general_config = vars(args)
    
    # Load model config
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})
    print("-"*50, f"\n>> Model API config: {model_api_config[args.model_name]}")
    
    if args.debug:
        # MAS inference
        sample = {"query": "If $|x+5|-|3x-6|=0$, find the largest possible value of $x$. Express your answer as an improper fraction. Only output the answer, without any explanation."}
        MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
        mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)

        response = mas.inference(sample)
        
        print(json.dumps(response, indent=4))
        print(f"\n>> Token stats: {json.dumps(mas.get_token_stats(), indent=4)}")
    
    else:
        print(f">> Method: {args.method_name} | Dataset: {args.test_dataset_name}")

        # load dataset
        if args.use_subset:
            if args.subset_framework:
                subset_framework = args.subset_framework
            else:
                framework_mapping = {
                    "dylan": "dylan",
                    "dylan_math": "dylan", 
                    "dylan_humaneval": "dylan",
                    "dylan_mmlu": "dylan",
                    "agentverse": "agentverse",
                    "agentverse_humaneval": "agentverse",
                    "agentverse_mgsm": "agentverse",
                    "llm_debate": "llm_debate",
                    "macnet": "macnet",
                    "macnet_srdd": "macnet"
                }
                subset_framework = framework_mapping.get(args.method_name, args.method_name)
            
            dataset_name_upper = args.test_dataset_name.upper()
            subset_file = f"./datasets/data/subset/{dataset_name_upper}+{subset_framework}.json"
            
            if os.path.exists(subset_file):
                print(f"📂 使用subset数据: {subset_file}")
                with open(subset_file, "r") as f:
                    test_dataset = json.load(f)
            else:
                print(f"⚠️ Subset文件不存在: {subset_file}")
                print(f"💡 回退到原始数据集: ./datasets/data/{args.test_dataset_name}.json")
                with open(f"./datasets/data/{args.test_dataset_name}.json", "r") as f:
                    test_dataset = json.load(f)
        else:
            with open(f"./datasets/data/{args.test_dataset_name}.json", "r") as f:
                test_dataset = json.load(f)
        
        if args.require_val:
            val_dataset_path = f"./datasets/data/{args.test_dataset_name}_val.json"
            if not os.path.exists(val_dataset_path):
                raise FileNotFoundError(f"Validation dataset not found at {val_dataset_path}. Please provide a valid path.")
            with open(val_dataset_path, "r") as f:
                val_dataset = json.load(f)
        
        import time
        if args.output_path is not None:
            output_path = args.output_path
        elif args.resume_from is not None:
            output_path = args.resume_from
        elif args.resume:
            results_dir = f"./results_right/{args.test_dataset_name}/{args.model_name}"
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.startswith(f"{args.method_name}_infer_") and f.endswith('.jsonl')]
                if result_files:
                    result_files.sort(reverse=True)
                    output_path = os.path.join(results_dir, result_files[0])
                    print(f"🔄 自动检测到最新结果文件: {output_path}")
                else:
                    output_path = f"./results_right/{args.test_dataset_name}/{args.model_name}/{args.method_name}_infer_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            else:
                output_path = f"./results_right/{args.test_dataset_name}/{args.model_name}/{args.method_name}_infer_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        else:
            output_path = f"./results_right/{args.test_dataset_name}/{args.model_name}/{args.method_name}_infer_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        processed_samples = load_existing_results(output_path)
        
        test_dataset = filter_processed_samples(test_dataset, processed_samples)
        
        if len(test_dataset) == 0:
            print("✅ 所有样本都已处理完成！")
            exit(0)
        
        # optimize mas if required (e.g., GPTSwarm, ADAS, and AFlow)
        if args.require_val:
            # get MAS instance
            MAS_METHOD = get_method_class(args.method_name, args.test_dataset_name)
            mas = MAS_METHOD(general_config, method_config_name=args.method_config_name)
            mas.optimizing(val_dataset)
        
        # inference the mas
        lock = threading.Lock()
        if args.sequential:
            for sample in tqdm(test_dataset, desc="Processing queries"):
                process_sample(args, general_config, sample, output_path, lock)
        else:
            max_workers = model_api_config[args.model_name]["max_workers"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for _ in tqdm(executor.map(lambda sample: process_sample(args, general_config, sample, output_path, lock), test_dataset), total=len(test_dataset), desc="Processing queries"):
                    pass