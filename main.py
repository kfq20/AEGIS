# file: main_fm.py

import yaml
import argparse
import asyncio
import importlib
import json
import os
import threading
import concurrent.futures
import traceback
from tqdm import tqdm

# Import our framework's components
from core.task import Task
# Assume utility functions for loading/writing data exist
from methods import get_method_class
from utils import reserve_unprocessed_queries, write_to_jsonl, load_model_api_config
from malicious_factory import FMMaliciousFactory, FMMaliciousAgent, FMErrorType, InjectionStrategy

def import_class(class_path: str):
    """Dynamically imports a class from a string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import class '{class_name}' from '{module_path}'. Error: {e}")

async def process_sample(args, sample: dict, lock: threading.Lock, output_path: str, exp_config: dict = None):
    """
    This function contains the core logic of our FM Malicious Injection framework.
    It processes a single data sample against a full FM injection experiment configuration.
    """
    if exp_config is None:
        print(f"--- Loading Experiment Config: {args.experiment_config} ---")
        with open(args.experiment_config, 'r') as f:
            exp_config = yaml.safe_load(f)
    
    general_config = vars(args).copy()
    model_api_config = load_model_api_config(args.model_api_config, args.model_name)
    general_config.update({"model_api_config": model_api_config})
    general_config.update({"experiment_config": exp_config})
    save_data = sample.copy()
    try:
        # 1. Load method-specific config
        method_config_path = exp_config['system_under_test']['method_config_path']
        
        # 🎯 为 MacNet 使用自动生成的配置
        if exp_config.get('macnet_config'):
            print(f"🤖 [MacNet] 使用自动生成的配置，type: {exp_config['macnet_config']['type']}")
            method_config = exp_config['macnet_config']
        else:
            # 其他框架使用原始配置文件
            with open(method_config_path, 'r') as f:
                method_config = yaml.safe_load(f)

        # For simplicity, we create the LLM config here. This can be expanded.
        llm_config = exp_config["llm_config"]

        # 2. Instantiate the Task from the benchmark sample
        task = Task(
            query=sample['query'],
            ground_truth=sample.get('answer') or sample.get('ground_truth'),
            metadata={'dataset': exp_config.get('benchmark_name'), 'id': sample.get('id')}
        )

        # 3. Instantiate the correct System Wrapper (每个线程独立创建)
        WrapperClass = import_class(exp_config['system_under_test']['wrapper_class_path'])
        system_wrapper = WrapperClass(
            general_config=general_config,
            method_config=method_config
        )

        # 4. Create the FM Malicious Agent(s) via the Factory
        fm_factory = FMMaliciousFactory(llm=system_wrapper.llm)
        
        # Parse injection configuration - support both single and multi-target formats
        injection_plan = exp_config['injection_plan']
        
        # Determine if we have single target (legacy) or multiple targets (new format)
        if 'target' in injection_plan:
            # Legacy single target format - convert to multi-target format for consistency
            target = injection_plan['target']
            attack_spec = injection_plan.get('attack_spec', {})
            
            # Move attack_spec into target if it's at the top level (legacy format)
            if 'attack_spec' not in target:
                target = target.copy()
                target['attack_spec'] = attack_spec
            
            injection_targets = [target]
        else:
            # New multi-target format
            injection_targets = injection_plan['targets']
        
        # Create malicious agents for each target
        malicious_agents = []
        for target in injection_targets:
            target_attack_spec = target.get('attack_spec', {})
            fm_error_type = target_attack_spec.get('fm_error_type', 'FM-2.3')
            injection_strategy = target_attack_spec.get('injection_strategy', 'prompt_injection')
            
            malicious_agent = await fm_factory.create_agent(
                task_query=task.query,
                target_agent_role=target.get('role', 'Assistant'),
                target_agent_index=target.get('role_index', 0),
                fm_error_type=fm_error_type,
                injection_strategy=injection_strategy
            )
            malicious_agents.append(malicious_agent)

        # 5. Run the full FM injection experiment with multiple agents
        final_output, log = system_wrapper.run_with_multi_injection(
            task=task,
            malicious_agents=malicious_agents,
            injection_targets=injection_targets
        )
        
        # 6. Collate results
        # 处理 final_output 的格式，避免嵌套的 response 字段
        if isinstance(final_output, dict) and 'response' in final_output:
            # 如果 final_output 是 {"response": "answer"} 格式，直接提取 answer
            save_data['response'] = final_output['response']
            # 将其他字段也保存到 save_data 中
            for key, value in final_output.items():
                if key != 'response':
                    save_data[key] = value
        else:
            # 如果 final_output 是字符串或其他格式，直接保存
            save_data['response'] = final_output
        
        save_data['injection_log'] = log
        
        # Handle multiple malicious agents
        if len(malicious_agents) == 1:
            # Single agent - maintain backward compatibility
            save_data['fm_error_type'] = malicious_agents[0].fm_error_type.value
            save_data['injection_strategy'] = malicious_agents[0].injection_strategy.value
        else:
            # Multiple agents - store as arrays
            save_data['fm_error_types'] = [agent.fm_error_type.value for agent in malicious_agents]
            save_data['injection_strategies'] = [agent.injection_strategy.value for agent in malicious_agents]
            save_data['num_injected_agents'] = len(malicious_agents)
        
        save_data['status'] = 'success'

    except Exception:
        save_data["error"] = f"Processing Error: {traceback.format_exc()}"
        save_data['status'] = 'error'
    
    # 7. Write results to file incrementally and safely
    write_to_jsonl(lock, output_path, save_data)

async def main(args):
    """Main entry point to load configs and orchestrate the FM experiment batch."""
    
    # Determine dataset file path
    if args.dataset_file:
        dataset_path = args.dataset_file
        print(f"--- Loading Dataset from custom file: {dataset_path} ---")
    else:
        dataset_path = f"./datasets/data/{args.test_dataset_name}.json"
        print(f"--- Loading Dataset: {args.test_dataset_name} ---")
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Load experiment config to get injection information
    print(f"--- Loading Experiment Config: {args.experiment_config} ---")
    with open(args.experiment_config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # Extract injection information for filename - support both single and multi-target
    injection_plan = exp_config.get('injection_plan', {})
    
    if 'target' in injection_plan:
        # Single target format
        injection_target = injection_plan['target']
        attack_spec = injection_plan.get('attack_spec', injection_target.get('attack_spec', {}))
        
        target_role = injection_target.get('role', 'unknown')
        target_index = injection_target.get('role_index', injection_target.get('call_index', 0))
        fm_error_type = attack_spec.get('fm_error_type', 'unknown')
        injection_strategy = attack_spec.get('injection_strategy', 'unknown')
        
        # Create agent identifier for filename
        if target_role == 'unknown':
            agent_identifier = f"agent{target_index+1}"
        else:
            agent_identifier = f"{target_role.lower()}{target_index+1}"
            
        multi_agent_suffix = ""
    else:
        # Multi-target format
        injection_targets = injection_plan['targets']
        
        # Create compound identifier for multiple targets
        agent_identifiers = []
        fm_error_types = []
        injection_strategies = []
        
        for target in injection_targets:
            target_role = target.get('role', 'unknown')
            target_index = target.get('role_index', target.get('call_index', 0))
            target_attack_spec = target.get('attack_spec', {})
            
            if target_role == 'unknown':
                target_identifier = f"agent{target_index+1}"
            else:
                target_identifier = f"{target_role.lower()}{target_index+1}"
            
            agent_identifiers.append(target_identifier)
            fm_error_types.append(target_attack_spec.get('fm_error_type', 'unknown'))
            injection_strategies.append(target_attack_spec.get('injection_strategy', 'unknown'))
        
        # For multi-agent, create compound identifiers
        agent_identifier = "multi_" + "-".join(agent_identifiers)
        fm_error_type = "-".join(fm_error_types)
        injection_strategy = "-".join(injection_strategies)
        multi_agent_suffix = f"_n{len(injection_targets)}"
    
    # Determine dataset name for output path
    if args.dataset_file:
        # Extract dataset name from file path
        dataset_name = os.path.splitext(os.path.basename(args.dataset_file))[0]
        # If it's a subset file like "HumanEval+dylan", keep the full name
        if '+' in dataset_name:
            base_dataset = dataset_name.split('+')[0]
        else:
            base_dataset = dataset_name
    else:
        dataset_name = args.test_dataset_name
        base_dataset = args.test_dataset_name
    
    # Determine output path, using a default if not provided
    import time
    if args.output_path is not None:
        output_path = args.output_path
    else:
        # Enhanced filename with FM injection information (without timestamp for consistency)
        filename = f"{args.method_name}_fm_{agent_identifier}_{fm_error_type}_{injection_strategy}{multi_agent_suffix}.jsonl"
        output_path = f"./results_inj/{base_dataset}/{args.model_name}/{filename}"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    print(f"--- Output will be saved to: {output_path} ---")
    
    # Display injection information
    if 'target' in injection_plan:
        print(f"--- FM Injection: {target_role} {target_index+1}, Type: {fm_error_type}, Strategy: {injection_strategy} ---")
    else:
        print(f"--- Multi-Agent FM Injection ({len(injection_targets)} targets): ---")
        for i, target in enumerate(injection_targets):
            t_role = target.get('role', 'unknown')
            t_index = target.get('role_index', target.get('call_index', 0))
            t_attack_spec = target.get('attack_spec', {})
            t_fm_error = t_attack_spec.get('fm_error_type', 'unknown')
            t_strategy = t_attack_spec.get('injection_strategy', 'unknown')
            print(f"    Target {i+1}: {t_role} {t_index+1}, Type: {t_fm_error}, Strategy: {t_strategy}")

    # Filter out already processed samples to allow resuming runs
    unprocessed_samples = reserve_unprocessed_queries(output_path, dataset)
    
    # Apply limit if specified
    if args.limit is not None and len(unprocessed_samples) > args.limit:
        unprocessed_samples = unprocessed_samples[:args.limit]
        print(f">> Applied limit: Processing only {args.limit} samples")
    
    print(f">> Total samples: {len(dataset)} | Unprocessed: {len(unprocessed_samples)}")

    if not unprocessed_samples:
        print("All samples have already been processed. Exiting.")
        return

    # Create a lock for thread-safe file writing
    lock = threading.Lock()

    # Define a wrapper for asyncio compatibility with ThreadPoolExecutor
    def run_async_process(sample):
        asyncio.run(process_sample(args, sample, lock, output_path, exp_config))

    if args.sequential or args.debug:
        for sample in tqdm(unprocessed_samples, desc="Processing sequentially"):
            await process_sample(args, sample, lock, output_path, exp_config)
    else:
        # 检查并行处理的线程安全
        if args.max_workers > 1:
            print(f"⚠️  Warning: Running with {args.max_workers} workers in parallel.")
            print("   Each thread will create its own wrapper instance to ensure thread safety.")
            print("   Make sure your LLM provider supports concurrent requests.")
        
        # Use ThreadPoolExecutor for parallel execution, similar to MASLab
        max_workers = args.max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(run_async_process, unprocessed_samples), total=len(unprocessed_samples), desc=f"Processing in parallel (max_workers={max_workers})"))
    
    print(f"\n--- FM EXPERIMENT BATCH COMPLETE ---")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FM Malicious Injection & Evaluation Pipeline.")
    parser.add_argument("--method_name", type=str, default="dylan", help="MAS name.")
    parser.add_argument("--experiment_config", type=str, default="./experiments/fm_injection/test_fm_dylan_math.yaml", help="Path to the experiment YAML configuration file.")
    parser.add_argument("--model_api_config", type=str, default="model_api_configs/model_api_config.json")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-azure", help="The agent backend to be used for inference.")
    parser.add_argument("--method_config_name", type=str, default=None, help="The config file name. If None, the default config file will be used.")
    parser.add_argument("--model_temperature", type=float, default=0, help="Temperature for sampling.")
    parser.add_argument("--model_max_tokens", type=int, default=2048, help="Maximum tokens for sampling.")
    parser.add_argument("--model_timeout", type=int, default=600, help="Timeout for sampling.")
    parser.add_argument("--test_dataset_name", type=str, default="MATH")
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to custom dataset file. If provided, overrides test_dataset_name.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output .jsonl file.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel workers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process.")
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially instead of in parallel.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (currently same as sequential).")
    
    args = parser.parse_args()

    asyncio.run(main(args)) 