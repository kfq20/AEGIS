#!/usr/bin/env python3
"""
Automated GAIA Malicious Injection Data Collection Script
自动化GAIA恶意注入数据收集脚本

Usage:
    python auto_collect_gaia_data.py --level all --dataset valid --count 100
    
This script automatically:
1. Randomly selects injection strategies, FM error types, and target agents
2. Runs gaia.py with different configurations
3. Collects specified amount of data with batch processing
4. Handles progress tracking and resume functionality
"""

import argparse
import json
import random
import os
import subprocess
import time
from typing import Dict, Tuple

# Available configurations based on gaia.py
TARGET_AGENTS = ["Orchestrator", "WebSurfer", "FileSurfer", "Coder", "ComputerTerminal"]

FM_ERROR_TYPES = [
    # FM-1.x: Specification Errors
    "FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5",
    # FM-2.x: Misalignment Errors  
    "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
    # FM-3.x: Verification Errors
    "FM-3.1", "FM-3.2", "FM-3.3"
]

INJECTION_STRATEGIES = ["prompt_injection"]

GAIA_LEVELS = ["1", "2", "3", "all"]
DATASET_TYPES = ["valid", "test"]
CAPTURE_MODES = ["stream", "memory", "direct", "none"]

def get_random_config() -> Tuple[str, str, str]:
    """Randomly select target agent, FM error type, and injection strategy"""
    target_agent = random.choice(TARGET_AGENTS)
    fm_error_type = random.choice(FM_ERROR_TYPES)
    injection_strategy = random.choice(INJECTION_STRATEGIES)
    return target_agent, fm_error_type, injection_strategy

def get_injection_metadata(target_agent: str, fm_error_type: str, injection_strategy: str, 
                          level: str, dataset_type: str) -> Dict[str, any]:
    """Generate comprehensive injection metadata for result recording"""
    return {
        # Injection configuration
        "target_agent": target_agent,
        "fm_error_type": fm_error_type,
        "injection_strategy": injection_strategy,
        
        # GAIA configuration
        "gaia_level": level,
        "dataset_type": dataset_type,
        
        # Available options for context
        "available_target_agents": TARGET_AGENTS,
        "available_fm_types": FM_ERROR_TYPES,
        "available_strategies": INJECTION_STRATEGIES,
        
        # Generation info
        "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "random_seed_info": "uniformly_random_selection"
    }

def get_output_path() -> str:
    """Get output directory path for injection logs"""
    output_dir = f"./magentic_one/injection_logs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def count_existing_results(output_dir: str, level: str, dataset_type: str) -> int:
    """Count existing result files to avoid duplication.
    Only counts final result files, skipping intermediate evaluation files like *_evaluated.json
    and *_incorrect_only.json.
    """
    if not os.path.exists(output_dir):
        return 0
    
    count = 0
    pattern = f"level_{level}_{dataset_type}_"
    
    for filename in os.listdir(output_dir):
        # Only consider final result files (exclude evaluated and incorrect_only artifacts)
        if not (filename.startswith(pattern) and filename.endswith(".json")):
            continue
        if filename.endswith("_evaluated.json") or filename.endswith("_incorrect_only.json"):
            continue
        
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Count completed tasks
                if isinstance(data, dict):
                    count += len(data)
        except Exception:
            continue
    return count

def check_experiment_exists(output_dir: str, level: str, dataset_type: str, 
                           target_agent: str, fm_error_type: str, injection_strategy: str,
                           min_tasks: int = 1) -> bool:
    """Check if experiment with specific configuration already exists with sufficient data"""
    if not os.path.exists(output_dir):
        return False
    
    # 构建预期的文件名（与 gaia.py 中的命名规则一致）
    filename = f"level_{level}_{dataset_type}_{target_agent}_{fm_error_type}_{injection_strategy}.json"
    filepath = os.path.join(output_dir, filename)
    
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            task_count = len(data)
            print(f"  📁 发现已存在配置文件: {filename} ({task_count} 个任务)")
            return task_count >= min_tasks
    except Exception:
        return False

def get_subset_file_path(level: str) -> str:
    """Get path to subset file based on level"""
    if level == "all":
        return os.getenv("GAIA_SUBSET_DIR", "./data/gaia_subset") + "/gaia_subset_all_levels.json"
    else:
        return os.getenv("GAIA_SUBSET_DIR", "./data/gaia_subset") + f"/gaia_subset_level_{level}.json"

def check_subset_file(level: str) -> bool:
    """Check if subset file exists"""
    subset_path = get_subset_file_path(level)
    return os.path.exists(subset_path)

def run_single_experiment(level: str, dataset_type: str, target_agent: str, 
                         fm_error_type: str, injection_strategy: str,
                         capture_mode: str = "stream", randomize: bool = True, 
                         batch_size: int = 1) -> bool:
    """Run a single gaia.py experiment with given parameters"""
    
    # Build command
    cmd = [
        "python", "magentic_one/gaia.py",
        "--inject", "True",
        "--target-agent", target_agent,
        "--fm-type", fm_error_type,
        "--injection-strategy", injection_strategy,
        "--level", level,
        "--on", dataset_type,
        "--capture-mode", capture_mode,
        "--limit", str(batch_size)  # 🎯 每次实验处理指定数量的任务
    ]
    
    if randomize:
        cmd.append("--randomize")
    
    print(f"🔧 Running: {' '.join(cmd)}")
    
    try:
        # Run the command with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print("✅ Experiment completed successfully")
            return True
        else:
            print(f"❌ Experiment failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out after 2 hours")
        return False
    except Exception as e:
        print(f"💥 Error running experiment: {e}")
        return False

def run_batch_collection(level: str, dataset_type: str, target_count: int,
                        capture_mode: str = "stream", randomize: bool = True, 
                        batch_size: int = 1) -> None:
    """Run batch data collection with multiple random configurations"""
    
    # Validate inputs
    if level not in GAIA_LEVELS:
        raise ValueError(f"Invalid level: {level}. Available: {GAIA_LEVELS}")
    
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Available: {DATASET_TYPES}")
    
    # Check subset file exists
    if not check_subset_file(level):
        print(f"⚠️  Warning: Subset file not found for level {level}")
        subset_path = get_subset_file_path(level)
        print(f"   Expected: {subset_path}")
    
    # Get output directory
    output_dir = get_output_path()
    
    # Check existing results
    existing_count = count_existing_results(output_dir, level, dataset_type)
    print(f"📊 Existing results: {existing_count}")
    
    if existing_count >= target_count:
        print(f"✅ Target count ({target_count}) already reached!")
        return
    
    remaining_count = target_count - existing_count
    print(f"🎯 Need to collect {remaining_count} more samples")
    
    collected = existing_count
    experiment_num = 1
    skip_count = 0
    
    while collected < target_count:
        print(f"\n🚀 Starting Experiment {experiment_num}")
        
        # Generate random configuration
        target_agent, fm_error_type, injection_strategy = get_random_config()
        
        print(f"🎲 Random config: {target_agent} + {fm_error_type} + {injection_strategy}")
        print(f"📋 Target: Level {level} {dataset_type} dataset (batch_size: {batch_size})")
        
        # 🔄 断点重续检查：如果该配置已存在足够的数据，跳过
        if check_experiment_exists(output_dir, level, dataset_type, target_agent, 
                                 fm_error_type, injection_strategy, min_tasks=batch_size):
            print(f"  ⏭️  配置已存在足够数据，跳过实验 {experiment_num}")
            skip_count += 1
            experiment_num += 1
            
            # 如果连续跳过太多次，可能需要增加随机性或停止
            if skip_count >= 50:
                print(f"  ⚠️  连续跳过 {skip_count} 个实验，可能大部分配置已完成")
                break
            continue
        
        # 重置跳过计数
        skip_count = 0
        
        # Generate metadata for tracking
        metadata = get_injection_metadata(
            target_agent, fm_error_type, injection_strategy, level, dataset_type
        )
        
        # Create metadata file for this experiment
        metadata_path = os.path.join(output_dir, f"experiment_{experiment_num}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 🔄 执行前最后检查：防止并发执行导致的重复
        print(f"🔍 执行前最后检查: {target_agent} + {fm_error_type} + {injection_strategy}")
        if check_experiment_exists(output_dir, level, dataset_type, target_agent, 
                                 fm_error_type, injection_strategy, min_tasks=batch_size):
            print(f"  ⚠️  执行前发现配置已存在足够数据，跳过执行")
            experiment_num += 1
            continue
        
        # Run experiment
        success = run_single_experiment(
            level=level,
            dataset_type=dataset_type,
            target_agent=target_agent,
            fm_error_type=fm_error_type,
            injection_strategy=injection_strategy,
            capture_mode=capture_mode,
            randomize=randomize,
            batch_size=batch_size
        )
        
        if success:
            # Update collected count
            new_count = count_existing_results(output_dir, level, dataset_type)
            experiment_collected = new_count - collected
            collected = new_count
            
            # 🔍 执行后验证：检查该配置实际产生的数据量
            actual_tasks = 0
            filename = f"level_{level}_{dataset_type}_{target_agent}_{fm_error_type}_{injection_strategy}.json"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        actual_tasks = len(data)
                except Exception:
                    pass
            
            print(f"✅ Experiment {experiment_num} completed:")
            print(f"   📊 该配置实际任务数: {actual_tasks}")
            print(f"   📈 新增样本数: {experiment_collected} (总计: {collected}/{target_count})")
        else:
            print(f"❌ Experiment {experiment_num} failed, continuing...")
        
        experiment_num += 1
        
        # Safety check to avoid infinite loops
        if experiment_num > 100:
            print("⚠️  Reached maximum experiment limit (100). Stopping.")
            break
        
        # Brief pause between experiments
        time.sleep(2)
    
    print(f"\n🎉 Data collection completed!")
    print(f"📊 Final count: {collected}/{target_count}")
    print(f"📁 Results saved in: {output_dir}")

def list_available_configs():
    """List all available configuration options"""
    print("🔧 Available Configuration Options:\n")
    print(f"🎯 Target Agents: {TARGET_AGENTS}")
    print(f"🧠 FM Error Types: {FM_ERROR_TYPES}")
    print(f"💉 Injection Strategies: {INJECTION_STRATEGIES}")
    print(f"📊 GAIA Levels: {GAIA_LEVELS}")
    print(f"📁 Dataset Types: {DATASET_TYPES}")
    print(f"📝 Capture Modes: {CAPTURE_MODES}")

def main():
    parser = argparse.ArgumentParser(description="Automated GAIA Malicious Injection Data Collection")
    parser.add_argument("--level", type=str, 
                       choices=GAIA_LEVELS,
                       default="all",
                       help="GAIA level: 1, 2, 3, or 'all'")
    parser.add_argument("--dataset", type=str,
                       choices=DATASET_TYPES,
                       default="valid",
                       help="Dataset type: valid or test")
    parser.add_argument("--count", type=int,
                       default=50,
                       help="Target number of experiments to run")
    parser.add_argument("--batch-size", type=int,
                       default=1,
                       help="Number of tasks per experiment batch (default: 1)")
    parser.add_argument("--capture-mode", type=str,
                       choices=CAPTURE_MODES,
                       default="stream",
                       help="Log capture mode")
    parser.add_argument("--no-randomize", action="store_true",
                       help="Disable task randomization")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available configuration options")
    
    args = parser.parse_args()
    
    if args.list_configs:
        list_available_configs()
        return
    
    print(f"🎯 Target: Run {args.count} experiments for GAIA level {args.level} {args.dataset} dataset")
    print(f"⚙️  Capture mode: {args.capture_mode}")
    print(f"📦 Batch size: {args.batch_size} tasks per experiment")
    print(f"🎲 Randomize tasks: {not args.no_randomize}")
    print(f"🧪 Configurations will be randomly selected from:")
    print(f"   • {len(TARGET_AGENTS)} target agents")
    print(f"   • {len(FM_ERROR_TYPES)} FM error types") 
    print(f"   • {len(INJECTION_STRATEGIES)} injection strategies")
    
    # Check if subset file exists
    subset_path = get_subset_file_path(args.level)
    if os.path.exists(subset_path):
        print(f"📁 Subset file found: {subset_path}")
    else:
        print(f"⚠️  Subset file not found: {subset_path}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return
    
    try:
        run_batch_collection(
            level=args.level,
            dataset_type=args.dataset,
            target_count=args.count,
            capture_mode=args.capture_mode,
            randomize=not args.no_randomize,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()