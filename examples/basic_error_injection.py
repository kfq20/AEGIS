#!/usr/bin/env python3
"""
Basic Error Injection Example

This script demonstrates how to use AEGIS to inject errors into a multi-agent system
and generate labeled failure trajectories for training error attribution models.
"""

import json
import asyncio
from pathlib import Path

from aegis_core.malicious_factory.fm_malicious_system import MaliciousSystem
from aegis_core.agent_systems.fm_dylan_wrapper import DylanMASWrapper
from aegis_core.utils.utils import load_config


async def basic_error_injection_example():
    """Demonstrate basic error injection workflow."""
    
    # Configuration
    config_path = "configs/config_main.yaml"
    output_dir = Path("outputs/basic_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize MAS wrapper (using DyLAN as example)
    mas_wrapper = DylanMASWrapper(config=config)
    
    # Initialize malicious system
    malicious_system = MaliciousSystem()
    
    # Define example tasks
    tasks = [
        {
            "id": "example_1",
            "type": "math",
            "content": "Solve the equation: 2x + 5 = 17. What is the value of x?",
            "expected_answer": "x = 6"
        },
        {
            "id": "example_2", 
            "type": "reasoning",
            "content": "A bakery sells 3 types of bread. If they sell 12 loaves of wheat, 8 loaves of rye, and 15 loaves of sourdough, how many loaves did they sell in total?",
            "expected_answer": "35 loaves"
        }
    ]
    
    # Define error modes to test
    error_modes = [
        "FM-1.1",  # Task specification deviation
        "FM-2.3",  # Deviate from main goal
        "FM-3.2",  # Remove verification steps
    ]
    
    results = []
    
    print("🚀 Starting AEGIS Error Injection Example")
    print(f"📋 Processing {len(tasks)} tasks with {len(error_modes)} error modes")
    
    for task in tasks:
        print(f"\n📝 Processing task: {task['id']}")
        
        # First, generate a successful baseline trajectory
        try:
            baseline_result = await mas_wrapper.run_task(task)
            if not baseline_result.get("success", False):
                print(f"⚠️  Skipping task {task['id']} - baseline failed")
                continue
                
            print(f"✅ Baseline successful for task {task['id']}")
            
        except Exception as e:
            print(f"❌ Error in baseline for task {task['id']}: {e}")
            continue
        
        # Now inject errors for each error mode
        for error_mode in error_modes:
            print(f"  🎯 Injecting error mode: {error_mode}")
            
            try:
                # Inject error and run task
                injected_result = await malicious_system.inject_and_run(
                    mas_wrapper=mas_wrapper,
                    task=task,
                    error_mode=error_mode,
                    injection_strategy="prompt_injection"  # or "response_corruption"
                )
                
                if injected_result.get("injection_successful", False):
                    # Create labeled training example
                    labeled_example = {
                        "task_id": task["id"],
                        "task_content": task["content"],
                        "error_mode": error_mode,
                        "trajectory": injected_result["trajectory"],
                        "ground_truth_label": injected_result["ground_truth_label"],
                        "baseline_trajectory": baseline_result["trajectory"],
                        "injection_metadata": injected_result["metadata"]
                    }
                    
                    results.append(labeled_example)
                    print(f"    ✅ Successfully injected {error_mode}")
                else:
                    print(f"    ⚠️  Injection failed for {error_mode}")
                    
            except Exception as e:
                print(f"    ❌ Error injecting {error_mode}: {e}")
    
    # Save results
    output_file = output_dir / "error_injection_results.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n🎉 Generated {len(results)} labeled error examples")
    print(f"💾 Results saved to: {output_file}")
    
    # Generate summary statistics
    error_mode_counts = {}
    for result in results:
        mode = result["error_mode"]
        error_mode_counts[mode] = error_mode_counts.get(mode, 0) + 1
    
    print("\n📊 Summary Statistics:")
    for mode, count in error_mode_counts.items():
        print(f"  {mode}: {count} examples")
    
    return results


def create_simple_config():
    """Create a simple configuration file for testing."""
    config = {
        "model": {
            "name": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 2048
        },
        "agents": {
            "num_agents": 3,
            "roles": ["planner", "executor", "critic"]
        },
        "injection": {
            "max_attempts": 3,
            "validation_threshold": 0.8
        }
    }
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    with open(config_dir / "config_main.yaml", 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    print("📝 Created basic configuration file")


if __name__ == "__main__":
    # Create basic config if it doesn't exist
    if not Path("configs/config_main.yaml").exists():
        create_simple_config()
    
    # Run the example
    asyncio.run(basic_error_injection_example())