#!/usr/bin/env python3
"""
Command Line Interface for AEGIS

Provides command-line tools for error injection and model evaluation.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from aegis_core.malicious_factory.fm_malicious_system import MaliciousSystem
from aegis_core.agent_systems import get_mas_wrapper
from aegis_core.utils.utils import load_config
from evaluation.evaluate import evaluate_model


def inject_errors():
    """Command-line interface for error injection."""
    parser = argparse.ArgumentParser(description="Inject errors into multi-agent systems")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--mas-framework", required=True, choices=["dylan", "agentverse", "llm_debate", "macnet"], help="MAS framework to use")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSON file")
    parser.add_argument("--error-modes", nargs="+", default=["FM-1.1", "FM-2.3", "FM-3.2"], help="Error modes to inject")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples per error mode")
    
    args = parser.parse_args()
    
    async def run_injection():
        # Load configuration and tasks
        config = load_config(args.config)
        with open(args.tasks, 'r') as f:
            tasks = json.load(f)
        
        # Initialize components
        mas_wrapper = get_mas_wrapper(args.mas_framework, config)
        malicious_system = MaliciousSystem()
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        total_tasks = len(tasks) * len(args.error_modes)
        completed = 0
        
        print(f"🚀 Starting error injection: {total_tasks} total experiments")
        
        for task in tasks:
            for error_mode in args.error_modes:
                try:
                    result = await malicious_system.inject_and_run(
                        mas_wrapper=mas_wrapper,
                        task=task,
                        error_mode=error_mode,
                        injection_strategy="prompt_injection"
                    )
                    
                    if result.get("injection_successful", False):
                        results.append(result)
                        print(f"✅ {completed+1}/{total_tasks}: {task['id']} + {error_mode}")
                    else:
                        print(f"⚠️  {completed+1}/{total_tasks}: Failed {task['id']} + {error_mode}")
                    
                except Exception as e:
                    print(f"❌ {completed+1}/{total_tasks}: Error in {task['id']} + {error_mode}: {e}")
                
                completed += 1
        
        # Save results
        output_file = output_dir / "injection_results.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"🎉 Completed! Generated {len(results)} labeled examples")
        print(f"💾 Results saved to: {output_file}")
    
    asyncio.run(run_injection())


def evaluate_model_cli():
    """Command-line interface for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate error attribution models")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, choices=["aegis_bench", "whowhen"], help="Evaluation dataset")
    parser.add_argument("--config", help="Model configuration file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--metrics", nargs="+", default=["micro_f1", "macro_f1"], help="Metrics to compute")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_name=args.model,
        dataset=args.dataset,
        config_path=args.config,
        metrics=args.metrics
    )
    
    # Print results
    print("📊 Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "inject":
        inject_errors()
    elif len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        evaluate_model_cli()
    else:
        print("Usage: python cli.py [inject|evaluate] [options]")
        sys.exit(1)