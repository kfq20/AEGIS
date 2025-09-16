#!/usr/bin/env python3
"""
Multi-Framework Evaluation Example

This script demonstrates how to compare error attribution performance
across different Multi-Agent System frameworks using AEGIS.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

from aegis_core.malicious_factory.fm_malicious_system import MaliciousSystem
from aegis_core.agent_systems import get_mas_wrapper
from aegis_core.utils.utils import load_config
from evaluation.evaluate import evaluate_model


async def multi_framework_evaluation():
    """Compare error attribution across multiple MAS frameworks."""
    
    # Configuration
    frameworks = ["dylan", "agentverse", "llm_debate", "macnet"]
    error_modes = ["FM-1.1", "FM-2.3", "FM-3.2"]
    output_dir = Path("outputs/multi_framework")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample tasks for each framework
    tasks = [
        {
            "id": "math_1",
            "type": "math", 
            "content": "Calculate the derivative of f(x) = 3x² + 2x - 1",
            "expected_answer": "f'(x) = 6x + 2"
        },
        {
            "id": "reasoning_1",
            "type": "reasoning",
            "content": "A train leaves Station A at 2:00 PM traveling at 60 mph. Another train leaves Station B at 2:30 PM traveling at 80 mph toward Station A. If the stations are 200 miles apart, when will the trains meet?",
            "expected_answer": "The trains will meet at 3:15 PM"
        }
    ]
    
    results = {}
    malicious_system = MaliciousSystem()
    
    print("🔄 Running Multi-Framework Evaluation")
    print(f"🎯 Testing {len(frameworks)} frameworks with {len(error_modes)} error modes")
    
    for framework in frameworks:
        print(f"\n🧪 Testing framework: {framework}")
        framework_results = {
            "success_rate": 0,
            "error_modes": {},
            "injection_examples": []
        }
        
        try:
            # Load framework-specific config
            config_path = f"configs/config_main.yaml"
            config = load_config(config_path)
            
            # Initialize MAS wrapper
            mas_wrapper = get_mas_wrapper(framework, config)
            
            total_attempts = 0
            successful_injections = 0
            
            for task in tasks:
                for error_mode in error_modes:
                    total_attempts += 1
                    print(f"  📝 {task['id']} + {error_mode}")
                    
                    try:
                        # Attempt error injection
                        result = await malicious_system.inject_and_run(
                            mas_wrapper=mas_wrapper,
                            task=task,
                            error_mode=error_mode,
                            injection_strategy="prompt_injection"
                        )
                        
                        if result.get("injection_successful", False):
                            successful_injections += 1
                            framework_results["injection_examples"].append({
                                "task_id": task["id"],
                                "error_mode": error_mode,
                                "trajectory": result["trajectory"],
                                "ground_truth": result["ground_truth_label"]
                            })
                            print(f"    ✅ Success")
                        else:
                            print(f"    ⚠️  Failed")
                            
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
            
            # Calculate success rate
            framework_results["success_rate"] = successful_injections / total_attempts if total_attempts > 0 else 0
            
            # Count by error mode
            for example in framework_results["injection_examples"]:
                mode = example["error_mode"]
                framework_results["error_modes"][mode] = framework_results["error_modes"].get(mode, 0) + 1
            
            results[framework] = framework_results
            print(f"  📊 Success rate: {framework_results['success_rate']:.2%}")
            
        except Exception as e:
            print(f"  ❌ Framework {framework} failed: {e}")
            results[framework] = {"error": str(e)}
    
    # Save detailed results
    results_file = output_dir / "framework_comparison.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary report
    generate_comparison_report(results, output_dir)
    
    # Create visualizations
    create_comparison_plots(results, output_dir)
    
    return results


def generate_comparison_report(results: Dict[str, Any], output_dir: Path):
    """Generate a comprehensive comparison report."""
    
    report_lines = [
        "# Multi-Framework Error Injection Comparison Report\n",
        "## Summary\n"
    ]
    
    # Overall statistics
    successful_frameworks = [f for f, r in results.items() if "success_rate" in r]
    avg_success_rate = sum(results[f]["success_rate"] for f in successful_frameworks) / len(successful_frameworks) if successful_frameworks else 0
    
    report_lines.extend([
        f"- **Frameworks tested:** {len(results)}",
        f"- **Successfully tested:** {len(successful_frameworks)}",
        f"- **Average success rate:** {avg_success_rate:.2%}\n",
        "## Framework Performance\n"
    ])
    
    # Per-framework results
    for framework, framework_results in results.items():
        if "error" in framework_results:
            report_lines.append(f"### {framework.upper()}")
            report_lines.append(f"❌ **Failed:** {framework_results['error']}\n")
            continue
            
        success_rate = framework_results.get("success_rate", 0)
        error_modes = framework_results.get("error_modes", {})
        
        report_lines.extend([
            f"### {framework.upper()}",
            f"- **Success rate:** {success_rate:.2%}",
            f"- **Total injections:** {len(framework_results.get('injection_examples', []))}",
            "- **Error mode distribution:**"
        ])
        
        for mode, count in error_modes.items():
            report_lines.append(f"  - {mode}: {count} examples")
        
        report_lines.append("")
    
    # Error mode analysis
    report_lines.extend([
        "## Error Mode Analysis\n",
        "Error mode success rates across frameworks:\n"
    ])
    
    # Collect error mode statistics
    error_mode_stats = {}
    for framework, framework_results in results.items():
        if "error_modes" not in framework_results:
            continue
        for mode, count in framework_results["error_modes"].items():
            if mode not in error_mode_stats:
                error_mode_stats[mode] = []
            error_mode_stats[mode].append(count)
    
    for mode, counts in error_mode_stats.items():
        avg_count = sum(counts) / len(counts) if counts else 0
        total_count = sum(counts)
        report_lines.append(f"- **{mode}:** {total_count} total, {avg_count:.1f} average per framework")
    
    # Save report
    report_file = output_dir / "comparison_report.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))
    
    print(f"📄 Report saved to: {report_file}")


def create_comparison_plots(results: Dict[str, Any], output_dir: Path):
    """Create visualization plots for the comparison results."""
    
    # Extract data for plotting
    frameworks = []
    success_rates = []
    
    for framework, framework_results in results.items():
        if "success_rate" in framework_results:
            frameworks.append(framework)
            success_rates.append(framework_results["success_rate"])
    
    if not frameworks:
        print("⚠️  No data available for plotting")
        return
    
    # Create plots
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Success rate comparison
    bars = ax1.bar(frameworks, success_rates, color='skyblue', alpha=0.7)
    ax1.set_title('Error Injection Success Rate by Framework')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.2%}', ha='center', va='bottom')
    
    # Error mode distribution heatmap
    error_modes = ["FM-1.1", "FM-2.3", "FM-3.2"]
    framework_mode_matrix = []
    
    for framework in frameworks:
        framework_results = results[framework]
        mode_counts = []
        for mode in error_modes:
            count = framework_results.get("error_modes", {}).get(mode, 0)
            mode_counts.append(count)
        framework_mode_matrix.append(mode_counts)
    
    if framework_mode_matrix:
        sns.heatmap(framework_mode_matrix, 
                   xticklabels=error_modes,
                   yticklabels=frameworks,
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd',
                   ax=ax2)
        ax2.set_title('Error Mode Distribution by Framework')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "framework_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to: {plot_file}")


if __name__ == "__main__":
    asyncio.run(multi_framework_evaluation())