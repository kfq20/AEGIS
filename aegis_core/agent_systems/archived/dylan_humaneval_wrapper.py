# file: agent_systems/dylan_humaneval_wrapper.py

import re
import asyncio
from .base_wrapper import SystemWrapper
from malicious_factory.agent import MaliciousAgent
from typing import Any, Dict, Tuple

from methods import get_method_class

class DyLANHumanEvalWrapper(SystemWrapper):
    """
    A wrapper for the DyLAN HumanEval system. It injects malicious behavior by monkey-patching
    the call_llm method and tracking agent activations across multiple rounds.
    Supports role-specific injection for both agents and judges.
    """
    def __init__(self, general_config: Dict[str, Any], method_config: Dict[str, Any]):
        from utils.async_llm import create_llm_instance
        exp_config = general_config.get('experiment_config', {})
        llm_config = exp_config.get('llm_config', {})
        if not llm_config:
            raise ValueError("No LLM configuration found for the specified model name.")

        self.llm = create_llm_instance(llm_config)
        method_name = exp_config['system_under_test']['name']  # "dylan"
        dataset_name = exp_config.get('benchmark_name', 'HumanEval')
        MAS_CLASS = get_method_class(method_name, dataset_name)
        self.dylan_instance = MAS_CLASS(general_config, method_config_name="config_humaneval")

        print(f"DyLANHumanEvalWrapper initialized with {MAS_CLASS.__name__}.")
        print(f"Available agent roles: {self.dylan_instance.agent_roles}")
        print(f"Available judge roles: {self.dylan_instance.judge_roles}")

    def run_with_injection(
        self,
        task: Any,
        malicious_agent: MaliciousAgent,
        injection_target: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:

        target_role = injection_target['role']
        target_role_index = injection_target.get('role_index', 0)

        # Store the original call_llm method
        original_llm_call = self.dylan_instance.call_llm
        
        current_node_id = None

        from malicious_factory import MaliciousAgentFactory
        factory = MaliciousAgentFactory(llm=self.llm)

        # --- Define the malicious call_llm method ---
        def sophisticated_malicious_llm_call(*args, **kwargs):
            nonlocal current_node_id
            
            # 1. Determine the current node and its role/index
            messages = args[2] if len(args) > 2 else kwargs.get('messages', [])
            
            # Find the active node
            if hasattr(self.dylan_instance, 'nodes'):
                for i, node in enumerate(self.dylan_instance.nodes):
                    if node.get('active', False) and node.get('question') == task.query:
                        current_node_id = i
                        break
            
            current_role = None
            current_role_index = None
            current_node_type = None
            
            if hasattr(self.dylan_instance, 'nodes') and current_node_id is not None:
                current_node = self.dylan_instance.nodes[current_node_id]
                current_role = current_node.get('role', None)
                current_node_type = current_node.get('type', None)  # 'agent' or 'judge'
                
                if current_node_type == 'agent':
                    current_role_index = current_node_id % self.dylan_instance.num_agents
                elif current_node_type == 'judge':
                    current_role_index = current_node_id % self.dylan_instance.num_judges
            
            print(f"[DyLAN HumanEval Runner Intercept] Node {current_node_id}: type='{current_node_type}', role='{current_role}', index={current_role_index}")

            is_malicious_call = (current_role == target_role and current_role_index == target_role_index)

            if not is_malicious_call:
                return original_llm_call(*args, **kwargs)

            print(f"*** Malicious Agent Activated on '{current_role}' (index {current_role_index}, node {current_node_id}, type {current_node_type}) ***")
            
            task_input = messages[-1]['content'] if messages else ""
            response = factory.inject_malicious_behavior(
                lambda: original_llm_call(*args, **kwargs),
                malicious_agent,
                task_input=task_input,
                messages=messages
            )
            
            return response

        # --- Apply the monkey patch ---
        self.dylan_instance.call_llm = sophisticated_malicious_llm_call
        print(f"[DyLAN HumanEval Runner] Monkey-patch applied. Target: role='{target_role}', index={target_role_index}.")

        # --- Execute the patched workflow ---
        sample = {"query": task.query}
        final_output = self.dylan_instance.inference(sample)

        # --- Cleanup ---
        self.dylan_instance.call_llm = original_llm_call
        print("[DyLAN HumanEval Runner] Original `call_llm` method restored.")
        
        # Collect history from DyLAN's network structure
        full_history = []
        if hasattr(self.dylan_instance, 'nodes'):
            for i, node in enumerate(self.dylan_instance.nodes):
                if node.get('active', False) and node.get('reply'):
                    role = node.get('role', f'unknown_{i}')
                    node_type = node.get('type', 'unknown')
                    
                    if node_type == 'agent':
                        role_index = i % self.dylan_instance.num_agents
                    elif node_type == 'judge':
                        role_index = i % self.dylan_instance.num_judges
                    else:
                        role_index = 0
                    
                    round_num = i // (self.dylan_instance.num_agents + self.dylan_instance.num_judges)
                    
                    full_history.append({
                        "role": role,
                        "role_index": role_index,
                        "node_type": node_type,
                        "content": node.get('reply', ''),
                        "node_id": i,
                        "round": round_num
                    })
        
        log = {
            "final_output": final_output,
            "full_history": full_history,
            "injected_role": target_role,
            "injected_role_index": target_role_index,
            "malicious_action_description": malicious_agent.description,
            "agent_roles": self.dylan_instance.agent_roles,
            "judge_roles": self.dylan_instance.judge_roles,
        }
        return final_output, log 